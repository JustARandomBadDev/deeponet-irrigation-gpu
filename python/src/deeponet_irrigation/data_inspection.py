from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data_download import DATASET_DOI, DATASET_SLUG, DATASET_VERSION

CHUNK_SIZE = 100_000


class DatasetInspectionError(RuntimeError):
    """Raised when the local dataset cannot be inspected as documented."""


@dataclass(frozen=True)
class TableRule:
    pattern: str
    timestamp_columns: tuple[str, ...] = ()
    sector_source: str | None = None


# These names and semantics come from the supplied DATA_DICTIONARY.md, not
# column-name heuristics. An unknown CSV is deliberately rejected.
TABLE_RULES = (
    TableRule(
        "01_raw_data/raw_sensor/MEASUREMENT_*.csv",
        ("measurement_date",),
        "node_id",
    ),
    TableRule(
        "01_raw_data/raw_irrigation_actuators/raw_actuators.csv",
        ("time",),
        "node_id",
    ),
    TableRule(
        "01_raw_data/raw_irrigation_actuators/raw_measurements.csv",
        ("time",),
        "node_id",
    ),
    TableRule(
        "01_raw_data/raw_irrigation_actuators/raw_nominal_flow_rates.csv",
        sector_source="node_id",
    ),
    TableRule(
        "02_processed_data/irrigation_events/final_irrigation_events.csv",
        ("start", "end"),
        "column",
    ),
    TableRule(
        "02_processed_data/irrigation_events/sector_statistics_report.csv",
        sector_source="column",
    ),
    TableRule("02_processed_data/merged/dataset_zone_*.csv", ("ts",), "filename"),
    TableRule(
        "02_processed_data/preprocessed/dataset_zone_*_preprocessed.csv",
        ("ts",),
        "filename",
    ),
)


@dataclass
class TableSummary:
    rows: int
    columns: list[str]
    dtypes: dict[str, str]
    missing: dict[str, int]
    timestamp_min: pd.Timestamp | None
    timestamp_max: pd.Timestamp | None
    sectors: list[int]


def _format_size(size: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{size} B"
        value /= 1024
    raise AssertionError("unreachable")


def _rule_for(relative_path: Path) -> TableRule:
    for rule in TABLE_RULES:
        if relative_path.match(rule.pattern):
            return rule
    raise DatasetInspectionError(
        f"No documentation-backed inspection rule for {relative_path.as_posix()}"
    )


def _sectors_from_nodes(nodes: set[str]) -> set[int]:
    sectors: set[int] = set()
    patterns = (
        re.compile(r"^(?:SUT|AIRR)0*(\d+)$"),
        re.compile(r"^SNPK0*(\d+)_(?:CE|PH)$"),
    )
    for node in nodes:
        for pattern in patterns:
            match = pattern.fullmatch(node)
            if match:
                sectors.add(int(match.group(1)))
                break
    return sectors


def _inspect_csv(path: Path, relative_path: Path) -> TableSummary:
    rule = _rule_for(relative_path)
    rows = 0
    columns: list[str] | None = None
    dtype_sets: dict[str, set[str]] = {}
    missing: dict[str, int] = {}
    timestamp_min: pd.Timestamp | None = None
    timestamp_max: pd.Timestamp | None = None
    sectors: set[int] = set()
    node_ids: set[str] = set()

    filename_match = re.search(r"dataset_zone_(\d+)", path.name)
    if rule.sector_source == "filename":
        if not filename_match:
            raise DatasetInspectionError(
                f"Documented sector filename pattern does not match {path.name}"
            )
        sectors.add(int(filename_match.group(1)))

    try:
        chunks = pd.read_csv(path, chunksize=CHUNK_SIZE)
        for chunk in chunks:
            if columns is None:
                columns = list(chunk.columns)
                missing = dict.fromkeys(columns, 0)
                dtype_sets = {column: set() for column in columns}
                required = set(rule.timestamp_columns)
                if rule.sector_source == "column":
                    required.add("sector")
                if rule.sector_source == "node_id":
                    required.add("fk_iot_node")
                absent = sorted(required - set(columns))
                if absent:
                    raise DatasetInspectionError(
                        f"{relative_path} is missing documented columns: {', '.join(absent)}"
                    )
            elif list(chunk.columns) != columns:
                raise DatasetInspectionError(
                    f"Column layout changes within {relative_path.as_posix()}"
                )

            rows += len(chunk)
            chunk_missing = chunk.isna().sum()
            for column in columns:
                missing[column] += int(chunk_missing[column])
                dtype_sets[column].add(str(chunk[column].dtype))

            for column in rule.timestamp_columns:
                timestamps = pd.to_datetime(chunk[column], errors="coerce")
                current_min = timestamps.min()
                current_max = timestamps.max()
                if pd.notna(current_min):
                    timestamp_min = (
                        current_min
                        if timestamp_min is None
                        else min(timestamp_min, current_min)
                    )
                if pd.notna(current_max):
                    timestamp_max = (
                        current_max
                        if timestamp_max is None
                        else max(timestamp_max, current_max)
                    )

            if rule.sector_source == "column":
                numeric = pd.to_numeric(chunk["sector"], errors="coerce").dropna()
                sectors.update(int(value) for value in numeric.unique())
            elif rule.sector_source == "node_id":
                node_ids.update(chunk["fk_iot_node"].dropna().astype(str).unique())
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DatasetInspectionError(f"Could not inspect {relative_path}: {exc}") from exc

    if columns is None:
        raise DatasetInspectionError(f"CSV has no header: {relative_path.as_posix()}")
    if node_ids:
        sectors.update(_sectors_from_nodes(node_ids))

    dtypes = {
        column: "/".join(sorted(observed_types))
        for column, observed_types in dtype_sets.items()
    }
    return TableSummary(
        rows=rows,
        columns=columns,
        dtypes=dtypes,
        missing=missing,
        timestamp_min=timestamp_min,
        timestamp_max=timestamp_max,
        sectors=sorted(sectors),
    )


def _extract_markdown_field(text: str, field: str) -> str | None:
    match = re.search(rf"^\*\*{re.escape(field)}:\*\*\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def _print_documentation_check(dataset_root: Path, metadata_path: Path) -> None:
    readme_path = dataset_root / "README.md"
    dictionary_path = dataset_root / "DATA_DICTIONARY.md"
    try:
        readme = readme_path.read_text(encoding="utf-8")
        dictionary = dictionary_path.read_text(encoding="utf-8")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DatasetInspectionError(f"Could not read dataset documentation: {exc}") from exc

    required_dictionary_sections = (
        "## 1. Raw layer",
        "## 2. Processed layer",
        "## 5. Node identifier scheme",
    )
    missing_sections = [
        heading for heading in required_dictionary_sections if heading not in dictionary
    ]
    if missing_sections:
        raise DatasetInspectionError(
            "DATA_DICTIONARY.md is missing expected sections: "
            + ", ".join(missing_sections)
        )

    declared_version = _extract_markdown_field(readme, "Version")
    declared_doi = _extract_markdown_field(readme, "DOI")
    print("Documentation")
    print(f"  README.md: {_format_size(readme_path.stat().st_size)}")
    print(f"  DATA_DICTIONARY.md: {_format_size(dictionary_path.stat().st_size)}")
    print(f"  Local provenance: DOI {metadata.get('doi')}, version {metadata.get('version')}")
    print(f"  Author README declares: Version {declared_version}; DOI {declared_doi}")
    if declared_version != str(DATASET_VERSION) or DATASET_DOI not in (declared_doi or ""):
        print(
            "  WARNING: the README's internal version/DOI fields do not match the "
            "Mendeley v2 release metadata."
        )
    documented_but_absent = [
        name
        for name in ("LICENSE.txt", "CITATION.cff", "file_manifest.csv")
        if name in readme and not (dataset_root / name).exists()
    ]
    if documented_but_absent:
        print(
            "  WARNING: README package entries absent from the downloaded archive: "
            + ", ".join(documented_but_absent)
        )
    print("  Table semantics below use DATA_DICTIONARY.md's explicit schemas.")


def _print_tree(dataset_root: Path) -> None:
    print("\nFiles")
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file() or path.name == ".extraction.json":
            continue
        relative = path.relative_to(dataset_root)
        print(f"  {relative.as_posix()} ({_format_size(path.stat().st_size)})")


def _print_summary(relative_path: Path, path: Path, summary: TableSummary) -> None:
    print(f"\n{relative_path.as_posix()} ({_format_size(path.stat().st_size)})")
    print(f"  shape: {summary.rows:,} rows x {len(summary.columns)} columns")
    print("  columns: " + ", ".join(summary.columns))
    print(
        "  dtypes: "
        + ", ".join(f"{column}={dtype}" for column, dtype in summary.dtypes.items())
    )
    nonzero_missing = {
        column: count for column, count in summary.missing.items() if count
    }
    if nonzero_missing:
        print(
            "  missing: "
            + ", ".join(
                f"{column}={count:,}" for column, count in nonzero_missing.items()
            )
        )
    else:
        print("  missing: none")
    if summary.timestamp_min is not None and summary.timestamp_max is not None:
        print(f"  timestamp range: {summary.timestamp_min} to {summary.timestamp_max}")
    if summary.sectors:
        print("  sectors: " + ", ".join(map(str, summary.sectors)))


def inspect_dataset(project_root: Path) -> None:
    """Inspect all documented CSV files in the local Arnesano v2 release."""
    dataset_root = project_root / "data" / "raw" / DATASET_SLUG
    metadata_path = project_root / "data" / "raw" / f"{DATASET_SLUG}.metadata.json"
    if not dataset_root.is_dir() or not metadata_path.is_file():
        raise DatasetInspectionError(
            "Arnesano v2 is not set up. Run python/scripts/setup_data.py through "
            "the documented uv workflow first."
        )

    _print_documentation_check(dataset_root, metadata_path)
    _print_tree(dataset_root)

    csv_paths = sorted(dataset_root.rglob("*.csv"))
    if not csv_paths:
        raise DatasetInspectionError("No CSV files found in the extracted dataset")
    print(f"\nTabular inspection ({len(csv_paths)} CSV files, chunk size {CHUNK_SIZE:,})")
    for path in csv_paths:
        relative_path = path.relative_to(dataset_root)
        _print_summary(relative_path, path, _inspect_csv(path, relative_path))

    preprocessed = sorted(
        (dataset_root / "02_processed_data" / "preprocessed").glob("*.csv")
    )
    print("\nRecommended author-provided starting point")
    for path in preprocessed:
        print(f"  {path.relative_to(dataset_root).as_posix()}")
    print(
        "  These are the data dictionary's analysis-ready tables. Retain its "
        "documented horizon, leakage, sensor-code, and sector-5 flow-rate caveats."
    )

