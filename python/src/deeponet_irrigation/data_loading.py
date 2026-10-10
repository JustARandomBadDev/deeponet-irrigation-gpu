from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

DATASET_DIRECTORY = "arnesano-v2"
EXPECTED_INTERVAL = pd.Timedelta(minutes=10)

MERGED_REQUIRED_COLUMNS = (
    "ts",
    "soil_moisture",
    "liters_total",
    "weather_rain",
    "weather_temp",
    "weather_humidity",
)
PREPROCESSED_REQUIRED_COLUMNS = (
    "ts",
    "soil_moisture",
    "irrigation_duration_minutes",
    "weather_rain",
    "weather_temp",
    "weather_humidity",
)


class DataValidationError(RuntimeError):
    """Raised when a source table violates a documented dataset assumption."""


@dataclass(frozen=True)
class SourceComparison:
    sector: int
    source: str
    rows: int
    start: str
    end: str
    regular_10min_percent: float
    timestamp_gaps: int
    maximum_gap: str
    soil_moisture_missing: int
    selected_weather_missing: int
    irrigation_missing: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def processed_source_root(project_root: Path) -> Path:
    return (
        project_root
        / "data"
        / "raw"
        / DATASET_DIRECTORY
        / "02_processed_data"
    )


def source_path(project_root: Path, source: str, sector: int) -> Path:
    root = processed_source_root(project_root)
    if source == "merged":
        return root / "merged" / f"dataset_zone_{sector}.csv"
    if source == "preprocessed":
        return root / "preprocessed" / f"dataset_zone_{sector}_preprocessed.csv"
    raise ValueError(f"Unknown data source: {source!r}")


def load_sector_table(project_root: Path, source: str, sector: int) -> pd.DataFrame:
    path = source_path(project_root, source, sector)
    if not path.is_file():
        raise DataValidationError(f"Missing Arnesano source table: {path}")

    required = (
        MERGED_REQUIRED_COLUMNS if source == "merged" else PREPROCESSED_REQUIRED_COLUMNS
    )
    try:
        frame = pd.read_csv(path, parse_dates=["ts"])
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DataValidationError(f"Could not read {path}: {exc}") from exc

    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise DataValidationError(
            f"{path.name} is missing documented columns: {', '.join(missing)}"
        )
    if frame.empty:
        raise DataValidationError(f"Source table is empty: {path}")
    if frame["ts"].isna().any():
        raise DataValidationError(f"Invalid timestamps in {path}")
    if not frame["ts"].is_monotonic_increasing:
        raise DataValidationError(f"Timestamps are not ordered in {path}")
    if frame["ts"].duplicated().any():
        raise DataValidationError(f"Duplicate timestamps in {path}")
    return frame.set_index("ts")


def require_regular_grid(frame: pd.DataFrame, label: str) -> None:
    deltas = frame.index.to_series().diff().dropna()
    irregular = deltas.ne(EXPECTED_INTERVAL)
    if irregular.any():
        examples = ", ".join(str(value) for value in deltas[irregular].head(3))
        raise DataValidationError(
            f"{label} is not a strict 10-minute grid; example deltas: {examples}"
        )


def compare_sources(project_root: Path, sectors: tuple[int, ...]) -> list[SourceComparison]:
    comparisons: list[SourceComparison] = []
    weather = ["weather_rain", "weather_temp", "weather_humidity"]
    for sector in sectors:
        for source in ("merged", "preprocessed"):
            frame = load_sector_table(project_root, source, sector)
            deltas = frame.index.to_series().diff().dropna()
            regular = deltas.eq(EXPECTED_INTERVAL)
            irrigation_column = (
                "liters_total"
                if source == "merged"
                else "irrigation_duration_minutes"
            )
            comparisons.append(
                SourceComparison(
                    sector=sector,
                    source=source,
                    rows=len(frame),
                    start=str(frame.index.min()),
                    end=str(frame.index.max()),
                    regular_10min_percent=round(float(regular.mean() * 100), 2),
                    timestamp_gaps=int((deltas > EXPECTED_INTERVAL).sum()),
                    maximum_gap=str(deltas.max()),
                    soil_moisture_missing=int(frame["soil_moisture"].isna().sum()),
                    selected_weather_missing=int(frame[weather].isna().sum().sum()),
                    irrigation_missing=int(frame[irrigation_column].isna().sum()),
                )
            )
    return comparisons
