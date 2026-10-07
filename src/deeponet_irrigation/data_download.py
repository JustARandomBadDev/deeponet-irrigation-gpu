from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import requests

DATASET_ID = "c837v6p8ph"
DATASET_VERSION = 2
DATASET_DOI = "10.17632/c837v6p8ph.2"
DATASET_SLUG = "arnesano-v2"

PUBLIC_API = "https://data.mendeley.com/public-api"
DATASETS_API = "https://data.mendeley.com/api/datasets-v2"
SNAPSHOT_URL = f"{PUBLIC_API}/datasets/{DATASET_ID}/snapshot/{DATASET_VERSION}"
ARCHIVE_METADATA_URL = (
    f"{DATASETS_API}/datasets/{DATASET_ID}/zip?version={DATASET_VERSION}"
)
ARCHIVE_DOWNLOAD_URL = (
    f"{PUBLIC_API}/zip/{DATASET_ID}/download/{DATASET_VERSION}"
)

EXPECTED_ROOT_ENTRIES = {
    "01_raw_data",
    "02_processed_data",
    "03_code",
    "DATA_DICTIONARY.md",
    "README.md",
}
EXPECTED_DATA_FILES = {
    "DATA_DICTIONARY.md",
    "README.md",
    "01_raw_data/raw_irrigation_actuators/raw_actuators.csv",
    "01_raw_data/raw_irrigation_actuators/raw_measurements.csv",
    "01_raw_data/raw_irrigation_actuators/raw_nominal_flow_rates.csv",
    "01_raw_data/raw_sensor/MEASUREMENT_all.csv",
    "01_raw_data/raw_sensor/MEASUREMENT_ce.csv",
    "01_raw_data/raw_sensor/MEASUREMENT_ph.csv",
    "01_raw_data/raw_sensor/MEASUREMENT_ste.csv",
    "01_raw_data/raw_sensor/MEASUREMENT_sti.csv",
    "01_raw_data/raw_sensor/MEASUREMENT_sue.csv",
    "01_raw_data/raw_sensor/MEASUREMENT_sui.csv",
    "01_raw_data/raw_sensor/MEASUREMENT_sut.csv",
    "02_processed_data/irrigation_events/final_irrigation_events.csv",
    "02_processed_data/irrigation_events/sector_statistics_report.csv",
    "02_processed_data/merged/dataset_zone_1.csv",
    "02_processed_data/merged/dataset_zone_2.csv",
    "02_processed_data/merged/dataset_zone_3.csv",
    "02_processed_data/merged/dataset_zone_4.csv",
    "02_processed_data/merged/dataset_zone_5.csv",
    "02_processed_data/preprocessed/dataset_zone_1_preprocessed.csv",
    "02_processed_data/preprocessed/dataset_zone_2_preprocessed.csv",
    "02_processed_data/preprocessed/dataset_zone_3_preprocessed.csv",
    "02_processed_data/preprocessed/dataset_zone_4_preprocessed.csv",
    "02_processed_data/preprocessed/dataset_zone_5_preprocessed.csv",
    "03_code/00_irrigation_actuators_cleaning.py",
    "03_code/01_merge_datasets.py",
    "03_code/02_preprocessing.py",
    "03_code/README.md",
    "03_code/config.py",
    "03_code/requirements.txt",
}


class DatasetError(RuntimeError):
    """Raised when the remote or local dataset does not match the pinned release."""


def _get_json(session: requests.Session, url: str) -> dict[str, Any]:
    try:
        response = session.get(url, timeout=(15, 60))
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise DatasetError(f"Could not read dataset metadata from {url}: {exc}") from exc

    if not isinstance(payload, dict):
        raise DatasetError(f"Unexpected metadata response from {url}: expected an object")
    return payload


def _validate_remote_metadata(
    snapshot: dict[str, Any], archive: dict[str, Any]
) -> tuple[int, str]:
    if snapshot.get("id") != DATASET_ID:
        raise DatasetError(
            f"Dataset ID mismatch: expected {DATASET_ID!r}, got {snapshot.get('id')!r}"
        )
    if snapshot.get("version") != DATASET_VERSION:
        raise DatasetError(
            "Dataset version mismatch: "
            f"expected {DATASET_VERSION}, got {snapshot.get('version')!r}"
        )
    if snapshot.get("doi") != DATASET_DOI:
        raise DatasetError(
            f"Dataset DOI mismatch: expected {DATASET_DOI!r}, got {snapshot.get('doi')!r}"
        )
    if archive.get("status") != "FINISH":
        raise DatasetError(
            f"Mendeley archive is not ready (status={archive.get('status')!r})"
        )

    size = archive.get("size")
    checksum = archive.get("sha256_hash")
    if not isinstance(size, int) or size <= 0:
        raise DatasetError(f"Invalid archive size in metadata: {size!r}")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise DatasetError(
            "Mendeley did not provide the expected SHA-256 checksum for this archive"
        )
    try:
        bytes.fromhex(checksum)
    except ValueError as exc:
        raise DatasetError(f"Invalid SHA-256 checksum in metadata: {checksum!r}") from exc
    return size, checksum.lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_is_valid(path: Path, expected_size: int, expected_sha256: str) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == expected_size
        and _sha256(path) == expected_sha256
    )


def _download_archive(
    session: requests.Session,
    destination: Path,
    expected_size: int,
    expected_sha256: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)

    print(f"Downloading Arnesano v{DATASET_VERSION} ({expected_size / 1_000_000:.1f} MB)...")
    try:
        with session.get(
            ARCHIVE_DOWNLOAD_URL, stream=True, timeout=(15, 120)
        ) as response:
            response.raise_for_status()
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
    except (OSError, requests.RequestException) as exc:
        temporary.unlink(missing_ok=True)
        raise DatasetError(f"Dataset download failed: {exc}") from exc

    if not _archive_is_valid(temporary, expected_size, expected_sha256):
        actual_size = temporary.stat().st_size if temporary.exists() else 0
        actual_sha256 = _sha256(temporary) if temporary.exists() else "missing"
        temporary.unlink(missing_ok=True)
        raise DatasetError(
            "Downloaded archive failed validation: "
            f"expected {expected_size} bytes / {expected_sha256}, "
            f"got {actual_size} bytes / {actual_sha256}"
        )
    temporary.replace(destination)
    print("Download complete; SHA-256 verified.")


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    if not members:
        raise DatasetError("The downloaded archive is empty")
    for member in members:
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise DatasetError(f"Unsafe path in downloaded archive: {member.filename!r}")
    return members


def _validate_extracted_structure(root: Path) -> None:
    missing = sorted(name for name in EXPECTED_ROOT_ENTRIES if not (root / name).exists())
    if missing:
        raise DatasetError(
            "Extracted dataset structure differs from Arnesano v2; missing: "
            + ", ".join(missing)
        )
    for directory in ("01_raw_data", "02_processed_data", "03_code"):
        if not (root / directory).is_dir():
            raise DatasetError(f"Expected a directory at {root / directory}")
    for document in ("README.md", "DATA_DICTIONARY.md"):
        if not (root / document).is_file():
            raise DatasetError(f"Expected a file at {root / document}")

    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != ".extraction.json"
    }
    missing_files = sorted(EXPECTED_DATA_FILES - actual_files)
    unexpected_files = sorted(actual_files - EXPECTED_DATA_FILES)
    if missing_files or unexpected_files:
        details = []
        if missing_files:
            details.append("missing files: " + ", ".join(missing_files))
        if unexpected_files:
            details.append("unexpected files: " + ", ".join(unexpected_files))
        raise DatasetError(
            "Extracted dataset file manifest differs from Arnesano v2 ("
            + "; ".join(details)
            + ")"
        )


def _find_dataset_root(extraction_root: Path) -> Path:
    """Return the dataset root, allowing Mendeley's single wrapper directory."""
    try:
        _validate_extracted_structure(extraction_root)
        return extraction_root
    except DatasetError as root_error:
        entries = list(extraction_root.iterdir())
        if len(entries) != 1 or not entries[0].is_dir():
            raise root_error
        try:
            _validate_extracted_structure(entries[0])
        except DatasetError as nested_error:
            raise DatasetError(
                "Extracted dataset does not have the documented Arnesano v2 layout "
                "at the archive root or inside one wrapper directory"
            ) from nested_error
        return entries[0]


def _extraction_is_current(root: Path, expected_sha256: str) -> bool:
    marker = root / ".extraction.json"
    if not marker.is_file():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        _validate_extracted_structure(root)
    except (OSError, ValueError, DatasetError):
        return False
    return data.get("archive_sha256") == expected_sha256


def _extract_archive(archive_path: Path, destination: Path, checksum: str) -> None:
    if destination.exists():
        raise DatasetError(
            f"Existing extraction at {destination} is incomplete or does not match v2. "
            "Move it aside or remove it, then run setup again."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{DATASET_SLUG}-", dir=destination.parent))
    print("Extracting archive...")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(temporary, members=_safe_members(archive))
        extracted_root = _find_dataset_root(temporary)
        (extracted_root / ".extraction.json").write_text(
            json.dumps(
                {
                    "archive_sha256": checksum,
                    "extracted_at_utc": datetime.now(UTC).isoformat(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        extracted_root.replace(destination)
    except (OSError, zipfile.BadZipFile) as exc:
        raise DatasetError(f"Could not extract {archive_path}: {exc}") from exc
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print("Extraction complete.")


def setup_dataset(project_root: Path) -> Path:
    """Download, validate, and extract the pinned Arnesano v2 dataset."""
    raw_root = project_root / "data" / "raw"
    archive_path = raw_root / f"{DATASET_SLUG}.zip"
    dataset_root = raw_root / DATASET_SLUG
    metadata_path = raw_root / f"{DATASET_SLUG}.metadata.json"

    session = requests.Session()
    session.headers["User-Agent"] = "deeponet-irrigation-data-setup/0.1"
    snapshot = _get_json(session, SNAPSHOT_URL)
    archive_metadata = _get_json(session, ARCHIVE_METADATA_URL)
    expected_size, expected_sha256 = _validate_remote_metadata(
        snapshot, archive_metadata
    )

    if _archive_is_valid(archive_path, expected_size, expected_sha256):
        print(f"Archive already present and valid: {archive_path}")
    else:
        if archive_path.exists():
            print(f"Existing archive is invalid; replacing: {archive_path}")
            archive_path.unlink()
        _download_archive(session, archive_path, expected_size, expected_sha256)

    if _extraction_is_current(dataset_root, expected_sha256):
        print(f"Extracted dataset already present: {dataset_root}")
    else:
        _extract_archive(archive_path, dataset_root, expected_sha256)

    metadata = {
        "dataset_id": DATASET_ID,
        "doi": DATASET_DOI,
        "version": DATASET_VERSION,
        "dataset_name": snapshot.get("name"),
        "published_at": snapshot.get("publish_date"),
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "snapshot_url": SNAPSHOT_URL,
        "archive_metadata_url": ARCHIVE_METADATA_URL,
        "download_url": ARCHIVE_DOWNLOAD_URL,
        "archive_filename": archive_path.name,
        "archive_size_bytes": expected_size,
        "archive_sha256": expected_sha256,
        "checksum_source": "Mendeley Data archive metadata (sha256_hash)",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Metadata written: {metadata_path}")
    print(f"Done: Arnesano v{DATASET_VERSION} is ready at {dataset_root}")
    return dataset_root
