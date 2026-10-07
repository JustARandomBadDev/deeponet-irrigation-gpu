from pathlib import Path

from deeponet_irrigation.data_inspection import (
    DatasetInspectionError,
    inspect_dataset,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    try:
        inspect_dataset(project_root)
    except DatasetInspectionError as exc:
        raise SystemExit(f"Dataset inspection failed: {exc}") from exc


if __name__ == "__main__":
    main()
