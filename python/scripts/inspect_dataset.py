from deeponet_irrigation.data_inspection import (
    DatasetInspectionError,
    inspect_dataset,
)
from deeponet_irrigation.project_paths import REPOSITORY_ROOT


def main() -> None:
    try:
        inspect_dataset(REPOSITORY_ROOT)
    except DatasetInspectionError as exc:
        raise SystemExit(f"Dataset inspection failed: {exc}") from exc


if __name__ == "__main__":
    main()
