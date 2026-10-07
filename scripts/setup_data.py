from pathlib import Path

from deeponet_irrigation.data_download import DatasetError, setup_dataset


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    try:
        setup_dataset(project_root)
    except DatasetError as exc:
        raise SystemExit(f"Data setup failed: {exc}") from exc


if __name__ == "__main__":
    main()

