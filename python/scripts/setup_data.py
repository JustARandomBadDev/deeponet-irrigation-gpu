from deeponet_irrigation.data_download import DatasetError, setup_dataset
from deeponet_irrigation.project_paths import REPOSITORY_ROOT


def main() -> None:
    try:
        setup_dataset(REPOSITORY_ROOT)
    except DatasetError as exc:
        raise SystemExit(f"Data setup failed: {exc}") from exc


if __name__ == "__main__":
    main()

