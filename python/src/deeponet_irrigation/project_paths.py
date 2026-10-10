from __future__ import annotations

from pathlib import Path


def find_repository_root() -> Path:
    """Locate the shared repository root from source or the working directory."""
    starts = (Path(__file__).resolve().parent, Path.cwd().resolve())
    for start in starts:
        for candidate in (start, *start.parents):
            if (
                (candidate / "README.md").is_file()
                and (candidate / "python" / "pyproject.toml").is_file()
            ):
                return candidate
    raise RuntimeError("Could not locate the deeponet-irrigation-gpu repository root")


REPOSITORY_ROOT = find_repository_root()
DATA_DIR = REPOSITORY_ROOT / "data"
MODELS_DIR = REPOSITORY_ROOT / "models"
RESULTS_DIR = REPOSITORY_ROOT / "results"
DOCS_DIR = REPOSITORY_ROOT / "docs"
