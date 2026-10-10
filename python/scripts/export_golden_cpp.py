from __future__ import annotations

import hashlib
import json
import struct

import numpy as np

from deeponet_irrigation.project_paths import REPOSITORY_ROOT


SOURCE = REPOSITORY_ROOT / "models" / "deeponet_reference_golden.npz"
BINARY_OUTPUT = REPOSITORY_ROOT / "models" / "deeponet_reference_golden.bin"
METADATA_OUTPUT = REPOSITORY_ROOT / "models" / "deeponet_reference_golden.json"
MAGIC = b"DONGOLD1"
BRANCH_DIMENSION = 723


def load_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(SOURCE, allow_pickle=False) as fixture:
        required = (
            "branch_input",
            "horizon",
            "current_moisture",
            "expected_prediction",
        )
        missing = set(required).difference(fixture.files)
        if missing:
            raise ValueError(f"Golden fixture is missing arrays: {sorted(missing)}")
        by_name = {name: np.asarray(fixture[name]) for name in required}

    branch = by_name["branch_input"]
    horizon = by_name["horizon"]
    current = by_name["current_moisture"]
    expected = by_name["expected_prediction"]
    case_count = branch.shape[0]
    expected_shapes = {
        "branch_input": (case_count, BRANCH_DIMENSION),
        "horizon": (case_count, 1),
        "current_moisture": (case_count, 1),
        "expected_prediction": (case_count, 1),
    }
    for name, array in by_name.items():
        if array.dtype != np.float32:
            raise ValueError(f"{name} must be float32, got {array.dtype}")
        if array.shape != expected_shapes[name]:
            raise ValueError(
                f"{name} has shape {array.shape}, expected {expected_shapes[name]}"
            )
    return branch, horizon, current, expected


def main() -> None:
    if not SOURCE.is_file():
        raise SystemExit(f"Golden NPZ fixture does not exist: {SOURCE}")

    branch, horizon, current, expected = load_arrays()
    case_count = branch.shape[0]
    with BINARY_OUTPUT.open("wb") as output:
        output.write(struct.pack("<8sII", MAGIC, case_count, BRANCH_DIMENSION))
        for i in range(case_count):
            output.write(branch[i].astype("<f4", copy=False).tobytes(order="C"))
            output.write(horizon[i].astype("<f4", copy=False).tobytes(order="C"))
            output.write(current[i].astype("<f4", copy=False).tobytes(order="C"))
            output.write(expected[i].astype("<f4", copy=False).tobytes(order="C"))

    metadata = {
        "format": "deeponet-golden-v1",
        "endianness": "little",
        "dtype": "float32",
        "header": {
            "magic": MAGIC.decode("ascii"),
            "case_count": case_count,
            "branch_dimension": BRANCH_DIMENSION,
        },
        "record_layout": [
            {"name": "branch_input", "float32_count": BRANCH_DIMENSION},
            {"name": "horizon", "float32_count": 1},
            {"name": "current_moisture", "float32_count": 1},
            {"name": "expected_prediction", "float32_count": 1},
        ],
        "source": str(SOURCE.relative_to(REPOSITORY_ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "binary": str(BINARY_OUTPUT.relative_to(REPOSITORY_ROOT)),
        "binary_size_bytes": BINARY_OUTPUT.stat().st_size,
        "expected_values_recomputed": False,
    }
    METADATA_OUTPUT.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Converted {case_count} existing golden cases")
    print(f"Binary: {BINARY_OUTPUT} ({BINARY_OUTPUT.stat().st_size} bytes)")
    print(f"Metadata: {METADATA_OUTPUT}")


if __name__ == "__main__":
    main()

