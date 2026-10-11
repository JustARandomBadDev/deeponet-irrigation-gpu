from __future__ import annotations

from deeponet_irrigation.onnx_deployment import export_golden_cpp_fixture
from deeponet_irrigation.project_paths import REPOSITORY_ROOT


SOURCE = REPOSITORY_ROOT / "models" / "deeponet_reference_golden.npz"
BINARY_OUTPUT = REPOSITORY_ROOT / "models" / "deeponet_reference_golden.bin"
METADATA_OUTPUT = REPOSITORY_ROOT / "models" / "deeponet_reference_golden.json"
BRANCH_DIMENSION = 723


def main() -> None:
    if not SOURCE.is_file():
        raise SystemExit(f"Golden NPZ fixture does not exist: {SOURCE}")
    case_count = export_golden_cpp_fixture(
        SOURCE,
        BINARY_OUTPUT,
        METADATA_OUTPUT,
        branch_input_dimension=BRANCH_DIMENSION,
        repository_root=REPOSITORY_ROOT,
    )
    print(f"Converted {case_count} existing golden cases")
    print(f"Binary: {BINARY_OUTPUT} ({BINARY_OUTPUT.stat().st_size} bytes)")
    print(f"Metadata: {METADATA_OUTPUT}")


if __name__ == "__main__":
    main()
