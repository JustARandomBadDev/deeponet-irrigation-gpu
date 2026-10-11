from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .onnx_deployment import (
    ONNX_INPUT_NAMES,
    ONNX_OUTPUT_NAME,
    DeepONetONNXWrapper,
    export_golden_cpp_fixture,
    export_onnx_model,
    load_frozen_deeponet,
    load_golden_deployment_arrays,
    pytorch_deployment_prediction,
    validate_onnx_model_contract,
)
from .project_paths import REPOSITORY_ROOT


MODELS_DIR = REPOSITORY_ROOT / "models"
CHECKPOINT = MODELS_DIR / "deeponet_reference.pt"
ONNX_MODEL = MODELS_DIR / "deeponet_reference.onnx"
DEPLOYMENT_METADATA = MODELS_DIR / "deeponet_reference.metadata.json"
GOLDEN_NPZ = MODELS_DIR / "deeponet_reference_golden.npz"
GOLDEN_BINARY = MODELS_DIR / "deeponet_reference_golden.bin"
GOLDEN_METADATA = MODELS_DIR / "deeponet_reference_golden.json"
ABSOLUTE_TOLERANCE = 1e-4
RELATIVE_TOLERANCE = 1e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _prepared_metadata(
    checkpoint: dict[str, Any], deployment: dict[str, Any]
) -> dict[str, Any]:
    contract = checkpoint["preprocessing_contract"]
    statistics = deployment["historical_feature_normalization"]
    return {
        **contract,
        "normalization": {"statistics": statistics},
    }


def _validate_checkpoint_contract(
    checkpoint: dict[str, Any], deployment: dict[str, Any]
) -> None:
    architecture = checkpoint["architecture"]
    contract = checkpoint["preprocessing_contract"]
    expected = {
        "model_type": "residual_deeponet",
        "branch_input_dimension": int(deployment["branch_input_dimension"]),
        "history_steps": int(deployment["history_steps"]),
        "feature_order": list(deployment["historical_feature_order"]),
        "temporal_features": list(deployment["derived_trend_features"]["order"]),
        "train_residual_std": float(
            deployment["residual_scaling"]["train_residual_std"]
        ),
    }
    actual = {
        "model_type": architecture.get("model_type"),
        "branch_input_dimension": int(architecture["branch_input_dimension"]),
        "history_steps": int(architecture["history_steps"]),
        "feature_order": list(architecture["feature_order"]),
        "temporal_features": list(architecture["temporal_features"]["names"]),
        "train_residual_std": float(architecture["train_residual_std"]),
    }
    if actual != expected:
        raise ValueError(
            "Frozen checkpoint architecture does not match deployment metadata"
        )
    if list(contract["prediction_horizons_hours"]) != list(
        deployment["supported_horizons_hours"]
    ):
        raise ValueError("Checkpoint horizons do not match deployment metadata")
    expected_inputs = {
        "branch_input": ["batch", int(deployment["branch_input_dimension"])],
        "horizon": ["batch", 1],
        "current_moisture": ["batch", 1],
    }
    actual_inputs = {
        item["name"]: item["shape"] for item in deployment["inputs"]
    }
    if deployment.get("dtype") != "float32" or actual_inputs != expected_inputs:
        raise ValueError("Deployment metadata input contract is invalid")
    output = deployment["output"]
    if output["name"] != ONNX_OUTPUT_NAME or output["shape"] != ["batch", 1]:
        raise ValueError("Deployment metadata output contract is invalid")


def _refresh_golden_npz(
    path: Path,
    arrays: dict[str, np.ndarray],
    expected_prediction: np.ndarray,
) -> None:
    refreshed = dict(arrays)
    refreshed["expected_prediction"] = expected_prediction.astype(np.float32)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **refreshed)
    temporary.replace(path)


def _refresh_metadata(
    path: Path,
    metadata: dict[str, Any],
    checkpoint_path: Path,
    onnx_path: Path,
    golden_path: Path,
    repository_root: Path,
) -> None:
    metadata["checkpoint_source"] = str(checkpoint_path.relative_to(repository_root))
    metadata["onnx_filename"] = onnx_path.name
    metadata["onnx_file_present"] = onnx_path.is_file()
    metadata["golden_test_vectors"] = golden_path.name
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def setup_deployment(
    *,
    checkpoint_path: Path = CHECKPOINT,
    onnx_path: Path = ONNX_MODEL,
    metadata_path: Path = DEPLOYMENT_METADATA,
    golden_npz_path: Path = GOLDEN_NPZ,
    golden_binary_path: Path = GOLDEN_BINARY,
    golden_metadata_path: Path = GOLDEN_METADATA,
    repository_root: Path = REPOSITORY_ROOT,
    print_summary: bool = True,
) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Frozen DeepONet checkpoint is missing; setup-onnx never trains a "
            f"replacement: {checkpoint_path}"
        )
    deployment = _load_json(metadata_path, "Deployment metadata")
    if not golden_npz_path.is_file():
        raise FileNotFoundError(
            f"Tracked golden NPZ fixture does not exist: {golden_npz_path}"
        )

    checkpoint_sha = sha256(checkpoint_path)
    onnx_sha_before = sha256(onnx_path) if onnx_path.is_file() else None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _validate_checkpoint_contract(checkpoint, deployment)
    model, checkpoint = load_frozen_deeponet(
        checkpoint_path,
        _prepared_metadata(checkpoint, deployment),
        torch.device("cpu"),
    )
    if model.training:
        raise RuntimeError("Frozen DeepONet was not placed in evaluation mode")
    wrapper = DeepONetONNXWrapper(model).eval()

    arrays = load_golden_deployment_arrays(
        golden_npz_path, wrapper.branch_input_dimension
    )
    inputs = {name: arrays[name] for name in ONNX_INPUT_NAMES}
    pytorch_prediction = pytorch_deployment_prediction(wrapper, inputs)
    repeated_prediction = pytorch_deployment_prediction(wrapper, inputs)
    np.testing.assert_array_equal(pytorch_prediction, repeated_prediction)
    np.testing.assert_allclose(
        pytorch_prediction,
        arrays["expected_prediction"],
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
        err_msg="Frozen checkpoint no longer matches the tracked golden fixture",
    )

    export_onnx_model(wrapper, onnx_path)
    _, session, providers = validate_onnx_model_contract(
        onnx_path, wrapper.branch_input_dimension
    )
    onnx_prediction = session.run(
        [ONNX_OUTPUT_NAME],
        {name: inputs[name] for name in ONNX_INPUT_NAMES},
    )[0]
    np.testing.assert_allclose(
        onnx_prediction,
        pytorch_prediction,
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
        err_msg="PyTorch and ONNX Runtime predictions differ",
    )
    absolute_error = np.abs(pytorch_prediction - onnx_prediction)
    relative_error = absolute_error / np.maximum(np.abs(pytorch_prediction), 1e-6)

    _refresh_golden_npz(golden_npz_path, arrays, pytorch_prediction)
    case_count = export_golden_cpp_fixture(
        golden_npz_path,
        golden_binary_path,
        golden_metadata_path,
        branch_input_dimension=wrapper.branch_input_dimension,
        repository_root=repository_root,
    )
    _refresh_metadata(
        metadata_path,
        deployment,
        checkpoint_path,
        onnx_path,
        golden_npz_path,
        repository_root,
    )

    result = {
        "checkpoint_sha256": checkpoint_sha,
        "onnx_sha256_before": onnx_sha_before,
        "onnx_sha256_after": sha256(onnx_path),
        "golden_npz_sha256": sha256(golden_npz_path),
        "golden_case_count": case_count,
        "max_absolute_error": float(absolute_error.max()),
        "mean_absolute_error": float(absolute_error.mean()),
        "max_relative_error": float(relative_error.max()),
        "providers": providers,
    }
    if print_summary:
        before = result["onnx_sha256_before"] or "not present"
        print("setup-onnx completed successfully (no training performed)")
        print(f"Checkpoint SHA-256: {result['checkpoint_sha256']}")
        print(f"ONNX SHA-256 before: {before}")
        print(f"ONNX SHA-256 after:  {result['onnx_sha256_after']}")
        print(f"Golden NPZ SHA-256: {result['golden_npz_sha256']}")
        print(
            "ONNX contract: branch_input [batch,723], horizon [batch,1], "
            "current_moisture [batch,1] -> prediction [batch,1], float32"
        )
        print(
            "PyTorch/ONNX error: "
            f"max_abs={result['max_absolute_error']:.9g}, "
            f"mean_abs={result['mean_absolute_error']:.9g}, "
            f"max_rel={result['max_relative_error']:.9g}"
        )
        print(f"Golden cases and C++ fixtures: {case_count}")
    return result


def main() -> None:
    setup_deployment()


if __name__ == "__main__":
    main()
