from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deeponet_irrigation.dataset import PreparedTemporalDataset
from deeponet_irrigation.onnx_deployment import (
    ONNX_INPUT_NAMES,
    ONNX_OPSET,
    ONNX_OUTPUT_NAME,
    DeepONetONNXWrapper,
    export_onnx_model,
    load_frozen_deeponet,
    prepare_onnx_inputs,
    pytorch_deployment_prediction,
    sample_batch,
    validate_onnx_model_contract,
)
from deeponet_irrigation.project_paths import REPOSITORY_ROOT
from deeponet_irrigation.training import write_json


ROOT = REPOSITORY_ROOT
DATA_DIR = ROOT / "data" / "processed" / "arnesano_v2"
DEFAULT_CHECKPOINT = ROOT / "models" / "deeponet_reference.pt"
DEFAULT_OUTPUT = ROOT / "models" / "deeponet_reference.onnx"
DEFAULT_EQUIVALENCE_REPORT = ROOT / "results" / "deeponet_onnx_equivalence.json"
HORIZONS = (1, 3, 6, 12, 24)
BATCH_SIZES = (1, 8, 32, 128)
ABSOLUTE_TOLERANCE = 1e-4
RELATIVE_TOLERANCE = 1e-5


def portable_artifact_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the frozen DeepONet to ONNX")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def audit_checkpoint(
    checkpoint_path: Path,
) -> tuple[Any, dict[str, Any], PreparedTemporalDataset]:
    validation = PreparedTemporalDataset(DATA_DIR, "validation")
    model, checkpoint = load_frozen_deeponet(
        checkpoint_path, validation.metadata, torch.device("cpu")
    )
    history, horizons, _, _ = sample_batch(
        validation, horizon_hours=24, batch_size=8
    )
    with torch.inference_mode():
        first = model(history, horizons)
        second = model(history, horizons)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    print("Checkpoint reconstruction: deterministic on 8 prepared samples")
    return model, checkpoint, validation


def print_onnx_contract(
    output_path: Path,
    graph: Any,
    session: Any,
    providers: list[str],
) -> None:
    import onnx
    import onnxruntime as ort

    print(f"ONNX: {onnx.__version__}")
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"Opset: {graph.opset_import[0].version}")
    for value in session.get_inputs():
        print(f"Input: {value.name} {value.shape} {value.type}")
    for value in session.get_outputs():
        print(f"Output: {value.name} {value.shape} {value.type}")
    print(f"Available providers: {providers}")
    print(f"Model size: {output_path.stat().st_size} bytes")


def equivalence_checks(
    model: Any,
    wrapper: DeepONetONNXWrapper,
    session: Any,
) -> dict[str, Any]:
    cases = []
    absolute_errors = []
    relative_errors = []
    for split in ("train", "validation", "test"):
        dataset = PreparedTemporalDataset(DATA_DIR, split)
        for horizon in HORIZONS:
            for batch_size in BATCH_SIZES:
                history, horizons, _, _ = sample_batch(
                    dataset, horizon_hours=horizon, batch_size=batch_size
                )
                inputs = prepare_onnx_inputs(model, history, horizons)
                with torch.inference_mode():
                    pytorch_prediction = model(history, horizons).cpu().numpy()
                wrapper_prediction = pytorch_deployment_prediction(wrapper, inputs)
                np.testing.assert_allclose(
                    wrapper_prediction,
                    pytorch_prediction,
                    rtol=1e-6,
                    atol=1e-6,
                )
                onnx_prediction = session.run([ONNX_OUTPUT_NAME], inputs)[0]
                absolute = np.abs(pytorch_prediction - onnx_prediction)
                relative = absolute / np.maximum(np.abs(pytorch_prediction), 1e-6)
                np.testing.assert_allclose(
                    onnx_prediction,
                    pytorch_prediction,
                    rtol=RELATIVE_TOLERANCE,
                    atol=ABSOLUTE_TOLERANCE,
                )
                result = {
                    "split": split,
                    "horizon_hours": horizon,
                    "batch_size": batch_size,
                    "max_absolute_error": float(absolute.max()),
                    "mean_absolute_error": float(absolute.mean()),
                    "max_relative_error": float(relative.max()),
                }
                cases.append(result)
                absolute_errors.append(absolute.reshape(-1))
                relative_errors.append(relative.reshape(-1))
    all_absolute = np.concatenate(absolute_errors)
    all_relative = np.concatenate(relative_errors)
    by_batch_size = {}
    for batch_size in BATCH_SIZES:
        selected = [case for case in cases if case["batch_size"] == batch_size]
        by_batch_size[str(batch_size)] = {
            "case_count": len(selected),
            "max_absolute_error": max(case["max_absolute_error"] for case in selected),
            "mean_absolute_error": float(
                np.mean([case["mean_absolute_error"] for case in selected])
            ),
            "max_relative_error": max(case["max_relative_error"] for case in selected),
        }
    return {
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "cases": cases,
        "overall": {
            "max_absolute_error": float(all_absolute.max()),
            "mean_absolute_error": float(all_absolute.mean()),
            "max_relative_error": float(all_relative.max()),
        },
        "by_batch_size": by_batch_size,
        "passed": True,
    }


def create_golden_vectors(
    model: Any,
    wrapper: DeepONetONNXWrapper,
    output_path: Path,
) -> None:
    dataset = PreparedTemporalDataset(DATA_DIR, "validation")
    histories = []
    horizons = []
    targets = []
    sample_indices = []
    for horizon in HORIZONS:
        history, query, target, indices = sample_batch(
            dataset, horizon_hours=horizon, batch_size=2
        )
        histories.append(history)
        horizons.append(query)
        targets.append(target)
        sample_indices.append(indices)
    history = torch.cat(histories)
    query = torch.cat(horizons)
    inputs = prepare_onnx_inputs(model, history, query)
    prediction = pytorch_deployment_prediction(wrapper, inputs)
    np.savez_compressed(
        output_path,
        **inputs,
        expected_prediction=prediction.astype(np.float32),
        target=np.concatenate([target.numpy() for target in targets]),
        horizon_hours=query.numpy().astype(np.float32),
        source_sample_index=np.concatenate(sample_indices),
    )


def deployment_metadata(
    model: Any,
    checkpoint_path: Path,
    onnx_path: Path,
    providers: list[str],
    equivalence: dict[str, Any],
    golden_path: Path,
) -> dict[str, Any]:
    trend_definitions = model.architecture()["temporal_features"]
    statistics = PreparedTemporalDataset(DATA_DIR, "train").metadata["normalization"][
        "statistics"
    ]
    return {
        "model_name": "deeponet_reference",
        "checkpoint_source": portable_artifact_path(checkpoint_path),
        "onnx_filename": onnx_path.name,
        "onnx_file_present": onnx_path.exists(),
        "onnx_opset": ONNX_OPSET,
        "dtype": "float32",
        "inputs": [
            {
                "name": "branch_input",
                "shape": ["batch", model.branch_input_dimension],
                "semantics": "720 normalized history values followed by 3 normalized moisture trends",
            },
            {
                "name": "horizon",
                "shape": ["batch", 1],
                "semantics": "physical horizon_hours divided by 24 before inference",
            },
            {
                "name": "current_moisture",
                "shape": ["batch", 1],
                "semantics": "soil moisture at prediction origin in physical dataset units",
            },
        ],
        "output": {
            "name": "prediction",
            "shape": ["batch", 1],
            "semantics": "future soil moisture in physical dataset units",
        },
        "branch_input_dimension": model.branch_input_dimension,
        "history_steps": model.history_steps,
        "sampling_interval_minutes": model.sampling_interval_minutes,
        "history_flattening": "time-major [144,5], feature order varies fastest",
        "historical_feature_order": list(model.feature_order),
        "historical_feature_normalization": {
            name: {
                "mean": float(statistics[name]["mean"]),
                "std": float(statistics[name]["std"]),
                "formula": "(physical_value - mean) / std",
            }
            for name in model.feature_order
        },
        "derived_trend_features": {
            "input_units_before_normalization": "physical soil-moisture units",
            "onnx_branch_values": "normalized",
            "order": trend_definitions["names"],
            "definitions": trend_definitions["definitions"],
            "means": trend_definitions["means"],
            "stds": trend_definitions["stds"],
            "formula": "(physical_trend - train_mean) / train_std",
        },
        "horizon_normalization": "horizon = horizon_hours / 24",
        "residual_scaling": {
            "rule": "physical_delta = normalized_delta * train_residual_std",
            "train_residual_std": float(model.target_std.item()),
            "mean_subtraction": False,
        },
        "current_moisture": {
            "units": "physical soil-moisture units",
            "normalization": "none",
            "semantics": "last soil_moisture value at the prediction origin",
        },
        "supported_horizons_hours": list(HORIZONS),
        "dynamic_batch": True,
        "required_runtime_validation_provider": "CPUExecutionProvider",
        "available_onnxruntime_providers": providers,
        "numerical_equivalence": {
            "status": equivalence.get(
                "status", "passed" if equivalence["passed"] else "failed"
            ),
            "passed": equivalence["passed"],
            "absolute_tolerance": equivalence["absolute_tolerance"],
            "relative_tolerance": equivalence["relative_tolerance"],
            "overall": equivalence["overall"],
            "report": portable_artifact_path(DEFAULT_EQUIVALENCE_REPORT),
        },
        "golden_test_vectors": golden_path.name,
    }


def main() -> None:
    args = parse_args()
    model, _, _ = audit_checkpoint(args.checkpoint)
    wrapper = DeepONetONNXWrapper(model).eval()
    export_onnx_model(wrapper, args.output)
    graph, session, providers = validate_onnx_model_contract(
        args.output, wrapper.branch_input_dimension
    )
    print_onnx_contract(args.output, graph, session, providers)
    equivalence = equivalence_checks(model, wrapper, session)
    metadata_path = args.output.with_suffix(".metadata.json")
    golden_path = args.output.with_name(f"{args.output.stem}_golden.npz")
    create_golden_vectors(model, wrapper, golden_path)
    metadata = deployment_metadata(
        model, args.checkpoint, args.output, providers, equivalence, golden_path
    )
    write_json(DEFAULT_EQUIVALENCE_REPORT, equivalence)
    write_json(metadata_path, metadata)
    print("Numerical equivalence:", json.dumps(equivalence["overall"], indent=2))
    print("By batch size:", json.dumps(equivalence["by_batch_size"], indent=2))
    print(f"Golden vectors: {golden_path}")
    print(f"Deployment metadata: {metadata_path}")


if __name__ == "__main__":
    main()
