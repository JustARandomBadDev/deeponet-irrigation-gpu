from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np
import torch
from torch import nn

from .evaluation import load_deeponet_checkpoint
from .models import ResidualDeepONet
from .temporal_features import derive_temporal_features


ONNX_INPUT_NAMES = ("branch_input", "horizon", "current_moisture")
ONNX_OUTPUT_NAME = "prediction"
ONNX_OPSET = 17
GOLDEN_MAGIC = b"DONGOLD1"


class DeepONetONNXWrapper(nn.Module):
    """Deployment graph with preprocessing-dependent values as explicit inputs."""

    def __init__(self, model: ResidualDeepONet) -> None:
        super().__init__()
        self.branch_network = model.branch_network
        self.trunk_network = model.trunk_network
        self.operator_bias = model.operator_bias
        self.register_buffer("residual_std", model.target_std.detach().clone())
        self.branch_input_dimension = model.branch_input_dimension

    def forward(
        self,
        branch_input: torch.Tensor,
        horizon: torch.Tensor,
        current_moisture: torch.Tensor,
    ) -> torch.Tensor:
        branch_latent = self.branch_network(branch_input)
        trunk_latent = self.trunk_network(horizon)
        normalized_delta = (
            branch_latent * trunk_latent
        ).sum(dim=-1, keepdim=True) + self.operator_bias
        return current_moisture + normalized_delta * self.residual_std


def export_onnx_model(
    wrapper: DeepONetONNXWrapper,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    example = (
        torch.zeros(2, wrapper.branch_input_dimension, dtype=torch.float32),
        torch.zeros(2, 1, dtype=torch.float32),
        torch.zeros(2, 1, dtype=torch.float32),
    )
    torch.onnx.export(
        wrapper,
        example,
        output_path,
        export_params=True,
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        input_names=list(ONNX_INPUT_NAMES),
        output_names=[ONNX_OUTPUT_NAME],
        dynamic_axes={
            "branch_input": {0: "batch"},
            "horizon": {0: "batch"},
            "current_moisture": {0: "batch"},
            "prediction": {0: "batch"},
        },
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=False,
    )


def validate_onnx_model_contract(
    output_path: str | Path,
    branch_input_dimension: int,
) -> tuple[Any, Any, list[str]]:
    import onnx
    import onnxruntime as ort

    output_path = Path(output_path)
    graph = onnx.load(output_path)
    onnx.checker.check_model(graph)
    providers = ort.get_available_providers()
    session = ort.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )
    expected_inputs = {
        "branch_input": ["batch", branch_input_dimension],
        "horizon": ["batch", 1],
        "current_moisture": ["batch", 1],
    }
    inputs = session.get_inputs()
    actual_inputs = {value.name: value.shape for value in inputs}
    if actual_inputs != expected_inputs:
        raise RuntimeError(f"Unexpected ONNX inputs: {actual_inputs}")
    non_float_inputs = [
        value.name for value in inputs if value.type != "tensor(float)"
    ]
    if non_float_inputs:
        raise RuntimeError(f"ONNX inputs are not float32: {non_float_inputs}")
    outputs = session.get_outputs()
    if len(outputs) != 1 or outputs[0].name != ONNX_OUTPUT_NAME:
        raise RuntimeError("Unexpected ONNX output names")
    if outputs[0].shape != ["batch", 1]:
        raise RuntimeError(f"Unexpected ONNX output shape: {outputs[0].shape}")
    if outputs[0].type != "tensor(float)":
        raise RuntimeError(f"ONNX output is not float32: {outputs[0].type}")
    return graph, session, providers


def load_golden_deployment_arrays(
    source: str | Path,
    branch_input_dimension: int,
) -> dict[str, np.ndarray]:
    source = Path(source)
    with np.load(source, allow_pickle=False) as fixture:
        required = (*ONNX_INPUT_NAMES, "expected_prediction")
        missing = set(required).difference(fixture.files)
        if missing:
            raise ValueError(f"Golden fixture is missing arrays: {sorted(missing)}")
        arrays = {name: np.asarray(fixture[name]) for name in fixture.files}

    case_count = arrays["branch_input"].shape[0]
    expected_shapes = {
        "branch_input": (case_count, branch_input_dimension),
        "horizon": (case_count, 1),
        "current_moisture": (case_count, 1),
        "expected_prediction": (case_count, 1),
    }
    for name, shape in expected_shapes.items():
        array = arrays[name]
        if array.dtype != np.float32:
            raise ValueError(f"{name} must be float32, got {array.dtype}")
        if array.shape != shape:
            raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    return arrays


def export_golden_cpp_fixture(
    source: str | Path,
    binary_output: str | Path,
    metadata_output: str | Path,
    *,
    branch_input_dimension: int,
    repository_root: str | Path,
) -> int:
    source = Path(source)
    binary_output = Path(binary_output)
    metadata_output = Path(metadata_output)
    repository_root = Path(repository_root)
    arrays = load_golden_deployment_arrays(source, branch_input_dimension)
    case_count = arrays["branch_input"].shape[0]

    binary_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    with binary_output.open("wb") as output:
        output.write(
            struct.pack(
                "<8sII", GOLDEN_MAGIC, case_count, branch_input_dimension
            )
        )
        for i in range(case_count):
            for name in (*ONNX_INPUT_NAMES, "expected_prediction"):
                output.write(
                    arrays[name][i].astype("<f4", copy=False).tobytes(order="C")
                )

    metadata = {
        "format": "deeponet-golden-v1",
        "endianness": "little",
        "dtype": "float32",
        "header": {
            "magic": GOLDEN_MAGIC.decode("ascii"),
            "case_count": case_count,
            "branch_dimension": branch_input_dimension,
        },
        "record_layout": [
            {"name": "branch_input", "float32_count": branch_input_dimension},
            {"name": "horizon", "float32_count": 1},
            {"name": "current_moisture", "float32_count": 1},
            {"name": "expected_prediction", "float32_count": 1},
        ],
        "source": str(source.relative_to(repository_root)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "binary": str(binary_output.relative_to(repository_root)),
        "binary_size_bytes": binary_output.stat().st_size,
        "expected_values_recomputed": False,
    }
    metadata_output.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return case_count


def load_frozen_deeponet(
    checkpoint_path: str | Path,
    metadata: dict[str, Any],
    device: torch.device = torch.device("cpu"),
) -> tuple[ResidualDeepONet, dict[str, Any]]:
    model, checkpoint = load_deeponet_checkpoint(checkpoint_path, metadata, device)
    if not checkpoint.get("reference_model"):
        raise ValueError("Checkpoint is not marked as the frozen reference model")
    if not checkpoint.get("selection_frozen_before_test"):
        raise ValueError("Checkpoint selection is not marked as frozen")
    model.eval()
    return model, checkpoint


def prepare_onnx_inputs(
    model: ResidualDeepONet,
    historical_input: torch.Tensor,
    horizon_hours: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Convert prepared model histories into the explicit ONNX contract."""
    if historical_input.ndim != 3:
        raise ValueError("historical_input must have shape [batch, history, features]")
    historical_input = historical_input.to(
        device=model.target_std.device, dtype=torch.float32
    )
    horizon_hours = horizon_hours.to(
        device=model.target_std.device, dtype=torch.float32
    ).reshape(-1, 1)
    if historical_input.shape[0] != horizon_hours.shape[0]:
        raise ValueError("Historical input and horizon batch sizes differ")

    branch_parts = [historical_input.flatten(start_dim=1)]
    if model.temporal_feature_names:
        physical_trends = derive_temporal_features(
            historical_input,
            model.temporal_feature_names,
            feature_order=model.feature_order,
            feature_means=model.input_feature_means,
            feature_stds=model.input_feature_stds,
        )
        branch_parts.append(
            (physical_trends - model.temporal_feature_means)
            / model.temporal_feature_stds
        )
    branch_input = torch.cat(branch_parts, dim=1)
    if branch_input.shape[1] != model.branch_input_dimension:
        raise RuntimeError("Prepared Branch input dimension does not match checkpoint")

    normalized_current = historical_input[
        :, -1, model.soil_feature_index : model.soil_feature_index + 1
    ]
    current_moisture = normalized_current * model.soil_std + model.soil_mean
    return {
        "branch_input": branch_input.detach().cpu().numpy().astype(np.float32),
        "horizon": (horizon_hours / 24.0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        "current_moisture": current_moisture.detach()
        .cpu()
        .numpy()
        .astype(np.float32),
    }


def pytorch_deployment_prediction(
    wrapper: DeepONetONNXWrapper,
    inputs: dict[str, np.ndarray],
) -> np.ndarray:
    device = wrapper.residual_std.device
    with torch.inference_mode():
        prediction = wrapper(
            torch.from_numpy(inputs["branch_input"]).to(device),
            torch.from_numpy(inputs["horizon"]).to(device),
            torch.from_numpy(inputs["current_moisture"]).to(device),
        )
    return prediction.cpu().numpy()


def sample_batch(
    dataset: Any,
    *,
    horizon_hours: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    matching = np.flatnonzero(
        np.asarray(dataset.horizon_hours).reshape(-1) == horizon_hours
    )
    if matching.size < batch_size:
        raise ValueError(
            f"Only {matching.size} samples available for {horizon_hours}h"
        )
    positions = np.linspace(0, matching.size - 1, batch_size, dtype=np.int64)
    indices = matching[positions]
    samples = [dataset[int(index)] for index in indices]
    return (
        torch.stack([sample["branch_input"] for sample in samples]),
        torch.stack([sample["horizon"] for sample in samples]),
        torch.stack([sample["target"] for sample in samples]),
        indices,
    )
