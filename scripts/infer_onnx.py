from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from deeponet_irrigation.dataset import PreparedTemporalDataset
from deeponet_irrigation.onnx_deployment import (
    ONNX_INPUT_NAMES,
    load_frozen_deeponet,
    prepare_onnx_inputs,
    sample_batch,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "arnesano_v2"
CHECKPOINT = ROOT / "models" / "deeponet_reference.pt"
ONNX_MODEL = ROOT / "models" / "deeponet_reference.onnx"


def main() -> None:
    dataset = PreparedTemporalDataset(DATA_DIR, "validation")
    model, _ = load_frozen_deeponet(
        CHECKPOINT, dataset.metadata, torch.device("cpu")
    )
    session = ort.InferenceSession(
        str(ONNX_MODEL), providers=["CPUExecutionProvider"]
    )
    print(f"Available providers: {ort.get_available_providers()}")
    for horizon in (1, 6, 24):
        history, horizon_hours, target, _ = sample_batch(
            dataset, horizon_hours=horizon, batch_size=1
        )
        inputs = prepare_onnx_inputs(model, history, horizon_hours)
        prediction = session.run(None, {name: inputs[name] for name in ONNX_INPUT_NAMES})[
            0
        ]
        print(
            f"current={inputs['current_moisture'][0, 0]:.6f} "
            f"horizon={horizon}h target={target[0, 0].item():.6f} "
            f"prediction={prediction[0, 0]:.6f} "
            f"absolute_error={abs(prediction[0, 0] - target[0, 0].item()):.6f}"
        )


if __name__ == "__main__":
    main()
