from __future__ import annotations

import json
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .models import MLPBaseline


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 512
    learning_rate: float = 1e-3
    max_epochs: int = 50
    patience: int = 8
    seed: int = 42


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _mean_loss(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    sample_count = 0
    with torch.inference_mode():
        for batch in loader:
            target = batch["target"].to(device, non_blocking=True)
            prediction = model(
                batch["branch_input"].to(device, non_blocking=True),
                batch["horizon"].to(device, non_blocking=True),
            )
            total += float(loss_function(prediction, target).item()) * len(target)
            sample_count += len(target)
    return total / sample_count


def _atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def train_mlp(
    model: MLPBaseline,
    train_loader: DataLoader[dict[str, torch.Tensor]],
    validation_loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    config: TrainingConfig,
    checkpoint_path: Path,
    preprocessing_metadata: dict[str, object],
) -> dict[str, object]:
    set_deterministic_seed(config.seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.MSELoss()
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total = 0.0
        sample_count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            target = batch["target"].to(device, non_blocking=True)
            prediction = model(
                batch["branch_input"].to(device, non_blocking=True),
                batch["horizon"].to(device, non_blocking=True),
            )
            loss = loss_function(prediction, target)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(target)
            sample_count += len(target)

        train_loss = total / sample_count
        validation_loss = _mean_loss(
            model, validation_loader, loss_function, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "validation_mse": validation_loss,
            }
        )
        print(
            f"Epoch {epoch:03d}: train MSE={train_loss:.6f}, "
            f"validation MSE={validation_loss:.6f}"
        )

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            _atomic_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": model.architecture(),
                    "training_config": asdict(config),
                    "best_epoch": best_epoch,
                    "best_validation_mse": best_loss,
                    "preprocessing_contract": {
                        "dataset": preprocessing_metadata["dataset"],
                        "selected_data_source": preprocessing_metadata[
                            "selected_data_source"
                        ],
                        "selected_sector": preprocessing_metadata["selected_sector"],
                        "feature_order": preprocessing_metadata["feature_order"],
                        "target_column": preprocessing_metadata["target_column"],
                        "history_steps": preprocessing_metadata["history_steps"],
                        "prediction_horizons_hours": preprocessing_metadata[
                            "prediction_horizons_hours"
                        ],
                    },
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    return {
        "torch_version": torch.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "architecture": model.architecture(),
        "trainable_parameters": model.trainable_parameter_count(),
        "training_config": asdict(config),
        "best_epoch": best_epoch,
        "best_validation_mse": best_loss,
        "epochs_completed": len(history),
        "duration_seconds": time.perf_counter() - started,
        "history": history,
        "checkpoint": str(checkpoint_path),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
