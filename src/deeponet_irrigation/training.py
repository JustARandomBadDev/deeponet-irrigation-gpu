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
    loss: str = "mse"
    weight_decay: float = 0.0
    change_weighting: dict[str, float] | None = None


def make_loss(name: str, *, reduction: str = "mean") -> nn.Module:
    if name == "mse":
        return nn.MSELoss(reduction=reduction)
    if name == "huber":
        return nn.HuberLoss(delta=1.0, reduction=reduction)
    raise ValueError(f"Unknown loss {name!r}")


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


def change_weights(
    physical_residual: torch.Tensor,
    configuration: dict[str, float] | None,
) -> torch.Tensor:
    weights = torch.ones_like(physical_residual)
    if configuration is None:
        return weights
    magnitude = physical_residual.abs()
    weights = torch.where(
        magnitude > configuration["change_threshold"],
        torch.full_like(weights, configuration["change_weight"]),
        weights,
    )
    return torch.where(
        magnitude > configuration["large_change_threshold"],
        torch.full_like(weights, configuration["large_change_weight"]),
        weights,
    )


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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = make_loss(config.loss)
    elementwise_loss = make_loss(config.loss, reduction="none")
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
            losses = elementwise_loss(prediction, target)
            weights = change_weights(
                batch["residual_target"].to(device, non_blocking=True),
                config.change_weighting,
            )
            loss = (losses * weights).sum() / weights.sum()
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
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"Epoch {epoch:03d}: train {config.loss}={train_loss:.6f}, "
            f"validation {config.loss}={validation_loss:.6f}"
        )

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            _atomic_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": model.architecture(),
                    "target_mode": model.target_mode,
                    "normalization_metadata": {
                        "output_mean": float(model.target_mean.item()),
                        "output_std": float(model.target_std.item()),
                        "fit_split": "train",
                    },
                    "training_config": asdict(config),
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
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
        "best_validation_loss": best_loss,
        "epochs_completed": len(history),
        "duration_seconds": time.perf_counter() - started,
        "history": history,
        "checkpoint": str(checkpoint_path),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validation_selection_key(experiment: dict[str, object]) -> tuple[float, ...]:
    """Rank an experiment without consulting any test-set information."""
    validation = experiment["validation_metrics"]
    assert isinstance(validation, dict)
    overall = validation["overall"]
    by_horizon = validation["by_horizon"]
    assert isinstance(overall, dict) and isinstance(by_horizon, dict)
    return (
        float(overall["rmse"]),
        float(by_horizon["24h"]["rmse"]),
        float(by_horizon["12h"]["rmse"]),
        -float(overall["r2"]),
        float(overall["mae"]),
    )


def select_best_experiment(
    experiments: list[dict[str, object]],
) -> dict[str, object]:
    if not experiments:
        raise ValueError("No experiments available for selection")
    return min(experiments, key=validation_selection_key)


def update_checkpoint_metadata(path: Path, updates: dict[str, object]) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint.update(updates)
    _atomic_torch_save(checkpoint, path)
