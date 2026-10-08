from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .temporal_features import derive_temporal_features


SPLIT_NAMES = ("train", "validation", "test")


class PreparedTemporalDataset(Dataset[dict[str, torch.Tensor]]):
    """Reconstruct temporal branches from the compact prepared-data indices."""

    def __init__(self, data_dir: Path | str, split: str) -> None:
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split {split!r}; expected one of {SPLIT_NAMES}")

        self.data_dir = Path(data_dir)
        self.split = split
        self.metadata: dict[str, Any] = json.loads(
            (self.data_dir / "metadata.json").read_text(encoding="utf-8")
        )
        self.features = np.load(self.data_dir / "features.npy", mmap_mode="r")
        self.timestamps = np.load(self.data_dir / "timestamps.npy", mmap_mode="r")

        with np.load(self.data_dir / f"{split}_samples.npz") as samples:
            required = {
                "history_start_index",
                "current_index",
                "target_index",
                "horizon_hours",
                "target",
            }
            if set(samples.files) != required:
                raise ValueError(
                    f"Unexpected arrays in {split}_samples.npz: {samples.files}"
                )
            self.history_start_index = samples["history_start_index"]
            self.current_index = samples["current_index"]
            self.target_index = samples["target_index"]
            self.horizon_hours = samples["horizon_hours"]
            self.targets = samples["target"]

        self.history_steps = int(self.metadata["history_steps"])
        self.feature_count = len(self.metadata["feature_order"])
        self.soil_index = self.metadata["feature_order"].index("soil_moisture")
        soil_statistics = self.metadata["normalization"]["statistics"][
            "soil_moisture"
        ]
        self.soil_mean = float(soil_statistics["mean"])
        self.soil_std = float(soil_statistics["std"])
        self._validate()

    def _validate(self) -> None:
        lengths = {
            len(self.history_start_index),
            len(self.current_index),
            len(self.target_index),
            len(self.horizon_hours),
            len(self.targets),
        }
        if len(lengths) != 1:
            raise ValueError(f"Sample arrays have inconsistent lengths in {self.split}")
        if self.features.ndim != 2 or self.features.shape[1] != self.feature_count:
            raise ValueError("features.npy does not match metadata feature order")
        if self.timestamps.shape[0] != self.features.shape[0]:
            raise ValueError("features.npy and timestamps.npy have different row counts")
        window_lengths = self.current_index - self.history_start_index + 1
        if not np.all(window_lengths == self.history_steps):
            raise ValueError(f"Inconsistent branch lengths in {self.split}")
        if not np.all(self.target_index > self.current_index):
            raise ValueError(f"Non-future targets found in {self.split}")

    def __len__(self) -> int:
        return len(self.current_index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = int(self.history_start_index[index])
        current = int(self.current_index[index])
        branch = np.asarray(self.features[start : current + 1], dtype=np.float32)
        if branch.shape != (self.history_steps, self.feature_count):
            raise RuntimeError(f"Invalid branch shape {branch.shape} at sample {index}")

        # Copy memmapped slices so PyTorch always receives writable storage.
        current_soil_moisture = (
            float(branch[-1, self.soil_index]) * self.soil_std + self.soil_mean
        )
        target = torch.from_numpy(self.targets[index].copy())
        return {
            "branch_input": torch.from_numpy(branch.copy()),
            "horizon": torch.from_numpy(self.horizon_hours[index].copy()),
            "target": target,
            "current_soil_moisture": torch.tensor(
                [current_soil_moisture], dtype=torch.float32
            ),
            "residual_target": target
            - torch.tensor([current_soil_moisture], dtype=torch.float32),
            "current_index": torch.tensor(current, dtype=torch.int64),
            "target_index": torch.tensor(
                int(self.target_index[index]), dtype=torch.int64
            ),
        }


def make_data_loader(
    dataset: PreparedTemporalDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
) -> DataLoader[dict[str, torch.Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def residual_targets(dataset: PreparedTemporalDataset) -> np.ndarray:
    """Return future minus current soil moisture in physical units."""
    normalized_current = np.asarray(
        dataset.features[dataset.current_index, dataset.soil_index],
        dtype=np.float64,
    )
    current = normalized_current * dataset.soil_std + dataset.soil_mean
    return np.asarray(dataset.targets, dtype=np.float64).reshape(-1) - current


def fit_residual_normalization(
    train_dataset: PreparedTemporalDataset,
) -> dict[str, float | str | int]:
    residual = residual_targets(train_dataset)
    standard_deviation = float(residual.std(ddof=0))
    if standard_deviation <= 0:
        raise ValueError("Training residual targets have zero variance")
    return {
        "mean": float(residual.mean()),
        "std": standard_deviation,
        "fit_split": "train",
        "sample_count": int(residual.size),
        "definition": "future_soil_moisture - current_soil_moisture",
    }


def summarize_change_weighting(
    train_dataset: PreparedTemporalDataset,
    configuration: dict[str, float],
) -> dict[str, float | str | int]:
    magnitude = np.abs(residual_targets(train_dataset))
    change = magnitude > configuration["change_threshold"]
    large = magnitude > configuration["large_change_threshold"]
    return {
        **configuration,
        "fit_split": "train",
        "sample_count": int(magnitude.size),
        "tiny_fraction": float(np.mean(~change)),
        "change_fraction": float(np.mean(change & ~large)),
        "large_change_fraction": float(np.mean(large)),
    }


def fit_temporal_feature_normalization(
    train_dataset: PreparedTemporalDataset,
    names: tuple[str, ...],
) -> dict[str, object]:
    if train_dataset.split != "train":
        raise ValueError("Temporal feature normalization must use the train split")
    statistics = train_dataset.metadata["normalization"]["statistics"]
    feature_order = tuple(train_dataset.metadata["feature_order"])
    means = torch.tensor(
        [statistics[name]["mean"] for name in feature_order], dtype=torch.float32
    )
    stds = torch.tensor(
        [statistics[name]["std"] for name in feature_order], dtype=torch.float32
    )
    values = []
    for batch in make_data_loader(
        train_dataset, batch_size=1024, shuffle=False, seed=42
    ):
        values.append(
            derive_temporal_features(
                batch["branch_input"],
                names,
                feature_order=feature_order,
                feature_means=means,
                feature_stds=stds,
            )
        )
    array = torch.cat(values).numpy().astype(np.float64)
    derived_means = array.mean(axis=0)
    derived_stds = array.std(axis=0, ddof=0)
    if np.any(derived_stds <= 0):
        raise ValueError("A derived temporal feature is constant in train")
    return {
        "names": list(names),
        "means": derived_means.tolist(),
        "stds": derived_stds.tolist(),
        "fit_split": "train",
        "sample_count": len(train_dataset),
    }
