from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


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
        return {
            "branch_input": torch.from_numpy(branch.copy()),
            "horizon": torch.from_numpy(self.horizon_hours[index].copy()),
            "target": torch.from_numpy(self.targets[index].copy()),
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
