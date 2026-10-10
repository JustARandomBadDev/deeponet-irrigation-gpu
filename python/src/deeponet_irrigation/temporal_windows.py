from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loading import EXPECTED_INTERVAL, DataValidationError
from .preprocessing import (
    FEATURE_COLUMNS,
    HISTORY_STEPS,
    HORIZON_HOURS,
    HORIZON_STEPS,
    TARGET_COLUMN,
)
from .splits import SplitRange


@dataclass
class SampleArrays:
    history_start_index: np.ndarray
    current_index: np.ndarray
    target_index: np.ndarray
    horizon_hours: np.ndarray
    target: np.ndarray

    def __len__(self) -> int:
        return len(self.current_index)


@dataclass(frozen=True)
class GenerationReport:
    temporal_candidates: int
    accepted_samples: int
    dropped_total: int
    dropped_percent: float
    dropped_split_boundary: int
    dropped_history_gap: int
    dropped_missing_target: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def select_modeling_support(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    complete = frame[list(FEATURE_COLUMNS)].notna().all(axis=1)
    valid_history = (
        complete.rolling(HISTORY_STEPS, min_periods=HISTORY_STEPS).sum()
        == HISTORY_STEPS
    )
    any_target = pd.Series(False, index=frame.index)
    for steps in HORIZON_STEPS:
        any_target |= frame[TARGET_COLUMN].shift(-steps).notna()
    eligible = valid_history & any_target
    positions = np.flatnonzero(eligible.to_numpy())
    if len(positions) == 0:
        raise DataValidationError("No complete 24-hour histories with a future target")

    first_current = int(positions[0])
    support_start = first_current - (HISTORY_STEPS - 1)
    target_positions = np.flatnonzero(frame[TARGET_COLUMN].notna().to_numpy())
    support_end = int(target_positions[-1])
    selected = frame.iloc[support_start : support_end + 1].copy()
    report = {
        "support_start": str(selected.index.min()),
        "support_end": str(selected.index.max()),
        "first_valid_prediction_origin": str(frame.index[first_current]),
        "source_rows_before_support": support_start,
        "source_rows_after_support": len(frame) - support_end - 1,
        "reason": (
            "Starts at the history start of the first complete 24-hour window; "
            "earlier data cannot form a valid history under the gap policy."
        ),
    }
    return selected, report


def _empty_samples() -> SampleArrays:
    return SampleArrays(
        history_start_index=np.empty(0, dtype=np.int32),
        current_index=np.empty(0, dtype=np.int32),
        target_index=np.empty(0, dtype=np.int32),
        horizon_hours=np.empty((0, 1), dtype=np.float32),
        target=np.empty((0, 1), dtype=np.float32),
    )


def generate_samples(
    frame: pd.DataFrame, splits: tuple[SplitRange, ...]
) -> tuple[dict[str, SampleArrays], GenerationReport]:
    values = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    target_values = frame[TARGET_COLUMN].to_numpy(dtype=np.float64)
    finite_rows = np.isfinite(values).all(axis=1)

    records: dict[str, dict[str, list[float | int]]] = {
        split.name: {
            "history_start_index": [],
            "current_index": [],
            "target_index": [],
            "horizon_hours": [],
            "target": [],
        }
        for split in splits
    }
    split_by_position = np.empty(len(frame), dtype=object)
    split_lookup = {split.name: split for split in splits}
    for split in splits:
        split_by_position[split.start_index : split.end_index + 1] = split.name

    candidates = 0
    dropped_boundary = 0
    dropped_history = 0
    dropped_target = 0
    for current in range(HISTORY_STEPS - 1, len(frame)):
        history_start = current - HISTORY_STEPS + 1
        split_name = str(split_by_position[current])
        split = split_lookup[split_name]
        for hours, steps in zip(HORIZON_HOURS, HORIZON_STEPS, strict=True):
            target_index = current + steps
            if target_index >= len(frame):
                continue
            candidates += 1
            if history_start < split.start_index or target_index > split.end_index:
                dropped_boundary += 1
                continue
            if not finite_rows[history_start : current + 1].all():
                dropped_history += 1
                continue
            if not np.isfinite(target_values[target_index]):
                dropped_target += 1
                continue

            record = records[split_name]
            record["history_start_index"].append(history_start)
            record["current_index"].append(current)
            record["target_index"].append(target_index)
            record["horizon_hours"].append(float(hours))
            record["target"].append(float(target_values[target_index]))

    samples: dict[str, SampleArrays] = {}
    for split_name, record in records.items():
        if not record["current_index"]:
            samples[split_name] = _empty_samples()
            continue
        samples[split_name] = SampleArrays(
            history_start_index=np.asarray(
                record["history_start_index"], dtype=np.int32
            ),
            current_index=np.asarray(record["current_index"], dtype=np.int32),
            target_index=np.asarray(record["target_index"], dtype=np.int32),
            horizon_hours=np.asarray(
                record["horizon_hours"], dtype=np.float32
            ).reshape(-1, 1),
            target=np.asarray(record["target"], dtype=np.float32).reshape(-1, 1),
        )

    accepted = sum(len(split_samples) for split_samples in samples.values())
    dropped = candidates - accepted
    report = GenerationReport(
        temporal_candidates=candidates,
        accepted_samples=accepted,
        dropped_total=dropped,
        dropped_percent=round(100.0 * dropped / candidates, 2),
        dropped_split_boundary=dropped_boundary,
        dropped_history_gap=dropped_history,
        dropped_missing_target=dropped_target,
    )
    return samples, report


def fit_feature_normalization(
    frame: pd.DataFrame, train_samples: SampleArrays
) -> tuple[dict[str, dict[str, float]], dict[str, object], np.ndarray]:
    if len(train_samples) == 0:
        raise DataValidationError("Training split contains no samples")

    difference = np.zeros(len(frame) + 1, dtype=np.int64)
    np.add.at(difference, train_samples.history_start_index, 1)
    np.add.at(difference, train_samples.current_index + 1, -1)
    training_rows = np.cumsum(difference[:-1]) > 0
    raw = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    fit_values = raw[training_rows]
    if not np.isfinite(fit_values).all():
        raise DataValidationError("Training normalization rows contain missing values")

    means = fit_values.mean(axis=0)
    standard_deviations = fit_values.std(axis=0, ddof=0)
    if (standard_deviations <= 0).any():
        constant = [
            name
            for name, value in zip(
                FEATURE_COLUMNS, standard_deviations, strict=True
            )
            if value <= 0
        ]
        raise DataValidationError(
            "Cannot normalize constant training features: " + ", ".join(constant)
        )

    statistics = {
        name: {"mean": float(mean), "std": float(std)}
        for name, mean, std in zip(
            FEATURE_COLUMNS, means, standard_deviations, strict=True
        )
    }
    used_positions = np.flatnonzero(training_rows)
    fit_report = {
        "method": "mean/std (population standard deviation)",
        "fit_split": "train",
        "unique_training_rows": int(training_rows.sum()),
        "fit_start": str(frame.index[used_positions[0]]),
        "fit_end": str(frame.index[used_positions[-1]]),
    }
    normalized = (raw - means) / standard_deviations
    return statistics, fit_report, normalized.astype(np.float32)


def validate_prepared_data(
    frame: pd.DataFrame,
    normalized_features: np.ndarray,
    samples: dict[str, SampleArrays],
    splits: tuple[SplitRange, ...],
    normalization_fit_report: dict[str, object],
) -> None:
    deltas = frame.index.to_numpy()[1:] - frame.index.to_numpy()[:-1]
    if not np.all(deltas == np.timedelta64(10, "m")):
        raise DataValidationError("Prepared timestamps are not a strict 10-minute grid")
    if normalized_features.shape != (len(frame), len(FEATURE_COLUMNS)):
        raise DataValidationError("Normalized feature matrix has the wrong shape")
    if any("future" in name or "water_vol_to" in name for name in FEATURE_COLUMNS):
        raise DataValidationError("A future-looking feature is configured")
    if "applied_water_liters" not in FEATURE_COLUMNS:
        raise DataValidationError("Historical applied water feature is missing")

    split_lookup = {split.name: split for split in splits}
    previous_end = -1
    for split in splits:
        if split.start_index <= previous_end:
            raise DataValidationError("Train/validation/test splits are not ordered")
        previous_end = split.end_index
        split_samples = samples[split.name]
        if len(split_samples) == 0:
            raise DataValidationError(f"{split.name} split contains no samples")
        lengths = split_samples.current_index - split_samples.history_start_index + 1
        if not np.all(lengths == HISTORY_STEPS):
            raise DataValidationError(f"Inconsistent branch shape in {split.name}")
        if not np.all(split_samples.target_index > split_samples.current_index):
            raise DataValidationError(f"Non-future target in {split.name}")
        if not np.all(split_samples.history_start_index >= split.start_index):
            raise DataValidationError(f"History crosses into the prior split in {split.name}")
        if not np.all(split_samples.target_index <= split.end_index):
            raise DataValidationError(f"Target crosses into the next split in {split.name}")
        if not np.all(np.diff(split_samples.current_index) >= 0):
            raise DataValidationError(f"Samples are not chronological in {split.name}")

        for position in range(len(split_samples)):
            start = int(split_samples.history_start_index[position])
            current = int(split_samples.current_index[position])
            target_index = int(split_samples.target_index[position])
            horizon = int(split_samples.horizon_hours[position, 0])
            expected_steps = HORIZON_STEPS[HORIZON_HOURS.index(horizon)]
            if target_index - current != expected_steps:
                raise DataValidationError("Target row offset does not match its horizon")
            if frame.index[target_index] - frame.index[current] != pd.Timedelta(
                hours=horizon
            ):
                raise DataValidationError("Target timestamp does not match its horizon")
            branch = normalized_features[start : current + 1]
            if branch.shape != (HISTORY_STEPS, len(FEATURE_COLUMNS)):
                raise DataValidationError("Resolved branch has the wrong shape")
            if not np.isfinite(branch).all():
                raise DataValidationError("A prepared branch violates the gap policy")
            expected_target = float(frame[TARGET_COLUMN].iloc[target_index])
            if not np.isclose(split_samples.target[position, 0], expected_target):
                raise DataValidationError("Stored target does not match source moisture")

    train_end = pd.Timestamp(split_lookup["train"].end_timestamp)
    if pd.Timestamp(str(normalization_fit_report["fit_end"])) > train_end:
        raise DataValidationError("Normalization uses data outside the training split")


def _atomic_numpy_write(path: Path, writer: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def save_prepared_data(
    output_dir: Path,
    timestamps: pd.DatetimeIndex,
    normalized_features: np.ndarray,
    samples: dict[str, SampleArrays],
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_numpy_write(
        output_dir / "features.npy",
        lambda stream: np.save(stream, normalized_features, allow_pickle=False),
    )
    timestamp_values = timestamps.to_numpy(dtype="datetime64[ns]")
    _atomic_numpy_write(
        output_dir / "timestamps.npy",
        lambda stream: np.save(stream, timestamp_values, allow_pickle=False),
    )
    for split_name, split_samples in samples.items():
        _atomic_numpy_write(
            output_dir / f"{split_name}_samples.npz",
            lambda stream, item=split_samples: np.savez_compressed(
                stream,
                history_start_index=item.history_start_index,
                current_index=item.current_index,
                target_index=item.target_index,
                horizon_hours=item.horizon_hours,
                target=item.target,
            ),
        )

    metadata_path = output_dir / "metadata.json"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{metadata_path.name}.", dir=output_dir, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, metadata_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_sample_arrays(path: Path) -> SampleArrays:
    try:
        with np.load(path, allow_pickle=False) as data:
            return SampleArrays(
                history_start_index=data["history_start_index"],
                current_index=data["current_index"],
                target_index=data["target_index"],
                horizon_hours=data["horizon_hours"],
                target=data["target"],
            )
    except (OSError, KeyError, ValueError) as exc:
        raise DataValidationError(f"Could not load prepared samples from {path}: {exc}") from exc
