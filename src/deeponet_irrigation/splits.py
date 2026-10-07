from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .data_loading import DataValidationError


@dataclass(frozen=True)
class SplitRange:
    name: str
    start_index: int
    end_index: int
    start_timestamp: str
    end_timestamp: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def make_chronological_splits(index: pd.DatetimeIndex) -> tuple[SplitRange, ...]:
    if len(index) < 3:
        raise DataValidationError("Not enough timestamps for train/validation/test splits")

    train_rows = int(len(index) * 0.70)
    validation_rows = int(len(index) * 0.15)
    boundaries = (
        ("train", 0, train_rows - 1),
        ("validation", train_rows, train_rows + validation_rows - 1),
        ("test", train_rows + validation_rows, len(index) - 1),
    )
    splits = tuple(
        SplitRange(
            name=name,
            start_index=start,
            end_index=end,
            start_timestamp=str(index[start]),
            end_timestamp=str(index[end]),
        )
        for name, start, end in boundaries
    )
    for previous, current in zip(splits, splits[1:], strict=False):
        if previous.end_index >= current.start_index:
            raise DataValidationError("Chronological split ranges overlap")
        if pd.Timestamp(previous.end_timestamp) >= pd.Timestamp(current.start_timestamp):
            raise DataValidationError("Chronological split timestamps overlap")
    return splits
