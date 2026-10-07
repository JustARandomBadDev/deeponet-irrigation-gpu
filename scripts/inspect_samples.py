from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from deeponet_irrigation.data_loading import DataValidationError
from deeponet_irrigation.temporal_windows import load_sample_arrays

OUTPUT_DIRECTORY = Path("data/processed/arnesano_v2")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / OUTPUT_DIRECTORY
    metadata_path = output_dir / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        features = np.load(output_dir / "features.npy", mmap_mode="r", allow_pickle=False)
        timestamps = np.load(
            output_dir / "timestamps.npy", mmap_mode="r", allow_pickle=False
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "Prepared data is unavailable or invalid. Run scripts/prepare_dataset.py "
            f"first: {exc}"
        ) from exc

    print(f"Source: {metadata['selected_data_source']}")
    print(f"Sector: {metadata['selected_sector']}")
    print("Features: " + ", ".join(metadata["feature_order"]))
    print(f"Target: {metadata['target_column']}")
    print(f"Sampling interval: {metadata['sampling_interval_minutes']} minutes")
    print(
        f"History: {metadata['history_duration_hours']} hours / "
        f"{metadata['history_steps']} steps"
    )
    offsets = metadata["horizon_offsets_steps"]
    print(
        "Horizons: "
        + ", ".join(
            f"{hours}h={offsets[f'{hours}h']} steps"
            for hours in metadata["prediction_horizons_hours"]
        )
    )
    generation = metadata["generation"]
    print(
        f"Dropped windows: {generation['dropped_total']:,} / "
        f"{generation['temporal_candidates']:,} ({generation['dropped_percent']:.2f}%)"
    )
    print(
        "  outside usable modeling support: "
        f"{generation['dropped_outside_modeling_support']:,}; split boundaries: "
        f"{generation['dropped_split_boundary']:,}; history gaps: "
        f"{generation['dropped_history_gap']:,}; missing targets: "
        f"{generation['dropped_missing_target']:,}"
    )
    support = metadata["modeling_support"]
    print(
        f"Modeling support: {support['support_start']} to {support['support_end']} "
        f"({generation['modeling_support_candidates']:,} candidates; "
        f"{generation['modeling_support_dropped_percent']:.2f}% dropped within support)"
    )
    cleaning = metadata["cleaning_reports"][str(metadata["selected_sector"])]
    print(
        f"Gaps: {cleaning['timestamp_gaps']} timestamp gaps; "
        f"{cleaning['interpolated_gap_runs']:,} short soil-moisture runs interpolated; "
        f"{cleaning['remaining_missing_runs']:,} runs remain missing"
    )
    print(f"Normalization statistics: {metadata_path}")

    print("\nSplits and examples")
    for split_name in ("train", "validation", "test"):
        split = metadata["splits"][split_name]
        summary = metadata["samples"][split_name]
        try:
            samples = load_sample_arrays(output_dir / f"{split_name}_samples.npz")
        except DataValidationError as exc:
            raise SystemExit(str(exc)) from exc
        print(
            f"\n{split_name}: {split['start_timestamp']} to {split['end_timestamp']}; "
            f"{len(samples):,} samples"
        )
        print(
            f"  origins {summary['current_min']} to {summary['current_max']}; "
            f"targets {summary['target_min']} to {summary['target_max']}"
        )
        available_horizons = samples.horizon_hours[:, 0].astype(int)
        example_positions = tuple(
            int(np.flatnonzero(available_horizons == horizon)[0])
            for horizon in (1, 6, 24)
        )
        for position in example_positions:
            start = int(samples.history_start_index[position])
            current = int(samples.current_index[position])
            target_index = int(samples.target_index[position])
            horizon = int(samples.horizon_hours[position, 0])
            branch = features[start : current + 1]
            actual_delta = pd.Timestamp(timestamps[target_index]) - pd.Timestamp(
                timestamps[current]
            )
            print(
                f"  sample {position:,}: history {pd.Timestamp(timestamps[start])} -> "
                f"{pd.Timestamp(timestamps[current])}; target "
                f"{pd.Timestamp(timestamps[target_index])}; horizon {horizon}h "
                f"(actual {actual_delta}); branch {branch.shape}; "
                f"target={samples.target[position, 0]:.3f}"
            )


if __name__ == "__main__":
    main()
