from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from deeponet_irrigation.data_download import DATASET_DOI, DATASET_VERSION
from deeponet_irrigation.data_loading import (
    DataValidationError,
    compare_sources,
    load_sector_table,
    source_path,
)
from deeponet_irrigation.preprocessing import (
    FEATURE_COLUMNS,
    FLATLINE_STD_THRESHOLD,
    FLATLINE_WINDOW_READINGS,
    HISTORY_HOURS,
    HISTORY_STEPS,
    HORIZON_HOURS,
    HORIZON_STEPS,
    INTERPOLATION_LIMIT_STEPS,
    MOISTURE_CLIP_MAX,
    MOISTURE_VALID_MAX,
    MOISTURE_VALID_MIN,
    SPIKE_THRESHOLD,
    SPIKE_WINDOW_STEPS,
    TARGET_COLUMN,
    clean_merged_sector,
    evaluate_sector,
)
from deeponet_irrigation.splits import make_chronological_splits
from deeponet_irrigation.temporal_windows import (
    fit_feature_normalization,
    generate_samples,
    save_prepared_data,
    select_modeling_support,
    validate_prepared_data,
)

EVALUATED_SECTORS = (1, 2, 3, 4)
SELECTION_CANDIDATES = (1, 2, 4)
OUTPUT_DIRECTORY = Path("data/processed/arnesano_v2")


def _print_source_comparison(comparisons: list[dict[str, object]]) -> None:
    columns = [
        "sector",
        "source",
        "rows",
        "regular_10min_percent",
        "timestamp_gaps",
        "maximum_gap",
        "soil_moisture_missing",
        "selected_weather_missing",
        "irrigation_missing",
    ]
    print("Merged vs. preprocessed")
    print(pd.DataFrame(comparisons)[columns].to_string(index=False))


def _print_sector_comparison(qualities: list[dict[str, object]]) -> None:
    columns = [
        "sector",
        "selection_status",
        "usable_start",
        "usable_end",
        "clean_soil_percent",
        "irrigation_completeness_percent",
        "weather_completeness_percent",
        "valid_history_windows",
        "valid_targets_1h",
        "valid_targets_3h",
        "valid_targets_6h",
        "valid_targets_12h",
        "valid_targets_24h",
        "origins_valid_all_horizons",
    ]
    print("\nSector quality after documented cleaning")
    print(pd.DataFrame(qualities)[columns].to_string(index=False))


def _sample_summary(samples: object, timestamps: pd.DatetimeIndex) -> dict[str, object]:
    count = len(samples)
    if count == 0:
        return {"count": 0}
    horizons = samples.horizon_hours[:, 0].astype(int)
    return {
        "count": count,
        "history_start_min": str(timestamps[samples.history_start_index.min()]),
        "current_min": str(timestamps[samples.current_index.min()]),
        "current_max": str(timestamps[samples.current_index.max()]),
        "target_min": str(timestamps[samples.target_index.min()]),
        "target_max": str(timestamps[samples.target_index.max()]),
        "counts_by_horizon": {
            f"{hours}h": int((horizons == hours).sum()) for hours in HORIZON_HOURS
        },
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    try:
        provenance_path = project_root / "data/raw/arnesano-v2.metadata.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("doi") != DATASET_DOI or provenance.get("version") != DATASET_VERSION:
            raise DataValidationError(
                "Local dataset provenance is not the pinned Arnesano v2 release"
            )

        source_comparison_objects = compare_sources(project_root, EVALUATED_SECTORS)
        source_comparisons = [item.to_dict() for item in source_comparison_objects]
        _print_source_comparison(source_comparisons)
        merged_reports = [item for item in source_comparison_objects if item.source == "merged"]
        preprocessed_reports = [
            item for item in source_comparison_objects if item.source == "preprocessed"
        ]
        if any(item.timestamp_gaps != 0 for item in merged_reports):
            raise DataValidationError("A merged source is not a complete 10-minute grid")
        if not any(item.timestamp_gaps > 0 for item in preprocessed_reports):
            raise DataValidationError(
                "Expected preprocessed timestamp gaps were not found; inspect dataset changes"
            )

        prepared_frames = {}
        cleaning_reports = {}
        quality_objects = []
        for sector in EVALUATED_SECTORS:
            merged = load_sector_table(project_root, "merged", sector)
            prepared, cleaning = clean_merged_sector(merged, sector)
            quality = evaluate_sector(
                sector, prepared, eligible_for_selection=sector in SELECTION_CANDIDATES
            )
            prepared_frames[sector] = prepared
            cleaning_reports[sector] = cleaning.to_dict()
            quality_objects.append(quality)

        qualities = [item.to_dict() for item in quality_objects]
        _print_sector_comparison(qualities)
        candidates = [
            item for item in quality_objects if item.sector in SELECTION_CANDIDATES
        ]
        selected_quality = max(
            candidates,
            key=lambda item: (
                item.origins_valid_all_horizons,
                item.valid_history_windows,
                item.clean_soil_percent,
            ),
        )
        selected_sector = selected_quality.sector
        selected = prepared_frames[selected_sector]
        modeling_frame, support_report = select_modeling_support(selected)
        splits = make_chronological_splits(modeling_frame.index)
        samples, generation_report = generate_samples(modeling_frame, splits)
        normalization, normalization_fit, normalized_features = (
            fit_feature_normalization(modeling_frame, samples["train"])
        )
        validate_prepared_data(
            modeling_frame,
            normalized_features,
            samples,
            splits,
            normalization_fit,
        )

        sample_summaries = {
            split.name: _sample_summary(samples[split.name], modeling_frame.index)
            for split in splits
        }
        full_source_candidates = sum(
            max(0, len(selected) - (HISTORY_STEPS - 1) - steps)
            for steps in HORIZON_STEPS
        )
        dropped_outside_support = (
            full_source_candidates - generation_report.temporal_candidates
        )
        full_source_dropped = full_source_candidates - generation_report.accepted_samples
        generation_metadata = {
            **generation_report.to_dict(),
            "temporal_candidates": full_source_candidates,
            "modeling_support_candidates": generation_report.temporal_candidates,
            "dropped_outside_modeling_support": dropped_outside_support,
            "dropped_total": full_source_dropped,
            "dropped_percent": round(
                100.0 * full_source_dropped / full_source_candidates, 2
            ),
            "modeling_support_dropped_percent": generation_report.dropped_percent,
        }
        output_dir = project_root / OUTPUT_DIRECTORY
        metadata = {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dataset": {
                "doi": DATASET_DOI,
                "version": DATASET_VERSION,
                "archive_sha256": provenance.get("archive_sha256"),
            },
            "selected_data_source": "merged",
            "source_file": str(
                source_path(project_root, "merged", selected_sector).relative_to(
                    project_root
                )
            ),
            "source_selection_reason": (
                "Merged tables preserve the complete strict 10-minute grid. The "
                "author-preprocessed tables remove rows and contain multi-day timestamp gaps."
            ),
            "selected_sector": selected_sector,
            "sector_selection_reason": (
                "Highest number of origins valid at every requested horizon under "
                "the same conservative gap policy among sectors 1, 2, and 4."
            ),
            "source_comparison": source_comparisons,
            "sector_comparison": qualities,
            "cleaning_reports": cleaning_reports,
            "modeling_support": support_report,
            "features": list(FEATURE_COLUMNS),
            "feature_order": list(FEATURE_COLUMNS),
            "feature_notes": {
                "applied_water_liters": (
                    "Author-provided liters_total lagged by one 10-minute bin so "
                    "only a fully elapsed irrigation interval is visible at time t."
                )
            },
            "target_column": TARGET_COLUMN,
            "sampling_interval_minutes": 10,
            "history_duration_hours": HISTORY_HOURS,
            "history_steps": HISTORY_STEPS,
            "history_policy": (
                "144 consecutive 10-minute observations ending at the prediction "
                "origin; any remaining missing feature rejects the sample."
            ),
            "prediction_horizons_hours": list(HORIZON_HOURS),
            "horizon_offsets_steps": {
                f"{hours}h": steps
                for hours, steps in zip(HORIZON_HOURS, HORIZON_STEPS, strict=True)
            },
            "split_policy": (
                "70/15/15 contiguous time ranges. Histories and targets must both "
                "stay inside their split; boundary-crossing samples are purged."
            ),
            "splits": {split.name: split.to_dict() for split in splits},
            "samples": sample_summaries,
            "generation": generation_metadata,
            "gap_policy": {
                "timestamp_grid": "Strict 10-minute grid; structural gaps are fatal.",
                "soil_moisture_valid_range": [
                    MOISTURE_VALID_MIN,
                    MOISTURE_VALID_MAX,
                ],
                "soil_moisture_clip_max": MOISTURE_CLIP_MAX,
                "short_gap_interpolation_max_steps": INTERPOLATION_LIMIT_STEPS,
                "short_gap_interpolation_max_minutes": (
                    INTERPOLATION_LIMIT_STEPS * 10
                ),
                "interpolation_rule": (
                    "Time interpolation only when an entire interior missing run "
                    "is at most two bins; longer runs remain missing."
                ),
                "spike_rule": (
                    f"Values more than {SPIKE_THRESHOLD:g} points from a centered "
                    f"{SPIKE_WINDOW_STEPS}-sample median are removed, then only "
                    "short complete gaps may be interpolated."
                ),
                "trailing_flatline_rule": (
                    f"Trim after the last point whose next {FLATLINE_WINDOW_READINGS} "
                    f"observed readings have standard deviation above "
                    f"{FLATLINE_STD_THRESHOLD:g}; this rejects stalled probe tails."
                ),
                "long_gap_policy": "Reject every history or target touching it.",
            },
            "normalization": {
                **normalization_fit,
                "statistics": normalization,
                "target_normalized": False,
            },
            "storage": {
                "features": "features.npy (normalized grid rows; may contain unused NaNs)",
                "timestamps": "timestamps.npy",
                "samples": "{train,validation,test}_samples.npz",
                "branch_resolution": (
                    "features[history_start_index:current_index + 1], shape [144, 5]"
                ),
            },
        }
        save_prepared_data(
            output_dir,
            modeling_frame.index,
            normalized_features,
            samples,
            metadata,
        )
    except (OSError, ValueError, DataValidationError) as exc:
        raise SystemExit(f"Dataset preparation failed: {exc}") from exc

    print(
        f"\nSelected merged sector {selected_sector}: "
        f"{selected_quality.origins_valid_all_horizons:,} origins valid at all horizons."
    )
    print("Features: " + ", ".join(FEATURE_COLUMNS))
    print(f"Target: {TARGET_COLUMN}")
    print("Splits")
    for split in splits:
        summary = sample_summaries[split.name]
        print(
            f"  {split.name}: {split.start_timestamp} to {split.end_timestamp}; "
            f"{summary['count']:,} samples"
        )
    print(
        f"Accepted {generation_report.accepted_samples:,} / "
        f"{generation_metadata['temporal_candidates']:,} full-source temporal "
        f"candidates; dropped {generation_metadata['dropped_total']:,} "
        f"({generation_metadata['dropped_percent']:.2f}%)."
    )
    print("Sanity checks passed.")
    print(f"Prepared data written to {output_dir}")


if __name__ == "__main__":
    main()
