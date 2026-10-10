from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .data_loading import EXPECTED_INTERVAL, DataValidationError, require_regular_grid

FEATURE_COLUMNS = (
    "soil_moisture",
    "applied_water_liters",
    "weather_rain",
    "weather_temp",
    "weather_humidity",
)
TARGET_COLUMN = "soil_moisture"
HISTORY_HOURS = 24
HISTORY_STEPS = 144
HORIZON_HOURS = (1, 3, 6, 12, 24)
HORIZON_STEPS = tuple(hours * 6 for hours in HORIZON_HOURS)

MOISTURE_VALID_MIN = 0.0
MOISTURE_VALID_MAX = 115.0
MOISTURE_CLIP_MAX = 100.0
INTERPOLATION_LIMIT_STEPS = 2
SPIKE_WINDOW_STEPS = 9
SPIKE_THRESHOLD = 15.0
FLATLINE_WINDOW_READINGS = 12
FLATLINE_STD_THRESHOLD = 0.001


@dataclass(frozen=True)
class CleaningReport:
    source_rows: int
    timestamp_gaps: int
    moisture_observed: int
    moisture_out_of_range_removed: int
    moisture_saturation_clipped: int
    moisture_spikes_removed: int
    trailing_flatline_values_removed: int
    trailing_flatline_cutoff: str
    interpolated_values: int
    interpolated_gap_runs: int
    remaining_missing_values: int
    remaining_missing_runs: int
    clean_moisture_rows: int
    clean_coverage_start: str
    clean_coverage_end: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SectorQuality:
    sector: int
    selection_status: str
    usable_start: str
    usable_end: str
    clean_soil_rows: int
    clean_soil_percent: float
    irrigation_completeness_percent: float
    weather_completeness_percent: float
    valid_history_windows: int
    valid_targets_1h: int
    valid_targets_3h: int
    valid_targets_6h: int
    valid_targets_12h: int
    valid_targets_24h: int
    origins_valid_all_horizons: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _missing_run_ids(missing: pd.Series) -> pd.Series:
    return missing.ne(missing.shift(fill_value=False)).cumsum()


def _count_missing_runs(series: pd.Series) -> int:
    missing = series.isna()
    if not missing.any():
        return 0
    groups = _missing_run_ids(missing)
    return int(missing.groupby(groups).first().sum())


def interpolate_short_complete_gaps(
    series: pd.Series, limit_steps: int = INTERPOLATION_LIMIT_STEPS
) -> tuple[pd.Series, int, int]:
    """Interpolate only entire bounded gaps no longer than ``limit_steps``."""
    missing = series.isna()
    if not missing.any():
        return series.copy(), 0, 0

    groups = _missing_run_ids(missing)
    run_sizes = missing.groupby(groups).transform("sum")
    interpolated = series.interpolate(method="time", limit_area="inside")
    fillable = missing & run_sizes.le(limit_steps) & interpolated.notna()
    result = series.copy()
    result.loc[fillable] = interpolated.loc[fillable]
    filled_runs = int(fillable.groupby(groups).any().sum())
    return result, int(fillable.sum()), filled_runs


def clean_merged_sector(frame: pd.DataFrame, sector: int) -> tuple[pd.DataFrame, CleaningReport]:
    require_regular_grid(frame, f"merged sector {sector}")
    required = {
        "soil_moisture",
        "liters_total",
        "weather_rain",
        "weather_temp",
        "weather_humidity",
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise DataValidationError(
            f"Merged sector {sector} is missing columns: {', '.join(missing_columns)}"
        )
    if frame[["liters_total", "weather_rain", "weather_temp", "weather_humidity"]].isna().any().any():
        raise DataValidationError(
            f"Merged sector {sector} has missing irrigation or selected weather values"
        )
    if (frame["liters_total"] < 0).any():
        raise DataValidationError(f"Merged sector {sector} has negative applied water")

    raw_moisture = frame["soil_moisture"].astype(float)
    observed = raw_moisture.notna()
    in_range = raw_moisture.between(MOISTURE_VALID_MIN, MOISTURE_VALID_MAX)
    out_of_range = observed & ~in_range
    saturation = raw_moisture.gt(MOISTURE_CLIP_MAX) & in_range

    moisture = raw_moisture.where(in_range).clip(upper=MOISTURE_CLIP_MAX)
    moisture, first_values, first_runs = interpolate_short_complete_gaps(moisture)

    rolling_median = moisture.rolling(
        SPIKE_WINDOW_STEPS, center=True, min_periods=5
    ).median()
    spikes = moisture.notna() & rolling_median.notna() & (
        (moisture - rolling_median).abs() > SPIKE_THRESHOLD
    )
    moisture = moisture.mask(spikes)
    moisture, second_values, second_runs = interpolate_short_complete_gaps(moisture)

    observed_after_cleaning = moisture.dropna()
    reversed_std = observed_after_cleaning.iloc[::-1].rolling(
        FLATLINE_WINDOW_READINGS
    ).std()
    varying = reversed_std[reversed_std > FLATLINE_STD_THRESHOLD]
    if varying.empty:
        raise DataValidationError(
            f"Merged sector {sector} moisture is entirely flat under the documented rule"
        )
    flatline_cutoff = varying.index[0]
    trailing_flatline = moisture.notna() & (moisture.index > flatline_cutoff)
    moisture = moisture.mask(moisture.index > flatline_cutoff)

    prepared = pd.DataFrame(index=frame.index)
    prepared["soil_moisture"] = moisture
    # `ts` labels the left edge of a bin. Lagging by one bin makes water at t
    # the fully observed interval ending at t, never the interval after t.
    prepared["applied_water_liters"] = frame["liters_total"].shift(1)
    prepared["weather_rain"] = frame["weather_rain"].astype(float)
    prepared["weather_temp"] = frame["weather_temp"].astype(float)
    prepared["weather_humidity"] = frame["weather_humidity"].astype(float)
    shifted_water = prepared["applied_water_liters"].iloc[1:].to_numpy()
    source_water = frame["liters_total"].iloc[:-1].to_numpy()
    if not (shifted_water == source_water).all():
        raise DataValidationError("Historical water lag does not match the prior bin")

    clean = moisture.dropna()
    if clean.empty:
        raise DataValidationError(f"Merged sector {sector} has no valid soil moisture")
    report = CleaningReport(
        source_rows=len(frame),
        timestamp_gaps=int(
            frame.index.to_series().diff().dropna().ne(EXPECTED_INTERVAL).sum()
        ),
        moisture_observed=int(observed.sum()),
        moisture_out_of_range_removed=int(out_of_range.sum()),
        moisture_saturation_clipped=int(saturation.sum()),
        moisture_spikes_removed=int(spikes.sum()),
        trailing_flatline_values_removed=int(trailing_flatline.sum()),
        trailing_flatline_cutoff=str(flatline_cutoff),
        interpolated_values=first_values + second_values,
        interpolated_gap_runs=first_runs + second_runs,
        remaining_missing_values=int(moisture.isna().sum()),
        remaining_missing_runs=_count_missing_runs(moisture),
        clean_moisture_rows=int(moisture.notna().sum()),
        clean_coverage_start=str(clean.index.min()),
        clean_coverage_end=str(clean.index.max()),
    )
    return prepared, report


def evaluate_sector(
    sector: int, prepared: pd.DataFrame, eligible_for_selection: bool
) -> SectorQuality:
    complete = prepared[list(FEATURE_COLUMNS)].notna().all(axis=1)
    valid_history = (
        complete.rolling(HISTORY_STEPS, min_periods=HISTORY_STEPS).sum()
        == HISTORY_STEPS
    )

    target_counts: dict[int, int] = {}
    all_horizons = valid_history.copy()
    for hours, steps in zip(HORIZON_HOURS, HORIZON_STEPS, strict=True):
        target_available = prepared[TARGET_COLUMN].shift(-steps).notna()
        target_counts[hours] = int((valid_history & target_available).sum())
        all_horizons &= target_available

    moisture = prepared[TARGET_COLUMN]
    start = moisture.first_valid_index()
    end = moisture.last_valid_index()
    if start is None or end is None:
        raise DataValidationError(f"Sector {sector} has no usable moisture coverage")
    coverage = prepared.loc[start:end]
    weather = ["weather_rain", "weather_temp", "weather_humidity"]
    return SectorQuality(
        sector=sector,
        selection_status="candidate" if eligible_for_selection else "diagnostic_only",
        usable_start=str(start),
        usable_end=str(end),
        clean_soil_rows=int(moisture.notna().sum()),
        clean_soil_percent=round(float(coverage[TARGET_COLUMN].notna().mean() * 100), 2),
        irrigation_completeness_percent=round(
            float(coverage["applied_water_liters"].notna().mean() * 100), 2
        ),
        weather_completeness_percent=round(
            float(coverage[weather].notna().all(axis=1).mean() * 100), 2
        ),
        valid_history_windows=int(valid_history.sum()),
        valid_targets_1h=target_counts[1],
        valid_targets_3h=target_counts[3],
        valid_targets_6h=target_counts[6],
        valid_targets_12h=target_counts[12],
        valid_targets_24h=target_counts[24],
        origins_valid_all_horizons=int(all_horizons.sum()),
    )
