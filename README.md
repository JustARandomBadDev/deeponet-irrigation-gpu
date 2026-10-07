# deeponet-irrigation-gpu

DeepONet-based soil-moisture prediction on the Arnesano precision-irrigation
dataset, with a reproducible Python pipeline and later GPU inference
optimization using C++, CUDA, and TensorRT.

## Dataset setup

The data tooling is pinned to version 2 of *Soil Moisture, Irrigation Actuator
and Weather Dataset from a Multi-Sector Precision-Irrigation*:
[DOI 10.17632/c837v6p8ph.2](https://doi.org/10.17632/c837v6p8ph.2).
It never requests the unversioned latest release.

```bash
uv sync
uv run python scripts/setup_data.py
uv run python scripts/inspect_dataset.py
```

`setup_data.py` downloads the official version-qualified Mendeley Data ZIP,
checks its server-published SHA-256, validates the documented directory layout,
and records provenance in `data/raw/arnesano-v2.metadata.json`. Re-running it
validates and reuses both the archive and extracted files.

`inspect_dataset.py` checks the supplied README and data dictionary, lists the
downloaded files, then streams every CSV in chunks to report shape, schema,
missing values, documented timestamp ranges, and documented sector identifiers.

Downloaded and generated artifacts under `data/`, `models/`, and `results/`
are intentionally excluded from Git.

## Temporal dataset

The leakage-safe temporal pipeline uses the author-provided **merged sector 4**
table. Unlike the preprocessed tables, merged data preserves the complete
10-minute grid; sector 4 provides the most valid 24-hour histories and future
targets among sectors 1, 2, and 4. Sector 3 is rejected because its probe has a
long trailing zero flatline, and sector 5 is excluded because of its documented
water-volume inconsistency.

The five input features, in fixed order, are `soil_moisture`,
`applied_water_liters`, `weather_rain`, `weather_temp`, and
`weather_humidity`. Applied water is lagged by one bin so it is strictly
historical. The target is soil moisture at 1, 3, 6, 12, or 24 hours after the
prediction origin. Each branch contains 144 consecutive 10-minute observations
(24 hours of bins).

Soil moisture is range-checked using the supplied data dictionary, clipped at
100, checked for documented spikes and trailing flatlines, and interpolated
only when an entire interior gap is at most 20 minutes. Longer gaps reject the
window. Splits are contiguous 70/15/15 time ranges; histories and targets that
cross a boundary are purged. Feature mean/std statistics use only unique rows
referenced by training histories.

```bash
uv run python scripts/prepare_dataset.py
uv run python scripts/inspect_samples.py
```

Compact indexed NumPy arrays and complete preprocessing provenance are written
under `data/processed/arnesano_v2/`. The normalization statistics and exact
split ranges are stored in `data/processed/arnesano_v2/metadata.json`.

## Baselines

The persistence baseline predicts the last soil-moisture value in the
historical window. The learned baseline is a deliberately small MLP: it
flattens the normalized `[144, 5]` history, appends the horizon in hours divided
by 24, and applies `Linear(721, 128) -> ReLU -> Linear(128, 64) -> ReLU ->
Linear(64, 1)`. Its output is mapped back to physical soil-moisture units using
the training-only soil-moisture mean and standard deviation. The dataset loader
reconstructs windows from the prepared indices; it does not duplicate
preprocessing or flatten model inputs.

```bash
uv run python scripts/train_mlp.py
uv run python scripts/evaluate_baselines.py
uv run python scripts/plot_baselines.py
uv run python -m unittest discover -s tests
```

Training uses Adam, MSE loss, validation after each epoch, a best-validation
checkpoint, and early stopping. CUDA is selected automatically when available.
The evaluation reports MAE, RMSE, and R² both overall and separately for every
forecast horizon. It also reports how often the future target equals, or is
within 0.5 of, the current soil moisture. Per-horizon results are essential:
long and quantized validation/test plateaus can make persistence deceptively
strong, especially at short horizons.

Generated checkpoints are stored under `models/`; metrics, prediction arrays,
and plots are stored under `results/`. Both directories are excluded from Git.
