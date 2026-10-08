from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deeponet-irrigation-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from deeponet_irrigation.dataset import PreparedTemporalDataset, make_data_loader
from deeponet_irrigation.evaluation import collect_predictions, load_deeponet_checkpoint
from deeponet_irrigation.metrics import (
    error_by_change_status,
    metrics_by_horizon,
    prediction_bias,
)
from deeponet_irrigation.training import select_device, write_json


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "arnesano_v2"
RESULTS_DIR = ROOT / "results"
CHECKPOINT = ROOT / "models" / "deeponet_reference.pt"
EXPERIMENT_LOG = RESULTS_DIR / "deeponet_experiments.json"
MLP_TEST_METRICS = RESULTS_DIR / "final_frozen_mlp_metrics.json"
MLP_TEST_PREDICTIONS = RESULTS_DIR / "final_frozen_mlp_predictions.npz"
OUTPUT_METRICS = RESULTS_DIR / "deeponet_final_test_metrics.json"
OUTPUT_PREDICTIONS = RESULTS_DIR / "deeponet_final_test_predictions.npz"
HORIZONS = (1, 3, 6, 12, 24)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-time final DeepONet test")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate plots from the immutable saved final-test artifacts",
    )
    return parser.parse_args()


def summarize(
    target: np.ndarray,
    prediction: np.ndarray,
    persistence: np.ndarray,
    horizons: np.ndarray,
) -> dict[str, Any]:
    return {
        "metrics": metrics_by_horizon(target, prediction, horizons),
        "change_status": error_by_change_status(
            target, prediction, persistence, tolerance=0.5
        ),
        "bias": prediction_bias(target, prediction),
    }


def evaluate_once() -> None:
    if OUTPUT_METRICS.exists() or OUTPUT_PREDICTIONS.exists():
        raise SystemExit(
            "Final DeepONet test artifacts already exist; repeat evaluation is forbidden."
        )
    experiment_log = json.loads(EXPERIMENT_LOG.read_text(encoding="utf-8"))
    selection = experiment_log.get("selected")
    if not selection or not CHECKPOINT.exists():
        raise SystemExit("Freeze the validation-selected DeepONet before test evaluation")
    checkpoint_metadata = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if not checkpoint_metadata.get("selection_frozen_before_test"):
        raise SystemExit("Checkpoint is not marked as frozen before test")

    # The official MLP comparison is reused from its immutable final evaluation.
    mlp_report = json.loads(MLP_TEST_METRICS.read_text(encoding="utf-8"))
    with np.load(MLP_TEST_PREDICTIONS) as saved:
        mlp_predictions = {name: saved[name].copy() for name in saved.files}

    test_dataset = PreparedTemporalDataset(DATA_DIR, "test")
    test_loader = make_data_loader(
        test_dataset, batch_size=512, shuffle=False, seed=42
    )
    device = select_device()
    model, _ = load_deeponet_checkpoint(CHECKPOINT, test_dataset.metadata, device)
    started = time.perf_counter()
    deep_predictions = collect_predictions(
        test_loader,
        test_dataset.metadata,
        device=device,
        model=model,
        prediction_name="deeponet",
    )
    inference_seconds = time.perf_counter() - started

    for name in ("target", "persistence", "horizon_hours", "target_index"):
        np.testing.assert_array_equal(deep_predictions[name], mlp_predictions[name])
    target_timestamps = np.asarray(
        test_dataset.timestamps[deep_predictions["target_index"]]
    ).astype("datetime64[ns]")
    output_arrays = {
        **deep_predictions,
        "frozen_mlp": mlp_predictions["final_mlp"],
        "target_timestamp": target_timestamps,
    }
    np.savez_compressed(OUTPUT_PREDICTIONS, **output_arrays)

    target = deep_predictions["target"]
    persistence = deep_predictions["persistence"]
    horizons = deep_predictions["horizon_hours"]
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "this_is_the_only_final_deeponet_test_evaluation": True,
        "selection_frozen_before_test": True,
        "selection": selection,
        "test": {
            "persistence": summarize(target, persistence, persistence, horizons),
            "frozen_mlp": summarize(
                target, mlp_predictions["final_mlp"], persistence, horizons
            ),
            "deeponet": summarize(
                target, deep_predictions["deeponet"], persistence, horizons
            ),
        },
        "frozen_mlp_source": {
            "metrics": str(MLP_TEST_METRICS),
            "predictions": str(MLP_TEST_PREDICTIONS),
            "prior_report_flag": mlp_report[
                "this_is_the_last_mlp_test_evaluation"
            ],
        },
        "runtime": {
            "torch_version": torch.__version__,
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(0) if device.type == "cuda" else None
            ),
            "deeponet_inference_seconds": inference_seconds,
        },
    }
    write_json(OUTPUT_METRICS, report)
    print_comparison(report)


def print_comparison(report: dict[str, Any]) -> None:
    print("\nFinal one-time DeepONet test")
    print(f"{'Model':<18} {'MAE':>10} {'RMSE':>10} {'R2':>10} {'Unchanged':>12} {'Changing':>10} {'Bias':>10}")
    for name, label in (
        ("persistence", "Persistence"),
        ("frozen_mlp", "Frozen MLP"),
        ("deeponet", "DeepONet"),
    ):
        item = report["test"][name]
        overall = item["metrics"]["overall"]
        print(
            f"{label:<18} {overall['mae']:>10.4f} {overall['rmse']:>10.4f} "
            f"{overall['r2']:>10.4f} "
            f"{item['change_status']['nearly_unchanged']['rmse']:>12.4f} "
            f"{item['change_status']['changing']['rmse']:>10.4f} "
            f"{item['bias']['mean_signed_error']:>10.4f}"
        )


def validation_plots() -> tuple[Path, Path]:
    log = json.loads(EXPERIMENT_LOG.read_text(encoding="utf-8"))
    series = {
        "Persistence": log["baselines"]["persistence"]["metrics"],
        "Frozen MLP": log["baselines"]["frozen_mlp"]["metrics"],
        "DeepONet": log["selected"]["validation_metrics"],
    }
    positions = np.arange(len(HORIZONS))
    width = 0.25
    figure, axis = plt.subplots(figsize=(9, 4.8))
    for offset, (label, metrics) in zip((-1, 0, 1), series.items(), strict=True):
        values = [metrics["by_horizon"][f"{h}h"]["rmse"] for h in HORIZONS]
        axis.bar(positions + offset * width, values, width, label=label)
    axis.set(
        title="Validation RMSE by prediction horizon",
        xlabel="Prediction horizon",
        ylabel="RMSE (soil-moisture units)",
        xticks=positions,
        xticklabels=[f"{h}h" for h in HORIZONS],
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    horizon_path = RESULTS_DIR / "deeponet_validation_rmse_by_horizon.png"
    figure.savefig(horizon_path, dpi=160)
    plt.close(figure)

    change_series = {
        "Persistence": log["baselines"]["persistence"]["change_status"],
        "Frozen MLP": log["baselines"]["frozen_mlp"]["change_status"],
        "DeepONet": log["selected"]["validation_change_status"],
    }
    positions = np.arange(2)
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for offset, (label, metrics) in zip(
        (-1, 0, 1), change_series.items(), strict=True
    ):
        values = [
            metrics["nearly_unchanged"]["rmse"],
            metrics["changing"]["rmse"],
        ]
        axis.bar(positions + offset * width, values, width, label=label)
    axis.set(
        title="Validation RMSE by change status",
        ylabel="RMSE (soil-moisture units)",
        xticks=positions,
        xticklabels=("Nearly unchanged", "Changing"),
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    change_path = RESULTS_DIR / "deeponet_validation_change_status_rmse.png"
    figure.savefig(change_path, dpi=160)
    plt.close(figure)
    return horizon_path, change_path


def test_plot() -> Path:
    with np.load(OUTPUT_PREDICTIONS) as values:
        horizons = values["horizon_hours"].reshape(-1)
        mask = horizons == 24
        time_values = values["target_timestamp"][mask]
        order = np.argsort(time_values)
        time_values = time_values[order]
        target = values["target"].reshape(-1)[mask][order]
        persistence = values["persistence"].reshape(-1)[mask][order]
        frozen_mlp = values["frozen_mlp"].reshape(-1)[mask][order]
        deeponet = values["deeponet"].reshape(-1)[mask][order]
    figure, axis = plt.subplots(figsize=(11, 4.5))
    axis.plot(time_values, target, label="Ground truth", linewidth=2)
    axis.plot(time_values, persistence, label="Persistence", alpha=0.75)
    axis.plot(time_values, frozen_mlp, label="Frozen MLP", alpha=0.8)
    axis.plot(time_values, deeponet, label="DeepONet", alpha=0.9)
    axis.set(
        title="Final test predictions at the 24-hour horizon",
        ylabel="Soil moisture",
    )
    axis.legend()
    axis.grid(alpha=0.25)
    figure.autofmt_xdate()
    figure.tight_layout()
    output = RESULTS_DIR / "deeponet_final_test_predictions_24h.png"
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def generate_plots() -> None:
    if not OUTPUT_METRICS.exists() or not OUTPUT_PREDICTIONS.exists():
        raise SystemExit("Final saved metrics and predictions are required for plots")
    for path in (*validation_plots(), test_plot()):
        print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    if not args.plots_only:
        evaluate_once()
    generate_plots()


if __name__ == "__main__":
    main()
