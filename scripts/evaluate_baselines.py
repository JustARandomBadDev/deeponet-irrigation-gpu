from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from deeponet_irrigation.dataset import PreparedTemporalDataset, make_data_loader
from deeponet_irrigation.evaluation import collect_predictions
from deeponet_irrigation.metrics import metrics_by_horizon, plateau_statistics
from deeponet_irrigation.models import MLPBaseline
from deeponet_irrigation.training import select_device, write_json


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "arnesano_v2"
CHECKPOINT_PATH = ROOT / "models" / "mlp_baseline.pt"
RESULTS_DIR = ROOT / "results"


def _load_model(
    dataset: PreparedTemporalDataset, device: torch.device
) -> tuple[MLPBaseline, dict[str, object]]:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CHECKPOINT_PATH}. Run scripts/train_mlp.py first."
        )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    architecture = checkpoint["architecture"]
    model = MLPBaseline(
        history_steps=int(architecture["history_steps"]),
        feature_count=int(architecture["feature_count"]),
        hidden_sizes=tuple(architecture["hidden_sizes"]),
        target_mean=float(architecture["output_scaling"]["mean"]),
        target_std=float(architecture["output_scaling"]["std"]),
    )
    contract = checkpoint["preprocessing_contract"]
    expected = {
        "dataset": dataset.metadata["dataset"],
        "selected_data_source": dataset.metadata["selected_data_source"],
        "selected_sector": dataset.metadata["selected_sector"],
        "feature_order": dataset.metadata["feature_order"],
        "target_column": dataset.metadata["target_column"],
        "history_steps": dataset.metadata["history_steps"],
        "prediction_horizons_hours": dataset.metadata["prediction_horizons_hours"],
    }
    if contract != expected:
        raise ValueError("Checkpoint preprocessing contract does not match prepared data")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint


def _print_comparison(
    split: str,
    persistence: dict[str, object],
    mlp: dict[str, object],
) -> None:
    print(f"\n{split.upper()}")
    print(f"{'Model':<13} {'Horizon':>8} {'N':>7} {'MAE':>10} {'RMSE':>10} {'R2':>10}")
    for label, metrics in (("Persistence", persistence), ("MLP", mlp)):
        rows = [("overall", metrics["overall"]), *metrics["by_horizon"].items()]
        for horizon, values in rows:
            print(
                f"{label:<13} {horizon:>8} {values['sample_count']:>7d} "
                f"{values['mae']:>10.4f} {values['rmse']:>10.4f} "
                f"{values['r2']:>10.4f}"
            )


def main() -> None:
    device = select_device()
    print(f"Evaluation device: {device}")
    persistence_results: dict[str, object] = {}
    mlp_results: dict[str, object] = {}
    plateau_results: dict[str, object] = {
        "description": (
            "Fractions compare soil moisture at the prediction origin with the future target."
        ),
        "splits": {},
    }
    prediction_files: list[str] = []
    checkpoint_summary: dict[str, object] | None = None

    for split in ("validation", "test"):
        dataset = PreparedTemporalDataset(DATA_DIR, split)
        model, checkpoint = _load_model(dataset, device)
        checkpoint_summary = {
            "path": str(CHECKPOINT_PATH),
            "best_epoch": checkpoint["best_epoch"],
            "best_validation_mse": checkpoint["best_validation_mse"],
        }
        loader = make_data_loader(dataset, batch_size=1024, shuffle=False)
        predictions = collect_predictions(
            loader, dataset.metadata, device=device, model=model
        )
        persistence_results[split] = metrics_by_horizon(
            predictions["target"],
            predictions["persistence"],
            predictions["horizon_hours"],
        )
        mlp_results[split] = metrics_by_horizon(
            predictions["target"],
            predictions["mlp"],
            predictions["horizon_hours"],
        )
        plateau_results["splits"][split] = plateau_statistics(
            predictions["target"],
            predictions["persistence"],
            predictions["horizon_hours"],
            tolerance=0.5,
        )
        _print_comparison(split, persistence_results[split], mlp_results[split])

        prediction_path = RESULTS_DIR / f"{split}_predictions.npz"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prediction_path,
            **predictions,
            target_timestamp=np.asarray(dataset.timestamps)[
                predictions["target_index"].astype(np.int64)
            ],
        )
        prediction_files.append(str(prediction_path))

    common = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "checkpoint": checkpoint_summary,
        "splits_evaluated": ["validation", "test"],
    }
    write_json(
        RESULTS_DIR / "persistence_metrics.json",
        {**common, "model": "persistence", "metrics": persistence_results},
    )
    write_json(
        RESULTS_DIR / "mlp_metrics.json",
        {**common, "model": "mlp_baseline", "metrics": mlp_results},
    )
    write_json(RESULTS_DIR / "plateau_analysis.json", plateau_results)
    print(f"\nResults: {RESULTS_DIR}")
    print("Prediction files: " + ", ".join(prediction_files))


if __name__ == "__main__":
    main()
