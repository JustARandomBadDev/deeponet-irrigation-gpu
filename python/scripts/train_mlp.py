from __future__ import annotations

import argparse
import torch

from deeponet_irrigation.dataset import PreparedTemporalDataset, make_data_loader
from deeponet_irrigation.models import MLPBaseline
from deeponet_irrigation.project_paths import REPOSITORY_ROOT
from deeponet_irrigation.training import (
    TrainingConfig,
    select_device,
    set_deterministic_seed,
    train_mlp,
    write_json,
)


ROOT = REPOSITORY_ROOT
DATA_DIR = ROOT / "data" / "processed" / "arnesano_v2"
CHECKPOINT_PATH = ROOT / "models" / "mlp_baseline.pt"
TRAINING_RESULTS_PATH = ROOT / "results" / "mlp_training.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the small MLP baseline")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
    )
    train_dataset = PreparedTemporalDataset(DATA_DIR, "train")
    validation_dataset = PreparedTemporalDataset(DATA_DIR, "validation")
    train_loader = make_data_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    validation_loader = make_data_loader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
    )
    set_deterministic_seed(config.seed)
    soil_statistics = train_dataset.metadata["normalization"]["statistics"][
        "soil_moisture"
    ]
    model = MLPBaseline(
        history_steps=train_dataset.history_steps,
        feature_count=train_dataset.feature_count,
        target_mean=soil_statistics["mean"],
        target_std=soil_statistics["std"],
    )
    device = select_device()
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Architecture: {model}")
    print(f"Trainable parameters: {model.trainable_parameter_count():,}")
    print(f"Training config: {config}")

    results = train_mlp(
        model,
        train_loader,
        validation_loader,
        device=device,
        config=config,
        checkpoint_path=CHECKPOINT_PATH,
        preprocessing_metadata=train_dataset.metadata,
    )
    write_json(TRAINING_RESULTS_PATH, results)
    print(f"Best epoch: {results['best_epoch']}")
    print(f"Best validation loss: {results['best_validation_loss']:.6f}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Training history: {TRAINING_RESULTS_PATH}")


if __name__ == "__main__":
    main()
