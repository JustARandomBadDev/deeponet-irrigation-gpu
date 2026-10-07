import unittest
from pathlib import Path

import numpy as np
import torch

from deeponet_irrigation.dataset import PreparedTemporalDataset
from deeponet_irrigation.evaluation import persistence_prediction
from deeponet_irrigation.metrics import metrics_by_horizon
from deeponet_irrigation.models import MLPBaseline


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "arnesano_v2"


class BaselineTests(unittest.TestCase):
    def test_prepared_branch_shape(self) -> None:
        dataset = PreparedTemporalDataset(DATA_DIR, "train")
        self.assertEqual(dataset[0]["branch_input"].shape, (144, 5))

    def test_mlp_output_shape(self) -> None:
        model = MLPBaseline(history_steps=144, feature_count=5)
        output = model(torch.zeros(7, 144, 5), torch.ones(7, 1))
        self.assertEqual(output.shape, (7, 1))

    def test_persistence_uses_last_historical_soil_value(self) -> None:
        branch = torch.zeros(2, 4, 3)
        branch[0, -1, 1] = 2.0
        branch[1, -1, 1] = -1.0
        prediction = persistence_prediction(
            branch, soil_index=1, soil_mean=10.0, soil_std=3.0
        )
        torch.testing.assert_close(prediction, torch.tensor([[16.0], [7.0]]))

    def test_metrics_are_grouped_by_horizon(self) -> None:
        target = np.array([1.0, 3.0, 10.0, 14.0])
        prediction = np.array([2.0, 2.0, 12.0, 12.0])
        horizon = np.array([1.0, 1.0, 24.0, 24.0])
        metrics = metrics_by_horizon(target, prediction, horizon)
        self.assertEqual(set(metrics["by_horizon"]), {"1h", "24h"})
        self.assertAlmostEqual(metrics["by_horizon"]["1h"]["mae"], 1.0)
        self.assertAlmostEqual(metrics["by_horizon"]["24h"]["mae"], 2.0)

    def test_validation_and_test_metrics_are_not_mixed(self) -> None:
        validation = metrics_by_horizon(
            np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0, 1.0])
        )
        test = metrics_by_horizon(
            np.array([1.0, 2.0]), np.array([2.0, 3.0]), np.array([1.0, 1.0])
        )
        self.assertEqual(validation["overall"]["mae"], 0.0)
        self.assertEqual(test["overall"]["mae"], 1.0)

        validation_data = PreparedTemporalDataset(DATA_DIR, "validation")
        test_data = PreparedTemporalDataset(DATA_DIR, "test")
        validation_range = validation_data.metadata["splits"]["validation"]
        test_range = test_data.metadata["splits"]["test"]
        self.assertLess(validation_data.target_index.max(), test_data.current_index.min())
        self.assertLessEqual(
            validation_data.target_index.max(), validation_range["end_index"]
        )
        self.assertGreaterEqual(test_data.history_start_index.min(), test_range["start_index"])


if __name__ == "__main__":
    unittest.main()
