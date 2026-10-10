import unittest

import numpy as np
import pandas as pd

from deeponet_irrigation.preprocessing import (
    FEATURE_COLUMNS,
    interpolate_short_complete_gaps,
)
from deeponet_irrigation.splits import make_chronological_splits
from deeponet_irrigation.temporal_windows import generate_samples


class TemporalPipelineTests(unittest.TestCase):
    def test_only_complete_short_gaps_are_interpolated(self) -> None:
        index = pd.date_range("2025-01-01", periods=12, freq="10min")
        values = pd.Series(
            [1.0, 2.0, np.nan, np.nan, 5.0, 6.0, np.nan, np.nan, np.nan, 10.0, 11.0, 12.0],
            index=index,
        )
        result, filled_values, filled_runs = interpolate_short_complete_gaps(values)
        self.assertEqual(filled_values, 2)
        self.assertEqual(filled_runs, 1)
        self.assertTrue(result.iloc[2:4].notna().all())
        self.assertTrue(result.iloc[6:9].isna().all())

    def test_samples_stay_inside_chronological_splits(self) -> None:
        index = pd.date_range("2025-01-01", periods=2_000, freq="10min")
        frame = pd.DataFrame(1.0, index=index, columns=FEATURE_COLUMNS)
        splits = make_chronological_splits(index)
        samples, _ = generate_samples(frame, splits)
        for split in splits:
            item = samples[split.name]
            self.assertGreater(len(item), 0)
            self.assertTrue((item.history_start_index >= split.start_index).all())
            self.assertTrue((item.target_index <= split.end_index).all())
            self.assertTrue((item.target_index > item.current_index).all())


if __name__ == "__main__":
    unittest.main()
