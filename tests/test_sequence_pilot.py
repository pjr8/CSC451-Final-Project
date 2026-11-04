import unittest

import numpy as np
import pandas as pd
import torch

from sequence_pilot import TemperatureGRU, make_sequences


class SequenceTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"DATE": pd.date_range("2023-01-02T00:30Z", periods=500, freq="h"),
                             "temperature": np.arange(500, dtype=np.float32)})

    def test_history_and_all_future_targets(self):
        x, y, origins = make_sequences(self.frame(), "2023-01-02", "2023-01-23")
        self.assertEqual(x.shape[1:], (336, 1))
        self.assertEqual(y.shape[1:], (24,))
        np.testing.assert_array_equal(x[0, :, 0], np.arange(336))
        np.testing.assert_array_equal(y[0], np.arange(336, 360))
        self.assertEqual(origins[0], pd.Timestamp("2023-01-16T00:00Z"))
        self.assertTrue((origins + pd.Timedelta(hours=24) < pd.Timestamp("2023-01-23T00:00Z")).all())

    def test_missing_future_hour_drops_entire_example(self):
        frame = self.frame().drop(index=345)
        _, _, origins = make_sequences(frame, "2023-01-02", "2023-01-23")
        self.assertNotIn(pd.Timestamp("2023-01-16T00:00Z"), origins)

    def test_short_partition_returns_correct_empty_shapes(self):
        x, y, origins = make_sequences(self.frame(), "2023-01-02", "2023-01-03")
        self.assertEqual(x.shape, (0, 336, 1))
        self.assertEqual(y.shape, (0, 24))
        self.assertEqual(len(origins), 0)

    def test_default_model_fits_parameter_limit(self):
        model = TemperatureGRU()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 997432)
        self.assertEqual(tuple(model(torch.zeros(2, 12, 1)).shape), (2, 24))


if __name__ == "__main__":
    unittest.main()
