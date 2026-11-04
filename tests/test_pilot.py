"""Small checks for time alignment, quality filtering, and the model shape."""
import unittest

import numpy as np
import pandas as pd
import torch

from pilot import GOOD_CODES, make_examples, make_model, read_data


class PilotTests(unittest.TestCase):
    def frame(self):
        dates = pd.date_range("2023-01-02T00:30Z", periods=96, freq="h")
        return pd.DataFrame({"DATE": dates, "temperature": np.arange(96, dtype=float)})

    def test_input_order_target_and_partition_boundaries(self):
        examples, _ = make_examples(self.frame(), "2023-01-02", "2023-01-05")
        np.testing.assert_array_equal(examples.iloc[0, :24], np.arange(24))
        self.assertEqual(examples.index[0], pd.Timestamp("2023-01-03T00:00Z"))
        self.assertEqual(examples.iloc[0]["target"], 26)
        self.assertTrue((examples.index - pd.Timedelta(hours=23) > pd.Timestamp("2023-01-02T00:00Z")).all())
        self.assertTrue((examples.index + pd.Timedelta(hours=3) < pd.Timestamp("2023-01-05T00:00Z")).all())

    def test_missing_hour_is_not_filled(self):
        frame = self.frame().drop(index=12)
        examples, counts = make_examples(frame, "2023-01-02", "2023-01-05")
        self.assertEqual(counts["missing_hours"], 1)
        self.assertEqual(examples.index[0], pd.Timestamp("2023-01-03T13:00Z"))

    def test_exact_hour_is_not_shifted_forward(self):
        frame = self.frame()
        exact = pd.DataFrame({"DATE": [pd.Timestamp("2023-01-02T01:00Z")], "temperature": [123.0]})
        frame = pd.concat([frame, exact], ignore_index=True)
        examples, _ = make_examples(frame, "2023-01-02", "2023-01-05")
        self.assertEqual(examples.iloc[0, 0], 123.0)

    def test_real_data_and_source_specific_quality(self):
        frame, counts = read_data()
        self.assertEqual(counts["raw_records"], 40052)
        self.assertTrue(np.isfinite(frame["temperature"]).all())
        self.assertFalse(frame.duplicated(["STATION", "DATE"]).any())
        self.assertTrue(all(row.temperature_Quality_Code in GOOD_CODES[row.temperature_Source_Code]
                            for row in frame.itertuples(index=False)))
        self.assertEqual(counts["raw_records"], counts["retained_observations"]
                         + counts["invalid_station_timestamp_or_temperature"]
                         + counts["rejected_quality_or_unrecognized_source"]
                         + counts["duplicate_station_timestamps"])

    def test_model_is_3713_parameters_with_one_output(self):
        model = make_model()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 3713)
        self.assertEqual(tuple(model(torch.zeros(4, 24)).shape), (4, 1))


if __name__ == "__main__":
    unittest.main()
