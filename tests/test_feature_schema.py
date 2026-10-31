import csv
import unittest
from pathlib import Path

import pandas as pd

from wsqs.features import construct_features
from wsqs.response_coding import coding, score_responses


ROOT = Path(__file__).resolve().parents[1]


class FeatureTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.example = pd.read_csv(
            ROOT / "example/synthetic25.csv",
            keep_default_na=False,
        )
        cls.X, cls.cat, cls.num = construct_features(cls.example)

    def test_feature_dimensions(self):
        self.assertEqual(self.X.shape, (25, 33))
        self.assertEqual(len(set(self.X.columns)), 33)
        self.assertEqual(len(self.cat) + len(self.num), 33)

    def test_response_coding(self):
        self.assertEqual(len(coding()), 43)

        with (
            ROOT / "example/expected_scores.csv"
        ).open(newline="") as f:
            expected = list(csv.DictReader(f))

        for i, row in self.example.iterrows():
            values = score_responses(row.to_dict())

            for name in ("CORE8", "CORE7A", "CORE7B"):
                self.assertAlmostEqual(
                    values[name],
                    float(expected[i][name]),
                )

    def test_provider_categories(self):
        self.assertEqual(
            self.X.loc[0, "service_Q3_5"],
            "__OUTSIDE_MUNICIPAL_EVIDENCE__",
        )
        self.assertEqual(
            self.X.loc[6, "service_Q3_5"],
            "__EXPLICIT_NONUSE__",
        )
        self.assertEqual(
            self.X.loc[6, "service_Q4_3"],
            "__NO_COMMUNICATION_STREAM__",
        )
        self.assertEqual(
            self.X.loc[6, "service_Q8_2"],
            "__QUANTITY_UNKNOWN__",
        )

    def test_native_ordinal_ranges(self):
        invalid = self.example.copy()
        invalid.loc[1, "Q4_4_backup_raw_severity"] = 5

        with self.assertRaises(ValueError):
            construct_features(invalid)

        invalid = self.example.copy()
        invalid.loc[
            1, "Q9_4_service_repair_delays_raw_severity"
        ] = 3

        with self.assertRaises(ValueError):
            construct_features(invalid)

    def test_household_size(self):
        self.assertIn("context_household_size", self.cat)
        self.assertEqual(
            self.X.loc[0, "context_household_size"],
            "8_plus",
        )


if __name__ == "__main__":
    unittest.main()
