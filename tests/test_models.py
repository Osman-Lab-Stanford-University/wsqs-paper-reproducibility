import unittest

import numpy as np
import pandas as pd

from wsqs.models import (
    LEARNERS,
    model_settings,
    partition_indices,
    prepare_catboost,
    prepare_sklearn,
    summary_metrics,
)


class ModelSpecificationTests(unittest.TestCase):

    def test_partition_integrity(self):
        fit, val, test, dev = partition_indices(
            6145, 20260714
        )

        self.assertEqual(
            (len(fit), len(val), len(test), len(dev)),
            (3932, 984, 1229, 4916),
        )

        self.assertEqual(
            len(set(fit) | set(val) | set(test)),
            6145,
        )

    def test_configurations(self):
        config = model_settings()

        self.assertEqual(
            config["catboost"]["iterations"], 1400
        )

        self.assertEqual(
            config["catboost"]["random_seed"], 20260714
        )

        self.assertEqual(
            config["ridge"]["alpha_candidates"],
            [0.1, 1.0, 10.0, 100.0],
        )

        self.assertEqual(
            len(config["repeated_split_seeds"]), 10
        )

        self.assertEqual(len(LEARNERS), 7)

    def test_categorical_preparation(self):
        X = pd.DataFrame({
            "category": ["A", None, "B"],
            "number": [1.0, np.nan, 3.0],
        })

        cb = prepare_catboost(
            X, ["category"], ["number"]
        )

        sk = prepare_sklearn(
            X, ["category"], ["number"]
        )

        self.assertEqual(
            cb.loc[1, "category"],
            "__UNANSWERED__",
        )

        self.assertTrue(
            pd.isna(sk.loc[1, "category"])
        )

        self.assertTrue(
            pd.isna(cb.loc[1, "number"])
        )

    def test_metrics(self):
        actual = np.array([0.0, 25.0, 50.0, 75.0])
        predicted = actual.copy()

        result = summary_metrics(actual, predicted)

        self.assertAlmostEqual(result["rmse_100"], 0)
        self.assertAlmostEqual(result["mae_100"], 0)
        self.assertAlmostEqual(result["r2"], 1)
        self.assertAlmostEqual(result["spearman"], 1)


if __name__ == "__main__":
    unittest.main()
