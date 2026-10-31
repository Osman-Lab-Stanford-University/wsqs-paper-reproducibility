import unittest

import numpy as np
import pandas as pd

from wsqs.extended import (
    DOMAINS,
    alternative_outcomes,
    block_shapley,
    block_columns,
    scientific_plans,
)


class ExtendedScientificTests(unittest.TestCase):

    def test_exact_block_membership(self):
        plans = scientific_plans()

        self.assertEqual(
            [len(plans["blocks"][k]) for k in ("S", "C", "P")],
            [22, 3, 8],
        )

        self.assertEqual(
            block_columns("SCP"), plans["full"]
        )

        self.assertEqual(
            len(plans["cross_domain"]), 8
        )

    def test_shapley_efficiency(self):
        r2 = {
            "": -0.000005,
            "S": 0.557162,
            "C": 0.015746,
            "P": 0.062838,
            "SC": 0.556957,
            "SP": 0.561507,
            "CP": 0.078701,
            "SCP": 0.564044,
        }

        result = block_shapley(r2)

        self.assertAlmostEqual(
            sum(result.values()),
            r2["SCP"] - r2[""],
            places=10,
        )

        self.assertAlmostEqual(
            result["S"], 0.520817, places=5
        )

        self.assertAlmostEqual(
            result["C"], 0.008705, places=5
        )

        self.assertAlmostEqual(
            result["P"], 0.034527, places=5
        )

    def test_alternative_outcome_definitions(self):
        frame = pd.DataFrame({
            **{
                f"{d}_raw_severity": [3.0, 3.0]
                for d in DOMAINS
            },
            **{
                f"{d}_priority_0_10": [5.0, 0.0]
                for d in DOMAINS
            },
        })

        frame.loc[
            0, "Acceptability_raw_severity"
        ] = 5.0

        outcomes, weights, fallbacks = (
            alternative_outcomes(frame, [0])
        )

        self.assertEqual(fallbacks, 1)

        self.assertAlmostEqual(
            outcomes["CORE8"].iloc[1], 50.0
        )

        self.assertGreater(
            outcomes["CORE7A"].iloc[0],
            outcomes["CORE8"].iloc[0],
        )

        self.assertAlmostEqual(
            weights.sum(), 1.0
        )

    def test_no_partial_block_manifest(self):
        with self.assertRaises(ValueError):
            block_columns("SPC_EXTRA")


if __name__ == "__main__":
    unittest.main()
