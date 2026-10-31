import unittest
from wsqs.extended import (
    OUTCOME_BLOCK,
    outcome_columns,
    scientific_plans,
)

class Core7PredictorTests(unittest.TestCase):

    def test_core7a_original_sc(self):
        self.assertEqual(OUTCOME_BLOCK["CORE7A"], "SC")
        self.assertEqual(len(outcome_columns("CORE7A")), 25)

    def test_core7b_original_scp(self):
        plans = scientific_plans()
        self.assertEqual(OUTCOME_BLOCK["CORE7B"], "SCP")
        self.assertEqual(outcome_columns("CORE7B"), plans["full"])
        self.assertEqual(len(outcome_columns("CORE7B")), 33)

    def test_weighted_outcomes_original_sc(self):
        for name in ("POPULATION_PRIORITY", "INDIVIDUAL_PRIORITY"):
            self.assertEqual(OUTCOME_BLOCK[name], "SC")
            self.assertEqual(len(outcome_columns(name)), 25)

    def test_invalid_outcome(self):
        with self.assertRaises(ValueError):
            outcome_columns("INVALID")
