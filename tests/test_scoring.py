import unittest

from wsqs.scoring import CORE8, core8, core7a, core7b


class ScoringTests(unittest.TestCase):

    def test_best_and_worst(self):
        best = {qid: 1 for qid, _ in CORE8}
        worst = {qid: 5 for qid, _ in CORE8}

        self.assertEqual(core8(best), 100.0)
        self.assertEqual(core8(worst), 0.0)

    def test_available_domain_denominator(self):
        values = {qid: 3 for qid, _ in CORE8}
        values["Q9.1"] = None
        values["Q2.2"] = 1

        self.assertAlmostEqual(
            core8(values), 400.0 / 7.0
        )

    def test_core7_exclusions(self):
        values = {qid: 3 for qid, _ in CORE8}
        values["Q6.1"] = 5
        values["Q9.1"] = 1

        self.assertAlmostEqual(
            core7a(values), 400.0 / 7.0
        )

        self.assertAlmostEqual(
            core7b(values), 300.0 / 7.0
        )

    def test_invalid_values(self):
        for invalid in (0, 2.5, 6, float("nan")):
            with self.assertRaises(ValueError):
                core8({"Q2.2": invalid})

        with self.assertRaises(ValueError):
            core8({"Q22.2": 3})


if __name__ == "__main__":
    unittest.main()
