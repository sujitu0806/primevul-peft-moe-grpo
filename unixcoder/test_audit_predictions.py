import unittest

from audit_predictions import capped_point, confusion, float32, pair_summary


class PredictionAuditTests(unittest.TestCase):
    def test_counts_and_tied_roc_threshold(self):
        rows = [(1, 0.8, None), (0, 0.8, None), (1, 0.7, None)]
        rows += [(0, 0.2, None)] * 199
        self.assertEqual(capped_point(rows), (0.7, {"tp": 2, "fp": 1, "tn": 199, "fn": 0}))
        self.assertEqual(confusion(rows, 0.5), {"tp": 2, "fp": 1, "tn": 199, "fn": 0})
        # The 0.8 group cannot be split just to keep the true positive.
        self.assertEqual(capped_point(rows, cap=0.004), (None, {"tp": 0, "fp": 0, "tn": 200, "fn": 2}))

    def test_all_pair_outcomes_and_ties(self):
        rows = [
            (0, 0.2, 0), (1, 0.9, 0), (1, 0.3, 1), (0, 0.3, 1),
            (0, 0.8, 2), (1, 0.7, 2), (1, 0.2, 3), (0, 0.8, 3),
        ]
        result = pair_summary(rows)
        self.assertEqual(result["outcomes"], {"P-C": 1, "P-V": 1, "P-B": 1, "P-R": 1})
        self.assertEqual(result["vulnerable_score_relative_to_fixed"], {"higher": 1, "tie": 1, "lower": 2})
        self.assertEqual(result["same_verdict"], 2)
        self.assertEqual(result["rank_accuracy_ties_half_credit_pct"], 37.5)

    def test_missing_pair_member_is_rejected(self):
        with self.assertRaises(ValueError):
            pair_summary([(1, 0.8, 0)])

    def test_colab_float32_threshold_round_trip(self):
        self.assertEqual(float32(0.5888891), 0.5888891220092773)
