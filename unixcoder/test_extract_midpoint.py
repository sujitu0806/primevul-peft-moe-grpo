import unittest

from extract_midpoint import (
    VALIDATION_HEADINGS, confusion_counts, count_from_pct, validation_history,
)


def validation_fragment(current_step, max_step, rows):
    header = "".join(f"<th>{column}</th>" for column in VALIDATION_HEADINGS)
    body = "".join("<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>" for row in rows)
    return {
        "data": {"text/html": f"<progress value='{current_step}' max='{max_step}'></progress>"
                              f"<table><tr>{header}</tr>{body}</table>"}
    }


class AggregateEvidenceTest(unittest.TestCase):
    def test_confusion_counts_distinguish_operating_points(self):
        full = {
            "accuracy": 75.0, "precision": 75.0, "recall": 75.0, "f1": 75.0,
            "VD-S": 50.0, "VD-S realized FPR": 0.0,
        }
        self.assertEqual(confusion_counts(full, {"vulnerable": 4, "benign": 4}), {
            "at_classifier_cutoff": {"tp": 3, "fp": 1, "tn": 3, "fn": 1},
            "at_vd_s_cutoff": {"tp": 2, "fp": 0, "tn": 4, "fn": 2},
        })

    def test_counts_reject_inconsistent_percentage(self):
        with self.assertRaisesRegex(ValueError, "disagrees"):
            count_from_pct(16.0, 5, "pair-correct")

    def test_validation_uses_completed_snapshot(self):
        row1 = ["1000", "0.100000", "0.200000", "0.900000", "0.400000", "0.300000", "0.342857"]
        row2 = ["2000", "0.090000", "0.190000", "0.910000", "0.400000", "0.400000", "0.400000"]
        cell = {"outputs": [validation_fragment(2100, 2100, [row1, row2]), validation_fragment(1000, 2100, [row1])]}
        self.assertEqual(validation_history(cell, 1000), (2100, [row1, row2]))

    def test_validation_rejects_missing_scheduled_step(self):
        row = ["1000", "0.1", "0.2", "0.9", "0.4", "0.3", "0.34"]
        with self.assertRaisesRegex(ValueError, "scheduled steps"):
            validation_history({"outputs": [validation_fragment(2000, 2000, [row])]}, 1000)


if __name__ == "__main__":
    unittest.main()
