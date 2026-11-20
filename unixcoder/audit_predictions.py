#!/usr/bin/env python3
"""Recompute aggregates from the two private Colab prediction exports.

This reads local CSVs only and prints summary counts and scores, never rows.
"""

import argparse
import csv
from collections import Counter, defaultdict
from itertools import groupby
import json
import math
from pathlib import Path
import struct


def float32(value):
    return struct.unpack(">f", struct.pack(">f", value))[0]


def read_predictions(path, *, paired):
    expected = ["idx", "target", "prediction", "vulnerability_score"]
    if paired:
        expected.append("pair_number")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames != expected:
            raise ValueError("Prediction CSV header does not match the Colab export")
        for row in rows:
            score = float32(float(row["vulnerability_score"]))
            target = int(row["target"])
            prediction = int(row["prediction"])
            if not math.isfinite(score) or not 0 <= score <= 1 or target not in (0, 1):
                raise ValueError("Expected finite scores in [0,1] and binary targets")
            if prediction != int(score >= 0.5):
                raise ValueError("Saved prediction does not agree with the 0.5 cutoff")
            yield target, score, int(row["pair_number"]) if paired else None


def confusion(rows, cutoff):
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for target, score, _ in rows:
        key = ("tp" if target else "fp") if score >= cutoff else ("fn" if target else "tn")
        counts[key] += 1
    return counts


def capped_point(rows, cap=0.005):
    negatives = sum(target == 0 for target, _, _ in rows)
    if not 0 < negatives < len(rows):
        raise ValueError("The full test export must contain both target classes")
    ordered = sorted(rows, key=lambda row: row[1], reverse=True)
    fp = tp = 0
    threshold = None
    for score, group in groupby(ordered, key=lambda row: row[1]):
        for target, _, _ in group:
            tp += target
            fp += 1 - target
        if fp / negatives > cap:
            break
        threshold = score
    counts = confusion(rows, threshold if threshold is not None else math.inf)
    return threshold, counts


def pair_summary(rows):
    pairs = defaultdict(list)
    for target, score, pair_number in rows:
        pairs[pair_number].append((target, score))
    if not pairs:
        raise ValueError("The paired export is empty")
    outcomes = Counter(dict.fromkeys(("P-C", "P-V", "P-B", "P-R"), 0))
    rank = Counter(dict.fromkeys(("higher", "tie", "lower"), 0))
    keys = {(True, False): "P-C", (True, True): "P-V", (False, False): "P-B", (False, True): "P-R"}
    for scores in pairs.values():
        if len(scores) != 2 or {target for target, _ in scores} != {0, 1}:
            raise ValueError("Each pair must contain one vulnerable and one benign function")
        by_label = dict(scores)
        v, b = by_label[1], by_label[0]
        outcomes[keys[v >= 0.5, b >= 0.5]] += 1
        rank["higher" if v > b else "tie" if v == b else "lower"] += 1
    return {
        "pairs": len(pairs),
        "outcomes": dict(outcomes),
        "same_verdict": outcomes["P-V"] + outcomes["P-B"],
        "vulnerable_score_relative_to_fixed": dict(rank),
        "rank_accuracy_ties_half_credit_pct": 100 * (rank["higher"] + 0.5 * rank["tie"]) / len(pairs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full_test_csv", type=Path)
    parser.add_argument("paired_test_csv", type=Path)
    args = parser.parse_args()
    full = list(read_predictions(args.full_test_csv, paired=False))
    paired = pair_summary(read_predictions(args.paired_test_csv, paired=True))
    threshold, vd_counts = capped_point(full)
    print(json.dumps({
        "full_test": {
            "rows": len(full),
            "counts_at_0_5": confusion(full, 0.5),
            "vd_s_probability_cutoff": threshold,
            "counts_at_vd_s_cutoff": vd_counts,
        },
        "paired_test": paired,
    }, indent=2))


if __name__ == "__main__":
    main()
