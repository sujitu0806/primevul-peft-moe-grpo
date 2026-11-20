#!/usr/bin/env python3
"""Export aggregate evidence from a completed UniXcoder notebook without running it.

Usage: python colab/extract_midpoint.py /path/to/completed.ipynb
The saved Trainer validation table is required. No dataset files are read.
"""

import argparse
import ast
import csv
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re


NOTEBOOK_CODE = (
    "https://github.com/sujitu0806/primevul-peft-moe-grpo/blob/"
    "8ed9f174d025f9be798c6d5b64d381db008c4d33/primevul_unixcoder_baseline_colab.ipynb"
)
PAPER = "https://arxiv.org/html/2403.18624v2#S5.T5"
DATASET = "https://github.com/DLVulDet/PrimeVul#-primevul-dataset"
OUTCOMES = ("P-C", "P-V", "P-B", "P-R")
FULL_METRICS = ("accuracy", "precision", "recall", "f1", "VD-S", "VD-S realized FPR")
PUBLISHED_METRICS = ("accuracy", "f1", "VD-S", *OUTCOMES)
VALIDATION_HEADINGS = (
    "Step", "Training Loss", "Validation Loss", "Accuracy", "Precision", "Recall", "F1"
)
CSV_HEADINGS = (
    "optimizer_step", "logged_training_loss", "validation_loss", "validation_accuracy",
    "validation_precision", "validation_recall", "validation_f1",
)


def text(value):
    return "".join(value) if isinstance(value, list) else value or ""


def matching_cell(cells, pattern):
    matches = [c for c in cells if c["cell_type"] == "code" and pattern in text(c["source"])]
    if len(matches) != 1:
        raise ValueError(f"Expected one code cell matching {pattern!r}; found {len(matches)}")
    return matches[0]


def streams(cell):
    return "\n".join(text(o.get("text")) for o in cell.get("outputs", []) if o["output_type"] == "stream")


def extract_line(pattern, output):
    match = re.search(pattern, output, re.MULTILINE)
    if match is None:
        raise ValueError(f"Required saved output not found: {pattern}")
    return match.group(1).strip()


def literals(cell, names):
    result = {}
    for statement in ast.parse(text(cell["source"])).body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    result[target.id] = ast.literal_eval(statement.value)
    if set(names) != result.keys():
        raise ValueError(f"Missing run settings: {set(names) - result.keys()}")
    return result


class CompletedTrainerDisplay(HTMLParser):
    def __init__(self):
        super().__init__()
        self.completed_step = None
        self.rows = []
        self.row = None
        self.field = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "progress" and attrs.get("value") == attrs.get("max") and attrs.get("max"):
            self.completed_step = int(attrs["max"])
        elif tag == "tr":
            self.row = []
        elif tag in {"th", "td"} and self.row is not None:
            self.field = []

    def handle_data(self, data):
        if self.field is not None:
            self.field.append(data)

    def handle_endtag(self, tag):
        if tag in {"th", "td"} and self.field is not None:
            self.row.append("".join(self.field).strip())
            self.field = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def validation_history(cell, interval):
    for output in reversed(cell.get("outputs", [])):
        parser = CompletedTrainerDisplay()
        parser.feed(text(output.get("data", {}).get("text/html")))
        if parser.completed_step and parser.rows and tuple(parser.rows[0]) == VALIDATION_HEADINGS:
            rows = parser.rows[1:]
            if [int(row[0]) for row in rows] != list(range(interval, parser.completed_step + 1, interval)):
                raise ValueError("Saved validation rows do not match the scheduled steps")
            if any(len(row) != len(CSV_HEADINGS) for row in rows):
                raise ValueError("Saved validation row has an unexpected number of fields")
            return parser.completed_step, rows
    raise ValueError("Missing completed Trainer progress output and validation table")


def assert_close(actual, reported, name, tolerance=1e-8):
    if not math.isclose(actual, reported, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{name} disagrees with saved output: {actual!r} vs {reported!r}")


def percentage(numerator, denominator):
    return 100.0 * numerator / denominator if denominator else 0.0


def count_from_pct(value, denominator, name):
    numerator = round(value * denominator / 100)
    if not 0 <= numerator <= denominator:
        raise ValueError(f"{name} is outside the range allowed by its denominator")
    assert_close(percentage(numerator, denominator), value, name)
    return numerator


def confusion_counts(full, test_split):
    positives = test_split["vulnerable"]
    negatives = test_split["benign"]
    tp = count_from_pct(full["recall"], positives, "recall")
    correct = count_from_pct(full["accuracy"], positives + negatives, "accuracy")
    tn = correct - tp
    fp, fn = negatives - tn, positives - tp
    if min(tp, tn, fp, fn) < 0:
        raise ValueError("Negative confusion count inferred from saved output")
    assert_close(percentage(tp, tp + fp), full["precision"], "precision")
    assert_close(percentage(2 * tp, 2 * tp + fp + fn), full["f1"], "F1")
    vd_fn = count_from_pct(full["VD-S"], positives, "VD-S")
    vd_fp = count_from_pct(full["VD-S realized FPR"], negatives, "VD-S realized FPR")
    return {
        "at_classifier_cutoff": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "at_vd_s_cutoff": {"tp": positives - vd_fn, "fp": vd_fp, "tn": negatives - vd_fp, "fn": vd_fn},
    }


def aggregate(notebook):
    cells = notebook["cells"]
    config_cell = matching_cell(cells, "MODEL_NAME =")
    settings = literals(config_cell, (
        "MODEL_NAME", "SEED", "MAX_LENGTH", "NUM_EPOCHS", "LEARNING_RATE", "WARMUP_STEPS",
        "TRAIN_BATCH_SIZE", "GRADIENT_ACCUMULATION_STEPS", "EVAL_BATCH_SIZE",
        "EVAL_AND_SAVE_STEPS", "USE_BF16", "THRESHOLD",
    ))
    load_output = streams(matching_cell(cells, 'print("Tokenized columns:"'))
    splits = {}
    for name, n, vulnerable, benign in re.findall(
        r"^(train|validation|test|test_paired)\s*:\s*([\d,]+) rows \| vulnerable=\s*([\d,]+) \| benign=\s*([\d,]+)$",
        load_output, re.MULTILINE,
    ):
        n, vulnerable, benign = (int(v.replace(",", "")) for v in (n, vulnerable, benign))
        if n != vulnerable + benign or min(vulnerable, benign) <= 0:
            raise ValueError(f"Invalid split counts: {name}")
        splits[name] = {"rows": n, "vulnerable": vulnerable, "benign": benign}
    if set(splits) != {"train", "validation", "test", "test_paired"}:
        raise ValueError("Missing saved data split counts")

    trainer_cell = matching_cell(cells, "training_args = TrainingArguments(")
    trainer_args_call = next(
        node for node in ast.walk(ast.parse(text(trainer_cell["source"])))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "TrainingArguments"
    )
    args = {k.arg: ast.literal_eval(k.value) for k in trainer_args_call.keywords if k.arg in {
        "optim", "weight_decay", "max_grad_norm", "eval_strategy", "save_strategy",
        "load_best_model_at_end", "metric_for_best_model",
    }}
    if args["eval_strategy"] != "steps" or args["save_strategy"] != "steps":
        raise ValueError("This exporter expects step-based evaluation and saving")
    if not args["load_best_model_at_end"] or args["metric_for_best_model"] != "f1":
        raise ValueError("This exporter expects the checkpoint selected by validation F1")

    training = matching_cell(cells, "train_result = trainer.train(")
    output = streams(training)
    last_step, history = validation_history(training, settings["EVAL_AND_SAVE_STEPS"])
    best_step = int(extract_line(r"^Best checkpoint: [^\n]*/checkpoint-(\d+)\s*$", output))
    best_f1 = float(extract_line(r"^Best validation F1: ([0-9.]+)\s*$", output))
    max_row = max(history, key=lambda row: float(row[-1]))
    if int(max_row[0]) != best_step:
        raise ValueError("Best checkpoint disagrees with saved validation history")
    assert_close(float(max_row[-1]), best_f1, "best validation F1", tolerance=5e-7)
    elapsed = extract_line(r"^\s*train_runtime\s*=\s*([\d:.]+)\s*$", output)
    epochs = float(extract_line(r"^\s*epoch\s*=\s*([\d.]+)\s*$", output))
    assert_close(epochs, settings["NUM_EPOCHS"], "number of completed epochs")
    h, m, s = elapsed.split(":")

    evaluated = streams(matching_cell(cells, 'print("Full-test metrics:")'))
    decode = json.JSONDecoder()
    full, _ = decode.raw_decode(evaluated.split("Full-test metrics:", 1)[1].lstrip())
    paired, _ = decode.raw_decode(evaluated.split("Paired metrics:", 1)[1].lstrip())
    n_pairs = paired["pairs"]
    paired_pct = {outcome: paired[outcome] for outcome in OUTCOMES}
    paired_counts = {outcome: count_from_pct(paired[outcome], n_pairs, outcome) for outcome in OUTCOMES}
    if sum(paired_counts.values()) != n_pairs or any(
        splits["test_paired"][key] != n_pairs for key in ("vulnerable", "benign")
    ):
        raise ValueError("Pair counts do not match the paired-test split")
    if full["VD-S realized FPR"] > 0.5 + 1e-8:
        raise ValueError("VD-S operating point exceeds the published false-positive cap")

    published = literals(matching_cell(cells, "PUBLISHED_UNIXCODER_PV_PV ="), ("PUBLISHED_UNIXCODER_PV_PV",))
    published = published["PUBLISHED_UNIXCODER_PV_PV"]
    config_output = streams(config_cell)
    evidence = {
        "schema_version": 1,
        "provenance": {
            "observed": "Settings and saved outputs of the completed Colab run; not a new run by this exporter.",
            "derived": "Integer counts reconstructed from saved percentages and denominators, not per-example predictions.",
            "dataset": "PrimeVul original release",
            "dataset_url": DATASET,
            "base_notebook_code_url": NOTEBOOK_CODE,
            "notebook_difference": "The executed notebook projects each JSONL row onto the three fields needed for training or evaluation and preserves row order.",
        },
        "run": {
            "model": settings["MODEL_NAME"],
            "gpu": extract_line(r"^GPU: (.+)$", config_output),
            "pytorch_version": extract_line(r"^PyTorch: (.+)$", config_output),
            "seed": settings["SEED"],
            "precision": "BF16" if settings["USE_BF16"] else "FP32",
            "initialization": "pretrained_weights" if "Resume checkpoint: none;" in streams(trainer_cell) else "resumed_from_checkpoint",
            "max_input_tokens": settings["MAX_LENGTH"],
            "epochs_completed": epochs,
            "learning_rate": settings["LEARNING_RATE"],
            "warmup_optimizer_steps": settings["WARMUP_STEPS"],
            "train_microbatch_size": settings["TRAIN_BATCH_SIZE"],
            "gradient_accumulation_steps": settings["GRADIENT_ACCUMULATION_STEPS"],
            "effective_train_batch_size": settings["TRAIN_BATCH_SIZE"] * settings["GRADIENT_ACCUMULATION_STEPS"],
            "eval_batch_size": settings["EVAL_BATCH_SIZE"],
            "eval_and_save_interval_optimizer_steps": settings["EVAL_AND_SAVE_STEPS"],
            "optimizer": args["optim"],
            "weight_decay": args["weight_decay"],
            "max_grad_norm": args["max_grad_norm"],
            "loss": "Unweighted binary cross-entropy with logits",
            "trainer_runtime_hms": elapsed,
            "trainer_runtime_seconds": round(int(h) * 3600 + int(m) * 60 + float(s), 2),
            "optimizer_steps_completed": last_step,
            "best_checkpoint_optimizer_step": best_step,
            "best_validation_f1": best_f1,
        },
        "split_counts": splits,
        "evaluation": {
            "classifier_probability_cutoff": settings["THRESHOLD"],
            "positive_when_score_is_at_least_cutoff": True,
            "vd_s_max_fpr_pct": 0.5,
            "vd_s_probability_cutoff": full["VD-S threshold"] if math.isfinite(full["VD-S threshold"]) else None,
            "paired_test_pairs": n_pairs,
        },
        "observed_metrics_pct": {
            "full_test": {key: full[key] for key in FULL_METRICS},
            "paired_test": paired_pct,
        },
        "derived_counts": {"full_test": confusion_counts(full, splits["test"]), "paired_test": paired_counts},
        "paper_reference": {
            "description": "Ding et al., Table V, UniXcoder trained and evaluated on PrimeVul (published rounded percentages).",
            "url": PAPER,
            "metrics_pct": {key: published[key] for key in PUBLISHED_METRICS},
        },
        "validation_history": {
            "file": "validation_history.csv",
            "rows": len(history),
            "metric_units": "Fractions at the precision displayed in the notebook.",
        },
    }
    return evidence, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path, help="The completed .ipynb with saved outputs")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    options = parser.parse_args()
    evidence, history = aggregate(json.loads(options.notebook.read_text(encoding="utf-8")))
    options.output_dir.mkdir(parents=True, exist_ok=True)
    (options.output_dir / "midpoint_baseline.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (options.output_dir / "validation_history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(CSV_HEADINGS)
        writer.writerows(history)


if __name__ == "__main__":
    main()
