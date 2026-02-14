"""
Evaluation script for decoder-based experiments on BLUFF benchmark.

Computes macro-averaged F1 scores across languages, resource categories,
and linguistic typological features. Supports analysis by language family,
script type, and syntactic word order.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import classification_report, f1_score

logger = logging.getLogger(__name__)


def load_predictions(predictions_path: str) -> list[dict]:
    """Load model predictions from JSONL file."""
    predictions = []
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def compute_metrics(y_true: list, y_pred: list, average: str = "macro") -> dict:
    """Compute classification metrics."""
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "accuracy": sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0,
        "n_samples": len(y_true),
    }


def evaluate_by_group(predictions: list[dict], group_key: str, label_key: str) -> dict:
    """Evaluate predictions grouped by a metadata field."""
    groups = defaultdict(lambda: {"true": [], "pred": []})
    for pred in predictions:
        group = pred.get(group_key, "unknown")
        groups[group]["true"].append(pred[label_key])
        groups[group]["pred"].append(pred["prediction"])

    results = {}
    for group, data in groups.items():
        results[group] = compute_metrics(data["true"], data["pred"])
    return results


def evaluate_by_resource_category(predictions: list[dict], label_key: str) -> dict:
    """Evaluate big-head vs. long-tail performance gap."""
    results = evaluate_by_group(predictions, "resource_category", label_key)
    big_head = results.get("big-head", {}).get("f1_macro", 0)
    long_tail = results.get("long-tail", {}).get("f1_macro", 0)
    results["gap"] = big_head - long_tail
    return results


def evaluate_transfer(predictions: list[dict], label_key: str) -> dict:
    """Evaluate cross-lingual transfer by family, script, and syntax."""
    return {
        "by_family": evaluate_by_group(predictions, "language_family", label_key),
        "by_script": evaluate_by_group(predictions, "script", label_key),
        "by_syntax": evaluate_by_group(predictions, "syntax", label_key),
        "by_resource": evaluate_by_resource_category(predictions, label_key),
    }


def generate_report(predictions: list[dict], task: str) -> dict:
    """Generate comprehensive evaluation report."""
    label_key_map = {
        "task1_veracity_binary": "veracity_label",
        "task2_veracity_multiclass": "veracity_label",
        "task3_authorship_binary": "authorship_type",
        "task4_authorship_multiclass": "authorship_type",
    }
    label_key = label_key_map.get(task, "veracity_label")

    y_true = [p[label_key] for p in predictions]
    y_pred = [p["prediction"] for p in predictions]

    report = {
        "task": task,
        "overall": compute_metrics(y_true, y_pred),
        "by_language": evaluate_by_group(predictions, "language", label_key),
        "transfer_analysis": evaluate_transfer(predictions, label_key),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate decoder predictions on BLUFF")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSONL")
    parser.add_argument("--task", required=True, choices=[
        "task1_veracity_binary", "task2_veracity_multiclass",
        "task3_authorship_binary", "task4_authorship_multiclass"
    ])
    parser.add_argument("--output", default=None, help="Output path for results JSON")
    args = parser.parse_args()

    predictions = load_predictions(args.predictions)
    report = generate_report(predictions, args.task)

    print(f"\n{'='*60}")
    print(f"BLUFF Decoder Evaluation: {args.task}")
    print(f"{'='*60}")
    print(f"Overall F1 (macro): {report['overall']['f1_macro']:.4f}")
    print(f"Samples: {report['overall']['n_samples']}")

    transfer = report["transfer_analysis"]["by_resource"]
    if "big-head" in transfer and "long-tail" in transfer:
        print(f"\nBig-Head F1: {transfer['big-head']['f1_macro']:.4f}")
        print(f"Long-Tail F1: {transfer['long-tail']['f1_macro']:.4f}")
        print(f"Gap: {transfer['gap']:.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report saved to {args.output}")


if __name__ == "__main__":
    main()
