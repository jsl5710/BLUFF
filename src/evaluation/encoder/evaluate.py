"""
BLUFF Encoder-Based Evaluation Script
Evaluates trained models with disaggregated results by language, family, script, and syntax.

Usage:
    python src/evaluation/encoder/evaluate.py \
        --model outputs/encoder/task1_veracity_binary/multilingual/xlm-roberta-large/best_model \
        --task task1_veracity_binary \
        --split test \
        --languages all \
        --disaggregate resource_category language_family script syntax
"""

import argparse
import json
import os
import logging
from collections import defaultdict

import numpy as np
import torch
import yaml
from datasets import load_dataset
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, classification_report,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_language_taxonomy(config_path: str = "configs/languages.yaml") -> dict:
    """Load language taxonomy for disaggregated evaluation."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_metrics_for_group(labels, preds):
    """Compute metrics for a specific group."""
    if len(labels) == 0:
        return {}
    return {
        "f1_macro": round(f1_score(labels, preds, average="macro"), 4),
        "precision_macro": round(precision_score(labels, preds, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(labels, preds, average="macro", zero_division=0), 4),
        "accuracy": round(accuracy_score(labels, preds), 4),
        "num_samples": len(labels),
    }


def disaggregate_results(examples, predictions, labels, taxonomy, dimensions):
    """Compute disaggregated results across specified dimensions."""
    results = {}

    for dim in dimensions:
        results[dim] = {}

        if dim == "resource_category":
            groups = defaultdict(lambda: {"labels": [], "preds": []})
            for ex, pred, label in zip(examples, predictions, labels):
                cat = ex.get("resource_category", "unknown")
                groups[cat]["labels"].append(label)
                groups[cat]["preds"].append(pred)

        elif dim == "language":
            groups = defaultdict(lambda: {"labels": [], "preds": []})
            for ex, pred, label in zip(examples, predictions, labels):
                lang = ex.get("language", "unknown")
                groups[lang]["labels"].append(label)
                groups[lang]["preds"].append(pred)

        elif dim == "language_family":
            lang_to_family = {}
            for family, info in taxonomy.get("language_families", {}).items():
                for lang in info.get("languages", []):
                    lang_to_family[lang] = family

            groups = defaultdict(lambda: {"labels": [], "preds": []})
            for ex, pred, label in zip(examples, predictions, labels):
                family = lang_to_family.get(ex.get("language", ""), "Unknown")
                groups[family]["labels"].append(label)
                groups[family]["preds"].append(pred)

        elif dim == "script":
            lang_to_script = {}
            for script, info in taxonomy.get("script_types", {}).items():
                for lang in info.get("languages", []):
                    lang_to_script[lang] = script

            groups = defaultdict(lambda: {"labels": [], "preds": []})
            for ex, pred, label in zip(examples, predictions, labels):
                script = lang_to_script.get(ex.get("language", ""), "Unknown")
                groups[script]["labels"].append(label)
                groups[script]["preds"].append(pred)

        elif dim == "syntax":
            lang_to_syntax = {}
            for syntax, info in taxonomy.get("syntax_types", {}).items():
                for lang in info.get("languages", []):
                    lang_to_syntax[lang] = syntax

            groups = defaultdict(lambda: {"labels": [], "preds": []})
            for ex, pred, label in zip(examples, predictions, labels):
                syn = lang_to_syntax.get(ex.get("language", ""), "Unknown")
                groups[syn]["labels"].append(label)
                groups[syn]["preds"].append(pred)

        else:
            logger.warning(f"Unknown dimension: {dim}")
            continue

        for group_name, group_data in groups.items():
            metrics = compute_metrics_for_group(group_data["labels"], group_data["preds"])
            results[dim][group_name] = metrics

    return results


def main():
    parser = argparse.ArgumentParser(description="BLUFF Encoder Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--languages", type=str, default="all")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--disaggregate", type=str, nargs="+",
                        default=["resource_category", "language_family", "script", "syntax"],
                        help="Dimensions for disaggregated evaluation")
    parser.add_argument("--taxonomy_config", type=str, default="configs/languages.yaml")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load taxonomy
    taxonomy = load_language_taxonomy(args.taxonomy_config)

    # Load model and tokenizer
    logger.info(f"Loading model from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    model.eval()

    # Load dataset
    logger.info(f"Loading dataset: {args.task} [{args.split}]")
    dataset = load_dataset("jsl5710/BLUFF", args.task, split=args.split)

    if args.languages != "all":
        lang_list = args.languages.split(",")
        dataset = dataset.filter(lambda x: x["language"] in lang_list)

    logger.info(f"Evaluating on {len(dataset)} samples")

    # Inference
    all_preds = []
    all_labels = []
    all_examples = []

    for i in tqdm(range(0, len(dataset), args.batch_size), desc="Evaluating"):
        batch = dataset[i:i + args.batch_size]
        inputs = tokenizer(
            batch["text"], truncation=True, max_length=args.max_length,
            padding=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(batch["label"] if "label" in batch else batch["veracity_label"])

        for j in range(len(preds)):
            all_examples.append({
                "language": batch["language"][j],
                "resource_category": batch.get("resource_category", ["unknown"])[j],
            })

    # Overall metrics
    overall = compute_metrics_for_group(all_labels, all_preds)
    logger.info(f"Overall F1 (macro): {overall['f1_macro']:.4f}")

    # Disaggregated results
    disagg = disaggregate_results(all_examples, all_preds, all_labels, taxonomy, args.disaggregate)

    # Compile and save results
    results = {
        "model": args.model,
        "task": args.task,
        "split": args.split,
        "num_samples": len(all_labels),
        "overall": overall,
        "disaggregated": disagg,
    }

    output_dir = args.output_dir or os.path.dirname(args.model)
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f"eval_{args.split}_results.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"  BLUFF Evaluation Results: {args.task}")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Split: {args.split} ({len(all_labels)} samples)")
    print(f"  Overall F1 (macro): {overall['f1_macro']:.4f}")
    print("-" * 60)

    for dim in args.disaggregate:
        if dim in disagg:
            print(f"\n  {dim.upper()}:")
            for group, metrics in sorted(disagg[dim].items()):
                print(f"    {group:25s}  F1={metrics['f1_macro']:.4f}  (n={metrics['num_samples']})")

    print("=" * 60)


if __name__ == "__main__":
    main()
