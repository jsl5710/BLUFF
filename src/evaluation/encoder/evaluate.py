"""
BLUFF Encoder-Based Evaluation Script
Evaluates trained models with disaggregated results by language, family, script, and syntax.

Usage:
    python src/evaluation/encoder/evaluate.py \
        --model outputs/encoder/task1_veracity_binary/multilingual/xlm-roberta-large/best_model \
        --task task1_veracity_binary \
        --data_dir data \
        --split_name multilingual \
        --disaggregate resource_category language_family script syntax
"""

import argparse
import json
import os
import logging
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, classification_report,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Task label mappings
TASK_LABELS = {
    "task1_veracity_binary": {"real": 0, "fake": 1},
    "task2_veracity_multiclass": {"real_hwt": 0, "fake_hwt": 1, "real_mgt": 2, "fake_mgt": 3},
    "task3_authorship_binary": {"human": 0, "machine": 1},
    "task4_authorship_multiclass": {"HWT": 0, "MGT": 1, "MTT": 2, "HAT": 3},
}


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


def load_eval_data(data_dir: str, split_dir: str, split_name: str, task: str, labels_map: dict):
    """Load evaluation data from local files."""

    # Load split UUIDs
    split_path = os.path.join(split_dir, f"{split_name}.json")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with open(split_path, "r") as f:
        uuids = set(json.load(f))

    # Load processed text data
    processed_dir = os.path.join(data_dir, "processed", "generated_data")
    text_dfs = []
    for csv_path in glob.glob(os.path.join(processed_dir, "**", "data.csv"), recursive=True):
        df = pd.read_csv(csv_path, low_memory=False)
        text_dfs.append(df)

    text_data = pd.concat(text_dfs, ignore_index=True)
    split_text = text_data[text_data["uuid"].isin(uuids)].copy()

    # Load metadata for resource_category info
    meta_ai_path = os.path.join(data_dir, "meta_data", "metadata_ai_generated.csv")
    if os.path.exists(meta_ai_path):
        meta_ai = pd.read_csv(meta_ai_path, usecols=["uuid", "language_category"], low_memory=False)
        split_text = split_text.merge(meta_ai[["uuid", "language_category"]], on="uuid", how="left")
        split_text["resource_category"] = split_text["language_category"].fillna("unknown")
    else:
        split_text["resource_category"] = "unknown"

    # Build text and labels
    if "veracity" in task:
        split_text["text"] = split_text["article_content"].fillna(split_text["post_content"])
        split_text["label_str"] = split_text["veracity"].map(
            lambda v: "fake" if "fake" in str(v).lower() else "real"
        )
    else:
        split_text["text"] = split_text["article_content"].fillna(split_text["post_content"])
        def get_authorship(row):
            if str(row.get("HWT", "")).lower() == "y":
                return "HWT" if "multiclass" in task else "human"
            elif str(row.get("MGT", "")).lower() == "y":
                return "MGT" if "multiclass" in task else "machine"
            elif str(row.get("MTT", "")).lower() == "y":
                return "MTT" if "multiclass" in task else "machine"
            elif str(row.get("HAT", "")).lower() == "y":
                return "HAT" if "multiclass" in task else "machine"
            return "human"
        split_text["label_str"] = split_text.apply(get_authorship, axis=1)

    split_text["label"] = split_text["label_str"].map(labels_map)
    split_text = split_text.dropna(subset=["text", "label"])
    split_text["label"] = split_text["label"].astype(int)

    return split_text


def main():
    parser = argparse.ArgumentParser(description="BLUFF Encoder Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--task", type=str, required=True, choices=list(TASK_LABELS.keys()))
    parser.add_argument("--split", type=str, default="val",
                        help="Split to evaluate on (train or val)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Root data directory")
    parser.add_argument("--split_name", type=str, default="multilingual",
                        help="Split setting name (e.g., multilingual, cross_lingual_family/Indo_European)")
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

    # Load dataset from local files
    split_dir = os.path.join(args.data_dir, "splits", "evaluation", args.split_name)
    logger.info(f"Loading data from: {args.data_dir}, split: {args.split}")
    labels_map = TASK_LABELS[args.task]
    eval_df = load_eval_data(args.data_dir, split_dir, args.split, args.task, labels_map)

    if args.languages != "all":
        lang_list = args.languages.split(",")
        eval_df = eval_df[eval_df["language"].isin(lang_list)]

    logger.info(f"Evaluating on {len(eval_df)} samples")

    # Inference
    all_preds = []
    all_labels = eval_df["label"].tolist()
    all_examples = eval_df[["language", "resource_category"]].to_dict("records")
    texts = eval_df["text"].tolist()

    for i in tqdm(range(0, len(texts), args.batch_size), desc="Evaluating"):
        batch_texts = texts[i:i + args.batch_size]
        inputs = tokenizer(
            batch_texts, truncation=True, max_length=args.max_length,
            padding=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

        all_preds.extend(preds.tolist())

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
        "split_name": args.split_name,
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
