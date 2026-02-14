"""
BLUFF Encoder-Based Training Script
Fine-tunes multilingual encoder models for BLUFF benchmark tasks.

Usage:
    python src/evaluation/encoder/train.py \
        --model xlm-roberta-large \
        --task task1_veracity_binary \
        --experiment multilingual \
        --config configs/encoder_models.yaml \
        --data_dir data
"""

import argparse
import json
import os
import logging
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Task label mappings
TASK_LABELS = {
    "task1_veracity_binary": {"real": 0, "fake": 1},
    "task2_veracity_multiclass": {"real_hwt": 0, "fake_hwt": 1, "real_mgt": 2, "fake_mgt": 3},
    "task3_authorship_binary": {"human": 0, "machine": 1},
    "task4_authorship_multiclass": {"HWT": 0, "MGT": 1, "MTT": 2, "HAT": 3},
}


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    return {
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision_macro": precision_score(labels, preds, average="macro"),
        "recall_macro": recall_score(labels, preds, average="macro"),
        "accuracy": accuracy_score(labels, preds),
    }


def filter_by_languages(dataset, languages, lang_field="language"):
    """Filter dataset to include only specified languages."""
    if languages == "all":
        return dataset
    return dataset.filter(lambda x: x[lang_field] in languages)


def load_split_uuids(split_dir: str, split_name: str) -> set:
    """Load UUIDs from a split JSON file."""
    split_path = os.path.join(split_dir, f"{split_name}.json")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with open(split_path, "r") as f:
        return set(json.load(f))


def load_bluff_data(data_dir: str, split_dir: str, task: str, labels_map: dict):
    """
    Load BLUFF data from local files.

    Args:
        data_dir: Root data directory (containing meta_data/, processed/, splits/)
        split_dir: Path to the specific split directory (e.g., data/splits/evaluation/multilingual)
        task: Task name for label mapping
        labels_map: Dict mapping label strings to integers

    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    # Load metadata
    meta_ai_path = os.path.join(data_dir, "meta_data", "metadata_ai_generated.csv")
    meta_hw_path = os.path.join(data_dir, "meta_data", "metadata_human_written.csv")

    logger.info("Loading metadata...")
    meta_ai = pd.read_csv(meta_ai_path, low_memory=False)
    meta_hw = pd.read_csv(meta_hw_path, low_memory=False)

    # Load processed text data
    logger.info("Loading processed text data...")
    processed_dir = os.path.join(data_dir, "processed", "generated_data")
    text_dfs = []

    for csv_path in glob.glob(os.path.join(processed_dir, "**", "data.csv"), recursive=True):
        df = pd.read_csv(csv_path, low_memory=False)
        text_dfs.append(df)

    if not text_dfs:
        raise FileNotFoundError(f"No data.csv files found in {processed_dir}")

    text_data = pd.concat(text_dfs, ignore_index=True)
    logger.info(f"Loaded {len(text_data)} text samples from processed data")

    # Build datasets for each split
    datasets = {}
    for split_name, hf_split_name in [("train", "train"), ("val", "validation")]:
        try:
            uuids = load_split_uuids(split_dir, split_name)
        except FileNotFoundError:
            logger.warning(f"Split '{split_name}' not found in {split_dir}, skipping")
            continue

        # Filter text data by UUIDs
        split_text = text_data[text_data["uuid"].isin(uuids)].copy()

        # Determine text and label columns based on task
        if "veracity" in task:
            split_text["text"] = split_text["article_content"].fillna(split_text["post_content"])
            split_text["label_str"] = split_text["veracity"].map(
                lambda v: "fake" if "fake" in str(v).lower() else "real"
            )
        else:  # authorship tasks
            split_text["text"] = split_text["article_content"].fillna(split_text["post_content"])
            # Determine authorship type from flags
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

        # Map labels to integers
        split_text["label"] = split_text["label_str"].map(labels_map)
        split_text = split_text.dropna(subset=["text", "label"])
        split_text["label"] = split_text["label"].astype(int)

        # Create HF Dataset
        ds = Dataset.from_pandas(split_text[["uuid", "text", "language", "label"]].reset_index(drop=True))
        datasets[hf_split_name] = ds
        logger.info(f"  {hf_split_name}: {len(ds)} samples")

    return DatasetDict(datasets)


def main():
    parser = argparse.ArgumentParser(description="BLUFF Encoder Training")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--task", type=str, required=True, choices=list(TASK_LABELS.keys()))
    parser.add_argument("--experiment", type=str, default="multilingual",
                        choices=["multilingual", "crosslingual", "external"])
    parser.add_argument("--config", type=str, default="configs/encoder_models.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/encoder")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Root data directory (containing meta_data/, processed/, splits/)")
    parser.add_argument("--split_name", type=str, default="multilingual",
                        help="Split setting name (e.g., multilingual, cross_lingual_family/Indo_European)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--languages", type=str, default="all",
                        help="Comma-separated language codes or 'all'")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Find model config
    model_config = None
    for key, cfg in config["models"].items():
        if cfg["name"] == args.model or key == args.model:
            model_config = cfg
            break

    if model_config is None:
        logger.warning(f"Model '{args.model}' not found in config. Using defaults.")
        model_config = {"name": args.model, "max_length": 512, "batch_size": 32,
                        "learning_rate": 2e-5, "epochs": 5, "warmup_ratio": 0.1}

    # Override config with CLI args
    max_length = args.max_length or model_config.get("max_length", 512)
    batch_size = args.batch_size or model_config.get("batch_size", 32)
    learning_rate = args.learning_rate or model_config.get("learning_rate", 2e-5)
    epochs = args.epochs or model_config.get("epochs", 5)

    set_seed(args.seed)
    model_name = model_config["name"]
    labels = TASK_LABELS[args.task]
    num_labels = len(labels)

    output_dir = os.path.join(args.output_dir, args.task, args.experiment,
                              model_name.replace("/", "_"))
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Task: {args.task} | Model: {model_name} | Experiment: {args.experiment}")
    logger.info(f"Labels: {labels} | Num labels: {num_labels}")

    # Load dataset from local files
    split_dir = os.path.join(args.data_dir, "splits", "evaluation", args.split_name)
    logger.info(f"Loading dataset from: {args.data_dir}")
    logger.info(f"Split directory: {split_dir}")
    dataset = load_bluff_data(args.data_dir, split_dir, args.task, labels)

    # Filter languages if specified
    if args.languages != "all":
        lang_list = args.languages.split(",")
        dataset = DatasetDict({
            split: filter_by_languages(ds, lang_list)
            for split, ds in dataset.items()
        })
        logger.info(f"Filtered to languages: {lang_list}")

    # Load tokenizer and model
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text", "uuid", "language"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=config.get("training", {}).get("weight_decay", 0.01),
        warmup_ratio=model_config.get("warmup_ratio", 0.1),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=config.get("training", {}).get(
            "gradient_accumulation_steps", 2
        ),
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        seed=args.seed,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    trainer.save_model(os.path.join(output_dir, "best_model"))

    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_results = trainer.evaluate(metric_key_prefix="val")

    # Save results
    results = {
        "model": model_name,
        "task": args.task,
        "experiment": args.experiment,
        "split_name": args.split_name,
        "languages": args.languages,
        "train_results": {k: float(v) for k, v in train_result.metrics.items()},
        "val_results": {k: float(v) for k, v in val_results.items()},
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")
    logger.info(f"Val F1 (macro): {val_results.get('val_f1_macro', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
