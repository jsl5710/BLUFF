"""
BLUFF Encoder-Based Training Script
Fine-tunes multilingual encoder models for BLUFF benchmark tasks.

Usage:
    python src/evaluation/encoder/train.py \
        --model xlm-roberta-large \
        --task task1_veracity_binary \
        --experiment multilingual \
        --config configs/encoder_models.yaml
"""

import argparse
import json
import os
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset, DatasetDict
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


def main():
    parser = argparse.ArgumentParser(description="BLUFF Encoder Training")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--task", type=str, required=True, choices=list(TASK_LABELS.keys()))
    parser.add_argument("--experiment", type=str, default="multilingual",
                        choices=["multilingual", "crosslingual", "external"])
    parser.add_argument("--config", type=str, default="configs/encoder_models.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/encoder")
    parser.add_argument("--data_dir", type=str, default="data/splits")
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

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("jsl5710/BLUFF", args.task)

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

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

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
        eval_dataset=tokenized["dev"] if "dev" in tokenized else tokenized["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    trainer.save_model(os.path.join(output_dir, "best_model"))

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(tokenized["test"], metric_key_prefix="test")

    # Save results
    results = {
        "model": model_name,
        "task": args.task,
        "experiment": args.experiment,
        "languages": args.languages,
        "train_results": {k: float(v) for k, v in train_result.metrics.items()},
        "test_results": {k: float(v) for k, v in test_results.items()},
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")
    logger.info(f"Test F1 (macro): {test_results.get('test_f1_macro', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
