"""
BLUFF Decoder-Based Inference Script
Evaluates multilingual decoder models with cross-lingual, native, and english-translated prompts.

Usage:
    python src/evaluation/decoder/inference.py \
        --model gpt-4o \
        --task task1_veracity_binary \
        --prompt_type crosslingual \
        --split test \
        --config configs/decoder_models.yaml
"""

import argparse
import json
import os
import logging
import time
from pathlib import Path

import yaml
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Prompt Templates
# ============================================================================

TASK_PROMPTS = {
    "task1_veracity_binary": {
        "crosslingual": {
            "system": "You are a fact-checking expert. Classify the following article as 'real' or 'fake'. Respond with ONLY the label.",
            "user": "Article:\n{text}\n\nLabel:"
        },
        "native": {
            # Prompts will be translated to target language at runtime
            "system": "You are a fact-checking expert. Classify the following article as 'real' or 'fake'. Respond with ONLY the label.",
            "user": "Article:\n{text}\n\nLabel:"
        },
        "english_translated": {
            "system": "You are a fact-checking expert. Classify the following article as 'real' or 'fake'. Respond with ONLY the label.",
            "user": "Article:\n{text}\n\nLabel:"
        },
    },
    "task3_authorship_binary": {
        "crosslingual": {
            "system": "You are a text forensics expert. Determine if the following text was written by a 'human' or 'machine'. Respond with ONLY the label.",
            "user": "Text:\n{text}\n\nLabel:"
        },
        "native": {
            "system": "You are a text forensics expert. Determine if the following text was written by a 'human' or 'machine'. Respond with ONLY the label.",
            "user": "Text:\n{text}\n\nLabel:"
        },
        "english_translated": {
            "system": "You are a text forensics expert. Determine if the following text was written by a 'human' or 'machine'. Respond with ONLY the label.",
            "user": "Text:\n{text}\n\nLabel:"
        },
    },
}

# Add multiclass variants
TASK_PROMPTS["task2_veracity_multiclass"] = {
    pt: {
        "system": p["system"].replace(
            "'real' or 'fake'",
            "'real_hwt', 'fake_hwt', 'real_mgt', or 'fake_mgt'"
        ),
        "user": p["user"],
    }
    for pt, p in TASK_PROMPTS["task1_veracity_binary"].items()
}

TASK_PROMPTS["task4_authorship_multiclass"] = {
    pt: {
        "system": p["system"].replace(
            "'human' or 'machine'",
            "'HWT' (human-written), 'MGT' (machine-generated), 'MTT' (machine-translated), or 'HAT' (human-AI hybrid)"
        ),
        "user": p["user"],
    }
    for pt, p in TASK_PROMPTS["task3_authorship_binary"].items()
}


# ============================================================================
# API Clients
# ============================================================================

def call_openai(model_name, system_prompt, user_prompt, config):
    """Call OpenAI API."""
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=config.get("max_tokens", 256),
        temperature=config.get("temperature", 0.0),
    )
    return response.choices[0].message.content.strip()


def call_anthropic(model_name, system_prompt, user_prompt, config):
    """Call Anthropic API."""
    from anthropic import Anthropic
    client = Anthropic()
    response = client.messages.create(
        model=model_name,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=config.get("max_tokens", 256),
        temperature=config.get("temperature", 0.0),
    )
    return response.content[0].text.strip()


def call_google(model_name, system_prompt, user_prompt, config):
    """Call Google Generative AI API."""
    import google.generativeai as genai
    model = genai.GenerativeModel(model_name, system_instruction=system_prompt)
    response = model.generate_content(
        user_prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=config.get("max_tokens", 256),
            temperature=config.get("temperature", 0.0),
        ),
    )
    return response.text.strip()


API_CLIENTS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "google": call_google,
}


def parse_prediction(response_text, task):
    """Parse model response into a standardized label."""
    text = response_text.lower().strip().strip("'\".,")

    if "binary" in task:
        if "veracity" in task:
            return "fake" if "fake" in text else "real"
        else:
            return "machine" if "machine" in text else "human"
    else:
        # Multiclass - attempt direct match
        if "veracity" in task:
            for label in ["real_hwt", "fake_hwt", "real_mgt", "fake_mgt"]:
                if label in text:
                    return label
            return "real_hwt"  # fallback
        else:
            for label in ["hwt", "mgt", "mtt", "hat"]:
                if label in text:
                    return label.upper()
            return "HWT"  # fallback


def main():
    parser = argparse.ArgumentParser(description="BLUFF Decoder Inference")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=list(TASK_PROMPTS.keys()))
    parser.add_argument("--prompt_type", type=str, required=True,
                        choices=["crosslingual", "native", "english_translated"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--config", type=str, default="configs/decoder_models.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/decoder")
    parser.add_argument("--languages", type=str, default="all")
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Find model config
    model_config = None
    for key, cfg in config["models"].items():
        if cfg["name"] == args.model or key == args.model:
            model_config = cfg
            break

    if model_config is None:
        raise ValueError(f"Model '{args.model}' not found in config")

    model_name = model_config["name"]
    provider = model_config["provider"]
    api_client = API_CLIENTS.get(provider)

    if api_client is None:
        raise ValueError(f"Unsupported provider: {provider}. Use: {list(API_CLIENTS.keys())}")

    # Setup output
    output_dir = os.path.join(args.output_dir, args.task, args.prompt_type,
                              model_name.replace("/", "_"))
    os.makedirs(output_dir, exist_ok=True)

    # Load prompts
    prompts = TASK_PROMPTS[args.task][args.prompt_type]

    # Load dataset
    logger.info(f"Loading dataset: {args.task} [{args.split}]")
    dataset = load_dataset("jsl5710/BLUFF", args.task, split=args.split)

    if args.languages != "all":
        lang_list = args.languages.split(",")
        dataset = dataset.filter(lambda x: x["language"] in lang_list)

    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    logger.info(f"Evaluating {len(dataset)} samples with {model_name} [{args.prompt_type}]")

    # Inference
    inference_config = config.get("inference", {})
    results = []
    errors = 0

    for i, example in enumerate(tqdm(dataset, desc="Inference")):
        text_field = "text"
        if args.prompt_type == "english_translated" and "text_en" in example:
            text_field = "text_en"

        user_prompt = prompts["user"].format(text=example[text_field])

        for attempt in range(inference_config.get("max_retries", 3)):
            try:
                response = api_client(model_name, prompts["system"], user_prompt, model_config)
                pred = parse_prediction(response, args.task)

                results.append({
                    "id": example.get("id", i),
                    "language": example["language"],
                    "true_label": example.get("label", example.get("veracity_label", "")),
                    "predicted_label": pred,
                    "raw_response": response,
                })
                break

            except Exception as e:
                if attempt < inference_config.get("max_retries", 3) - 1:
                    time.sleep(inference_config.get("retry_delay", 5))
                else:
                    errors += 1
                    results.append({
                        "id": example.get("id", i),
                        "language": example["language"],
                        "true_label": example.get("label", ""),
                        "predicted_label": "ERROR",
                        "raw_response": str(e),
                    })

        # Rate limiting
        time.sleep(inference_config.get("rate_limit_delay", 1))

    # Save predictions
    preds_path = os.path.join(output_dir, f"predictions_{args.split}.jsonl")
    with open(preds_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Compute metrics
    valid = [r for r in results if r["predicted_label"] != "ERROR"]
    if valid:
        from sklearn.metrics import f1_score, accuracy_score

        true_labels = [r["true_label"] for r in valid]
        pred_labels = [r["predicted_label"] for r in valid]

        metrics = {
            "f1_macro": round(f1_score(true_labels, pred_labels, average="macro", zero_division=0), 4),
            "accuracy": round(accuracy_score(true_labels, pred_labels), 4),
            "num_samples": len(valid),
            "num_errors": errors,
        }
    else:
        metrics = {"error": "No valid predictions"}

    summary = {
        "model": model_name,
        "task": args.task,
        "prompt_type": args.prompt_type,
        "split": args.split,
        "metrics": metrics,
    }

    summary_path = os.path.join(output_dir, f"results_{args.split}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Predictions: {preds_path}")
    logger.info(f"Results: {summary_path}")
    logger.info(f"F1 (macro): {metrics.get('f1_macro', 'N/A')}")


if __name__ == "__main__":
    main()
