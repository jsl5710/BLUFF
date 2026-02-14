"""
Content validation utilities for BLUFF benchmark data quality assurance.

Provides validation functions for checking text quality, metadata consistency,
deduplication, and format compliance.
"""

import hashlib
import json
import logging
import re
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = [
    "id", "text", "language", "language_family", "script", "syntax",
    "resource_category", "veracity_label", "authorship_type", "source", "split"
]

VALID_AUTHORSHIP_TYPES = {"HWT", "MGT", "MTT", "HAT"}
VALID_VERACITY_LABELS = {"real", "fake"}
VALID_SPLITS = {"train", "dev", "test"}
VALID_RESOURCE_CATEGORIES = {"big-head", "long-tail"}
VALID_INTENSITIES = {"low", "medium", "high", None}


def validate_schema(sample: dict) -> list[str]:
    """Validate that a sample has all required fields with valid values."""
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in sample or sample[field] is None:
            errors.append(f"Missing required field: {field}")

    if sample.get("veracity_label") not in VALID_VERACITY_LABELS:
        errors.append(f"Invalid veracity_label: {sample.get('veracity_label')}")
    if sample.get("authorship_type") not in VALID_AUTHORSHIP_TYPES:
        errors.append(f"Invalid authorship_type: {sample.get('authorship_type')}")
    if sample.get("split") not in VALID_SPLITS:
        errors.append(f"Invalid split: {sample.get('split')}")
    if sample.get("resource_category") not in VALID_RESOURCE_CATEGORIES:
        errors.append(f"Invalid resource_category: {sample.get('resource_category')}")
    if sample.get("edit_intensity") not in VALID_INTENSITIES:
        errors.append(f"Invalid edit_intensity: {sample.get('edit_intensity')}")

    return errors


def validate_text_quality(text: str, min_length: int = 50, max_length: int = 50000) -> list[str]:
    """Check text quality: length, encoding, repetition, completeness."""
    errors = []
    if not text or not text.strip():
        return ["Empty text"]
    if len(text) < min_length:
        errors.append(f"Text too short: {len(text)} < {min_length}")
    if len(text) > max_length:
        errors.append(f"Text too long: {len(text)} > {max_length}")

    # Check for repetition artifacts
    words = text.split()
    if len(words) > 10:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        most_common_count = bigram_counts.most_common(1)[0][1] if bigram_counts else 0
        if most_common_count > len(words) * 0.1:
            errors.append("Excessive repetition detected")

    # Check for instruction leakage
    leakage_patterns = [
        r"as an ai", r"i cannot", r"i'm sorry", r"as a language model",
        r"here is the", r"sure, here", r"certainly!", r"\[inst\]", r"\[/inst\]"
    ]
    for pattern in leakage_patterns:
        if re.search(pattern, text.lower()):
            errors.append(f"Possible instruction leakage: '{pattern}'")
            break

    return errors


def compute_fingerprint(text: str) -> str:
    """Compute a text fingerprint for deduplication."""
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def check_near_duplicate(text1: str, text2: str, threshold: float = 0.85) -> bool:
    """Check if two texts are near-duplicates using Jaccard similarity on word n-grams."""
    def get_ngrams(text, n=3):
        words = text.lower().split()
        return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))

    ngrams1 = get_ngrams(text1)
    ngrams2 = get_ngrams(text2)

    if not ngrams1 or not ngrams2:
        return False

    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    jaccard = intersection / union if union > 0 else 0

    return jaccard >= threshold


def validate_metadata_consistency(sample: dict, languages_config: dict) -> list[str]:
    """Validate that metadata fields are consistent with language configuration."""
    errors = []
    lang = sample.get("language")

    if lang and languages_config:
        expected_family = languages_config.get(lang, {}).get("family")
        expected_script = languages_config.get(lang, {}).get("script")
        if expected_family and sample.get("language_family") != expected_family:
            errors.append(
                f"Family mismatch: {sample.get('language_family')} != {expected_family}"
            )
        if expected_script and sample.get("script") != expected_script:
            errors.append(
                f"Script mismatch: {sample.get('script')} != {expected_script}"
            )

    # HWT samples should not have generation metadata
    if sample.get("authorship_type") == "HWT":
        if sample.get("generation_model"):
            errors.append("HWT sample should not have generation_model")
        if sample.get("manipulation_tactic"):
            errors.append("HWT sample should not have manipulation_tactic")

    return errors


def validate_batch(samples: list[dict], languages_config: Optional[dict] = None) -> dict:
    """Validate a batch of samples and return summary statistics."""
    results = {"valid": 0, "invalid": 0, "errors": {}}

    for sample in samples:
        all_errors = []
        all_errors.extend(validate_schema(sample))
        all_errors.extend(validate_text_quality(sample.get("text", "")))
        if languages_config:
            all_errors.extend(validate_metadata_consistency(sample, languages_config))

        if all_errors:
            results["invalid"] += 1
            for err in all_errors:
                results["errors"][err] = results["errors"].get(err, 0) + 1
        else:
            results["valid"] += 1

    return results
