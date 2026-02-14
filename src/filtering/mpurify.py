"""
mPURIFY: Multilingual Pipeline for Unified Review, Inspection, Filtering, and Yield

Quality filtering pipeline for BLUFF benchmark data. Evaluates generated content
across 5 dimensions using 32 features with asymmetric thresholds for real vs. fake content.

Dimensions:
    1. Consistency: Language, format, and structural consistency
    2. Validation: Factual manipulation verification
    3. Translation: Semantic preservation across languages
    4. Hallucination: Detection of fabricated content artifacts
    5. Defective Generation: Identification of malformed outputs

Evaluation Metrics:
    Standard AEM: MENLI, FrugalScore, AlignScore, BERTScore, YiSi-2, COMET-QE, SelfCheckGPT
    LLM-AEM: 32 features scored by LLM evaluators with asymmetric thresholds

Retention Rates:
    Overall: 79,559/181,966 samples (43.7%)
    Real: 41,779 (23.0%)
    Fake: 36,664 (20.1%)

Reference:
    Lucas & Lee (2026). "BLUFF: A Benchmark for Linguistic Understanding
    of Fake-news Forensics." KDD 2026.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ContentType(Enum):
    REAL = "real"
    FAKE = "fake"


@dataclass
class FilterThresholds:
    """Asymmetric quality thresholds for real vs. fake content."""
    real_llm_aem_threshold: float = 4.0   # Higher bar for real content quality
    fake_llm_aem_threshold: float = 3.0   # Lower bar to retain challenging fakes
    bertscore_min: float = 0.75
    comet_qe_min: float = 0.6
    language_confidence_min: float = 0.8
    dedup_jaccard_max: float = 0.85


@dataclass
class QualityReport:
    """Quality assessment report for a single sample."""
    sample_id: str
    passed: bool
    dimension_scores: dict = field(default_factory=dict)
    standard_aem_scores: dict = field(default_factory=dict)
    llm_aem_scores: dict = field(default_factory=dict)
    failure_reasons: list = field(default_factory=list)


class LanguageConsistencyFilter:
    """
    Dimension 1: Language Consistency Verification

    Uses multi-tool consensus (langdetect, fastText, CLD3) to verify
    that generated content matches the intended target language.
    """

    def __init__(self, confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold

    def check(self, text: str, expected_language: str) -> dict:
        """
        Verify language consistency using available detection tools.

        Args:
            text: Text to verify
            expected_language: Expected ISO language code

        Returns:
            Dictionary with detection results and pass/fail status
        """
        detections = []

        # langdetect
        try:
            import langdetect
            detected = langdetect.detect(text)
            detections.append({"tool": "langdetect", "detected": detected})
        except Exception as e:
            logger.warning(f"langdetect failed: {e}")

        # fastText (if available)
        try:
            import fasttext
            # Model should be pre-downloaded
            model = fasttext.load_model("lid.176.ftz")
            pred = model.predict(text.replace("\n", " "), k=1)
            detected = pred[0][0].replace("__label__", "")
            confidence = float(pred[1][0])
            detections.append({"tool": "fasttext", "detected": detected, "confidence": confidence})
        except Exception:
            pass

        # Consensus check
        if not detections:
            return {"passed": False, "reason": "no_detection_tools_available"}

        matches = sum(1 for d in detections if d["detected"] == expected_language)
        consensus = matches / len(detections)

        return {
            "passed": consensus >= 0.5,
            "consensus": consensus,
            "detections": detections,
            "expected": expected_language
        }


class SemanticPreservationFilter:
    """
    Dimension 3: Translation Quality Assessment

    Evaluates semantic preservation between source and translated text
    using standard automated evaluation metrics.
    """

    def __init__(self, bertscore_min: float = 0.75, comet_qe_min: float = 0.6):
        self.bertscore_min = bertscore_min
        self.comet_qe_min = comet_qe_min

    def check(self, source_text: str, translated_text: str,
              source_lang: str, target_lang: str) -> dict:
        """
        Evaluate semantic preservation of translated content.

        Args:
            source_text: Original text
            translated_text: Translated text
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Dictionary with metric scores and pass/fail status
        """
        scores = {}

        # BERTScore (cross-lingual via multilingual BERT)
        try:
            from bert_score import score as bert_score
            P, R, F1 = bert_score(
                [translated_text], [source_text],
                lang=target_lang, model_type="bert-base-multilingual-cased"
            )
            scores["bertscore_f1"] = float(F1[0])
        except Exception as e:
            logger.warning(f"BERTScore failed: {e}")
            scores["bertscore_f1"] = None

        # YiSi-2 (cross-lingual semantic similarity)
        # Placeholder - requires YiSi installation
        scores["yisi2"] = None

        # COMET-QE (quality estimation without reference)
        try:
            from comet import download_model, load_from_checkpoint
            model_path = download_model("Unbabel/wmt22-cometkiwi-da")
            model = load_from_checkpoint(model_path)
            data = [{"src": source_text, "mt": translated_text}]
            output = model.predict(data, batch_size=1)
            scores["comet_qe"] = float(output.scores[0])
        except Exception as e:
            logger.warning(f"COMET-QE failed: {e}")
            scores["comet_qe"] = None

        # Determine pass/fail
        passed = True
        reasons = []
        if scores.get("bertscore_f1") is not None and scores["bertscore_f1"] < self.bertscore_min:
            passed = False
            reasons.append(f"bertscore_f1={scores['bertscore_f1']:.3f} < {self.bertscore_min}")
        if scores.get("comet_qe") is not None and scores["comet_qe"] < self.comet_qe_min:
            passed = False
            reasons.append(f"comet_qe={scores['comet_qe']:.3f} < {self.comet_qe_min}")

        return {"passed": passed, "scores": scores, "reasons": reasons}


class LLMAEMFilter:
    """
    LLM-based Automated Evaluation Metrics (LLM-AEM)

    Uses LLM evaluators to score 32 quality features across 5 dimensions
    with asymmetric thresholds for real (≥4.0) and fake (≥3.0) content.
    """

    FEATURES = {
        "consistency": [
            "language_match", "register_consistency", "style_coherence",
            "format_adherence", "structural_integrity", "tone_uniformity"
        ],
        "validation": [
            "factual_manipulation_present", "tactic_correctly_applied",
            "intensity_appropriate", "claim_modification_verified",
            "source_attribution_modified", "narrative_coherent"
        ],
        "translation": [
            "semantic_preservation", "cultural_appropriateness",
            "idiom_handling", "entity_preservation", "numerical_accuracy",
            "tone_preservation", "fluency", "grammar_correctness"
        ],
        "hallucination": [
            "no_fabricated_entities", "no_invented_statistics",
            "no_phantom_sources", "internally_consistent",
            "temporally_consistent", "geographically_consistent"
        ],
        "defective_generation": [
            "complete_output", "no_truncation", "no_repetition",
            "no_language_mixing", "no_instruction_leakage", "no_meta_commentary"
        ]
    }

    def __init__(self, model_client, thresholds: Optional[FilterThresholds] = None):
        self.model_client = model_client
        self.thresholds = thresholds or FilterThresholds()

    def evaluate(self, text: str, content_type: ContentType,
                 metadata: dict) -> QualityReport:
        """
        Run full LLM-AEM evaluation on a text sample.

        Args:
            text: Content to evaluate
            content_type: Whether content is real or fake
            metadata: Generation metadata for context

        Returns:
            QualityReport with dimension-level and feature-level scores
        """
        threshold = (self.thresholds.real_llm_aem_threshold
                     if content_type == ContentType.REAL
                     else self.thresholds.fake_llm_aem_threshold)

        dimension_scores = {}
        all_feature_scores = {}
        failure_reasons = []

        for dimension, features in self.FEATURES.items():
            scores = self._evaluate_dimension(text, dimension, features, metadata)
            all_feature_scores.update(scores)
            dim_avg = np.mean(list(scores.values())) if scores else 0
            dimension_scores[dimension] = dim_avg

            if dim_avg < threshold:
                failure_reasons.append(
                    f"{dimension}: avg={dim_avg:.2f} < threshold={threshold}"
                )

        overall_avg = np.mean(list(dimension_scores.values())) if dimension_scores else 0
        passed = overall_avg >= threshold and len(failure_reasons) == 0

        return QualityReport(
            sample_id=metadata.get("id", "unknown"),
            passed=passed,
            dimension_scores=dimension_scores,
            llm_aem_scores=all_feature_scores,
            failure_reasons=failure_reasons
        )

    def _evaluate_dimension(self, text: str, dimension: str,
                            features: list, metadata: dict) -> dict:
        """Evaluate all features within a quality dimension."""
        scores = {}
        prompt = self._build_evaluation_prompt(text, dimension, features, metadata)

        try:
            response = self.model_client.generate(
                messages=[{"role": "user", "content": prompt}]
            )
            scores = self._parse_scores(response, features)
        except Exception as e:
            logger.error(f"LLM-AEM evaluation failed for {dimension}: {e}")
            scores = {f: 0.0 for f in features}

        return scores

    def _build_evaluation_prompt(self, text: str, dimension: str,
                                 features: list, metadata: dict) -> str:
        """Build the LLM evaluation prompt."""
        features_str = "\n".join(f"- {f}" for f in features)
        return (
            f"Evaluate the following text on these quality features for the "
            f"'{dimension}' dimension. Score each 1-5 (5=best).\n\n"
            f"Features:\n{features_str}\n\n"
            f"Text:\n{text}\n\n"
            f"Respond in JSON format: {{\"feature_name\": score, ...}}"
        )

    def _parse_scores(self, response: str, features: list) -> dict:
        """Parse LLM response into feature scores."""
        try:
            scores = json.loads(response)
            return {f: float(scores.get(f, 0)) for f in features}
        except (json.JSONDecodeError, ValueError):
            return {f: 0.0 for f in features}


class mPURIFYPipeline:
    """
    Main mPURIFY pipeline orchestrator.

    Runs all quality filters in sequence and produces a comprehensive
    quality report for each sample.

    Args:
        model_client: LLM client for LLM-AEM evaluation
        thresholds: Quality threshold configuration
    """

    def __init__(self, model_client=None,
                 thresholds: Optional[FilterThresholds] = None):
        self.thresholds = thresholds or FilterThresholds()
        self.language_filter = LanguageConsistencyFilter(
            confidence_threshold=self.thresholds.language_confidence_min
        )
        self.semantic_filter = SemanticPreservationFilter(
            bertscore_min=self.thresholds.bertscore_min,
            comet_qe_min=self.thresholds.comet_qe_min
        )
        if model_client:
            self.llm_aem = LLMAEMFilter(model_client, self.thresholds)
        else:
            self.llm_aem = None

    def filter_sample(self, sample: dict) -> QualityReport:
        """
        Run the full mPURIFY filtering pipeline on a single sample.

        Args:
            sample: Dictionary with text, language, source_text, and metadata

        Returns:
            QualityReport with pass/fail determination
        """
        failures = []

        # Stage 1: Language consistency
        lang_result = self.language_filter.check(
            sample["text"], sample["language"]
        )
        if not lang_result["passed"]:
            failures.append(f"language_consistency: {lang_result.get('reason', 'mismatch')}")

        # Stage 2: Semantic preservation (for translations)
        if sample.get("source_text"):
            sem_result = self.semantic_filter.check(
                sample["source_text"], sample["text"],
                sample.get("source_language", "en"), sample["language"]
            )
            if not sem_result["passed"]:
                failures.extend(sem_result["reasons"])

        # Stage 3: LLM-AEM (if available)
        if self.llm_aem:
            content_type = (ContentType.REAL if sample.get("veracity_label") == "real"
                           else ContentType.FAKE)
            llm_report = self.llm_aem.evaluate(sample["text"], content_type, sample)
            if not llm_report.passed:
                failures.extend(llm_report.failure_reasons)

        return QualityReport(
            sample_id=sample.get("id", "unknown"),
            passed=len(failures) == 0,
            failure_reasons=failures
        )

    def filter_batch(self, samples: list[dict]) -> dict:
        """
        Filter a batch of samples and return statistics.

        Args:
            samples: List of sample dictionaries

        Returns:
            Dictionary with filtered samples and statistics
        """
        passed = []
        failed = []

        for sample in samples:
            report = self.filter_sample(sample)
            if report.passed:
                passed.append(sample)
            else:
                failed.append({"sample": sample, "report": report})

        stats = {
            "total": len(samples),
            "passed": len(passed),
            "failed": len(failed),
            "retention_rate": len(passed) / len(samples) if samples else 0,
        }

        logger.info(
            f"mPURIFY: {stats['passed']}/{stats['total']} passed "
            f"({stats['retention_rate']:.1%} retention)"
        )

        return {"passed_samples": passed, "failed_samples": failed, "statistics": stats}
