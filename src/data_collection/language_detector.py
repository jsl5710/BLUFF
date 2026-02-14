"""
Multi-tool language identification for BLUFF benchmark.

Uses consensus across multiple detection tools (langdetect, fastText, CLD3)
to reliably identify the language of text content, with special attention
to low-resource languages where individual tools may be unreliable.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result from a single language detection tool."""
    tool: str
    language: str
    confidence: float
    is_reliable: bool = True


@dataclass
class ConsensusResult:
    """Consensus result from multiple detection tools."""
    detected_language: str
    consensus_score: float
    individual_results: list
    is_reliable: bool


class MultiToolLanguageDetector:
    """
    Language detection using multi-tool consensus.

    Combines results from langdetect, fastText lid.176.ftz, and
    Google CLD3 to achieve reliable detection especially for
    low-resource languages.

    Args:
        min_consensus: Minimum agreement ratio for reliable detection (default: 0.5)
        min_confidence: Minimum confidence score per tool (default: 0.7)
    """

    def __init__(self, min_consensus: float = 0.5, min_confidence: float = 0.7):
        self.min_consensus = min_consensus
        self.min_confidence = min_confidence
        self._init_tools()

    def _init_tools(self):
        """Initialize available detection tools."""
        self.available_tools = []

        try:
            import langdetect
            self.available_tools.append("langdetect")
        except ImportError:
            logger.warning("langdetect not available")

        try:
            import fasttext
            self.available_tools.append("fasttext")
        except ImportError:
            logger.warning("fasttext not available")

        try:
            import gcld3
            self.available_tools.append("cld3")
        except ImportError:
            logger.warning("gcld3 not available")

        logger.info(f"Language detection tools: {self.available_tools}")

    def detect(self, text: str) -> ConsensusResult:
        """
        Detect language using all available tools and return consensus.

        Args:
            text: Text to detect language for

        Returns:
            ConsensusResult with detected language and reliability info
        """
        results = []

        if "langdetect" in self.available_tools:
            results.append(self._detect_langdetect(text))
        if "fasttext" in self.available_tools:
            results.append(self._detect_fasttext(text))
        if "cld3" in self.available_tools:
            results.append(self._detect_cld3(text))

        return self._compute_consensus(results)

    def _detect_langdetect(self, text: str) -> DetectionResult:
        """Run langdetect."""
        try:
            import langdetect
            lang = langdetect.detect(text)
            return DetectionResult(tool="langdetect", language=lang, confidence=0.8)
        except Exception as e:
            return DetectionResult(tool="langdetect", language="unk",
                                confidence=0.0, is_reliable=False)

    def _detect_fasttext(self, text: str) -> DetectionResult:
        """Run fastText language identification."""
        try:
            import fasttext
            model = fasttext.load_model("lid.176.ftz")
            pred = model.predict(text.replace("\n", " "), k=1)
            lang = pred[0][0].replace("__label__", "")
            conf = float(pred[1][0])
            return DetectionResult(tool="fasttext", language=lang, confidence=conf)
        except Exception:
            return DetectionResult(tool="fasttext", language="unk",
                                confidence=0.0, is_reliable=False)

    def _detect_cld3(self, text: str) -> DetectionResult:
        """Run Google CLD3."""
        try:
            import gcld3
            detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)
            result = detector.FindLanguage(text=text)
            return DetectionResult(
                tool="cld3", language=result.language,
                confidence=result.probability, is_reliable=result.is_reliable
            )
        except Exception:
            return DetectionResult(tool="cld3", language="unk",
                                confidence=0.0, is_reliable=False)

    def _compute_consensus(self, results: list[DetectionResult]) -> ConsensusResult:
        """Compute consensus across detection results."""
        reliable = [r for r in results if r.is_reliable and r.confidence >= self.min_confidence]

        if not reliable:
            # Fall back to all results
            reliable = [r for r in results if r.is_reliable]

        if not reliable:
            return ConsensusResult(
                detected_language="unk", consensus_score=0.0,
                individual_results=results, is_reliable=False
            )

        # Majority vote
        from collections import Counter
        votes = Counter(r.language for r in reliable)
        best_lang, best_count = votes.most_common(1)[0]
        consensus = best_count / len(reliable)

        return ConsensusResult(
            detected_language=best_lang,
            consensus_score=consensus,
            individual_results=results,
            is_reliable=consensus >= self.min_consensus
        )
