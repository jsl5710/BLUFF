"""
AXL-CoI: Adversarial Cross-Lingual Agentic Chain-of-Interactions

This module implements the core generation framework for the BLUFF benchmark.
AXL-CoI orchestrates multi-agent chains to produce high-quality multilingual
fake and real news content through controlled adversarial generation.

Architecture:
    Fake News Pipeline (10 chains):
        Analyst → Manipulator → Auditor → Editor → Validator →
        Adjuster → Translator → Localization QA → Evaluator → Formatter

    Real News Pipeline (8 chains):
        Analyst → Dynamic Editor → Auditor → Editor → Validator →
        Translator → Localization QA → Formatter

Reference:
    Lucas & Lee (2026). "BLUFF: A Benchmark for Linguistic Understanding
    of Fake-news Forensics." KDD 2026.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


class PipelineType(Enum):
    """Pipeline types for content generation."""
    FAKE_NEWS = "fake_news"
    REAL_NEWS = "real_news"


class EditIntensity(Enum):
    """Edit intensity levels for content manipulation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ChainInput:
    """Input to a chain agent."""
    source_text: str
    source_language: str
    target_language: str
    pipeline_type: PipelineType
    manipulation_tactic: Optional[str] = None
    edit_intensity: EditIntensity = EditIntensity.MEDIUM
    metadata: dict = field(default_factory=dict)


@dataclass
class ChainOutput:
    """Output from a chain agent."""
    text: str
    agent_name: str
    status: str  # "pass", "fail", "revision_needed"
    scores: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class AgentBase:
    """Base class for all chain agents in the AXL-CoI framework."""

    def __init__(self, name: str, model_client, system_prompt: str = ""):
        self.name = name
        self.model_client = model_client
        self.system_prompt = system_prompt

    def execute(self, chain_input: ChainInput, previous_outputs: list[ChainOutput]) -> ChainOutput:
        """Execute the agent's task. Override in subclasses."""
        raise NotImplementedError

    def _call_model(self, messages: list[dict]) -> str:
        """Call the underlying LLM with the given messages."""
        response = self.model_client.generate(
            messages=[{"role": "system", "content": self.system_prompt}] + messages
        )
        return response


class AnalystAgent(AgentBase):
    """
    Chain 1: Analyst Agent
    Analyzes source content to identify key claims, entities, and structure
    for downstream manipulation or editing.
    """

    def __init__(self, model_client):
        super().__init__(
            name="Analyst",
            model_client=model_client,
            system_prompt=(
                "You are a content analyst. Identify the key claims, entities, "
                "statistics, causal relationships, and narrative structure in the "
                "given text. Output a structured analysis."
            )
        )

    def execute(self, chain_input: ChainInput, previous_outputs: list[ChainOutput]) -> ChainOutput:
        messages = [{"role": "user", "content": (
            f"Analyze the following article for key claims, entities, and structure:\n\n"
            f"{chain_input.source_text}"
        )}]
        result = self._call_model(messages)
        return ChainOutput(text=result, agent_name=self.name, status="pass")


class ManipulatorAgent(AgentBase):
    """
    Chain 2 (Fake News): Manipulator Agent
    Applies the specified manipulation tactic to the analyzed content.
    Supports 36 distinct manipulation tactics across 5 categories.
    """

    TACTICS = {
        "factual_distortion": [
            "fact_inversion", "numerical_manipulation", "source_fabrication",
            "causal_reversal", "selective_omission", "detail_substitution",
            "timeline_distortion", "scope_manipulation"
        ],
        "emotional_rhetorical": [
            "emotional_amplification", "fear_inducement", "outrage_fabrication",
            "sensationalization", "appeal_to_authority", "victimhood_framing",
            "us_vs_them_polarization", "moral_panic_construction"
        ],
        "narrative_restructuring": [
            "context_transplanting", "narrative_grafting", "headline_body_disconnect",
            "cherry_picking", "straw_man_construction", "false_equivalence",
            "conspiracy_framing", "whataboutism"
        ],
        "technical_manipulation": [
            "scientific_misinterpretation", "statistical_deception",
            "jargon_obfuscation", "correlation_causation_conflation",
            "sample_size_exploitation", "methodology_hiding"
        ],
        "identity_attribution": [
            "impersonation", "false_attribution", "satire_as_news",
            "astroturfing_simulation", "credential_inflation", "anonymous_sourcing_abuse"
        ]
    }

    def __init__(self, model_client, tactic_prompts_dir: str = "prompts/tactics"):
        super().__init__(
            name="Manipulator",
            model_client=model_client,
            system_prompt="You are a content manipulation specialist for research purposes."
        )
        self.tactic_prompts_dir = Path(tactic_prompts_dir)

    def execute(self, chain_input: ChainInput, previous_outputs: list[ChainOutput]) -> ChainOutput:
        analysis = previous_outputs[-1].text if previous_outputs else ""
        tactic = chain_input.manipulation_tactic
        intensity = chain_input.edit_intensity.value

        prompt = self._build_tactic_prompt(tactic, intensity, chain_input.source_text, analysis)
        messages = [{"role": "user", "content": prompt}]
        result = self._call_model(messages)

        return ChainOutput(
            text=result, agent_name=self.name, status="pass",
            metadata={"tactic": tactic, "intensity": intensity}
        )

    def _build_tactic_prompt(self, tactic: str, intensity: str, text: str, analysis: str) -> str:
        """Build the manipulation prompt for the given tactic and intensity."""
        prompt_file = self.tactic_prompts_dir / f"{tactic}.txt"
        if prompt_file.exists():
            template = prompt_file.read_text()
            return template.format(text=text, analysis=analysis, intensity=intensity)
        return (
            f"Apply the '{tactic}' manipulation tactic at '{intensity}' intensity "
            f"to the following article based on this analysis:\n\n"
            f"Analysis: {analysis}\n\nArticle: {text}"
        )


class DynamicEditorAgent(AgentBase):
    """
    Chain 2 (Real News): Dynamic Editor Agent
    Applies controlled AI editing strategies to real news content
    to create HAT (Human-AI Hybrid Text) while preserving factual accuracy.
    """

    STRATEGIES = ["stylistic_enhancement", "structural_reorganization", "summarization_expansion"]

    def __init__(self, model_client):
        super().__init__(
            name="DynamicEditor",
            model_client=model_client,
            system_prompt=(
                "You are an AI editor. Apply the specified editing strategy while "
                "preserving ALL factual claims. Do not add, remove, or alter any facts."
            )
        )

    def execute(self, chain_input: ChainInput, previous_outputs: list[ChainOutput]) -> ChainOutput:
        strategy = chain_input.manipulation_tactic or random.choice(self.STRATEGIES)
        messages = [{"role": "user", "content": (
            f"Apply '{strategy}' editing to this article. Preserve all facts:\n\n"
            f"{chain_input.source_text}"
        )}]
        result = self._call_model(messages)
        return ChainOutput(
            text=result, agent_name=self.name, status="pass",
            metadata={"strategy": strategy}
        )


class AuditorAgent(AgentBase):
    """
    Chain 3: Auditor Agent
    Verifies that the manipulation/editing was applied correctly and
    meets quality standards for the specified intensity level.
    """

    def __init__(self, model_client):
        super().__init__(
            name="Auditor",
            model_client=model_client,
            system_prompt="You are a quality auditor. Verify content modifications meet specifications."
        )

    def execute(self, chain_input: ChainInput, previous_outputs: list[ChainOutput]) -> ChainOutput:
        original = chain_input.source_text
        modified = previous_outputs[-1].text if previous_outputs else ""
        messages = [{"role": "user", "content": (
            f"Compare the original and modified text. Verify the modification is "
            f"appropriate for the specified task.\n\n"
            f"Original:\n{original}\n\nModified:\n{modified}"
        )}]
        result = self._call_model(messages)
        status = "pass" if "APPROVED" in result.upper() else "revision_needed"
        return ChainOutput(text=result, agent_name=self.name, status=status)


class TranslatorAgent(AgentBase):
    """
    Chain 7 (Fake) / Chain 6 (Real): Translator Agent
    Performs bidirectional translation: Eng→X (70 languages) and X→Eng (50 languages).
    Uses specialized prompts for linguistic accuracy.
    """

    def __init__(self, model_client):
        super().__init__(
            name="Translator",
            model_client=model_client,
            system_prompt=(
                "You are an expert multilingual translator. Translate accurately while "
                "preserving meaning, tone, and cultural context. Maintain all factual claims."
            )
        )

    def execute(self, chain_input: ChainInput, previous_outputs: list[ChainOutput]) -> ChainOutput:
        text_to_translate = previous_outputs[-1].text if previous_outputs else chain_input.source_text
        messages = [{"role": "user", "content": (
            f"Translate the following text from {chain_input.source_language} to "
            f"{chain_input.target_language}. Preserve meaning and tone:\n\n{text_to_translate}"
        )}]
        result = self._call_model(messages)
        return ChainOutput(
            text=result, agent_name=self.name, status="pass",
            metadata={"source_lang": chain_input.source_language,
                      "target_lang": chain_input.target_language}
        )


class LocalizationQAAgent(AgentBase):
    """
    Chain 8 (Fake) / Chain 7 (Real): Localization QA Agent
    Verifies translation quality including semantic preservation,
    cultural appropriateness, and linguistic naturalness.
    """

    def __init__(self, model_client):
        super().__init__(
            name="LocalizationQA",
            model_client=model_client,
            system_prompt="You are a localization quality assurance specialist."
        )

    def execute(self, chain_input: ChainInput, previous_outputs: list[ChainOutput]) -> ChainOutput:
        translated = previous_outputs[-1].text if previous_outputs else ""
        messages = [{"role": "user", "content": (
            f"Evaluate translation quality for {chain_input.target_language}. "
            f"Check: semantic preservation, cultural appropriateness, fluency, "
            f"grammar.\n\nTranslation:\n{translated}"
        )}]
        result = self._call_model(messages)
        status = "pass" if "PASS" in result.upper() else "revision_needed"
        return ChainOutput(text=result, agent_name=self.name, status=status)


class AXLCoIPipeline:
    """
    Main AXL-CoI pipeline orchestrator.

    Manages the sequential execution of chain agents for both fake news
    and real news content generation pipelines.

    Args:
        model_client: LLM API client for agent interactions
        config_path: Path to pipeline configuration YAML
        max_retries: Maximum revision cycles per agent (default: 3)
    """

    def __init__(self, model_client, config_path: str = "configs/pipeline.yaml",
                 max_retries: int = 3):
        self.model_client = model_client
        self.max_retries = max_retries
        self.config = self._load_config(config_path)
        self._initialize_agents()

    def _load_config(self, path: str) -> dict:
        """Load pipeline configuration."""
        config_path = Path(path)
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}

    def _initialize_agents(self):
        """Initialize all chain agents."""
        self.analyst = AnalystAgent(self.model_client)
        self.manipulator = ManipulatorAgent(self.model_client)
        self.dynamic_editor = DynamicEditorAgent(self.model_client)
        self.auditor = AuditorAgent(self.model_client)
        self.translator = TranslatorAgent(self.model_client)
        self.localization_qa = LocalizationQAAgent(self.model_client)

    def generate(self, chain_input: ChainInput) -> dict:
        """
        Execute the full generation pipeline.

        Args:
            chain_input: Input specification for content generation

        Returns:
            Dictionary containing generated text, metadata, and quality scores
        """
        outputs = []

        # Chain 1: Analysis
        logger.info(f"Executing Analyst for {chain_input.target_language}")
        analysis = self.analyst.execute(chain_input, outputs)
        outputs.append(analysis)

        # Chain 2: Manipulation or Editing
        if chain_input.pipeline_type == PipelineType.FAKE_NEWS:
            logger.info(f"Executing Manipulator: {chain_input.manipulation_tactic}")
            content = self._execute_with_retry(self.manipulator, chain_input, outputs)
        else:
            logger.info("Executing DynamicEditor")
            content = self._execute_with_retry(self.dynamic_editor, chain_input, outputs)
        outputs.append(content)

        # Chain 3: Audit
        audit = self.auditor.execute(chain_input, outputs)
        outputs.append(audit)

        # Chain 6/7: Translation (if target != source)
        if chain_input.target_language != chain_input.source_language:
            translation = self._execute_with_retry(self.translator, chain_input, outputs)
            outputs.append(translation)

            # Chain 7/8: Localization QA
            qa = self.localization_qa.execute(chain_input, outputs)
            outputs.append(qa)

        return self._compile_output(chain_input, outputs)

    def _execute_with_retry(self, agent: AgentBase, chain_input: ChainInput,
                            outputs: list[ChainOutput]) -> ChainOutput:
        """Execute an agent with retry logic for revision cycles."""
        for attempt in range(self.max_retries):
            result = agent.execute(chain_input, outputs)
            if result.status == "pass":
                return result
            logger.warning(f"{agent.name} revision needed (attempt {attempt + 1}/{self.max_retries})")
        logger.error(f"{agent.name} failed after {self.max_retries} attempts")
        return result

    def _compile_output(self, chain_input: ChainInput, outputs: list[ChainOutput]) -> dict:
        """Compile all chain outputs into the final result."""
        final_text = outputs[-1].text
        return {
            "text": final_text,
            "source_language": chain_input.source_language,
            "target_language": chain_input.target_language,
            "pipeline_type": chain_input.pipeline_type.value,
            "manipulation_tactic": chain_input.manipulation_tactic,
            "edit_intensity": chain_input.edit_intensity.value,
            "chain_trace": [
                {"agent": o.agent_name, "status": o.status, "scores": o.scores}
                for o in outputs
            ],
            "num_chains": len(outputs),
        }

    def generate_batch(self, inputs: list[ChainInput], num_workers: int = 1) -> list[dict]:
        """
        Generate content for a batch of inputs.

        Args:
            inputs: List of ChainInput specifications
            num_workers: Number of parallel workers (default: 1)

        Returns:
            List of generation results
        """
        results = []
        for i, inp in enumerate(inputs):
            logger.info(f"Processing {i + 1}/{len(inputs)}: {inp.target_language}")
            result = self.generate(inp)
            results.append(result)
        return results
