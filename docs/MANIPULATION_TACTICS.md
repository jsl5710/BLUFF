# BLUFF Textual Modification Techniques

This document describes the 39 textual modification techniques used in BLUFF's AXL-CoI framework: 36 manipulation tactics for fake news generation and 3 AI-editing strategies for real news augmentation.

---

## Overview

BLUFF employs a diverse set of content manipulation and editing strategies to create realistic, challenging test samples. These techniques are organized into categories reflecting real-world disinformation tactics documented by media literacy organizations and academic research.

---

## Fake News Manipulation Tactics (36)

### Category 1: Factual Distortion (8 tactics)

| # | Tactic | Description |
|---|--------|-------------|
| 1 | **Fact Inversion** | Reverses key factual claims (e.g., "increased" → "decreased") |
| 2 | **Numerical Manipulation** | Alters statistics, dates, or quantities to mislead |
| 3 | **Source Fabrication** | Attributes claims to non-existent or unrelated sources |
| 4 | **Causal Reversal** | Swaps cause-and-effect relationships |
| 5 | **Selective Omission** | Removes critical context that changes interpretation |
| 6 | **Detail Substitution** | Replaces specific facts (names, locations, organizations) |
| 7 | **Timeline Distortion** | Manipulates chronological ordering of events |
| 8 | **Scope Manipulation** | Generalizes limited findings or narrows broad trends |

### Category 2: Emotional & Rhetorical Manipulation (8 tactics)

| # | Tactic | Description |
|---|--------|-------------|
| 9 | **Emotional Amplification** | Heightens emotional language to provoke reactions |
| 10 | **Fear Inducement** | Introduces threat-based framing to create anxiety |
| 11 | **Outrage Fabrication** | Constructs narratives designed to generate anger |
| 12 | **Sensationalization** | Exaggerates claims with hyperbolic language |
| 13 | **Appeal to Authority** | Invokes false or misleading expert endorsements |
| 14 | **Victimhood Framing** | Repositions actors as victims to gain sympathy |
| 15 | **Us-vs-Them Polarization** | Creates artificial group divisions |
| 16 | **Moral Panic Construction** | Frames ordinary events as existential threats |

### Category 3: Narrative Restructuring (8 tactics)

| # | Tactic | Description |
|---|--------|-------------|
| 17 | **Context Transplanting** | Places real quotes/events in misleading contexts |
| 18 | **Narrative Grafting** | Combines elements from unrelated stories |
| 19 | **Headline-Body Disconnect** | Creates misleading headlines that contradict body text |
| 20 | **Cherry-Picking** | Selects only supporting evidence, ignoring contradictions |
| 21 | **Straw Man Construction** | Misrepresents opposing positions for easy rebuttal |
| 22 | **False Equivalence** | Presents unequal positions as equally valid |
| 23 | **Conspiracy Framing** | Introduces unfounded conspiratorial explanations |
| 24 | **Whataboutism** | Deflects by pointing to unrelated issues |

### Category 4: Technical Manipulation (6 tactics)

| # | Tactic | Description |
|---|--------|-------------|
| 25 | **Scientific Misinterpretation** | Distorts research findings or methodology |
| 26 | **Statistical Deception** | Uses misleading visualizations or cherry-picked data |
| 27 | **Jargon Obfuscation** | Uses technical language to obscure weak arguments |
| 28 | **Correlation-Causation Conflation** | Presents correlations as causal relationships |
| 29 | **Sample Size Exploitation** | Draws broad conclusions from limited data |
| 30 | **Methodology Hiding** | Omits methodological limitations |

### Category 5: Identity & Attribution Manipulation (6 tactics)

| # | Tactic | Description |
|---|--------|-------------|
| 31 | **Impersonation** | Mimics writing style of trusted sources |
| 32 | **False Attribution** | Assigns statements to wrong individuals |
| 33 | **Satire-as-News** | Presents satirical content as factual reporting |
| 34 | **Astroturfing Simulation** | Creates appearance of grassroots support |
| 35 | **Credential Inflation** | Exaggerates qualifications of cited sources |
| 36 | **Anonymous Sourcing Abuse** | Uses vague anonymous sources to make unverifiable claims |

---

## Real News AI-Editing Strategies (3)

These strategies apply controlled AI editing to real (human-written, truthful) content to create the HAT (Human-AI Hybrid Text) authorship category while preserving factual accuracy.

| # | Strategy | Description |
|---|----------|-------------|
| 37 | **Stylistic Enhancement** | AI improves clarity, grammar, and readability while preserving all facts |
| 38 | **Structural Reorganization** | AI restructures article flow (paragraph reordering, section consolidation) without altering content |
| 39 | **Summarization & Expansion** | AI condenses verbose sections or expands terse passages while maintaining factual integrity |

---

## Edit Intensity Levels

Each tactic is applied at one of three intensity levels:

| Level | Description | Characteristics |
|-------|-------------|-----------------|
| **Low** | Subtle modifications | Minor changes that are difficult to detect; preserve most of the original text |
| **Medium** | Moderate alterations | Noticeable changes that require careful reading to identify |
| **High** | Substantial rewriting | Major content changes that significantly alter the narrative |

---

## Tactic Distribution in BLUFF

Tactics are distributed across the dataset to ensure diversity. Each LLM-generated sample is tagged with its primary manipulation tactic and edit intensity level, enabling fine-grained analysis of detection model vulnerabilities.

The AXL-CoI framework's Manipulator agent selects tactics based on the source content characteristics, ensuring realistic application. The Auditor and Validator agents verify that the selected tactic was effectively applied while maintaining linguistic naturalness.

---

## References

The manipulation taxonomy draws on established frameworks including:

- First Draft's [Information Disorder Framework](https://firstdraftnews.org/)
- UNESCO's [Journalism, Fake News & Disinformation Handbook](https://en.unesco.org/fightfakenews)
- Wardle & Derakhshan (2017): Information Disorder: Toward an interdisciplinary framework for research and policymaking
- Zellers et al. (2019): Defending Against Neural Fake News
- Zhou & Zafarani (2020): A Survey of Fake News: Fundamental Theories, Detection Methods, and Opportunities
