# Datasheet for BLUFF

*Following the framework proposed by [Gebru et al. (2021)](https://arxiv.org/abs/1803.09010)*

---

## Motivation

**For what purpose was the dataset created?**
BLUFF was created to address the critical gap in multilingual fake news detection benchmarks. Existing resources are overwhelmingly English-centric or cover only a handful of high-resource languages, leaving low-resource linguistic communities — where disinformation often causes the greatest harm — without robust evaluation tools. BLUFF provides a standardized benchmark spanning 78 languages to enable equitable research in multilingual disinformation detection.

**Who created the dataset and on behalf of which entity?**
Jason Lucas and Dongwon Lee at the PIKE Research Lab, Penn State University, College of Information Sciences and Technology.

**Who funded the creation of the dataset?**
This work was supported in part by the Penn State College of Information Sciences and Technology.

---

## Composition

**What do the instances that comprise the dataset represent?**
Each instance is a news article or article segment labeled with veracity (real/fake), authorship type (human-written, machine-generated, machine-translated, or human-AI hybrid), language metadata, and content generation metadata.

**How many instances are there in total?**
Over 201,000 samples: approximately 122,836 human-written and 78,443 LLM-generated.

**Does the dataset contain all possible instances or is it a sample?**
It is a curated sample. Human-written content is sourced from publicly available fact-checking organizations. LLM-generated content is produced using the AXL-CoI framework with controlled generation parameters.

**What data does each instance consist of?**
- `text`: The article content (string)
- `language`: ISO 639-1/3 language code
- `language_family`: Linguistic family classification
- `script`: Writing system
- `syntax`: Word order typology (SVO, SOV, VSO, VOS)
- `resource_category`: "big-head" or "long-tail"
- `veracity_label`: "real" or "fake"
- `authorship_type`: HWT, MGT, MTT, or HAT
- `generation_model`: Model used for generation (if applicable)
- `manipulation_tactic`: Specific tactic applied (if applicable)
- `edit_intensity`: low, medium, or high (if applicable)
- `source`: Original fact-checking organization
- `split`: train, dev, or test

**Is any information missing from individual instances?**
Some fields (e.g., `generation_model`, `manipulation_tactic`, `edit_intensity`) are only applicable to LLM-generated content and are null for human-written instances.

**Are there any errors, sources of noise, or redundancies?**
The mPURIFY pipeline addresses quality issues including language mismatches, semantic drift, and near-duplicates. Some residual noise may exist, particularly in low-resource languages where language identification tools have lower accuracy. We document known limitations in the paper.

**Is the dataset self-contained?**
Yes. All text content is included directly in the dataset files. No external data retrieval is required for evaluation.

**Does the dataset contain data that might be considered confidential?**
No. All human-written content is sourced from publicly published fact-checking articles. No personally identifiable information (PII), user data, or private communications are included.

**Does the dataset contain data that might be considered offensive or harmful?**
The dataset contains disinformation content by design — both real-world fact-checked false claims and synthetically generated fake news. This content is provided strictly for research purposes to improve detection capabilities. All synthetic content is watermarked as research material.

---

## Collection Process

**How was the data associated with each instance acquired?**
- *Human-Written Text (HWT):* Collected from IFCN-certified fact-checking organizations and CredCatalog-indexed sources via web scraping and API access.
- *LLM-Generated Content (MGT/MTT/HAT):* Produced using the AXL-CoI framework with 19 multilingual LLMs, applying 39 textual modification techniques.

**What mechanisms or procedures were used to collect the data?**
- Automated web scrapers for fact-checking articles
- Language identification using multiple tools (langdetect, fastText, CLD3)
- AXL-CoI agentic pipeline for controlled content generation
- mPURIFY quality filtering pipeline
- Human spot-check validation

**If the dataset relates to people, were they informed and did they consent?**
The dataset does not contain data about private individuals. All source content is from professionally published fact-checking articles intended for public consumption.

**Was any preprocessing/cleaning/labeling applied?**
Yes:
1. Deduplication (exact and near-duplicate removal)
2. Language verification (multi-tool consensus)
3. Quality filtering via mPURIFY (semantic preservation, factual manipulation validation)
4. Standardized formatting and metadata alignment
5. Train/dev/test split stratification by language and label

---

## Uses

**What tasks has the dataset been used for?**
Four benchmark tasks: binary veracity classification, multi-class veracity classification, binary authorship detection, and multi-class authorship attribution. Cross-lingual transfer experiments across language families, scripts, and syntactic typologies.

**Is there anything about the composition or collection that might impact future uses?**
- Geographic concentration: HWT data skews toward regions with established fact-checking infrastructure
- Topical bias: Fact-checked content tends toward politically salient topics
- Linguistic quality: Big-head languages likely have higher generation quality due to LLM training data distributions

**Are there tasks for which the dataset should not be used?**
The dataset should NOT be used to generate or distribute disinformation, train models intended to produce misleading content, or any purpose that undermines information integrity.

---

## Distribution

**How will the dataset be distributed?**
- Primary: HuggingFace Datasets (`jsl5710/BLUFF`)
- Code & Documentation: GitHub (`jsl5710/BLUFF`)
- Baseline models: HuggingFace Model Hub

**When will the dataset be released?**
Upon publication at KDD 2026.

**Will the dataset be distributed under a copyright or IP license?**
CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International). Code is released under the MIT License.

---

## Maintenance

**Who will maintain the dataset?**
Jason Lucas and the PIKE Research Lab at Penn State University.

**How can the dataset be updated?**
Community contributions of additional language data are welcome via GitHub pull requests. Updates will be versioned and documented in the repository changelog.

**Will older versions be supported?**
Yes. All versions will remain accessible via HuggingFace with version tags.

**How will the maintainers be contacted?**
Via GitHub issues or email (jsl5710@psu.edu).
