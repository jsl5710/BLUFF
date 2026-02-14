# Dataset Card for BLUFF

## Dataset Description

- **Homepage:** [https://github.com/jsl5710/BLUFF](https://github.com/jsl5710/BLUFF)
- **Repository:** [https://github.com/jsl5710/BLUFF](https://github.com/jsl5710/BLUFF)
- **HuggingFace:** [https://huggingface.co/datasets/jsl5710/BLUFF](https://huggingface.co/datasets/jsl5710/BLUFF)
- **Paper:** BLUFF: A Benchmark for Linguistic Understanding of Fake-news Forensics (under review)
- **Point of Contact:** Jason Lucas (jsl5710@psu.edu)

### Dataset Summary

BLUFF is a multilingual fake news detection benchmark covering 78 languages with over 201K samples. It combines human-written fact-checked content (122K samples across 57 languages) and LLM-generated content (78K samples across 71 languages using 19 diverse mLLMs) to provide the first comprehensive testbed for disinformation detection beyond high-resource settings.

### Supported Tasks

| Task | Description | Classes | Metric |
|------|-------------|---------|--------|
| **Task 1** | Binary Veracity Classification | Real / Fake | F1 (macro) |
| **Task 2** | Multi-class Veracity Classification | Real / Fake × Source Type | F1 (macro) |
| **Task 3** | Binary Authorship Detection | Human / Machine | F1 (macro) |
| **Task 4** | Multi-class Authorship Attribution | HWT / MGT / MTT / HAT | F1 (macro) |

### Languages

78 languages across 12 families:

**Big-Head (20):** ar, bn, de, en, es, fa, fr, hi, id, it, ja, ko, nl, pl, pt, ru, sv, tr, uk, zh

**Long-Tail (58):** af, am, az, bg, ca, ceb, cs, cy, da, el, et, eu, fi, ga, gl, gu, ha, he, hr, hu, hy, is, ka, kk, km, kn, ku, ky, lo, lt, lv, mk, ml, mr, ms, mt, my, ne, no, pa, ps, ro, si, sk, sl, sq, sr, sw, ta, te, tg, th, tl, ur, uz, vi, xh, yo

## Dataset Structure

### Data Organization

The dataset is hosted on HuggingFace at [`jsl5710/BLUFF`](https://huggingface.co/datasets/jsl5710/BLUFF) with the following structure:

```
data/
├── meta_data/                     # Sample metadata
│   ├── metadata_human_written.csv   (122K rows, 33 columns)
│   └── metadata_ai_generated.csv    (78K rows, 29 columns)
├── processed/                     # Cleaned text data
│   └── generated_data/
│       ├── ai_generated/            Per-model, per-language: {model}/{lang}/data.csv
│       └── human_written/           Per-organization, per-language: {org}/{lang}/data.csv
├── raw/                           # Original source data
│   └── source_data/
│       ├── human/                   Raw fact-check articles (CSV per source)
│       ├── sd_eng_x_f/              English→X fake news source data
│       ├── sd_eng_x_r/              English→X real news source data
│       ├── sd_x_eng_f/              X→English fake news source data
│       └── sd_x_eng_r/              X→English real news source data
└── splits/                        # Evaluation split definitions
    └── evaluation/
        ├── multilingual/            train.json, val.json, stats.json
        ├── cross_lingual_bighead_longtail/
        ├── cross_lingual_family/{Family}/
        ├── cross_lingual_script/{Script}/
        ├── cross_lingual_syntax/{Order}/
        ├── external_evaluation/
        └── small_test_50/           Balanced subsets (50 per class per lang)
```

### Data Fields — Processed Data

| Field | Type | Description |
|-------|------|-------------|
| `uuid` | string | Unique sample identifier |
| `article_content` | string | Full article text in the original language |
| `translated_content` | string | English translation of the article |
| `post_content` | string | Social media post version in the original language |
| `translated_post` | string | English translation of the post |
| `language` | string | ISO 639-3 language code |
| `translation_directionality` | string | `eng_x` or `x_eng` |
| `model` | string | Generating model name |
| `veracity` | string | `fake_news` or `real_news` |
| `technique_keys` | list | Manipulation technique IDs |
| `degree` | string | Edit intensity: `minor`, `moderate`, or `critical` |
| `source_dataset` | string | Original source dataset name |
| `HAT` | string | Human-AI Hybrid flag (`y`/`n`) |
| `MGT` | string | Machine-Generated flag (`y`/`n`) |
| `MTT` | string | Machine-Translated flag (`y`/`n`) |
| `HWT` | string | Human-Written flag (`y`/`n`) |

### Data Fields — Human-Written Metadata

Key columns in `metadata_human_written.csv`:

| Field | Description |
|-------|-------------|
| `uuid` | Unique identifier (links to splits and processed data) |
| `language` | ISO 639-3 language code |
| `veracity` | Fact-check verdict |
| `organization` | Fact-checking organization name |
| `country` | Country code |
| `category` | Content category |
| `topic` | Topic classification |
| `source_content_type` | `article` or `post` |
| `extraction_path` | Path to processed data file |

### Data Fields — AI-Generated Metadata

Key columns in `metadata_ai_generated.csv`:

| Field | Description |
|-------|-------------|
| `uuid` | Unique identifier |
| `language` | ISO 639-3 language code |
| `language_category` | `head` or `tail` |
| `transform_technique` | Manipulation techniques applied |
| `technique_keys` | Technique ID numbers |
| `degree` | Edit intensity level |
| `veracity` | `fake_news` or `real_news` |
| `mLLM` | Generating model name |
| `mPURIFY` | Quality filtering status |
| `translation_directionality` | Generation direction |

### Data Splits

Split files contain lists of UUIDs (JSON arrays) that map to samples in the metadata and processed data.

| Split | Purpose | Available |
|-------|---------|-----------|
| `train.json` | Model training | Yes |
| `val.json` | Hyperparameter tuning / validation | Yes |
| `test` | Final evaluation | Held out (contact authors) |

**Note:** Test splits are withheld to preserve benchmark integrity.

### Training Settings

| Setting | Split Directory | Description |
|---------|----------------|-------------|
| Multilingual | `multilingual/` | Train on all languages |
| Cross-lingual (Head→Tail) | `cross_lingual_bighead_longtail/` | Train big-head, eval long-tail |
| Cross-lingual (Family) | `cross_lingual_family/{Family}/` | Leave-one-family-out |
| Cross-lingual (Script) | `cross_lingual_script/{Script}/` | Leave-one-script-out |
| Cross-lingual (Syntax) | `cross_lingual_syntax/{Order}/` | Leave-one-syntax-out |
| External | `external_evaluation/` | Held-out external datasets |

## Dataset Creation

### Curation Rationale

Existing fake news detection benchmarks are overwhelmingly English-centric. BLUFF addresses the digital language divide by providing evaluation resources for 58 low-resource languages alongside 20 high-resource languages, enabling research on equitable disinformation detection.

### Source Data

- **Human-Written:** IFCN-certified fact-checking organizations and CredCatalog-indexed publishers
- **LLM-Generated:** AXL-CoI (Adversarial Cross-Lingual Agentic Chain-of-Interactions) framework using 19 multilingual LLMs with 39 textual modification techniques
- **Quality Filtering:** mPURIFY pipeline ensuring language consistency, semantic preservation, and deduplication

### Generation Models (19)

GPT-4.1, o1, Gemini 1.5 Flash/Pro, Gemini 2.0 Flash, Llama 3.3 70B, Llama 4 Maverick/Scout, DeepSeek-R1/R1-Turbo/R1-Distill, Aya Expanse 32B, Qwen3-Next 80B, QwQ-32B, Mistral Large, Phi-4 Multimodal

### Annotations

Veracity labels for human-written content are inherited from professional fact-checkers. Authorship and generation metadata are recorded during the controlled generation process.

### Personal and Sensitive Information

The dataset contains no personally identifiable information. All source content is from publicly published articles.

## Considerations for Using the Data

### Social Impact

BLUFF aims to improve disinformation detection for underserved linguistic communities. The dataset contains realistic synthetic disinformation for research purposes only.

### Known Limitations

- Geographic bias toward regions with fact-checking infrastructure
- Topical skew toward politically salient content
- Variable generation quality across languages due to LLM capabilities
- 9.0–25.3% performance degradation for long-tail languages in cross-lingual settings

### Licensing

- **Dataset:** CC BY-NC-SA 4.0
- **Code:** MIT License
