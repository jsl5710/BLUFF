# BLUFF: Benchmark for Linguistic Understanding of Fake-news Forensics

<p align="center">
  <img src="figures/bluff_overview.png" alt="BLUFF Overview" width="800"/>
</p>

<p align="center">
  <!-- <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg" alt="arXiv"></a> -->
  <a href="https://huggingface.co/datasets/jsl5710/BLUFF"><img src="https://img.shields.io/badge/🤗_HuggingFace-Dataset-yellow.svg" alt="HuggingFace"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+"></a>
</p>

---

**BLUFF** is a comprehensive multilingual benchmark for fake news detection spanning **78 languages** with over **201K samples**. It uniquely covers both high-resource "big-head" (20) and low-resource "long-tail" (58) languages, addressing critical gaps in multilingual disinformation research.

> **Paper:** *BLUFF: A Benchmark for Linguistic Understanding of Fake-news Forensics*  
> **Authors:** Jason Lucas, Dongwon Lee  
> **Venue:** KDD 2026 — Datasets and Benchmarks Track  

---

## 🔑 Key Features

- **78 Languages** across 12 language families, 10 script types, and 4 syntactic orders
- **201K+ Samples** combining human-written (122K) and LLM-generated (78K) content
- **4 Content Types:** Human-Written (HWT), Machine-Generated (MGT), Machine-Translated (MTT), and Human-AI Hybrid (HAT)
- **39 Textual Modification Techniques:** 36 manipulation tactics for fake news + 3 AI-editing strategies for real news
- **19 Diverse mLLMs** used for content generation
- **4 Benchmark Tasks** with standardized train/dev/test splits
- **AXL-CoI Framework:** Adversarial Cross-Lingual Agentic Chain-of-Interactions for controlled generation
- **mPURIFY Pipeline:** Quality filtering ensuring dataset integrity

---

## 📂 Repository Structure

```
BLUFF/
├── README.md                     # This file
├── LICENSE                       # MIT License (code) + CC BY-NC-SA 4.0 (data)
├── DATASHEET.md                  # Datasheet for BLUFF (Gebru et al., 2021)
├── DATASET_CARD.md               # HuggingFace-style dataset card
├── CHANGELOG.md                  # Version history
├── CODE_OF_CONDUCT.md            # Community guidelines
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installation
│
├── data/
│   ├── raw/                      # Raw source data (see Data Access)
│   ├── processed/                # Processed and cleaned data
│   └── splits/                   # Standardized train/dev/test splits
│       ├── task1_veracity_binary/
│       ├── task2_veracity_multiclass/
│       ├── task3_authorship_binary/
│       └── task4_authorship_multiclass/
│
├── src/
│   ├── data_collection/          # Data collection scripts
│   │   ├── source_scraper.py     # Fact-check article collection from 331 sources
│   │   └── language_detector.py  # Multi-tool language identification
│   ├── generation/               # Content generation pipeline
│   │   ├── axl_coi.py            # AXL-CoI agentic framework (10+8 chain agents)
│   │   ├── adis.py               # ADIS adversarial prompt engineering
│   │   └── prompts/              # Prompt templates for 39 techniques
│   ├── filtering/                # Quality filtering
│   │   ├── mpurify.py            # mPURIFY pipeline (32 features, 5 dimensions)
│   │   └── validators.py         # Schema, text quality, deduplication validators
│   └── evaluation/               # Evaluation scripts
│       ├── encoder/              # Encoder-based experiments
│       │   ├── train.py          # Fine-tuning script
│       │   ├── evaluate.py       # Evaluation with grouped metrics
│       │   └── configs/          # Multilingual & cross-lingual configs
│       └── decoder/              # Decoder-based experiments
│           ├── inference.py      # Prompt-based inference
│           ├── evaluate.py       # Evaluation with transfer analysis
│           └── prompts/          # 3 prompt regimens (crosslingual, native, translated)
│
├── scripts/
│   ├── download_data.sh          # Download dataset from HuggingFace
│   ├── run_encoder_experiments.sh
│   ├── run_decoder_experiments.sh
│   └── generate_results_tables.py
│
├── configs/
│   ├── encoder_models.yaml       # 9 encoder model configurations
│   ├── decoder_models.yaml       # 7 decoder model configurations
│   ├── languages.yaml            # 78-language metadata and taxonomy
│   └── tasks.yaml                # 4 task definitions
│
├── docs/
│   ├── LANGUAGE_TAXONOMY.md      # Full 78-language classification
│   ├── MANIPULATION_TACTICS.md   # 39 modification techniques documentation
│   ├── EXPERIMENT_GUIDE.md       # Step-by-step experiment reproduction
│   ├── RESULTS.md                # Comprehensive results summary
│   ├── ETHICS.md                 # Ethics, fairness, and responsible use
│   └── CONTRIBUTING.md           # Contribution guidelines
│
├── figures/                      # Paper figures and visualizations
│
├── experiments/                  # Detailed evaluation framework and experiment runner
│   └── README.md                # Comprehensive experiments guide (tasks, models, settings)
│
└── baselines/                    # Pre-trained baseline checkpoints (links)
    └── README.md
```

---

## 📊 Benchmark Tasks

| Task | Description | Classes | Metric |
|------|-------------|---------|--------|
| **Task 1** | Binary Veracity Classification | Real / Fake | F1 (macro) |
| **Task 2** | Multi-class Veracity Classification | Real / Fake × Source Type | F1 (macro) |
| **Task 3** | Binary Authorship Detection | Human / Machine | F1 (macro) |
| **Task 4** | Multi-class Authorship Attribution | HWT / MGT / MTT / HAT | F1 (macro) |

---

## 🌐 Language Coverage

BLUFF covers **78 languages** organized into:

| Category | Count | Examples |
|----------|-------|---------|
| **Big-Head (High-Resource)** | 20 | English, Spanish, French, Chinese, Arabic, Hindi, ... |
| **Long-Tail (Low-Resource)** | 58 | Yoruba, Amharic, Khmer, Lao, Quechua, Malagasy, ... |

**Language Families:** Indo-European, Sino-Tibetan, Afro-Asiatic, Niger-Congo, Austronesian, Dravidian, Turkic, Uralic, Koreanic, Japonic, Tai-Kadai, Austroasiatic

**Scripts:** Latin, Cyrillic, Arabic, Devanagari, CJK, Thai, Ethiopic, Khmer, Bengali, Georgian

See [`docs/LANGUAGE_TAXONOMY.md`](docs/LANGUAGE_TAXONOMY.md) for the complete linguistic classification.

---

## 🧪 Experiments

For the complete evaluation framework, including all 6 training settings, 18 models, prompt regimens, data preparation scripts, and full reproducibility instructions, see the **[Experiments Guide](experiments/README.md)**.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/jsl5710/BLUFF.git
cd BLUFF
pip install -r requirements.txt
```

### Download Dataset

```bash
# Option 1: Download everything (~3.9 GB)
bash scripts/download_data.sh

# Option 2: Download specific subsets
bash scripts/download_data.sh --subset splits      # Only split definitions
bash scripts/download_data.sh --subset meta_data   # Only metadata CSVs
bash scripts/download_data.sh --subset processed   # Cleaned text data

# Option 3: Python API
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='jsl5710/BLUFF', repo_type='dataset', local_dir='./data')
"
```

The dataset on HuggingFace is organized as:
- `data/meta_data/` — Metadata CSVs (`metadata_human_written.csv`, `metadata_ai_generated.csv`)
- `data/processed/` — Cleaned text data organized by model and language
- `data/raw/` — Original source data from fact-checking organizations
- `data/splits/` — Train/val split definitions (JSON files with sample UUIDs)

> **Note:** Test splits are held out to preserve benchmark integrity. Contact the authors for test set evaluation.

See the [HuggingFace dataset page](https://huggingface.co/datasets/jsl5710/BLUFF) for detailed field descriptions and usage examples.

### Run Evaluation

```bash
# Encoder-based training (e.g., XLM-RoBERTa)
python src/evaluation/encoder/train.py \
    --model xlm-roberta-large \
    --task task1_veracity_binary \
    --experiment multilingual \
    --data_dir data

# Encoder-based evaluation
python src/evaluation/encoder/evaluate.py \
    --model outputs/encoder/task1_veracity_binary/multilingual/xlm-roberta-large/best_model \
    --task task1_veracity_binary \
    --data_dir data

# Decoder-based inference (e.g., GPT-4o)
python src/evaluation/decoder/inference.py \
    --model gpt4o \
    --task task1_veracity_binary \
    --prompt_type crosslingual \
    --data_dir data
```

---

## 🔬 Reproducing Experiments

### Encoder Models

We evaluate the following multilingual encoder models:

| Model | Parameters | Languages |
|-------|-----------|-----------|
| mBERT | 110M | 104 |
| XLM-RoBERTa-base | 270M | 100 |
| XLM-RoBERTa-large | 550M | 100 |
| mDeBERTa-v3 | 86M / 304M | 100 |
| Glot500 | 395M | 511 |

**Experiment Types:**
- **Multilingual:** Train on all languages, evaluate head vs. tail performance
- **Cross-lingual:** Train on subset, evaluate transfer by language family, syntax, and script
- **External:** Evaluate on held-out datasets

```bash
# Run all encoder experiments
bash scripts/run_encoder_experiments.sh

# Run specific configuration
python src/evaluation/encoder/train.py \
    --config configs/encoder_models.yaml \
    --experiment multilingual \
    --task task1_veracity_binary
```

### Decoder Models

We evaluate multilingual decoder models with three prompt regimens:

| Prompt Type | Description |
|-------------|-------------|
| **Cross-lingual** | Prompt in English, input in original language |
| **Native** | Prompt and input in target language |
| **English-Translated** | Both prompt and input in English (translated) |

```bash
# Run all decoder experiments
bash scripts/run_decoder_experiments.sh

# Run specific prompt regimen
python src/evaluation/decoder/inference.py \
    --config configs/decoder_models.yaml \
    --prompt_type native \
    --task task1_veracity_binary
```

---

## 📋 Data Format

### Processed Text Data (`data/processed/generated_data/`)

Each processed CSV file contains cleaned, model-ready text data:

| Field | Description |
|-------|-------------|
| `uuid` | Unique sample identifier (links to metadata and splits) |
| `article_content` | Full article text in the original language |
| `translated_content` | English translation of the article |
| `post_content` | Social media post version in the original language |
| `translated_post` | English translation of the post |
| `language` | ISO 639-3 language code |
| `model` | Generating model name |
| `veracity` | `fake_news` or `real_news` |
| `technique_keys` | Manipulation technique IDs applied |
| `degree` | Edit intensity (`minor`, `moderate`, `critical`) |
| `HAT` / `MGT` / `MTT` / `HWT` | Content type flags (`y`/`n`) |

### Split Files (`data/splits/evaluation/`)

Each split directory contains `train.json` and `val.json` — lists of UUIDs that map to samples in the metadata and processed data. Use these to construct train/val datasets for each experimental setting.

### Metadata (`data/meta_data/`)

Rich per-sample metadata including quality filtering (mPURIFY) results, source information, language classification, and generation details.

---

## 📑 Data Collection & Methodology

### Human-Written Text (HWT)
- **Sources:** IFCN-certified fact-checking organizations and CredCatalog-indexed publishers
- **Languages:** 57 languages, 122,836 samples
- **Process:** Collected, deduplicated, language-verified, and labeled by professional fact-checkers

### LLM-Generated Content (MGT/MTT/HAT)
- **Framework:** AXL-CoI (Adversarial Cross-Lingual Agentic Chain-of-Interactions)
- **Models:** 19 multilingual LLMs (GPT-4o, Claude, Gemini, Llama, Qwen, Aya, etc.)
- **Languages:** 71 languages, 78,443 samples
- **Tactics:** 36 manipulation tactics for fake news, 3 editing strategies for real news
- **Translation:** Bidirectional (English ↔ X) with quality verification

### Quality Filtering (mPURIFY)
- Language consistency verification
- Semantic preservation scoring
- Factual manipulation validation
- Deduplication and near-duplicate detection
- Human spot-check validation

See the paper and [`DATASHEET.md`](DATASHEET.md) for full methodology details.

---

## 📄 Citation

If you use BLUFF in your research, please cite:

Paper currently under review. Citation will be provided upon acceptance.

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) for details.

Areas where contributions are especially welcome:
- Additional low-resource language data
- New baseline model evaluations
- Improved evaluation metrics for multilingual settings
- Bug fixes and documentation improvements

---

## ⚖️ Ethics & Responsible Use

BLUFF contains realistic synthetic disinformation for research purposes only. By accessing this dataset, you agree to:

1. **Use the data solely for research** aimed at improving disinformation detection
2. **Not redistribute** generated fake news content outside research contexts
3. **Cite the dataset** in any publications using BLUFF
4. **Report any misuse** discovered to the authors

All generated content includes metadata watermarks identifying it as synthetic research material. Detection models are released alongside the dataset.

See [`docs/ETHICS.md`](docs/ETHICS.md) for comprehensive ethics documentation including bias analysis, misuse safeguards, and fairness considerations. See [`DATASHEET.md`](DATASHEET.md) for the full Gebru et al. (2021) datasheet.

---

## ⚠️ Known Limitations

- **Geographic bias:** HWT data coverage correlates with global fact-checking infrastructure; some regions are underrepresented
- **Topical bias:** Fact-checked content skews toward politically salient topics
- **Generation quality:** Big-head languages likely have higher LLM generation quality than long-tail languages
- **Cross-lingual gaps:** 9.0–25.3% performance degradation for long-tail languages in cross-lingual settings
- **Temporal scope:** Dataset reflects disinformation patterns at time of collection
- **Decoder limitations:** Current decoder models fail on fine-grained classification tasks (below random baseline)

See the paper and [`docs/RESULTS.md`](docs/RESULTS.md) for detailed analysis of these limitations.

---

## 🏆 Leaderboard

We maintain benchmark results for community reference. Submit new results via pull request following the format in [`docs/RESULTS.md`](docs/RESULTS.md).

| Rank | Model | Task 1 (F1) | Task 3 (F1) | Task 4 (F1) | Setting |
|------|-------|-------------|-------------|-------------|---------|
| 1 | S-BERT (LaBSE) | 97.2 | 93.2 | 82.0 | Multilingual |
| 2 | mDeBERTa-v3 | 98.3* | 87.3 | 80.6 | Multilingual |
| 3 | XLM-R-large | 84.7 | 87.3 | — | Multilingual |

*Big-head only; see full results in [`docs/RESULTS.md`](docs/RESULTS.md)

---

## 📜 License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.

The dataset is released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) for research use.

---

## 📧 Contact

- **Jason Lucas** — [jsl5710@psu.edu](mailto:jsl5710@psu.edu) | [Website](https://jasonlucas.info)
- **Dongwon Lee** — [dongwon@psu.edu](mailto:dongwon@psu.edu)
- **PIKE Research Lab** — Penn State University, College of IST

---

## 🙏 Acknowledgments

This work was supported in part by the Penn State College of Information Sciences and Technology. We thank the IFCN-certified fact-checking organizations whose publicly available work made this benchmark possible.
