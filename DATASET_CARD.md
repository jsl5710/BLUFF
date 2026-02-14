# Dataset Card for BLUFF

## Dataset Description

- **Homepage:** [https://github.com/jsl5710/BLUFF](https://github.com/jsl5710/BLUFF)
- **Repository:** [https://github.com/jsl5710/BLUFF](https://github.com/jsl5710/BLUFF)
- **Paper:** BLUFF: A Benchmark for Linguistic Understanding of Fake-news Forensics (KDD 2026)
- **Point of Contact:** Jason Lucas (jsl5710@psu.edu)

### Dataset Summary

BLUFF is a multilingual fake news detection benchmark covering 78 languages with over 201K samples. It combines human-written fact-checked content (122K samples across 57 languages) and LLM-generated content (78K samples across 71 languages) to provide the first comprehensive testbed for disinformation detection beyond high-resource settings.

### Supported Tasks and Leaderboards

| Task ID | Task | Type |
|---------|------|------|
| `task1_veracity_binary` | Binary Veracity Classification | Real vs. Fake |
| `task2_veracity_multiclass` | Multi-class Veracity Classification | Real/Fake × Source Type |
| `task3_authorship_binary` | Binary Authorship Detection | Human vs. Machine |
| `task4_authorship_multiclass` | Multi-class Authorship Attribution | HWT / MGT / MTT / HAT |

### Languages

78 languages across 12 families:

**Big-Head (20):** ar, bn, de, en, es, fa, fr, hi, id, it, ja, ko, nl, pl, pt, ru, sv, tr, uk, zh

**Long-Tail (58):** af, am, az, bg, ca, ceb, cs, cy, da, el, et, eu, fi, ga, gl, gu, ha, he, hr, hu, hy, is, ka, kk, km, kn, ku, ky, lo, lt, lv, mk, ml, mr, ms, mt, my, ne, no, pa, ps, ro, si, sk, sl, sq, sr, sw, ta, te, tg, th, tl, ur, uz, vi, xh, yo

## Dataset Structure

### Data Instances

```json
{
    "id": "BLUFF-EN-001234",
    "text": "Article text...",
    "language": "en",
    "language_family": "Indo-European",
    "script": "Latin",
    "syntax": "SVO",
    "resource_category": "big-head",
    "veracity_label": "fake",
    "authorship_type": "MGT",
    "generation_model": "gpt-4o",
    "manipulation_tactic": "emotional_amplification",
    "edit_intensity": "high",
    "source": "PolitiFact",
    "split": "train"
}
```

### Data Splits

| Split | Samples | Purpose |
|-------|---------|---------|
| Train | ~140K | Model training |
| Dev | ~30K | Hyperparameter tuning |
| Test | ~31K | Final evaluation |

Splits are stratified by language and label distribution.

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique sample identifier |
| `text` | string | Article content |
| `language` | string | ISO 639 language code |
| `language_family` | string | Linguistic family |
| `script` | string | Writing system |
| `syntax` | string | Word order typology |
| `resource_category` | string | "big-head" or "long-tail" |
| `veracity_label` | string | "real" or "fake" |
| `authorship_type` | string | HWT, MGT, MTT, or HAT |
| `generation_model` | string | LLM used (null for HWT) |
| `manipulation_tactic` | string | Tactic applied (null for HWT/real) |
| `edit_intensity` | string | low/medium/high (null for HWT) |
| `source` | string | Fact-checking organization |
| `split` | string | train/dev/test |

## Dataset Creation

### Curation Rationale

Existing fake news detection benchmarks are overwhelmingly English-centric. BLUFF addresses the digital language divide by providing evaluation resources for 58 low-resource languages alongside 20 high-resource languages, enabling research on equitable disinformation detection.

### Source Data

- **Human-Written:** IFCN-certified fact-checking organizations and CredCatalog-indexed publishers
- **LLM-Generated:** AXL-CoI framework using 19 multilingual LLMs with 39 textual modification techniques

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

### Licensing

- **Dataset:** CC BY-NC-SA 4.0
- **Code:** MIT License
