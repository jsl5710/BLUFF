# Changelog

All notable changes to the BLUFF benchmark will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-XX-XX

### Added
- Initial release of BLUFF benchmark
- 202,395 samples across 79 languages (20 big-head + 59 long-tail)
- 122,836 human-written samples from 331 IFCN-certified sources
- 79,559 LLM-generated samples using AXL-CoI framework with 19 mLLMs
- 4 benchmark tasks: binary/multiclass veracity, binary/multiclass authorship
- 39 textual modification techniques (36 manipulation + 3 editing)
- mPURIFY quality filtering pipeline
- Encoder evaluation scripts (9 multilingual models)
- Decoder evaluation scripts (7 models, 3 prompt regimens)
- Comprehensive documentation: README, DATASHEET, DATASET_CARD
- Language taxonomy covering 12 families, 10+ script types, 4 syntactic orders
- Cross-lingual transfer analysis by script, syntax, and family
- External evaluation on 28 held-out sources (36,612 samples, 53 languages)
- Standardized train/dev/test splits stratified by language and label
