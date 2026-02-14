# Contributing to BLUFF

Thank you for your interest in contributing to the BLUFF benchmark! We welcome contributions that expand language coverage, improve evaluation methods, fix bugs, or enhance documentation.

---

## How to Contribute

### Reporting Issues

- Use [GitHub Issues](https://github.com/jsl5710/BLUFF/issues) for bug reports, feature requests, or questions
- Include your Python version, OS, and relevant error messages
- For data quality issues, include the sample ID and language

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes with clear commit messages
4. Run existing tests to ensure nothing breaks
5. Submit a pull request with a description of your changes

### Areas Where Contributions Are Welcome

- **Additional language data:** Expanding coverage to new low-resource languages
- **New baseline evaluations:** Running additional models on BLUFF tasks
- **Evaluation metrics:** Proposing multilingual-aware metrics
- **Bug fixes:** Addressing issues in code or data
- **Documentation:** Improving guides, examples, and translations

---

## Code Style

- Python code follows PEP 8 conventions
- Use type hints for function signatures
- Include docstrings for all public functions and classes
- YAML configuration files use consistent 2-space indentation

## Adding New Languages

If you want to contribute data for new languages:

1. Ensure data follows the schema in `DATASET_CARD.md`
2. Include language metadata (family, script, syntax, resource category)
3. Provide sourcing documentation and licensing information
4. Run the quality validation pipeline (`src/filtering/validators.py`)
5. Submit with a description of collection methodology

## Adding New Baselines

1. Add model configuration to `configs/encoder_models.yaml` or `configs/decoder_models.yaml`
2. Run experiments using the standardized evaluation scripts
3. Report results following the format in `docs/RESULTS.md`
4. Include training logs and hyperparameters

---

## Code of Conduct

All contributors are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). We are committed to providing a welcoming and inclusive experience for everyone.

## Questions?

Open a GitHub issue or contact the maintainers at jsl5710@psu.edu.
