# AXL-CoI Prompt Templates

This directory contains prompt templates for the 39 textual modification techniques
used in the BLUFF benchmark.

## Structure

- `tactics/` — Templates for 36 fake news manipulation tactics
- `editing/` — Templates for 3 real news AI-editing strategies
- `translation/` — Templates for bidirectional translation prompts

## Template Variables

Templates use Python string formatting with these variables:
- `{text}` — Source article text
- `{analysis}` — Analyst agent's content analysis
- `{intensity}` — Edit intensity level (low/medium/high)
- `{source_language}` — Source language name
- `{target_language}` — Target language name

## Usage

Templates are loaded automatically by the AXL-CoI pipeline.
See `src/generation/axl_coi.py` for implementation details.
