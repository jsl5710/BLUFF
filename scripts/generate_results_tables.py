"""
Generate results tables from BLUFF benchmark experiments.

Reads experiment output JSON files and produces formatted tables
for paper inclusion (LaTeX) and documentation (Markdown).

Usage:
    python scripts/generate_results_tables.py \
        --results_dir results/ \
        --output_format markdown \
        --output_file docs/RESULTS.md
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def load_results(results_dir: str) -> dict:
    """Load all experiment results from a directory."""
    results = {}
    results_path = Path(results_dir)
    for json_file in results_path.glob("**/*.json"):
        with open(json_file) as f:
            data = json.load(f)
            key = json_file.stem
            results[key] = data
    return results


def format_markdown_table(headers: list, rows: list) -> str:
    """Format data as a Markdown table."""
    col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows))
                  for i, h in enumerate(headers)]
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    separator = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
    data_lines = []
    for row in rows:
        data_lines.append(
            "| " + " | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)) + " |"
        )
    return "\n".join([header_line, separator] + data_lines)


def format_latex_table(headers: list, rows: list, caption: str = "") -> str:
    """Format data as a LaTeX table."""
    n_cols = len(headers)
    col_spec = "l" + "c" * (n_cols - 1)
    lines = [
        f"\\begin{{table}}[t]",
        f"\\caption{{{caption}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(v) for v in row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def generate_summary_table(results: dict, output_format: str = "markdown") -> str:
    """Generate a summary table across all tasks and models."""
    headers = ["Model", "Task 1 (F1)", "Task 2 (F1)", "Task 3 (F1)", "Task 4 (F1)"]
    rows = []

    for model_name, model_results in sorted(results.items()):
        if "overall" in model_results:
            row = [model_name]
            for task_id in ["task1", "task2", "task3", "task4"]:
                score = model_results.get(task_id, {}).get("f1_macro", "-")
                if isinstance(score, float):
                    score = f"{score:.1f}"
                row.append(score)
            rows.append(row)

    if output_format == "latex":
        return format_latex_table(headers, rows, "BLUFF Benchmark Results Summary")
    return format_markdown_table(headers, rows)


def main():
    parser = argparse.ArgumentParser(description="Generate BLUFF results tables")
    parser.add_argument("--results_dir", default="results/", help="Directory with result JSONs")
    parser.add_argument("--output_format", choices=["markdown", "latex"], default="markdown")
    parser.add_argument("--output_file", default=None, help="Output file path")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    table = generate_summary_table(results, args.output_format)
    print(table)

    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(table)
        print(f"\nTable saved to {args.output_file}")


if __name__ == "__main__":
    main()
