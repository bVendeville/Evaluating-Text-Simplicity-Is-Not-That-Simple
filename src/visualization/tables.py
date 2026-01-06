"""
Table generation utilities for LaTeX and Markdown output.

This module provides functions for creating publication-ready tables:
- LaTeX correlation tables
- LaTeX significance tables
- Markdown summary tables
"""

from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd


def generate_correlation_latex(
    results: Dict[str, Dict],
    method: str = 'pearson',
    bold_top: bool = True,
    decimals: int = 3,
    include_ci: bool = True,
    caption: str = "Correlations with human scores",
) -> str:
    """
    Generate LaTeX table comparing metric correlations across datasets.

    Args:
        results: Dict mapping dataset name to analysis results
        method: Correlation method used
        bold_top: Bold top metrics
        decimals: Decimal places
        include_ci: Include confidence intervals
        caption: Table caption

    Returns:
        LaTeX table string
    """
    # Get all metrics and datasets
    all_metrics = set()
    for r in results.values():
        all_metrics.update(r['correlations_df']['metric'].tolist())
    metrics = sorted(all_metrics)
    datasets = list(results.keys())

    # Build data matrix
    data = {}
    top_metrics = {}
    for dataset_name, dataset_results in results.items():
        top_metrics[dataset_name] = set(dataset_results['top_metrics'])
        corr_df = dataset_results['correlations_df']
        data[dataset_name] = dict(zip(corr_df['metric'], corr_df['correlation']))

    # Start LaTeX
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(f"\\caption{{{caption}}}")

    # Column specification
    n_cols = len(datasets) + 2  # Metric + datasets + Average
    col_spec = "l" + "c" * (n_cols - 1)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header row
    header = "Metric"
    for ds in datasets:
        header += f" & {ds}"
    header += " & Average \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Data rows
    for metric in metrics:
        values = []
        for ds in datasets:
            val = data[ds].get(metric, np.nan)
            values.append(val)

        avg = np.nanmean(values)

        # Format row
        row = metric.replace("_", "\\_")
        for i, (ds, val) in enumerate(zip(datasets, values)):
            if np.isnan(val):
                formatted = "-"
            else:
                formatted = f"{val:.{decimals}f}"
                if bold_top and metric in top_metrics[ds]:
                    formatted = f"\\textbf{{{formatted}}}"
            row += f" & {formatted}"

        # Average
        if not np.isnan(avg):
            row += f" & {avg:.{decimals}f}"
        else:
            row += " & -"

        row += " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_combined_results_latex(
    combined_df: pd.DataFrame,
    caption: str = "Correlation with human scores across datasets",
    bold_top_n: int = 3,
    decimals: int = 3,
) -> str:
    """
    Generate LaTeX table from combined results DataFrame.

    Args:
        combined_df: DataFrame with metrics as rows, datasets as columns
        caption: Table caption
        bold_top_n: Number of top values to bold per column
        decimals: Decimal places

    Returns:
        LaTeX table string
    """
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(f"\\caption{{{caption}}}")

    # Column specification
    col_spec = "l" + "c" * len(combined_df.columns)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header
    header = "Metric"
    for col in combined_df.columns:
        header += f" & {col}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Find top values per column
    top_values: Dict[str, Set] = {}
    for col in combined_df.columns:
        sorted_vals = combined_df[col].dropna().sort_values(ascending=False)
        top_values[col] = set(sorted_vals.head(bold_top_n).index)

    # Data rows
    for metric in combined_df.index:
        row = str(metric).replace("_", "\\_")
        for col in combined_df.columns:
            val = combined_df.loc[metric, col]
            if pd.isna(val):
                formatted = "-"
            else:
                formatted = f"{val:.{decimals}f}"
                if metric in top_values[col]:
                    formatted = f"\\textbf{{{formatted}}}"
            row += f" & {formatted}"
        row += " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_significance_summary_latex(
    results: Dict[str, Dict],
    caption: str = "Top metrics by dataset",
) -> str:
    """
    Generate LaTeX table summarizing top metrics per dataset.

    Args:
        results: Dict mapping dataset name to analysis results
        caption: Table caption

    Returns:
        LaTeX table string
    """
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\begin{tabular}{lll}")
    lines.append("\\toprule")
    lines.append("Dataset & $n$ & Top Metrics (not significantly outperformed) \\\\")
    lines.append("\\midrule")

    for dataset_name, dataset_results in results.items():
        n = dataset_results.get('n_samples', '?')
        top = dataset_results.get('top_metrics', [])
        top_str = ", ".join(top) if top else "None"
        top_str = top_str.replace("_", "\\_")

        lines.append(f"{dataset_name} & {n} & {top_str} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_model_comparison_latex(
    results_df: pd.DataFrame,
    metric_cols: List[str] = ['R2', 'Pearson', 'Spearman', 'RMSE'],
    caption: str = "Model comparison",
    bold_best: bool = True,
    decimals: int = 3,
) -> str:
    """
    Generate LaTeX table comparing model performance.

    Args:
        results_df: DataFrame with model results
        metric_cols: Columns to include
        caption: Table caption
        bold_best: Bold best value per column
        decimals: Decimal places

    Returns:
        LaTeX table string
    """
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")

    available_cols = [c for c in metric_cols if c in results_df.columns]
    col_spec = "l" + "c" * len(available_cols)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header
    header = "Model"
    for col in available_cols:
        header += f" & {col}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Find best values
    best_vals = {}
    for col in available_cols:
        if col in ['RMSE', 'MSE', 'MAE']:
            best_vals[col] = results_df[col].min()
        else:
            best_vals[col] = results_df[col].max()

    # Data rows
    name_col = 'Model' if 'Model' in results_df.columns else results_df.index.name
    for idx, row in results_df.iterrows():
        model_name = row.get('Model', idx) if 'Model' in results_df.columns else idx
        line = str(model_name).replace("_", "\\_")

        for col in available_cols:
            val = row[col]
            if pd.isna(val):
                formatted = "-"
            else:
                formatted = f"{val:.{decimals}f}"
                if bold_best and val == best_vals[col]:
                    formatted = f"\\textbf{{{formatted}}}"
            line += f" & {formatted}"

        line += " \\\\"
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_markdown_summary(
    results: Dict[str, Dict],
    decimals: int = 3,
) -> str:
    """
    Generate Markdown summary table.

    Args:
        results: Dict mapping dataset name to analysis results
        decimals: Decimal places

    Returns:
        Markdown table string
    """
    lines = []
    lines.append("| Dataset | n | Best Metric | Correlation | Top Metrics |")
    lines.append("|---------|---|-------------|-------------|-------------|")

    for dataset_name, r in results.items():
        n = r.get('n_samples', '?')
        summary = r.get('summary', {})
        best = summary.get('best_by_correlation', '-')
        best_corr = summary.get('best_correlation', 0)
        top = r.get('top_metrics', [])

        lines.append(
            f"| {dataset_name} | {n} | {best} | {best_corr:.{decimals}f} | "
            f"{', '.join(top[:3])}{'...' if len(top) > 3 else ''} |"
        )

    return "\n".join(lines)


def save_latex_table(
    latex_str: str,
    filepath: str,
) -> None:
    """Save LaTeX table to file."""
    with open(filepath, 'w') as f:
        f.write(latex_str)


if __name__ == '__main__':
    # Quick test
    print("Testing LaTeX table generation...")

    # Fake results
    results = {
        'ARTS94': {
            'correlations_df': pd.DataFrame({
                'metric': ['A', 'B', 'C'],
                'correlation': [0.85, 0.72, 0.65],
            }),
            'top_metrics': ['A'],
            'n_samples': 94,
            'summary': {
                'best_by_correlation': 'A',
                'best_correlation': 0.85,
            },
        },
    }

    latex = generate_correlation_latex(results)
    print("LaTeX output:")
    print(latex[:500] + "...")

    md = generate_markdown_summary(results)
    print("\nMarkdown output:")
    print(md)
