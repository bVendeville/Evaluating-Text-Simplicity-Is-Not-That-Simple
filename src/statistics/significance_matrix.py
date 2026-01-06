"""
Full significance analysis combining correlations and Williams tests.

This module provides high-level functions for comprehensive significance analysis:
- Compute correlations with human scores for all metrics
- Compute inter-metric correlations
- Apply Williams test to all metric pairs
- Generate significance matrices and identify top metrics
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .correlation import pearson_ci, spearman_ci
from .williams_test import williams_test


def compute_significance_matrix(
    metric_scores: Dict[str, np.ndarray],
    human_scores: np.ndarray,
    correlation_type: str = 'pearson',
    alpha: float = 0.05
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise Williams test significance matrix.

    This is the main function for significance analysis, matching the reference
    implementation in 08_significance_analysis.ipynb.

    Args:
        metric_scores: Dict mapping metric name to score array
        human_scores: Human judgment scores
        correlation_type: 'pearson' or 'spearman'
        alpha: Significance level

    Returns:
        correlations_df: DataFrame with correlations for each metric
        p_value_matrix: DataFrame with p-values (i,j) = p-value for metric i > metric j
        significance_matrix: DataFrame with boolean (i,j) = metric i significantly > metric j
    """
    metrics = list(metric_scores.keys())
    n = len(human_scores)

    corr_func = pearsonr if correlation_type == 'pearson' else spearmanr
    ci_func = pearson_ci if correlation_type == 'pearson' else spearman_ci

    # Compute correlations with human scores
    correlations = {}
    for metric in metrics:
        scores = metric_scores[metric]
        valid = ~(np.isnan(scores) | np.isnan(human_scores))
        if valid.sum() > 3:
            r, p = corr_func(scores[valid], human_scores[valid])
            correlations[metric] = {'correlation': r, 'p_value': p, 'n': valid.sum()}
        else:
            correlations[metric] = {'correlation': np.nan, 'p_value': np.nan, 'n': 0}

    correlations_df = pd.DataFrame(correlations).T

    # Compute confidence intervals
    correlations_df['ci_lower'] = correlations_df.apply(
        lambda row: ci_func(row['correlation'], int(row['n']))[0] if row['n'] > 0 else np.nan, axis=1
    )
    correlations_df['ci_upper'] = correlations_df.apply(
        lambda row: ci_func(row['correlation'], int(row['n']))[1] if row['n'] > 0 else np.nan, axis=1
    )
    correlations_df['ci_width'] = correlations_df['ci_upper'] - correlations_df['ci_lower']

    # Sort by ABSOLUTE correlation (strongest relationship first)
    correlations_df['abs_correlation'] = correlations_df['correlation'].abs()
    correlations_df = correlations_df.sort_values('abs_correlation', ascending=False)
    correlations_df = correlations_df.drop(columns=['abs_correlation'])
    metrics_ordered = correlations_df.index.tolist()

    # Compute correlation between metrics
    metric_metric_corr = {}
    for m1 in metrics_ordered:
        for m2 in metrics_ordered:
            s1, s2 = metric_scores[m1], metric_scores[m2]
            valid = ~(np.isnan(s1) | np.isnan(s2))
            if valid.sum() > 3:
                r, _ = corr_func(s1[valid], s2[valid])
                metric_metric_corr[(m1, m2)] = r
            else:
                metric_metric_corr[(m1, m2)] = np.nan

    # Compute Williams test p-values
    p_value_matrix = pd.DataFrame(index=metrics_ordered, columns=metrics_ordered, dtype=float)
    significance_matrix = pd.DataFrame(index=metrics_ordered, columns=metrics_ordered, dtype=bool)

    for m1 in metrics_ordered:
        r_m1_human = correlations[m1]['correlation']
        for m2 in metrics_ordered:
            if m1 == m2:
                p_value_matrix.loc[m1, m2] = np.nan
                significance_matrix.loc[m1, m2] = False
            else:
                r_m2_human = correlations[m2]['correlation']
                r_m1_m2 = metric_metric_corr.get((m1, m2), np.nan)

                if np.isnan(r_m1_human) or np.isnan(r_m2_human) or np.isnan(r_m1_m2):
                    p_value_matrix.loc[m1, m2] = np.nan
                    significance_matrix.loc[m1, m2] = False
                else:
                    # Use ABSOLUTE values for Williams test
                    r12 = abs(r_m1_human)
                    r13 = abs(r_m2_human)
                    r23 = abs(r_m1_m2)

                    _, p = williams_test(r12, r13, r23, n)
                    p_value_matrix.loc[m1, m2] = p
                    significance_matrix.loc[m1, m2] = p < alpha

    return correlations_df, p_value_matrix, significance_matrix


def identify_top_metrics(significance_matrix: pd.DataFrame) -> List[str]:
    """Identify metrics not significantly outperformed by any other metric."""
    top_metrics = []
    metrics = significance_matrix.index.tolist()

    for metric in metrics:
        is_outperformed = False
        for other_metric in metrics:
            if other_metric != metric:
                if significance_matrix.loc[other_metric, metric]:
                    is_outperformed = True
                    break
        if not is_outperformed:
            top_metrics.append(metric)

    return top_metrics


def compute_significance_analysis(
    metric_scores: Dict[str, np.ndarray],
    human_scores: np.ndarray,
    method: str = 'pearson',
    alpha: float = 0.05
) -> Dict:
    """
    Perform full significance analysis for a set of metrics.

    Wrapper around compute_significance_matrix that returns a dict with
    additional analysis results.

    Args:
        metric_scores: Dict mapping metric name to score array
        human_scores: Human judgment scores
        method: 'pearson' or 'spearman'
        alpha: Significance level

    Returns:
        Dict containing:
        - correlations_df: DataFrame with correlations and CIs
        - p_value_matrix: Williams test p-values
        - significance_matrix: Boolean significance matrix
        - top_metrics: List of top-performing metrics
        - n_samples: Sample size
    """
    correlations_df, p_value_matrix, significance_matrix = compute_significance_matrix(
        metric_scores, human_scores, correlation_type=method, alpha=alpha
    )

    top_metrics = identify_top_metrics(significance_matrix)

    # Reformat correlations_df to have 'metric' column
    correlations_df = correlations_df.reset_index()
    correlations_df = correlations_df.rename(columns={'index': 'metric'})

    return {
        'correlations_df': correlations_df,
        'p_value_matrix': p_value_matrix,
        'significance_matrix': significance_matrix,
        'top_metrics': top_metrics,
        'n_samples': len(human_scores),
        'method': method,
        'alpha': alpha,
    }


def analyze_multiple_datasets(
    datasets: Dict[str, Dict],
    metric_key: str = 'metric_scores',
    human_key: str = 'human_scores',
    method: str = 'pearson',
    alpha: float = 0.05
) -> Dict[str, Dict]:
    """
    Perform significance analysis across multiple datasets.

    Args:
        datasets: Dict mapping dataset name to dict with metric_scores and human_scores
        metric_key: Key for metric scores in each dataset dict
        human_key: Key for human scores in each dataset dict
        method: 'pearson' or 'spearman'
        alpha: Significance level

    Returns:
        Dict mapping dataset name to analysis results
    """
    results = {}

    for dataset_name, data in datasets.items():
        print(f"Analyzing {dataset_name}...")

        metric_scores = data[metric_key]
        human_scores = data[human_key]

        results[dataset_name] = compute_significance_analysis(
            metric_scores, human_scores, method=method, alpha=alpha
        )

    return results


def combine_dataset_results(
    results: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Combine correlation results from multiple datasets into a single table.

    Args:
        results: Dict mapping dataset name to analysis results

    Returns:
        DataFrame with metrics as rows and datasets as columns
    """
    # Get all unique metrics across datasets
    all_metrics = set()
    for dataset_results in results.values():
        all_metrics.update(dataset_results['correlations_df']['metric'].tolist())

    all_metrics = sorted(all_metrics)

    # Build combined table
    combined_data = []
    for metric in all_metrics:
        row = {'metric': metric}
        for dataset_name, dataset_results in results.items():
            corr_df = dataset_results['correlations_df']
            metric_row = corr_df[corr_df['metric'] == metric]
            if len(metric_row) > 0:
                row[dataset_name] = metric_row['correlation'].values[0]
            else:
                row[dataset_name] = np.nan
        combined_data.append(row)

    df = pd.DataFrame(combined_data)
    df = df.set_index('metric')

    # Add average column (using absolute values)
    df['Average'] = df.abs().mean(axis=1)

    return df.sort_values('Average', ascending=False)


def find_globally_top_metrics(
    results: Dict[str, Dict]
) -> List[str]:
    """
    Find metrics that are in the top group for ALL datasets.

    Args:
        results: Dict mapping dataset name to analysis results

    Returns:
        List of globally top-performing metrics
    """
    if not results:
        return []

    # Get top metrics for each dataset
    top_sets = [
        set(r['top_metrics']) for r in results.values()
    ]

    # Intersection of all top metric sets
    global_top = top_sets[0]
    for top_set in top_sets[1:]:
        global_top = global_top.intersection(top_set)

    return sorted(list(global_top))


def generate_latex_table(
    results: Dict,
    dataset_name: str,
    bold_top: bool = True,
    decimals: int = 3
) -> str:
    """
    Generate LaTeX table for publication.

    Args:
        results: Analysis results from compute_significance_analysis
        dataset_name: Name for table caption
        bold_top: Bold the top-performing metrics
        decimals: Decimal places for correlations

    Returns:
        LaTeX table string
    """
    corr_df = results['correlations_df']
    top_metrics = set(results['top_metrics'])

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Correlation with human scores ({dataset_name})}}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Metric & Correlation & 95\\% CI & $p$-value & $n$ \\\\")
    lines.append("\\midrule")

    for _, row in corr_df.iterrows():
        metric = row['metric']
        r = row['correlation']
        ci_l = row['ci_lower']
        ci_u = row['ci_upper']
        p = row['p_value']
        n = row['n']

        # Format values
        r_str = f"{r:.{decimals}f}"
        ci_str = f"[{ci_l:.{decimals}f}, {ci_u:.{decimals}f}]"
        p_str = f"{p:.4f}" if p >= 0.0001 else "$<$0.0001"

        # Bold if top metric
        if bold_top and metric in top_metrics:
            r_str = f"\\textbf{{{r_str}}}"
            metric = f"\\textbf{{{metric}}}"

        lines.append(f"{metric} & {r_str} & {ci_str} & {p_str} & {int(n)} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


if __name__ == '__main__':
    # Quick test with synthetic data
    np.random.seed(42)
    n = 100

    # Simulate human scores
    human = np.random.randn(n)

    # Simulate three metrics with different correlations to human
    metric_a = 0.8 * human + 0.2 * np.random.randn(n)  # Strong positive
    metric_b = -0.7 * human + 0.3 * np.random.randn(n)  # Strong negative
    metric_c = 0.3 * human + 0.7 * np.random.randn(n)  # Weak

    metric_scores = {
        'Metric_A': metric_a,
        'Metric_B': metric_b,
        'Metric_C': metric_c,
    }

    print("Significance Analysis Test")
    print("=" * 50)

    correlations_df, p_value_matrix, significance_matrix = compute_significance_matrix(
        metric_scores, human
    )

    print("\nCorrelations with human scores (sorted by |r|):")
    print(correlations_df[['correlation', 'p_value', 'ci_lower', 'ci_upper']])

    print(f"\nTop metrics: {identify_top_metrics(significance_matrix)}")

    print("\nSignificance matrix:")
    print(significance_matrix)
