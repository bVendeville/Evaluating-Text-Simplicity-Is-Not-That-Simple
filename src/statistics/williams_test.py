"""
Williams significance test for comparing dependent correlations.

This module implements the Williams test (Williams, 1959) for testing whether
one correlation is significantly greater than another when both correlations
share a common variable (e.g., comparing two metrics against human scores).

Reference:
Williams, E. J. (1959). Regression Analysis, volume 14. Wiley, New York.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def williams_test(
    r12: float,
    r13: float,
    r23: float,
    n: int
) -> Tuple[float, float]:
    """
    Williams test for comparing two dependent correlations.

    Tests if correlation r12 (metric A vs human) is significantly GREATER than
    r13 (metric B vs human), given r23 (correlation between metrics A and B).

    Args:
        r12: Correlation between metric A and human scores
        r13: Correlation between metric B and human scores
        r23: Correlation between metric A and metric B
        n: Sample size

    Returns:
        t_stat: Test statistic
        p_value: One-tailed p-value (significant if < 0.05)
    """
    if n <= 3:
        return np.nan, np.nan

    # If r12 <= r13, metric A is not better
    if r12 <= r13:
        return 0.0, 0.5

    # Compute K (determinant-related term)
    K = 1 - r12**2 - r13**2 - r23**2 + 2 * r12 * r13 * r23
    if K < 0:
        K = 0

    # Compute denominator
    denominator_sq = (
        2 * K * (n - 1) / (n - 3) +
        (((r12 + r13)**2) / 4) * ((1 - r23)**3)
    )

    if denominator_sq <= 0:
        return np.nan, np.nan

    denominator = np.sqrt(denominator_sq)

    # Compute t-statistic
    t_stat = (r12 - r13) * np.sqrt((n - 1) * (1 + r23)) / denominator

    # One-tailed p-value (testing if r12 > r13)
    p_value = 1 - scipy_stats.t.cdf(t_stat, df=n - 3)

    return t_stat, p_value


def williams_test_symmetric(
    r12: float,
    r13: float,
    r23: float,
    n: int
) -> Tuple[float, float]:
    """
    Two-tailed Williams test for testing if correlations are different.

    Args:
        r12: Correlation between metric A and human scores
        r13: Correlation between metric B and human scores
        r23: Correlation between metric A and metric B
        n: Sample size

    Returns:
        t_stat: Test statistic
        p_value: Two-tailed p-value
    """
    t_stat, p_one = williams_test(r12, r13, r23, n)

    if np.isnan(t_stat):
        return np.nan, np.nan

    # Two-tailed p-value
    p_value = 2 * min(p_one, 1 - p_one)

    return t_stat, p_value


def compute_pairwise_williams(
    metric_correlations: Dict[str, float],
    inter_metric_correlations: pd.DataFrame,
    n: int,
    alpha: float = 0.05
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise Williams tests for all metric pairs.

    Uses ABSOLUTE values for correlations to handle metrics with negative
    correlations (e.g., complexity metrics on ARTS datasets).

    Args:
        metric_correlations: Dict mapping metric name to correlation with human
        inter_metric_correlations: DataFrame of inter-metric correlations
        n: Sample size
        alpha: Significance level

    Returns:
        p_value_matrix: DataFrame of Williams test p-values (row > col)
        significance_matrix: Boolean DataFrame (True if row significantly > col)
    """
    metrics = list(metric_correlations.keys())
    n_metrics = len(metrics)

    p_values = np.full((n_metrics, n_metrics), np.nan)
    significance = np.full((n_metrics, n_metrics), False)

    for i, metric_a in enumerate(metrics):
        for j, metric_b in enumerate(metrics):
            if i == j:
                continue

            # Use ABSOLUTE values for Williams test
            r12 = abs(metric_correlations[metric_a])  # |Metric A vs Human|
            r13 = abs(metric_correlations[metric_b])  # |Metric B vs Human|
            r23 = abs(inter_metric_correlations.loc[metric_a, metric_b])  # |A vs B|

            _, p_value = williams_test(r12, r13, r23, n)

            p_values[i, j] = p_value
            significance[i, j] = p_value < alpha

    p_value_df = pd.DataFrame(p_values, index=metrics, columns=metrics)
    significance_df = pd.DataFrame(significance, index=metrics, columns=metrics)

    return p_value_df, significance_df


def identify_top_metrics(
    significance_matrix: pd.DataFrame
) -> List[str]:
    """
    Identify metrics that are not significantly outperformed by any other metric.

    A metric is "top" if no other metric is significantly better than it.

    Args:
        significance_matrix: Boolean DataFrame where (i,j)=True means
                            metric i is significantly > metric j

    Returns:
        List of top metric names
    """
    metrics = list(significance_matrix.columns)
    top_metrics = []

    for metric in metrics:
        # Check if any other metric significantly beats this one
        is_beaten = any(
            significance_matrix.loc[other, metric]
            for other in metrics
            if other != metric
        )
        if not is_beaten:
            top_metrics.append(metric)

    return top_metrics


def count_significant_wins(
    significance_matrix: pd.DataFrame
) -> pd.Series:
    """
    Count how many metrics each metric significantly outperforms.

    Args:
        significance_matrix: Boolean significance matrix

    Returns:
        Series with win counts per metric
    """
    return significance_matrix.sum(axis=1)


def count_significant_losses(
    significance_matrix: pd.DataFrame
) -> pd.Series:
    """
    Count how many metrics significantly outperform each metric.

    Args:
        significance_matrix: Boolean significance matrix

    Returns:
        Series with loss counts per metric
    """
    return significance_matrix.sum(axis=0)


def rank_metrics_by_wins(
    significance_matrix: pd.DataFrame
) -> pd.DataFrame:
    """
    Rank metrics by number of significant wins (ties broken by losses).

    Args:
        significance_matrix: Boolean significance matrix

    Returns:
        DataFrame with metric, wins, losses, net_wins columns
    """
    wins = count_significant_wins(significance_matrix)
    losses = count_significant_losses(significance_matrix)
    net = wins - losses

    df = pd.DataFrame({
        'metric': wins.index,
        'wins': wins.values,
        'losses': losses.values,
        'net_wins': net.values,
    })

    df = df.sort_values(['wins', 'losses'], ascending=[False, True])
    df['rank'] = range(1, len(df) + 1)

    return df


def summarize_williams_results(
    metric_correlations: Dict[str, float],
    p_value_matrix: pd.DataFrame,
    significance_matrix: pd.DataFrame,
    alpha: float = 0.05
) -> Dict:
    """
    Create a summary of Williams test results.

    Args:
        metric_correlations: Correlations with human scores
        p_value_matrix: Williams test p-values
        significance_matrix: Significance matrix
        alpha: Significance level used

    Returns:
        Dict with summary statistics and rankings
    """
    top_metrics = identify_top_metrics(significance_matrix)
    rankings = rank_metrics_by_wins(significance_matrix)

    # Find best metric by correlation
    best_by_corr = max(metric_correlations, key=metric_correlations.get)

    return {
        'n_metrics': len(metric_correlations),
        'alpha': alpha,
        'top_metrics': top_metrics,
        'n_top_metrics': len(top_metrics),
        'best_by_correlation': best_by_corr,
        'best_correlation': metric_correlations[best_by_corr],
        'rankings': rankings,
        'is_best_in_top': best_by_corr in top_metrics,
    }


if __name__ == '__main__':
    # Quick test
    print("Williams Test Example:")
    print("-" * 40)

    # Example: comparing two metrics against human scores
    r_metric_a_human = 0.85  # Metric A vs Human
    r_metric_b_human = 0.75  # Metric B vs Human
    r_metric_a_b = 0.60      # Metric A vs Metric B
    n = 100                   # Sample size

    t_stat, p_value = williams_test(
        r_metric_a_human,
        r_metric_b_human,
        r_metric_a_b,
        n
    )

    print(f"Metric A vs Human: r = {r_metric_a_human}")
    print(f"Metric B vs Human: r = {r_metric_b_human}")
    print(f"Metric A vs B: r = {r_metric_a_b}")
    print(f"Sample size: n = {n}")
    print()
    print(f"Williams t-statistic: {t_stat:.3f}")
    print(f"One-tailed p-value: {p_value:.4f}")
    print(f"Metric A significantly better: {p_value < 0.05}")
