"""
Correlation analysis utilities.

This module provides functions for computing correlations and confidence intervals:
- Pearson correlation with Fisher's z-transformation CI
- Spearman rank correlation with adjusted CI
- Correlation tables across datasets and metrics
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def pearson_ci(r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Compute confidence interval for Pearson correlation using Fisher's z-transformation.

    Args:
        r: Pearson correlation coefficient
        n: Sample size
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        (lower, upper): Confidence interval bounds
    """
    if n <= 3 or np.isnan(r):
        return np.nan, np.nan

    # Fisher's z-transformation
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = scipy_stats.norm.ppf(1 - alpha / 2)

    # Back-transform to correlation scale
    lower = np.tanh(z - z_crit * se)
    upper = np.tanh(z + z_crit * se)

    return lower, upper


def spearman_ci(r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Compute confidence interval for Spearman correlation.
    Uses Fisher z-transformation with Bonett & Wright (2000) SE adjustment.

    Args:
        r: Spearman correlation coefficient
        n: Sample size
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        (lower, upper): Confidence interval bounds
    """
    if n <= 3 or np.isnan(r):
        return np.nan, np.nan

    z = np.arctanh(r)
    # Adjusted SE for Spearman (Bonett & Wright, 2000)
    se = np.sqrt((1 + r**2 / 2) / (n - 3))
    z_crit = scipy_stats.norm.ppf(1 - alpha / 2)

    lower = np.tanh(z - z_crit * se)
    upper = np.tanh(z + z_crit * se)

    return lower, upper


def compute_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str = 'pearson',
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Compute correlation with p-value and confidence interval.

    Args:
        x: First array
        y: Second array
        method: 'pearson' or 'spearman'
        alpha: Significance level for CI

    Returns:
        Dict with keys: correlation, p_value, ci_lower, ci_upper, n
    """
    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    n = len(x_clean)

    if n < 3:
        return {
            'correlation': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n': n,
        }

    if method == 'pearson':
        r, p = scipy_stats.pearsonr(x_clean, y_clean)
        ci_lower, ci_upper = pearson_ci(r, n, alpha)
    else:  # spearman
        r, p = scipy_stats.spearmanr(x_clean, y_clean)
        ci_lower, ci_upper = spearman_ci(r, n, alpha)

    return {
        'correlation': r,
        'p_value': p,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n': n,
    }


def compute_correlation_matrix(
    data: pd.DataFrame,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Compute pairwise correlation matrix for all columns.

    Args:
        data: DataFrame with numeric columns
        method: 'pearson' or 'spearman'

    Returns:
        Correlation matrix DataFrame
    """
    if method == 'pearson':
        return data.corr(method='pearson')
    else:
        return data.corr(method='spearman')


def compute_metric_correlations(
    metric_scores: Dict[str, np.ndarray],
    human_scores: np.ndarray,
    method: str = 'pearson',
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Compute correlations between multiple metrics and human scores.

    Args:
        metric_scores: Dict mapping metric name to score array
        human_scores: Human judgment scores
        method: 'pearson' or 'spearman'
        alpha: Significance level for CI

    Returns:
        DataFrame with correlation results for each metric
    """
    results = []

    for metric_name, scores in metric_scores.items():
        result = compute_correlation(
            scores,
            human_scores,
            method=method,
            alpha=alpha
        )
        result['metric'] = metric_name
        results.append(result)

    df = pd.DataFrame(results)
    df = df[['metric', 'correlation', 'p_value', 'ci_lower', 'ci_upper', 'n']]
    # Sort by absolute correlation (strongest relationship first)
    df = df.sort_values('correlation', key=abs, ascending=False)

    return df


def compute_inter_metric_correlations(
    metric_scores: Dict[str, np.ndarray],
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Compute pairwise correlations between all metrics.

    Args:
        metric_scores: Dict mapping metric name to score array
        method: 'pearson' or 'spearman'

    Returns:
        DataFrame correlation matrix
    """
    df = pd.DataFrame(metric_scores)
    return compute_correlation_matrix(df, method=method)


def format_correlation_with_ci(
    r: float,
    ci_lower: float,
    ci_upper: float,
    decimals: int = 3
) -> str:
    """Format correlation with CI as string: r [lower, upper]"""
    if np.isnan(r):
        return "N/A"
    return f"{r:.{decimals}f} [{ci_lower:.{decimals}f}, {ci_upper:.{decimals}f}]"


def format_correlation_table(
    results: pd.DataFrame,
    include_ci: bool = True,
    include_p: bool = False,
    decimals: int = 3
) -> pd.DataFrame:
    """
    Format correlation results for display/export.

    Args:
        results: DataFrame with correlation results
        include_ci: Include confidence intervals
        include_p: Include p-values
        decimals: Decimal places

    Returns:
        Formatted DataFrame
    """
    formatted = results.copy()

    if include_ci:
        formatted['correlation_ci'] = formatted.apply(
            lambda row: format_correlation_with_ci(
                row['correlation'], row['ci_lower'], row['ci_upper'], decimals
            ),
            axis=1
        )

    # Round numeric columns
    for col in ['correlation', 'p_value', 'ci_lower', 'ci_upper']:
        if col in formatted.columns:
            formatted[col] = formatted[col].round(decimals)

    # Select columns to display
    cols = ['metric']
    if include_ci:
        cols.append('correlation_ci')
    else:
        cols.append('correlation')
    if include_p:
        cols.append('p_value')
    cols.append('n')

    return formatted[cols]


if __name__ == '__main__':
    # Quick test
    np.random.seed(42)
    x = np.random.randn(100)
    y = 0.7 * x + 0.3 * np.random.randn(100)

    result = compute_correlation(x, y, method='pearson')
    print("Pearson correlation test:")
    print(f"  r = {result['correlation']:.3f}")
    print(f"  p = {result['p_value']:.4f}")
    print(f"  95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
