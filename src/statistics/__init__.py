"""
Statistics module for correlation and significance analysis.
"""

from .correlation import (
    pearson_ci,
    spearman_ci,
    kendall_ci,
    compute_correlation,
    compute_correlation_matrix,
    compute_metric_correlations,
    compute_all_correlations,
    compute_inter_metric_correlations,
)

from .williams_test import (
    williams_test,
    williams_test_symmetric,
    compute_pairwise_williams,
    count_significant_wins,
    count_significant_losses,
    rank_metrics_by_wins,
)

from .significance_matrix import (
    compute_significance_matrix,
    compute_significance_analysis,
    identify_top_metrics,
    analyze_multiple_datasets,
    combine_dataset_results,
    find_globally_top_metrics,
    generate_latex_table,
)

__all__ = [
    # Correlation
    'pearson_ci',
    'spearman_ci',
    'kendall_ci',
    'compute_correlation',
    'compute_correlation_matrix',
    'compute_metric_correlations',
    'compute_all_correlations',
    'compute_inter_metric_correlations',
    # Williams test
    'williams_test',
    'williams_test_symmetric',
    'compute_pairwise_williams',
    'count_significant_wins',
    'count_significant_losses',
    'rank_metrics_by_wins',
    # Significance analysis
    'compute_significance_matrix',
    'compute_significance_analysis',
    'identify_top_metrics',
    'analyze_multiple_datasets',
    'combine_dataset_results',
    'find_globally_top_metrics',
    'generate_latex_table',
]
