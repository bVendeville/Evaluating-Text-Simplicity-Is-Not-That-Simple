"""
Visualization module for heatmaps, tables, and plots.
"""

from .heatmaps import (
    plot_correlation_heatmap,
    plot_williams_heatmap,
    plot_combined_heatmaps,
    plot_metric_comparison_heatmap,
)

from .tables import (
    generate_correlation_latex,
    generate_combined_results_latex,
    generate_significance_summary_latex,
    generate_model_comparison_latex,
    generate_markdown_summary,
    save_latex_table,
)

from .plots import (
    plot_metric_comparison_bars,
    plot_model_comparison,
    plot_scatter_with_regression,
    plot_scatter_grid,
    plot_ranking_comparison,
    set_publication_style,
    COLORS,
    MODEL_COLORS,
)

__all__ = [
    # Heatmaps
    'plot_correlation_heatmap',
    'plot_williams_heatmap',
    'plot_combined_heatmaps',
    'plot_metric_comparison_heatmap',
    # Tables
    'generate_correlation_latex',
    'generate_combined_results_latex',
    'generate_significance_summary_latex',
    'generate_model_comparison_latex',
    'generate_markdown_summary',
    'save_latex_table',
    # Plots
    'plot_metric_comparison_bars',
    'plot_model_comparison',
    'plot_scatter_with_regression',
    'plot_scatter_grid',
    'plot_ranking_comparison',
    'set_publication_style',
    'COLORS',
    'MODEL_COLORS',
]
