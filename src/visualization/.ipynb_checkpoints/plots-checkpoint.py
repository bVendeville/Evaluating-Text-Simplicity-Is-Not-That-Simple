"""
General plotting utilities.

This module provides functions for creating:
- Bar charts for model/metric comparison
- Scatter plots for metric vs human correlations
- Performance comparison plots
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Color palette for consistent styling
COLORS = {
    'primary': '#2ecc71',
    'secondary': '#3498db',
    'accent': '#e74c3c',
    'neutral': '#95a5a6',
    'dark': '#2c3e50',
}

MODEL_COLORS = {
    'Ridge': '#3498db',
    'RandomForest': '#2ecc71',
    'XGBoost': '#e74c3c',
    'LightGBM': '#9b59b6',
    'GradientBoosting': '#f39c12',
    'SVR': '#1abc9c',
    'MLP': '#e67e22',
}


def plot_metric_comparison_bars(
    correlations: Dict[str, float],
    title: str = "Metric Comparison",
    xlabel: str = "Correlation with Human Scores",
    figsize: Tuple[int, int] = (10, 6),
    color: str = None,
    highlight_top: int = 0,
    horizontal: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot bar chart comparing metric correlations.

    Args:
        correlations: Dict mapping metric name to correlation value
        title: Plot title
        xlabel: X-axis label
        figsize: Figure size
        color: Bar color (None for auto)
        highlight_top: Number of top metrics to highlight
        horizontal: Horizontal bars (True) or vertical (False)
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    # Sort by correlation
    sorted_items = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    metrics = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    # Colors
    if color is None:
        colors = [COLORS['primary'] if i < highlight_top else COLORS['secondary']
                  for i in range(len(metrics))]
    else:
        colors = [color] * len(metrics)

    fig, ax = plt.subplots(figsize=figsize)

    if horizontal:
        y_pos = range(len(metrics))
        bars = ax.barh(y_pos, values, color=colors, edgecolor='black', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics)
        ax.set_xlabel(xlabel)
        ax.invert_yaxis()  # Top metric at top
    else:
        x_pos = range(len(metrics))
        bars = ax.bar(x_pos, values, color=colors, edgecolor='black', alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_ylabel(xlabel)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x' if horizontal else 'y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, values):
        if horizontal:
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f'{val:.3f}', ha='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = 'R2',
    title: str = "Model Performance Comparison",
    figsize: Tuple[int, int] = (12, 6),
    show_error_bars: bool = True,
    std_col: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot model comparison bar chart.

    Args:
        results_df: DataFrame with model results
        metric: Metric column to plot
        title: Plot title
        figsize: Figure size
        show_error_bars: Show standard deviation error bars
        std_col: Column name for std (default: metric + '_std')
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    if 'Model' in results_df.columns:
        models = results_df['Model'].tolist()
    else:
        models = results_df.index.tolist()

    values = results_df[metric].values

    if std_col is None:
        std_col = f'{metric}_std'
    stds = results_df[std_col].values if std_col in results_df.columns else None

    # Colors based on model name
    colors = [MODEL_COLORS.get(m, COLORS['neutral']) for m in models]

    fig, ax = plt.subplots(figsize=figsize)

    x_pos = range(len(models))
    bars = ax.bar(x_pos, values, color=colors, edgecolor='black', alpha=0.8)

    if show_error_bars and stds is not None:
        ax.errorbar(x_pos, values, yerr=stds, fmt='none', color='black', capsize=5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel(metric)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f'{val:.3f}', ha='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_scatter_with_regression(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str = "Predicted",
    ylabel: str = "Human Score",
    title: str = "Metric vs Human Score",
    figsize: Tuple[int, int] = (8, 6),
    show_regression: bool = True,
    show_identity: bool = False,
    color: str = None,
    alpha: float = 0.6,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot scatter plot with optional regression line.

    Args:
        x: X-axis values (e.g., predicted scores)
        y: Y-axis values (e.g., human scores)
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        figsize: Figure size
        show_regression: Show regression line
        show_identity: Show y=x identity line
        color: Point color
        alpha: Point transparency
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    c = color or COLORS['primary']
    ax.scatter(x, y, c=c, alpha=alpha, edgecolors='white', s=50)

    if show_regression:
        # Fit regression line
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean, y_clean = x[mask], y[mask]
        z = np.polyfit(x_clean, y_clean, 1)
        p = np.poly1d(z)

        x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
        ax.plot(x_line, p(x_line), color=COLORS['accent'], linewidth=2,
                label=f'Regression (slope={z[0]:.2f})')

    if show_identity:
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, '--', color=COLORS['neutral'], alpha=0.8, label='Identity')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_scatter_grid(
    data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_cols: int = 3,
    figsize_per_plot: Tuple[int, int] = (4, 4),
    title: str = "Metric vs Human Score Comparison",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot grid of scatter plots for multiple metrics.

    Args:
        data: Dict mapping metric name to (predictions, human_scores) tuple
        n_cols: Number of columns in grid
        figsize_per_plot: Size per subplot
        title: Figure title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    n_metrics = len(data)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows),
    )

    # Flatten axes
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for idx, (metric_name, (pred, human)) in enumerate(data.items()):
        ax = axes[idx]

        # Remove NaN
        mask = ~(np.isnan(pred) | np.isnan(human))
        x, y = pred[mask], human[mask]

        ax.scatter(x, y, alpha=0.5, s=30, c=COLORS['primary'])

        # Regression line
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 50)
            ax.plot(x_line, p(x_line), color=COLORS['accent'], linewidth=1.5)

            # Calculate correlation
            r = np.corrcoef(x, y)[0, 1]
            ax.set_title(f"{metric_name}\nr = {r:.3f}", fontsize=10)
        else:
            ax.set_title(metric_name, fontsize=10)

        ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_ranking_comparison(
    rankings: pd.DataFrame,
    title: str = "Metric Rankings",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot metric rankings with wins/losses.

    Args:
        rankings: DataFrame with metric, wins, losses columns
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    metrics = rankings['metric'].tolist()
    wins = rankings['wins'].values
    losses = rankings['losses'].values
    y_pos = range(len(metrics))

    # Stacked horizontal bars
    bars_wins = ax.barh(y_pos, wins, color=COLORS['primary'],
                        label='Significant Wins', edgecolor='black', alpha=0.8)
    bars_losses = ax.barh(y_pos, -losses, color=COLORS['accent'],
                          label='Significant Losses', edgecolor='black', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics)
    ax.set_xlabel('Count')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


if __name__ == '__main__':
    print("Testing plot functions...")

    # Test bar chart
    correlations = {
        'Metric A': 0.85,
        'Metric B': 0.72,
        'Metric C': 0.68,
        'Metric D': 0.55,
    }
    fig = plot_metric_comparison_bars(correlations, highlight_top=2)
    plt.close(fig)
    print("  Bar chart: OK")

    # Test scatter
    np.random.seed(42)
    x = np.random.randn(100)
    y = 0.7 * x + 0.3 * np.random.randn(100)
    fig = plot_scatter_with_regression(x, y)
    plt.close(fig)
    print("  Scatter plot: OK")

    print("All tests passed!")
