"""
Heatmap visualization utilities.

This module provides functions for creating:
- Correlation heatmaps
- Williams significance heatmaps
- Combined multi-dataset heatmaps
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'coolwarm',
    annot: bool = True,
    fmt: str = '.2f',
    center: float = 0,
    vmin: Optional[float] = -1,
    vmax: Optional[float] = 1,
    square: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a correlation heatmap.

    Args:
        corr_matrix: Correlation matrix DataFrame
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        annot: Show annotations
        fmt: Annotation format
        center: Center value for colormap
        vmin, vmax: Color scale limits
        square: Square cells
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        corr_matrix,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        square=square,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_williams_heatmap(
    p_value_matrix: pd.DataFrame,
    alpha: float = 0.05,
    title: str = "Williams Test Significance",
    figsize: Tuple[int, int] = (10, 8),
    annot: bool = True,
    show_p_values: bool = True,
    show_only_significant: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot Williams test significance heatmap.

    Cell (i,j) shows whether metric i significantly outperforms metric j.
    Green = highly significant, Orange = less significant.

    Args:
        p_value_matrix: DataFrame of Williams test p-values
        alpha: Significance threshold
        title: Plot title
        figsize: Figure size
        annot: Show annotations
        show_p_values: If True, show p-values; if False, show significance markers
        show_only_significant: If True, only show significant cells (p < alpha)
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Apply significance masking
    matrix = p_value_matrix.copy().astype(float)
    if show_only_significant:
        matrix = matrix.where(matrix < alpha)

    # Custom color palette (green to orange)
    palette_list = ["#00a600", "#bddc00", "#eebaa0"]
    palette = sns.blend_palette(palette_list, n_colors=11)

    # Dynamic font sizes based on matrix size
    n = matrix.shape[0]
    if n <= 12:
        label_fs, title_fs, annot_fs = 12, 14, 9
    else:
        label_fs, title_fs, annot_fs = 8, 12, 6

    sns.heatmap(
        matrix,
        annot=annot if show_p_values else False,
        fmt='.3f',
        cmap=palette,
        center=0.01,
        vmin=0,
        vmax=alpha,
        square=True,
        linewidths=0.5,
        linecolor='lightgray',
        cbar_kws={"shrink": 0.8, "label": "p-value"},
        annot_kws={"size": annot_fs},
        ax=ax,
    )

    # Force all ticks to be shown
    ax.set_xticks(np.arange(len(matrix.columns)) + 0.5)
    ax.set_yticks(np.arange(len(matrix.index)) + 0.5)
    ax.set_xticklabels(matrix.columns, rotation=90, ha='right', fontsize=label_fs)
    ax.set_yticklabels(matrix.index, rotation=0, fontsize=label_fs)

    ax.set_title(title, fontsize=title_fs, fontweight='bold')
    ax.set_xlabel("Column metric")
    ax.set_ylabel("Row metric")

    # Add note
    ax.text(
        0.5, -0.1,
        f"Cell (i,j): Row metric significantly > Column metric (p < {alpha})",
        transform=ax.transAxes,
        ha='center',
        fontsize=9,
        style='italic',
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_combined_heatmaps(
    results: Dict[str, Dict],
    n_cols: int = 3,
    figsize_per_plot: Tuple[int, int] = (8, 7),
    alpha: float = 0.05,
    annotate_cell: bool = False,
    selected_metrics: Optional[List[str]] = None,
    selected_datasets: Optional[List[str]] = None,
    sort_by_wins: bool = False,
    general_fig_title: bool = False,
    title: str = "Significance Heatmaps - Comparison Across Datasets",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot Williams heatmaps for multiple datasets in a grid.

    Matches the visualization style from 08_significance_analysis.ipynb:
    - Only displays significant values (p < alpha)
    - Custom green-to-orange color palette
    - Inverted colorbar (lower p-values at top)
    - Consistent scale across all datasets

    Args:
        results: Dict mapping dataset name to analysis results (must have 'p_value_matrix')
        n_cols: Number of columns in grid
        figsize_per_plot: Size per subplot
        alpha: Significance threshold
        annotate_cell: Show p-values in cells
        selected_metrics: Subset of metrics to show (None = all)
        selected_datasets: Subset/ordering of datasets (None = all)
        sort_by_wins: Sort metrics by number of significant wins
        general_fig_title: Show overall figure title
        title: Figure title (if general_fig_title=True)
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    # Use selected_datasets if provided, otherwise use all datasets
    if selected_datasets:
        dataset_names = [d for d in selected_datasets if d in results]
    else:
        dataset_names = list(results.keys())
    n_datasets = len(dataset_names)

    # Calculate grid dimensions
    n_rows = (n_datasets + n_cols - 1) // n_cols

    # Set up the figure size dynamically
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows))

    if general_fig_title:
        fig.suptitle(
            title,
            fontsize=16, fontweight='bold', y=0.98 if n_rows > 1 else 1.05
        )

    # Flatten axes array for easy iteration
    if n_datasets > 1:
        if n_rows == 1:
            axes_flat = list(axes) if n_cols > 1 else [axes]
        else:
            axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    # --- Compute global vmin and vmax across all datasets ---
    all_values = []

    for dataset_name in dataset_names:
        dataset_results = results[dataset_name]
        p_matrix = dataset_results['p_value_matrix']
        matrix = p_matrix.astype(float)

        if selected_metrics:
            valid_metrics = [m for m in selected_metrics if m in matrix.index]
            matrix = matrix.loc[valid_metrics, valid_metrics]

        matrix = matrix.where(matrix < alpha)
        all_values.append(matrix.values.flatten())

    if all_values:
        all_values = np.concatenate(all_values)
        global_vmin = 0
        global_vmax = min(np.nanmax(all_values), alpha)  # cap at alpha
    else:
        global_vmin, global_vmax = 0, alpha

    # Custom color palette (green to orange)
    palette_list = ["#00a600", "#bddc00", "#eebaa0"]

    # --- Plot each dataset ---
    for idx, dataset_name in enumerate(dataset_names):
        ax = axes_flat[idx]

        dataset_results = results[dataset_name]
        p_matrix = dataset_results['p_value_matrix'].copy().astype(float)

        # Filter metrics if requested
        if selected_metrics:
            valid_metrics = [m for m in selected_metrics if m in p_matrix.index]
            p_matrix = p_matrix.loc[valid_metrics, valid_metrics]

        # Sort by number of significant wins if requested
        if sort_by_wins:
            wins = (p_matrix < alpha).sum(axis=1)
            sorted_order = wins.sort_values(ascending=False).index.tolist()
            p_matrix = p_matrix.loc[sorted_order, sorted_order]

        # Only show significant values
        matrix = p_matrix.where(p_matrix < alpha)

        n = matrix.shape[0]
        i = 6
        if n <= 12:  # 10x10 matrices
            label_fs, title_fs, annot_fs = 14 + i, 16 + i, 10 + i
        else:        # 22x22 matrices
            label_fs, title_fs, annot_fs = 7 + i, 14 + i, 5 + i

        palette = sns.blend_palette(palette_list, n_colors=11)

        sns.heatmap(
            matrix,
            cmap=palette,
            center=0.01,
            vmin=global_vmin,
            vmax=global_vmax,
            linewidth=0.5,
            linecolor='lightgray',
            cbar=False,
            square=True,
            annot=annotate_cell,
            fmt=".3f",
            annot_kws={"size": annot_fs},
            ax=ax
        )

        # Force all x and y ticks to be shown
        ax.set_xticks(np.arange(len(matrix.columns)) + 0.5)
        ax.set_yticks(np.arange(len(matrix.index)) + 0.5)

        ax.set_xticklabels(matrix.columns, rotation=90, ha='right', fontsize=label_fs)
        ax.set_yticklabels(matrix.index, rotation=0, fontsize=label_fs)

        # Disable automatic tick pruning
        ax.tick_params(axis='x', bottom=True, labelbottom=True)
        ax.tick_params(axis='y', left=True, labelleft=True)

        # Title with sample size
        n_samples = dataset_results.get('n_samples', '?')
        ax.set_title(f"{dataset_name} (n={n_samples})", fontsize=title_fs)
        plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=label_fs)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=label_fs)

    # Hide unused subplots
    for i in range(idx + 1, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout(rect=[0, 0, 0.93, 1])  # reserves 7% on the right for colorbar

    # Create reversed color palette manually
    palette = sns.blend_palette(
        palette_list,
        n_colors=50
    )

    # Normalize the values from 0 to 0.05 (best at top, worst at bottom)
    norm = mpl.colors.Normalize(vmin=0, vmax=alpha)

    # Create the ScalarMappable object for the colorbar
    sm = mpl.cm.ScalarMappable(cmap=mpl.colors.ListedColormap(palette), norm=norm)
    sm.set_array([])

    # Create a full-height colorbar
    cbar_ax = fig.add_axes([0.94, 0.12, 0.015, 0.80])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')

    # Invert colorbar so that the best (0) is at the top and the worst (0.05) is at the bottom
    cbar.ax.invert_yaxis()

    # Set explicit ticks for cleaner labels
    ticks = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
    cbar.set_ticks(ticks)

    # Set the colorbar label
    cbar.set_label('p-value', fontsize=18)

    # Format colorbar tick labels
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_metric_comparison_heatmap(
    combined_df: pd.DataFrame,
    title: str = "Metric Correlations Across Datasets",
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = 'RdYlGn',
    annot: bool = True,
    fmt: str = '.3f',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot heatmap comparing metric performance across datasets.

    Args:
        combined_df: DataFrame with metrics as rows, datasets as columns
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        annot: Show annotations
        fmt: Annotation format
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        combined_df,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        center=0.5,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Metric")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


if __name__ == '__main__':
    # Quick test with synthetic data
    np.random.seed(42)

    # Create fake correlation matrix
    metrics = ['Metric_A', 'Metric_B', 'Metric_C', 'Metric_D']
    corr = np.random.uniform(0.3, 0.9, (4, 4))
    np.fill_diagonal(corr, 1.0)
    corr = (corr + corr.T) / 2  # Symmetrize
    corr_df = pd.DataFrame(corr, index=metrics, columns=metrics)

    print("Testing correlation heatmap...")
    fig = plot_correlation_heatmap(corr_df, title="Test Correlation Heatmap")
    plt.close(fig)
    print("  Success!")

    # Create fake p-value matrix
    p_values = np.random.uniform(0, 0.2, (4, 4))
    np.fill_diagonal(p_values, 1.0)
    p_df = pd.DataFrame(p_values, index=metrics, columns=metrics)

    print("Testing Williams heatmap...")
    fig = plot_williams_heatmap(p_df, title="Test Williams Heatmap")
    plt.close(fig)
    print("  Success!")
