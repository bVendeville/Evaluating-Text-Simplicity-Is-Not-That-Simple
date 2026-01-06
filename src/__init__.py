"""
Paper Materials source package.

Modules:
- data_loader: Dataset loading utilities
- statistics: Statistical analysis (correlation, Williams test)
- visualization: Plotting and table generation
"""

from . import data_loader
from . import statistics
from . import visualization

__all__ = [
    'data_loader',
    'statistics',
    'visualization',
]
