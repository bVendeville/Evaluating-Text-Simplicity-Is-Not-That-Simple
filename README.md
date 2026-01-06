# Evaluating Text Simplicity Is Not That Simple

Code and data for evaluating automatic text simplicity metrics against human judgments.

## Overview

This repository provides:
- Pre-computed metric scores for 10 datasets (7 original + 3 length-residualized)
- Statistical analysis code (correlations, Williams significance test)
- Publication-quality visualization generation

## Quick Start

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook
```

## Datasets

| Dataset | Samples | Score Range | Description |
|---------|---------|-------------|-------------|
| ARTS94 | 94 | [0, 1] | ARTS small subset |
| ARTS300 | 300 | [0, 1] | ARTS medium subset |
| ARTS3k | 3000 | [0, 1] | ARTS full dataset |
| LR-ARTS94 | 94 | residuals | Length-residualized ARTS94 |
| LR-ARTS300 | 300 | residuals | Length-residualized ARTS300 |
| LR-ARTS3k | 3000 | residuals | Length-residualized ARTS3k |
| SDA | 600 | DA scores | Simplicity Direct Assessment |
| ST-sent | 1620 | [1, 7] | SimpleText sentence-level |
| ST-para | 271 | [1, 7] | SimpleText paragraph-level |
| D-Wiki | 601 | [1, 5] | D-Wikipedia |

**LR-ARTS datasets**: Human scores with length (word count) effects regressed out, revealing which metrics capture simplicity beyond text length.

## Metrics Evaluated

### Baselines
- **Random**: Uniform random scores
- **Shuffle**: Permuted human scores

### Readability
- **FKGL**: Flesch-Kincaid Grade Level (negated: higher = simpler)

### Neural Metrics
- **SLE**: Simplicity Level Estimate (RoBERTa-based)
- **LENS-SALSA**: Reference-free LENS metric
- **BATS-GB/RF**: BATS with Gradient Boosting/Random Forest

### LLM-based
- **Claude**: Claude Haiku simplicity scores (with variants: simp, simp+src, simp+ref, simp+src+ref)

### Reference-based (where applicable)
- **SARI**: System output Against References and Input
- **BERTScore**: BERT-based similarity
- **BLEU**, **ROUGE-L**: N-gram overlap metrics

## Repository Structure

```
data/
├── arts94.pkl          # ARTS94 metric scores
├── arts300.pkl         # ARTS300 metric scores
├── arts3k.pkl          # ARTS3k metric scores
├── lr_arts94.pkl       # Length-residualized ARTS94
├── lr_arts300.pkl      # Length-residualized ARTS300
├── lr_arts3k.pkl       # Length-residualized ARTS3k
├── sda.pkl             # SDA metric scores
├── st_sent.pkl         # ST-sent metric scores
├── st_para.pkl         # ST-para metric scores
└── d_wiki.pkl          # D-Wiki metric scores

src/
├── data_loader.py      # Dataset loading utilities
├── statistics/         # Correlation & Williams test
└── visualization/      # Heatmaps, tables, plots

notebooks/
├── 01_Data_Overview.ipynb        # Dataset overview and metrics
├── 02_Correlation_Analysis.ipynb # Pearson/Spearman correlations
├── 03_Williams_Significance.ipynb # Williams significance test
└── 04_Visualization.ipynb        # Generate publication figures
```

## Usage

```python
from src.data_loader import load_all_datasets, get_metric_names

# Load all datasets
datasets = load_all_datasets()

# Get metric scores for ARTS3k
arts3k = datasets['ARTS3k']
human_scores = arts3k['human_scores']
metric_scores = arts3k['metric_scores']

# Available metrics
print(get_metric_names(arts3k))
# ['SLE', 'LENS-SALSA', 'FKGL', 'Random', 'Shuffle', 'BATS-GB', 'BATS-RF', 'Claude(simp)']
```

## Statistical Methods

### Williams Test
Compares two dependent correlations sharing a common variable (human scores):

```python
from src.statistics import williams_test

t_stat, p_value = williams_test(r12=0.75, r13=0.65, r23=0.80, n=100)
```

### Correlation Analysis
```python
from src.statistics import compute_significance_analysis

result = compute_significance_analysis(
    metric_scores,  # Dict[str, np.ndarray]
    human_scores,   # np.ndarray
    method='pearson',
    alpha=0.05
)
```

## Requirements

- Python 3.8+
- numpy, pandas, scipy
- scikit-learn
- matplotlib, seaborn
- jupyter

## License

MIT License
