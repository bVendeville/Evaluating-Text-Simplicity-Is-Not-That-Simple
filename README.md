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

| Dataset | Samples | Score Range | LLM judges | Description |
|---------|---------|-------------|------------|-------------|
| ARTS94 | 94 | [0, 1] | all 5 (`output` only) | ARTS small subset |
| ARTS300 | 300 | [0, 1] | all 5 (`output` only) | ARTS medium subset |
| ARTS3k | 2999 | [0, 1] | all 5 (`output` only) | ARTS full dataset |
| LR-ARTS94 | 94 | residuals | all 5 (residualized) | Length-residualized ARTS94 |
| LR-ARTS300 | 300 | residuals | all 5 (residualized) | Length-residualized ARTS300 |
| LR-ARTS3k | 2999 | residuals | all 5 (residualized) | Length-residualized ARTS3k |
| SDA | 600 | DA scores | all 5 direct + 4 pairwise (Haiku also `+src`/`+ref`/`+src+ref`) | Simplicity Direct Assessment |
| ST-sent | 1620 | [1, 7] | all 5 direct + 4 pairwise (Haiku also `+src`/`+ref`/`+src+ref`) | SimpleText sentence-level |
| ST-para | 271 | [1, 7] | all 5 direct + 4 pairwise (Haiku also `+src`/`+ref`/`+src+ref`) | SimpleText paragraph-level |
| D-Wiki | 601 | [1, 5] | all 5 direct + 4 pairwise (Haiku also `+src`/`+ref`/`+src+ref`) | D-Wikipedia |

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

Five large language models are used as judges, scored at temperature 0 to keep results
deterministic. Direct-scoring columns are written under the model name; pairwise variants
add ` (pairwise)`.

| Column                                                     | Model                            | Notes                                                                                            |
|------------------------------------------------------------|----------------------------------|--------------------------------------------------------------------------------------------------|
| `Claude(simp)`, `Claude(simp+src)`, `Claude(simp+ref)`, `Claude(simp+src+ref)` | Claude Haiku 4.5 | Direct scoring on output (and on output + source / + reference / + both for the 4 human-annotated datasets). |
| `GPT-4o-mini`                                              | GPT-4o-mini (`gpt-4o-mini-2024-07-18`) | Direct scoring on output text only. Prompts from Kreutz et al. (2024). |
| `Gemma-4-E4B`                                              | `google/gemma-4-E4B-it`          | Same prompt as above (Kreutz et al.).                                                            |
| `Ministral-8B`                                             | `mistralai/Ministral-8B-Instruct-2410` | Same.                                                                                      |
| `Llama-3.1-8B`                                             | `meta-llama/Llama-3.1-8B-Instruct` | Same.                                                                                            |
| `GPT-4o-mini (pairwise)`, `Gemma-4-E4B (pairwise)`, `Ministral-8B (pairwise)`, `Llama-3.1-8B (pairwise)` | Same 4 non-Haiku models | Pairwise prompting on the 4 human-annotated datasets. Each sampled pair is queried in both orderings to detect position bias; surviving comparisons are aggregated with a Bradley-Terry model and rescaled to a 1-10 scalar **within each (model, dataset) cell**. Across-cell comparisons require rank correlation, not absolute values. |

**Caveat — Ministral-8B pairwise on SDA.** Ministral-8B returned an empty response on ~85% of the
Kreutz et al. pairwise prompts. After dropping parse failures and position-biased ties, the
Bradley-Terry fit had zero surviving comparisons; we therefore ship the column as NaN, mirroring
the '-' cell in the paper's Table 12. The position-bias rate (0.07) is still recorded in
`data/pairwise_position_bias.json`.

**Caveat — pairwise position bias.** The fraction of pairs on which a model disagreed with
itself when the ordering was swapped ranges from ~17% (Gemma-4 on D-Wiki) to ~68% (Ministral-8B
on ST-sent). Higher bias means the Bradley-Terry fit drops more pairs and the resulting score
is less stable. Read pairwise correlations alongside `data/pairwise_position_bias.json`.

### Reference-based (where applicable)
- **SARI**: System output Against References and Input
- **BERTScore**: BERT-based similarity
- **BLEU**, **ROUGE-L**: N-gram overlap metrics

## Repository Structure

```
data/
├── arts94.pkl                    # ARTS94 metric scores (incl. LLM judges)
├── arts300.pkl                   # ARTS300 metric scores
├── arts3k.pkl                    # ARTS3k metric scores
├── lr_arts94.pkl                 # Length-residualized ARTS94
├── lr_arts300.pkl                # Length-residualized ARTS300
├── lr_arts3k.pkl                 # Length-residualized ARTS3k
├── sda.pkl                       # SDA metric scores
├── st_sent.pkl                   # ST-sent metric scores
├── st_para.pkl                   # ST-para metric scores
├── d_wiki.pkl                    # D-Wiki metric scores
└── pairwise_position_bias.json   # Bias rates for pairwise LLM cells

src/
├── data_loader.py        # Dataset loading utilities
├── statistics/           # Correlation & Williams test
└── visualization/        # Heatmaps, tables, plots

scripts/
└── build_release_data.py # Re-build pickles from raw scoring CSVs (maintainer-only)

notebooks/
├── 01_Data_Overview.ipynb         # Dataset overview and metrics
├── 02_Correlation_Analysis.ipynb  # Pearson/Spearman correlations
├── 03_Williams_Significance.ipynb # Williams significance test
├── 04_Visualization.ipynb         # Generate publication figures
└── 05_LLM_as_Judge.ipynb          # Paper-faithful Tables 3, 4, 5, 10, 11, 12, 13, 14

results/
└── tables/               # CSV outputs from 05_LLM_as_Judge.ipynb
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

# Available metrics (includes LLM-as-judge columns)
print(get_metric_names(arts3k))
# ['SLE', 'LENS-SALSA', 'FKGL', 'Random', 'Shuffle', 'BATS-GB', 'BATS-RF',
#  'Claude(simp)', 'GPT-4o-mini', 'Gemma-4-E4B', 'Ministral-8B', 'Llama-3.1-8B']
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
