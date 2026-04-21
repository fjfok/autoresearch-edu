# /autoresearch

Run an **AutoResearch ratchet loop** on a classification dataset.

Inspired by Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) framework (March 2026), this command autonomously proposes ML hypotheses, trains models, evaluates them, and **keeps only improvements** — the ratchet can only move forward.

## Usage

```
/autoresearch [dataset_path_or_name] [target_column] [metric] [iterations]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `dataset_path_or_name` | `heart-disease` | Path to a CSV file, or a named Kaggle/OpenML dataset |
| `target_column` | `target` | Name of the binary/multiclass target column |
| `metric` | `roc_auc` | Evaluation metric: `roc_auc`, `accuracy`, `f1`, `rmse` |
| `iterations` | `8` | Number of ratchet loop iterations to run |

### Examples

```bash
# Run on the built-in Heart Disease dataset (default)
/autoresearch

# Run on a custom CSV with 10 iterations
/autoresearch ./data/my_dataset.csv churn roc_auc 10

# Run on a Kaggle titanic dataset
/autoresearch ./data/titanic.csv Survived accuracy 12
```

## What This Command Does

When invoked, this command will:

1. **Load & explore** the dataset — shape, class balance, missing values, feature types
2. **Establish a baseline** — fit a Logistic Regression as the human-researcher starting point
3. **Run FLAML one-shot** — Microsoft's lightweight AutoML for a 45-second automated search
4. **Execute the ratchet loop** for `N` iterations:
   - Propose a hypothesis (feature engineering, model selection, hyperparameters)
   - Train and evaluate with cross-validation
   - **Keep** if Test AUC improves over current best
   - **Discard** (revert) if no improvement — the ratchet never goes backward
5. **Generate visualizations** — 4-panel results figure, benchmark comparison, feature analysis
6. **Save results** to `results.json` and print a final summary table

## Output Files

After running, the following files are created in the working directory:

```
results.json                  # Full experiment log (all iterations)
autoresearch_results.png      # 4-panel: AUC progress, ratchet diagram, waterfall, kept/discarded
benchmark_comparison.png      # AutoResearch vs Manual vs AMLB benchmark context
feature_analysis.png          # Feature importance + class separation charts
flaml_log.txt                 # FLAML internal search log
```

## The Ratchet Loop Explained

```
┌─────────────────────────────────────────────────────────┐
│                  AUTORESEARCH RATCHET                    │
│                                                         │
│  Read program.md  ──►  Propose Hypothesis               │
│       ▲                      │                          │
│       │                      ▼                          │
│  Keep or Revert  ◄──  Modify Config/Code                │
│       ▲                      │                          │
│       │                      ▼                          │
│  Evaluate Metric ◄──  Run Experiment (fixed budget)     │
│                                                         │
│  RATCHET: Best AUC can only go UP. Never backward.      │
└─────────────────────────────────────────────────────────┘
```

The key invariant: `best_auc[t+1] >= best_auc[t]` for all `t`.

## Requirements

Install dependencies with `uv` (recommended):

```bash
uv sync
```

Or with pip:

```bash
pip install flaml scikit-learn xgboost lightgbm numpy pandas matplotlib
```

## References

- Karpathy, A. (2026). *autoresearch*. https://github.com/karpathy/autoresearch
- Gijsbers, P. et al. (2024). *AMLB: an AutoML Benchmark*. JMLR. https://openml.github.io/automlbenchmark/
- Wang, C. et al. (2021). *FLAML*. MLSys 2021. https://microsoft.github.io/FLAML/
