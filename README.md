# AutoResearch Educational Example

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Claude Command](https://img.shields.io/badge/Claude-Slash_Command-purple)](#run-it)

An educational implementation of Andrej Karpathy's **AutoResearch ratchet loop** [1], adapted for tabular ML. A Claude Code agent edits a single file, measures one number, keeps only improvements — autonomously, forever. Runs on a laptop CPU.

## What is AutoResearch?

Karpathy released AutoResearch in March 2026 [1] to let AI agents conduct ML research autonomously. The pattern is a **ratchet loop**: the agent proposes a hypothesis, implements it, trains under a fixed budget, evaluates, then either keeps the improvement or reverts. The codebase only moves forward. At scale this pattern has shown ~11% gains on LLM training tasks [1].

This repo ports that pattern to the [UCI Heart Disease dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) [2] — small, CPU-friendly, done in minutes. One file (`train.py`) is mutable. Everything else is frozen.

## What you see when it's running

Each iteration, the agent emits a single line:

```
iter 3 | val_auc=0.8996 (Δ +0.0012) | KEPT       — LightGBM num_leaves=31, reg_lambda=1.0
iter 4 | val_auc=0.8921 (Δ -0.0075) | DISCARDED  — stacking LogReg+RF+XGB
iter 5 | val_auc=0.9042 (Δ +0.0046) | KEPT       — + clinical interaction features
```

`KEPT` rows remain in `git log`. `DISCARDED` rows are `git reset --hard`'d out of history but logged in `results.tsv`. Leave it overnight; review in the morning.

## Run it

Requires [`uv`](https://github.com/astral-sh/uv) and [Claude Code](https://claude.com/claude-code).

```bash
git clone https://github.com/fjfok/autoresearch-edu.git
cd autoresearch-edu
uv run prepare.py        # one-time: cache data/heart.csv + sanity-check the harness
```

Then, inside Claude Code in this repo:

```
/autoresearch my-run
```

The agent creates an `autoresearch/my-run` branch, warms the cache, runs a baseline, and enters the loop. It stops only when you interrupt it, the harness breaks, or `val_auc ≥ 0.99` (inspect for leakage if that happens).

## How it works

| File | Role | Who edits |
|---|---|---|
| [`prepare.py`](prepare.py) | Frozen harness: loads data, 80/20 split, 5-fold stratified CV, 180 s wall-clock budget, prints `val_auc` / `test_auc` / `wall_time_s` / `status`. | Never. |
| [`train.py`](train.py) | Mutable artifact: `HYPOTHESIS` string, `feature_fn`, `build_pipeline`. | Agent, every iteration. |
| [`program.md`](program.md) | The skill: Setup, Rules, 9-step loop, NEVER STOP. | You, between runs. |
| [`.claude/commands/autoresearch.md`](.claude/commands/autoresearch.md) | Slash command that loads `program.md` and enters the loop. | You, rarely. |
| `results.tsv` | Append-only ratchet log — one row per iteration, including discards. Untracked. | Agent appends. |

**Ratchet metric:** `val_auc`, the mean of 5-fold stratified CV on the train split. Strict inequality advances the ratchet. `test_auc` is reported for audit but never drives decisions.

The 9-step loop, summarised: **state-check → edit `train.py` → commit → run → grep metrics → handle crash → append to `results.tsv` → ratchet keep/revert → loop.** Full spec in [`program.md`](program.md).

### Sanity-check the harness by hand

No agent required — useful when you're debugging `prepare.py` or poking at `train.py` yourself:

```bash
uv run train.py > run.log 2>&1
grep "^val_auc:\|^test_auc:\|^wall_time_s:\|^status:" run.log
```

You should see `status: ok` and a numeric `val_auc`. If not, the harness is broken — fix it before running the agent.

## Tuning the loop

Edit [`program.md`](program.md) between runs to narrow the hypothesis space — add hints, rule out dead ends, adjust the "what good iterations look like" examples. That file is the agent's only standing context; the leverage is there.

## Results (reference run)

One full `/autoresearch` run on this dataset produced the following, benchmarked against the [AMLB AutoML Benchmark](https://openml.github.io/automlbenchmark/) framework [3]:

| Approach | Test AUC | Test Acc | Wall time |
|----------|----------|----------|-----------|
| Manual baseline (LogisticRegression) | 0.8474 | 76.32% | ~2 s |
| FLAML one-shot AutoML [4] | 0.8836 | 77.63% | 45 s |
| **AutoResearch ratchet (best of 8 iters)** | **0.9091** | **78.95%** | ~10 min |

A **+6.17% AUC** lift over the manual baseline. The winning hypothesis combined clinical interaction features with FLAML's automated model search — neither on its own reached this number.

Your own run will produce different numbers and a different winning hypothesis; that's the point. These are one team's results, not a script you re-execute to reproduce the table.

### Key lessons

1. **Most hypotheses fail — and that's fine.** 7 of 8 iterations were discarded in the reference run. The ratchet enforces discipline.
2. **Winning hypotheses are usually combinatorial.** The breakthrough combined two ideas that individually failed.
3. **AutoResearch generalises AutoML.** AutoML searches a fixed space (hyperparameters). AutoResearch lets the agent propose *any* code change.
4. **Monotonically non-decreasing.** The best-so-far never moves backward. Safe to run unattended.

## Repository structure

```text
autoresearch-edu/
├── .claude/commands/autoresearch.md   # Slash command
├── prepare.py                         # FROZEN harness
├── train.py                           # MUTABLE (the only file the agent edits)
├── program.md                         # The skill
├── data/heart.csv                     # Cached at first run
├── pyproject.toml
├── LICENSE
└── README.md
```

`results.tsv`, `run.log`, and `data/` are gitignored runtime artefacts.

## References

[1] Karpathy, A. (2026). *autoresearch: AI agents running research on single-GPU nanochat training automatically*. https://github.com/karpathy/autoresearch

[2] Detrano, R., et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease*. American Journal of Cardiology, 64(5), 304–310. (UCI Heart Disease)

[3] Gijsbers, P., et al. (2024). *AMLB: an AutoML Benchmark*. JMLR. https://openml.github.io/automlbenchmark/

[4] Wang, C., et al. (2021). *FLAML: A Fast and Lightweight AutoML Library*. MLSys 2021. https://microsoft.github.io/FLAML/
