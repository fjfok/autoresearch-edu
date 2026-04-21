# AutoResearch — Tabular ML Ratchet Loop

You will run a Karpathy-style autoresearch ratchet loop over `train.py` on the
UCI Heart Disease dataset. You modify one file, measure one number, keep only
improvements. Forever.

Three sections: **Setup** (once), **Experimentation** (rules), **The Loop**
(9 steps, never stop).

---

## Setup

Do these once, before the first iteration.

1. **Pick a run tag.** Short kebab-case (e.g. `feat-interactions`, `gbm-sweep`,
   `run-20260421`). Use whatever the slash command passed in, or default to
   `run-<YYYYMMDD-HHMM>`.

2. **Branch.** `git checkout -b autoresearch/<tag>`. If you are already on an
   `autoresearch/*` branch (a resumed session), stay on it.

3. **Read the in-scope files in full.** `train.py`, `prepare.py`,
   `pyproject.toml`, and the existing `results.tsv` (if present, `tail -n 20`).
   Skim `README.md` once for context. Never read `run.log`.

4. **Warm the dataset cache.** `uv run prepare.py` — creates `data/heart.csv`.
   Skip if the file already exists.

5. **Sanity-run the baseline.**
   ```bash
   uv run train.py > run.log 2>&1
   grep "^val_auc:\|^test_auc:\|^wall_time_s:\|^status:" run.log
   ```
   You must see `status: ok` and a numeric `val_auc`. If not, stop the loop
   and surface the error to the user — the harness is broken and fixing it is
   out of scope for the agent (see Rules below).

6. **Initialise `results.tsv`** if it does not exist. Exact header, tab-separated:

   ```
   commit	val_auc	test_auc	wall_time_s	status	description
   ```

   Append one row for the current HEAD with the baseline metrics you just
   measured. `results.tsv` is untracked — never `git add` it.

---

## Experimentation

**Ratchet metric.** `val_auc` — the mean of 5-fold stratified CV AUC on the
train split, as printed by `prepare.evaluate`. A run is an improvement iff
`val_auc > best_val_auc_so_far` (strict inequality). Ties do not advance the
ratchet, except the simplicity tiebreak below.

**Audit metric.** `test_auc` is reported for every run but must never drive a
ratchet decision. Treat it as write-only — do not peek at it to pick
hypotheses, otherwise you are optimising against the holdout.

**In scope.** `train.py` only. You can add or remove imports from packages
already declared in `pyproject.toml`: sklearn, xgboost, lightgbm, flaml, numpy,
pandas, scipy, matplotlib. Anything they expose is fair game — feature
engineering, model choice, hyperparameters, ensembling, calibration,
preprocessing, feature selection.

**Out of scope — do NOT edit:**
- `prepare.py`, `RANDOM_STATE`, `TEST_SIZE`, `CV_FOLDS`, `TIME_BUDGET_S`.
- `pyproject.toml` — no new dependencies, period. If an idea needs a new
  library, pick a different idea.
- The split, the metric, the CV fold count. Changing any of these invalidates
  every prior row in `results.tsv`.
- `results.tsv` is untracked and append-only. Do not rewrite history in it.

**Budgets.**
- Wall clock per run: 180 s (the harness kills at that mark via SIGALRM).
- If `train.py` hangs from your side (rare), `kill` the process after ~240 s
  and record `status: crashed (timeout)` in `results.tsv`.
- CPU only. No GPU assumptions.

**Simplicity tiebreak.** If a new commit matches the previous `val_auc` within
1e-6 *and* strictly deletes lines from `train.py`, keep it. Otherwise on a tie
or regression, revert.

**Anti-gaming (non-negotiable).**
- Never import `y_test`, the test split, or anything from `prepare.load_data`
  that exposes the test labels. The harness owns the split.
- Do not re-seed or introduce nondeterminism that changes `val_auc` run to run
  with the same code (set `random_state=42` on every stochastic estimator).
- Do not cache fitted models between runs. Every iteration trains from scratch.
- Do not early-exit or short-circuit `evaluate()`. Always let it complete.
- If an idea needs to peek at `test_auc` to decide ratchet — drop the idea.

**What good iterations look like.**
- Feature engineering: interactions, polynomials, binning, log/quantile
  transforms, ratios between clinically meaningful columns.
- Model choice: LightGBM, XGBoost, ExtraTrees, GradientBoosting, SVC(rbf), MLP.
- Preprocessing: RobustScaler, QuantileTransformer, PowerTransformer, PCA,
  SelectKBest / variance thresholds.
- Regularisation sweeps: `C`, `alpha`, `max_depth`, `min_samples_leaf`,
  `num_leaves`, `reg_lambda`.
- Ensembling: `VotingClassifier(soft)`, `StackingClassifier`.
- Calibration: `CalibratedClassifierCV(method='isotonic')`.
- AutoML-in-loop: brief `FLAML.AutoML.fit` with a 30–60 s `time_budget` — still
  inside the 180 s wall-clock cap.

---

## The Loop (9 steps — NEVER STOP)

Run these in order. Then go back to step 1. Do not ask the user for permission
between iterations.

1. **State check.**
   ```bash
   git status                    # must be clean
   git rev-parse --short HEAD
   tail -n 20 results.tsv
   ```
   Note the current best `val_auc` — the max across all `status=ok` rows.

2. **Edit `train.py`** with ONE new idea. Change `HYPOTHESIS` to a one-line,
   tab-free description of what you are trying. If the idea builds on the last
   kept commit, diff against HEAD; if it is a net-new direction, consider
   reverting to the baseline shape first.

3. **Commit.**
   ```bash
   git add train.py
   git commit -m "<short hypothesis>"
   ```

4. **Run.** Always redirect to a file. Never `tee`, never `cat run.log`,
   never pipe stdout of `train.py` into your own context.
   ```bash
   uv run train.py > run.log 2>&1
   ```

5. **Extract metrics.**
   ```bash
   grep "^val_auc:\|^test_auc:\|^wall_time_s:\|^status:" run.log
   ```

6. **Handle crashes.** If `status:` is missing from the grep output, or is not
   exactly `ok`:
   - `tail -n 50 run.log` for the traceback.
   - If the cause is a one-line bug in *your* edit (typo, wrong import, shape
     mismatch): fix, `git add train.py && git commit --amend --no-edit`, rerun
     once. Max one amend per iteration.
   - Otherwise: record `status=crashed` in `results.tsv`,
     `git reset --hard HEAD~1`, go to step 1 with a different idea.

7. **Append to `results.tsv`.** Tab-separated. Description must not contain
   tabs or newlines; commas are fine.
   ```
   <short-sha>\t<val_auc>\t<test_auc>\t<wall_time_s>\t<ok|crashed>\t<hypothesis>
   ```

8. **Ratchet decision.**
   - `val_auc > best_so_far` → keep. Ratchet advances.
   - `val_auc <= best_so_far` → `git reset --hard HEAD~1`. Ratchet does not
     move. The commit vanishes from history; the `results.tsv` row stays.
   - Simplicity tiebreak (see above) may override a soft tie.

9. **Loop.** Go to step 1 with a fresh idea. Every ~5 iterations, scan
   `results.tsv` for the shape of ideas you have already discarded and avoid
   proposing them again. Do not pause to ask the user whether to continue.

---

## NEVER STOP

Do not stop the loop unless one of the following is true:
- The user interrupts.
- The harness reports an unrecoverable environment failure that cannot be
  diagnosed without a human (e.g. sklearn import error, `data/` unwriteable).
- `val_auc` reaches ≥ 0.99 — investigate for data leakage before celebrating.

"Should I continue?" is never a question you ask. The whole point of the
ratchet is that it runs autonomously.

---

## Reporting to the user

After each iteration, emit **exactly one line** of prose to the user:

```
iter N | val_auc=0.xxxx (Δ ±0.yyyy) | KEPT|DISCARDED — <hypothesis>
```

No tracebacks, no reasoning, no multi-line summaries. If the user wants
detail they will ask; `results.tsv` and git history have everything.

Every ~10 iterations, add one extra line with the running best:

```
best so far: val_auc=0.xxxx @ <short-sha> — <hypothesis of best>
```

That's it.
