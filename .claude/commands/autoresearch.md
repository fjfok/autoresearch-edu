---
description: Run the Karpathy-style AutoResearch ratchet loop on train.py — autonomous, never stops until interrupted
argument-hint: [run-tag]
---

# /autoresearch — launch the ratchet loop

You are about to run the AutoResearch ratchet loop on this repository.
Your job is to follow `program.md` verbatim and iterate on `train.py` until
the user interrupts. **Do not ask for confirmation between steps.**

## Argument

`$ARGUMENTS` — an optional short kebab-case run tag (e.g. `gbm-sweep`,
`feat-interactions`). If empty, default to `run-<YYYYMMDD-HHMM>` using the
current local time.

## What to do right now

1. **Read `program.md`** in the repo root, in full. It is the canonical spec
   for this loop. If anything here conflicts with `program.md`, `program.md`
   wins.
2. **Complete `program.md` § Setup** using `$ARGUMENTS` (or the default tag)
   as the branch suffix. Create the branch, warm the dataset, sanity-run the
   baseline, initialise `results.tsv` with the header + baseline row.
3. **Enter `program.md` § The Loop.** Execute the 9 steps in order. Repeat
   forever. One hypothesis per iteration, one commit per iteration, one row
   in `results.tsv` per iteration.
4. **Report each iteration as a single line** to the user, using the exact
   format in `program.md` § Reporting. No narration, no tracebacks, no
   multi-paragraph summaries.
5. **NEVER STOP** (see `program.md` § NEVER STOP). The only exits are user
   interrupt, unrecoverable environment failure, or `val_auc ≥ 0.99` (in
   which case investigate leakage).

## Hard rules (lifted from program.md — do not violate)

- Only `train.py` may change between iterations.
- `prepare.py`, the split, the metric, `pyproject.toml` are frozen. If you
  think you need to edit them, surface it and stop — do not silently change
  the harness.
- `run.log` must never be read whole; use `grep` and `tail -n 50`.
- `results.tsv` is untracked and append-only. Never `git add` it.
- `git reset --hard HEAD~1` on any non-improvement. No exceptions.
- Never peek at `test_auc` when deciding whether to keep a commit.
- No new dependencies.

Begin now by reading `program.md`.
