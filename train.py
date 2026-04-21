"""
train.py — the single MUTABLE artifact for the AutoResearch ratchet loop.

The agent edits this file (and only this file) per iteration: one new idea,
one commit, one row in results.tsv. The harness in prepare.py is frozen.

Contract with the harness:
  - Export  build_pipeline()  returning a fresh sklearn-compatible estimator.
  - Export  feature_fn(X)     returning a feature-transformed ndarray
                              (row count must be preserved; default: identity).
  - Update  HYPOTHESIS        to a one-line description of this iteration's idea.
  - __main__ calls evaluate(build_pipeline, feature_fn) and prints the
    grep-able metrics defined in prepare.py.

Run once:
    uv run train.py > run.log 2>&1
    grep "^val_auc:\\|^test_auc:\\|^wall_time_s:\\|^status:" run.log
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from prepare import evaluate

HYPOTHESIS = "baseline — standard-scaled logistic regression (C=1.0)"


def feature_fn(X):
    """Transform raw features. Default: identity."""
    return X


def build_pipeline():
    """Return a fresh sklearn estimator/pipeline. Called once per CV fold + once for test."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=1.0, max_iter=500, random_state=42)),
    ])


if __name__ == "__main__":
    print(f"hypothesis: {HYPOTHESIS}", flush=True)
    evaluate(build_pipeline, feature_fn)
