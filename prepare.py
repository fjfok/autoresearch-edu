"""
prepare.py — READ-ONLY harness for the AutoResearch ratchet loop.

DO NOT MODIFY. The evaluation harness must stay frozen — every prior
val_auc row in results.tsv was measured against *this exact file*, and
changing it silently invalidates the entire ratchet history.

What this file does:
  1. Loads UCI Heart Disease (303 rows, 13 features) via fetch_openml.
  2. Caches the cleaned frame to data/heart.csv so subsequent runs skip I/O.
  3. Produces a deterministic 80/20 stratified train/test split.
  4. Exposes evaluate(build_pipeline, feature_fn) which:
       - runs 5-fold stratified CV on the train split (the ratchet metric),
       - fits once on the full train split and scores the held-out test,
       - prints val_auc / test_auc / wall_time_s / status, one per line,
       - enforces a hard 180 s wall-clock budget via SIGALRM.

Ratchet metric: val_auc  (mean of CV folds on the train split)
Audit metric  : test_auc (reported but never used for ratchet decisions)

Run directly to warm the dataset cache:
    uv run prepare.py
"""

import os
import sys
import time
import signal
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

warnings.filterwarnings("ignore")

# ─── Frozen constants — changing any of these invalidates results.tsv ──
RANDOM_STATE   = 42
TEST_SIZE      = 0.20
CV_FOLDS       = 5
TIME_BUDGET_S  = 180
DATA_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATA_CSV       = os.path.join(DATA_DIR, "heart.csv")


def _download_and_cache() -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATA_CSV):
        return pd.read_csv(DATA_CSV)
    heart = fetch_openml(name="heart-disease", version=1, as_frame=True)
    df = heart.frame.copy()
    df = df.replace("?", np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(df.median(numeric_only=True))
    df["target"] = (df["target"] > 0).astype(int)
    df.to_csv(DATA_CSV, index=False)
    return df


def load_data():
    """Return (X_train, y_train, X_test, y_test, feature_names) — deterministic."""
    df = _download_and_cache()
    feature_names = [c for c in df.columns if c != "target"]
    X = df[feature_names].values.astype(float)
    y = df["target"].values.astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_tr, y_tr, X_te, y_te, feature_names


def _timeout_handler(signum, frame):
    raise TimeoutError(f"wall-clock budget of {TIME_BUDGET_S}s exceeded")


def _predict_scores(pipe, X):
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    if hasattr(pipe, "decision_function"):
        return pipe.decision_function(X)
    raise TypeError(
        f"pipeline of type {type(pipe).__name__} exposes neither "
        "predict_proba nor decision_function"
    )


def evaluate(build_pipeline, feature_fn=None):
    """Run the frozen evaluation and print grep-able metrics.

    Prints exactly these lines (one value each):
        val_auc:     <float>    # ratchet metric (5-fold CV mean on train)
        test_auc:    <float>    # audit only — do NOT optimise against it
        test_acc:    <float>
        test_f1:     <float>
        wall_time_s: <float>
        status:      ok | crashed (<reason>)

    The function swallows exceptions so that crashes still produce a single
    'status: crashed' line — the agent's grep picks that up and reverts.
    Exit code mirrors the outcome (0 ok / 2 timeout / 3 other).
    """
    if feature_fn is None:
        feature_fn = lambda X: X

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(TIME_BUDGET_S)

    t0 = time.time()
    rc = 0
    val_auc = test_auc = test_acc = test_f1 = float("nan")
    try:
        X_tr, y_tr, X_te, y_te, _ = load_data()
        X_tr_f = np.asarray(feature_fn(X_tr))
        X_te_f = np.asarray(feature_fn(X_te))
        if X_tr_f.shape[0] != X_tr.shape[0] or X_te_f.shape[0] != X_te.shape[0]:
            raise ValueError(
                "feature_fn must preserve row count — "
                f"got train {X_tr.shape[0]}→{X_tr_f.shape[0]}, "
                f"test {X_te.shape[0]}→{X_te_f.shape[0]}"
            )

        cv = StratifiedKFold(
            n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
        )
        cv_scores = cross_val_score(
            build_pipeline(), X_tr_f, y_tr, cv=cv, scoring="roc_auc", n_jobs=1
        )
        val_auc = float(np.mean(cv_scores))

        final = build_pipeline()
        final.fit(X_tr_f, y_tr)
        y_prob = _predict_scores(final, X_te_f)
        y_pred = (y_prob >= 0.5).astype(int)
        test_auc = float(roc_auc_score(y_te, y_prob))
        test_acc = float(accuracy_score(y_te, y_pred))
        test_f1  = float(f1_score(y_te, y_pred))
        status = "ok"
    except TimeoutError as e:
        status = f"crashed (timeout: {e})"
        rc = 2
    except Exception as e:
        status = f"crashed ({type(e).__name__}: {e})"
        rc = 3
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)

    wall = time.time() - t0
    print(f"val_auc: {val_auc:.6f}",      flush=True)
    print(f"test_auc: {test_auc:.6f}",    flush=True)
    print(f"test_acc: {test_acc:.6f}",    flush=True)
    print(f"test_f1: {test_f1:.6f}",      flush=True)
    print(f"wall_time_s: {wall:.2f}",     flush=True)
    print(f"cv_folds: {CV_FOLDS}",        flush=True)
    print(f"status: {status}",            flush=True)
    if rc:
        sys.exit(rc)


if __name__ == "__main__":
    df = _download_and_cache()
    print(
        f"cached heart-disease dataset: {df.shape[0]} rows × "
        f"{df.shape[1] - 1} features → {DATA_CSV}"
    )
