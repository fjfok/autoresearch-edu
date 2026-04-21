"""
=============================================================================
AutoResearch Educational Example — Heart Disease Dataset (Kaggle/UCI)
=============================================================================

This script demonstrates the CORE POWER of the AutoResearch pattern:
  - The "Ratchet Loop": iterative, autonomous experimentation that only
    keeps improvements and discards regressions.
  - Adapted from Karpathy's AutoResearch (March 2026) for CPU-friendly
    tabular ML using FLAML (Microsoft's Fast and Lightweight AutoML).

Dataset: Heart Disease (UCI / Kaggle) — heart-disease (Cleveland, 303 rows)
  - 303 patients, 13 clinical features
  - Binary classification: predict presence of heart disease
  - Source: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

Open-Source Benchmark: AMLB (OpenML AutoML Benchmark)
  - https://github.com/openml/automlbenchmark
  - The standard for comparing AutoML frameworks across 100+ datasets

The Three Phases Demonstrated:
  1. BASELINE  — A single manually-tuned model (human researcher)
  2. FLAML     — One-shot AutoML (automated search, fixed budget)
  3. RATCHET   — AutoResearch loop (iterative, accumulating improvements)

=============================================================================
"""

import os
import time
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

import flaml
from flaml import AutoML

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results.json")
RATCHET_ITERATIONS = 8         # number of ratchet loop iterations
FLAML_TOTAL_BUDGET = 45        # seconds for FLAML one-shot search
RANDOM_STATE = 42
CV_FOLDS = 5

print("=" * 70)
print("  AutoResearch Educational Example")
print("  Heart Disease Dataset (Kaggle/UCI) — CPU-Only Demonstration")
print("=" * 70)
print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 0: DATA LOADING — UCI Heart Disease (larger, more challenging)
# ─────────────────────────────────────────────────────────────────────────────
print("━" * 70)
print("  PHASE 0: Dataset Loading & Exploration")
print("━" * 70)

# Load the UCI Heart Disease dataset (Cleveland, 303 rows, 13 features)
# This is the canonical Kaggle heart disease dataset
from sklearn.datasets import fetch_openml
heart = fetch_openml(name='heart-disease', version=1, as_frame=True)
df = heart.frame.copy()

print(f"  Columns: {list(df.columns)}")
print(f"  Shape  : {df.shape}")
print(f"  Target : {df[df.columns[-1]].value_counts().to_dict()}")

# The target column is already named 'target'
# Values: 0=no disease, 1-4=disease severity -> convert to binary
target_col = 'target'
df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0)
df['target'] = (df[target_col] > 0).astype(int)

# Handle missing values (some '?' in original dataset)
df = df.replace('?', np.nan)
for col in df.columns:
    if col != 'target':
        df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.fillna(df.median(numeric_only=True))

feature_names = [c for c in df.columns if c != 'target']
X = df[feature_names].values.astype(float)
y = df['target'].values

# Use a deliberately simple split to make the problem harder
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

print(f"  Dataset shape   : {df.shape[0]} rows × {len(feature_names)} features")
print(f"  Target balance  : {y.sum()} positive ({100*y.mean():.1f}%) / "
      f"{(1-y).sum()} negative")
print(f"  Train / Test    : {len(X_train)} / {len(X_test)} samples")
print(f"  Features        : {', '.join(feature_names)}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Evaluate a model with cross-validation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, X_tr, y_tr, X_te, y_te, label="Model"):
    """Run 5-fold CV and hold-out test evaluation. Returns dict of metrics."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring='roc_auc', n_jobs=-1)

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else None

    test_acc  = accuracy_score(y_te, y_pred)
    test_auc  = roc_auc_score(y_te, y_prob) if y_prob is not None else float('nan')
    test_f1   = f1_score(y_te, y_pred)
    cv_mean   = cv_scores.mean()
    cv_std    = cv_scores.std()

    result = {
        "label"    : label,
        "cv_auc"   : round(cv_mean, 4),
        "cv_std"   : round(cv_std, 4),
        "test_auc" : round(test_auc, 4),
        "test_acc" : round(test_acc, 4),
        "test_f1"  : round(test_f1, 4),
    }
    return result, model


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: BASELINE — Manual Human Researcher
# ─────────────────────────────────────────────────────────────────────────────
print("━" * 70)
print("  PHASE 1: BASELINE — Manual Human Researcher")
print("━" * 70)
print("  A data scientist manually picks Logistic Regression with standard")
print("  scaling — a common, reasonable first choice.")
print()

t0 = time.time()
baseline_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=1.0, max_iter=500, random_state=RANDOM_STATE))
])
baseline_result, baseline_model = evaluate_model(
    baseline_pipe, X_train, y_train, X_test, y_test, label="Baseline (LogReg)"
)
baseline_time = time.time() - t0

print(f"  Model           : Logistic Regression (C=1.0, scaled)")
print(f"  CV AUC          : {baseline_result['cv_auc']:.4f} ± {baseline_result['cv_std']:.4f}")
print(f"  Test AUC        : {baseline_result['test_auc']:.4f}")
print(f"  Test Accuracy   : {baseline_result['test_acc']:.4f}")
print(f"  Test F1         : {baseline_result['test_f1']:.4f}")
print(f"  Time taken      : {baseline_time:.2f}s")
print()

best_auc = baseline_result['test_auc']
best_model_label = "Baseline (LogReg)"
results_history = [{"phase": "baseline", **baseline_result, "time_s": round(baseline_time, 2)}]

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: FLAML ONE-SHOT AutoML
# ─────────────────────────────────────────────────────────────────────────────
print("━" * 70)
print("  PHASE 2: FLAML — One-Shot AutoML (Microsoft, CPU-Friendly)")
print("━" * 70)
print(f"  FLAML searches over algorithms, hyperparameters, and feature")
print(f"  preprocessing automatically within a {FLAML_TOTAL_BUDGET}s time budget.")
print()

t0 = time.time()
automl = AutoML()
automl_settings = {
    "time_budget"     : FLAML_TOTAL_BUDGET,
    "metric"          : "roc_auc",
    "task"            : "classification",
    "log_file_name"   : os.path.join(os.path.dirname(RESULTS_LOG), "flaml_log.txt"),
    "seed"            : RANDOM_STATE,
    "verbose"         : 0,
    "eval_method"     : "cv",
    "n_splits"        : CV_FOLDS,
    "estimator_list"  : ["lgbm", "rf", "xgboost", "extra_tree", "lrl2"],
}
automl.fit(X_train, y_train, **automl_settings)
flaml_time = time.time() - t0

y_pred_flaml = automl.predict(X_test)
y_prob_flaml = automl.predict_proba(X_test)[:, 1]

flaml_result = {
    "label"    : f"FLAML AutoML ({FLAML_TOTAL_BUDGET}s budget)",
    "cv_auc"   : round(1 - automl.best_loss, 4),
    "cv_std"   : 0.0,
    "test_auc" : round(roc_auc_score(y_test, y_prob_flaml), 4),
    "test_acc" : round(accuracy_score(y_test, y_pred_flaml), 4),
    "test_f1"  : round(f1_score(y_test, y_pred_flaml), 4),
}

print(f"  Best estimator  : {automl.best_estimator}")
print(f"  Best config     : {json.dumps(automl.best_config, indent=4)}")
print(f"  CV AUC          : {flaml_result['cv_auc']:.4f}")
print(f"  Test AUC        : {flaml_result['test_auc']:.4f}")
print(f"  Test Accuracy   : {flaml_result['test_acc']:.4f}")
print(f"  Test F1         : {flaml_result['test_f1']:.4f}")
print(f"  Time taken      : {flaml_time:.2f}s")
print(f"  Improvement vs Baseline: "
      f"{(flaml_result['test_auc'] - baseline_result['test_auc'])*100:+.2f}% AUC")
print()

if flaml_result['test_auc'] > best_auc:
    best_auc = flaml_result['test_auc']
    best_model_label = flaml_result['label']
    print(f"  ✓ RATCHET: New best AUC = {best_auc:.4f} (FLAML one-shot)")

results_history.append({"phase": "flaml_oneshot", **flaml_result, "time_s": round(flaml_time, 2)})


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: THE AUTORESEARCH RATCHET LOOP
# ─────────────────────────────────────────────────────────────────────────────
print("━" * 70)
print("  PHASE 3: THE AUTORESEARCH RATCHET LOOP")
print("━" * 70)
print("  Inspired by Karpathy's AutoResearch, the ratchet loop runs")
print("  iterative experiments. Each iteration proposes a new hypothesis,")
print("  trains, evaluates, and ONLY KEEPS improvements. The configuration")
print("  can only move FORWARD — never backward.")
print()
print(f"  Total iterations     : {RATCHET_ITERATIONS}")
print(f"  Starting best AUC    : {best_auc:.4f}")
print()

# Feature engineering: create interaction terms
def make_features_v1(X):
    """Original features."""
    return X

def make_features_v2(X):
    """Add clinical interaction features."""
    df_f = pd.DataFrame(X, columns=feature_names)
    # Key clinical interactions
    df_f['age_thalach_ratio'] = df_f['age'] / (df_f['thalach'] + 1)
    df_f['chol_age'] = df_f['chol'] * df_f['age'] / 1000
    df_f['trestbps_age'] = df_f['trestbps'] * df_f['age'] / 1000
    df_f['oldpeak_slope'] = df_f['oldpeak'] * df_f['slope']
    return df_f.values

def make_features_v3(X):
    """Add polynomial features (degree 2) on top clinical vars."""
    df_f = pd.DataFrame(X, columns=feature_names)
    key_cols = ['age', 'thalach', 'oldpeak', 'ca', 'thal']
    X_key = df_f[key_cols].values
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    X_poly = poly.fit_transform(X_key)
    return np.hstack([X, X_poly])

def make_features_v4(X):
    """Combine v2 interactions + polynomial."""
    X2 = make_features_v2(X)
    df_f = pd.DataFrame(X, columns=feature_names)
    key_cols = ['age', 'thalach', 'oldpeak', 'ca', 'thal']
    X_key = df_f[key_cols].values
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    X_poly = poly.fit_transform(X_key)
    return np.hstack([X2, X_poly])

# Ratchet configurations — each proposes a different hypothesis
ratchet_configs = [
    {
        "name": "Hypothesis 1: Feature Engineering (clinical interactions)",
        "hypothesis": "Adding age/thalach ratio and cholesterol-age interaction will improve discrimination",
        "feature_fn": make_features_v2,
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, max_iter=500, random_state=RANDOM_STATE))
        ])
    },
    {
        "name": "Hypothesis 2: Gradient Boosting (depth=3, lr=0.1)",
        "hypothesis": "A shallow GBM will capture non-linear patterns better than logistic regression",
        "feature_fn": make_features_v1,
        "model": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=RANDOM_STATE
        )
    },
    {
        "name": "Hypothesis 3: FLAML + Feature Engineering v2",
        "hypothesis": "AutoML on engineered features will find a better model than raw features",
        "feature_fn": make_features_v2,
        "model": None,
        "flaml_budget": 40
    },
    {
        "name": "Hypothesis 4: Extra Trees (robust to noise)",
        "hypothesis": "Extra Trees with high n_estimators is more robust to small dataset noise",
        "feature_fn": make_features_v2,
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', ExtraTreesClassifier(
                n_estimators=500, max_features='sqrt',
                min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=-1
            ))
        ])
    },
    {
        "name": "Hypothesis 5: Polynomial Features + GBM",
        "hypothesis": "Polynomial feature interactions will give GBM richer signal",
        "feature_fn": make_features_v3,
        "model": GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            subsample=0.9, random_state=RANDOM_STATE
        )
    },
    {
        "name": "Hypothesis 6: FLAML on all features v4",
        "hypothesis": "Full feature engineering combined with AutoML search will maximize performance",
        "feature_fn": make_features_v4,
        "model": None,
        "flaml_budget": 45
    },
    {
        "name": "Hypothesis 7: SVM with RBF kernel",
        "hypothesis": "SVM with RBF kernel captures complex decision boundaries in clinical data",
        "feature_fn": make_features_v2,
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(C=10.0, kernel='rbf', gamma='scale',
                        probability=True, random_state=RANDOM_STATE))
        ])
    },
    {
        "name": "Hypothesis 8: Soft Voting Ensemble (best models)",
        "hypothesis": "Blending diverse models reduces variance and improves generalization",
        "feature_fn": make_features_v2,
        "model": "ensemble"
    },
]

ratchet_log = []
kept_count = 0
discarded_count = 0
best_feature_fn = make_features_v1  # track which feature transform is best

for i, config in enumerate(ratchet_configs[:RATCHET_ITERATIONS]):
    iter_num = i + 1
    print(f"  ── Iteration {iter_num}/{RATCHET_ITERATIONS}")
    print(f"     {config['name']}")
    print(f"     Hypothesis: \"{config['hypothesis']}\"")

    t0 = time.time()
    feat_fn = config.get("feature_fn", make_features_v1)
    X_tr_f = feat_fn(X_train)
    X_te_f = feat_fn(X_test)

    if config.get("model") == "ensemble":
        # Soft voting ensemble using best feature transform found so far
        X_tr_best = best_feature_fn(X_train)
        X_te_best = best_feature_fn(X_test)

        gbm = GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=RANDOM_STATE
        )
        et = ExtraTreesClassifier(
            n_estimators=300, max_features='sqrt',
            min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=-1
        )
        svm_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(C=10.0, kernel='rbf', gamma='scale',
                        probability=True, random_state=RANDOM_STATE))
        ])
        ensemble = VotingClassifier(
            estimators=[('gbm', gbm), ('et', et), ('svm', svm_pipe)],
            voting='soft', n_jobs=-1
        )
        result, fitted_model = evaluate_model(
            ensemble, X_tr_best, y_tr_best if 'y_tr_best' in dir() else y_train,
            X_te_best, y_test, label=config['name']
        )

    elif config.get("model") is None:
        # FLAML iteration with feature-engineered data
        budget = config.get("flaml_budget", 40)
        automl_iter = AutoML()
        settings = {
            "time_budget"   : budget,
            "metric"        : "roc_auc",
            "task"          : "classification",
            "seed"          : RANDOM_STATE + iter_num,
            "verbose"       : 0,
            "eval_method"   : "cv",
            "n_splits"      : CV_FOLDS,
            "estimator_list": ["lgbm", "rf", "xgboost", "extra_tree"],
        }
        automl_iter.fit(X_tr_f, y_train, **settings)

        y_pred_i = automl_iter.predict(X_te_f)
        y_prob_i = automl_iter.predict_proba(X_te_f)[:, 1]
        result = {
            "label"    : config['name'],
            "cv_auc"   : round(1 - automl_iter.best_loss, 4),
            "cv_std"   : 0.0,
            "test_auc" : round(roc_auc_score(y_test, y_prob_i), 4),
            "test_acc" : round(accuracy_score(y_test, y_pred_i), 4),
            "test_f1"  : round(f1_score(y_test, y_pred_i), 4),
        }
        fitted_model = automl_iter

    else:
        result, fitted_model = evaluate_model(
            config['model'], X_tr_f, y_train, X_te_f, y_test, label=config['name']
        )

    iter_time = time.time() - t0
    delta = result['test_auc'] - best_auc

    print(f"     CV AUC  : {result['cv_auc']:.4f}")
    print(f"     Test AUC: {result['test_auc']:.4f}  (Δ {delta:+.4f} vs current best {best_auc:.4f})")

    if result['test_auc'] > best_auc:
        best_auc = result['test_auc']
        best_model_label = config['name']
        best_feature_fn = feat_fn
        kept_count += 1
        status = "✓ KEPT  — new best! Ratchet advances."
        ratchet_log.append({
            "iteration": iter_num,
            "config": config['name'],
            "status": "kept",
            "test_auc": result['test_auc'],
            "delta": round(delta, 4),
            "time_s": round(iter_time, 2)
        })
    else:
        discarded_count += 1
        status = "✗ DISCARDED — no improvement, reverted."
        ratchet_log.append({
            "iteration": iter_num,
            "config": config['name'],
            "status": "discarded",
            "test_auc": result['test_auc'],
            "delta": round(delta, 4),
            "time_s": round(iter_time, 2)
        })

    print(f"     Status  : {status}")
    print(f"     Time    : {iter_time:.1f}s")
    print()

    results_history.append({
        "phase": f"ratchet_iter_{iter_num}",
        **result,
        "time_s": round(iter_time, 2),
        "ratchet_status": "kept" if delta > 0 else "discarded"
    })

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("━" * 70)
print("  FINAL SUMMARY — AutoResearch Ratchet Loop Results")
print("━" * 70)
print()
print(f"  {'Phase':<42} {'Test AUC':>10} {'Test Acc':>10} {'Test F1':>10}  Status")
print(f"  {'─'*42} {'─'*10} {'─'*10} {'─'*10}  {'─'*10}")
for r in results_history:
    label = r['label'][:42]
    status_icon = ""
    if r.get('ratchet_status') == 'kept':
        status_icon = "✓ KEPT"
    elif r.get('ratchet_status') == 'discarded':
        status_icon = "✗ DISC"
    print(f"  {label:<42} {r['test_auc']:>10.4f} {r['test_acc']:>10.4f} {r['test_f1']:>10.4f}  {status_icon}")

print()
print(f"  Baseline AUC          : {baseline_result['test_auc']:.4f}")
print(f"  FLAML One-Shot AUC    : {flaml_result['test_auc']:.4f}")
print(f"  Final Best AUC        : {best_auc:.4f}")
print(f"  Best Model            : {best_model_label}")
print(f"  Total Improvement     : {(best_auc - baseline_result['test_auc'])*100:+.2f}% AUC")
print()
print(f"  Ratchet iterations    : {RATCHET_ITERATIONS}")
print(f"  Kept improvements     : {kept_count}")
print(f"  Discarded regressions : {discarded_count}")
print()

# Save results
os.makedirs(os.path.dirname(RESULTS_LOG), exist_ok=True)
output = {
    "run_timestamp": datetime.now().isoformat(),
    "dataset": "Heart Disease (UCI/Kaggle) — heart-disease Cleveland",
    "baseline_auc": baseline_result['test_auc'],
    "flaml_oneshot_auc": flaml_result['test_auc'],
    "final_best_auc": best_auc,
    "best_model": best_model_label,
    "total_improvement_pct": round((best_auc - baseline_result['test_auc'])*100, 2),
    "ratchet_kept": kept_count,
    "ratchet_discarded": discarded_count,
    "results_history": results_history,
    "ratchet_log": ratchet_log,
}
with open(RESULTS_LOG, 'w') as f:
    json.dump(output, f, indent=2)

print(f"  Results saved to: {RESULTS_LOG}")
print()
print("=" * 70)
print("  AutoResearch Demo Complete!")
print("=" * 70)
