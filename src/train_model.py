"""
train_model.py — Phase 3 model training.

Time-based train/test split, not random. Randomly shuffling rows before
splitting would let the model train on trades that happened AFTER some of
its test trades - a temporal leakage that would make the reported accuracy
meaningless as an estimate of real-world (out-of-sample, forward-looking)
performance. Instead: sort by filing_date, train on the earliest 80%,
test on the most recent 20% - the same constraint a live trading signal
would actually face.

Also runs a walk-forward cross-validation (multiple time-based folds) and
a majority-class baseline, rather than reporting a single train/test split
as the final answer. A single fold's AUC on this dataset's size (~2-3k
rows) has a standard error large enough that one number alone isn't a
reliable verdict - and if that one fold happens to land entirely in an
unusual regime (e.g. the 2020 COVID crash/recovery), it says as much about
that period as about the model. Multiple folds + a baseline turn "is 0.45
good or bad?" into an answerable question.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.dummy import DummyClassifier

DATA_PATH = Path(__file__).parent.parent / "data" / "ml_features.csv"

CATEGORICAL_FEATURES = ["transaction_type", "owner", "asset_type"]
NUMERIC_FEATURES = ["log_amount_mid", "disclosure_delay_days", "ou_zscore"]
LABEL_COL = "beat_market_30d"
N_FOLDS = 5


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("filing_date").reset_index(drop=True)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    return df


def evaluate_split(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str]) -> dict:
    X_train, y_train = train_df[features], train_df[LABEL_COL]
    X_test, y_test = test_df[features], test_df[LABEL_COL]

    if y_train.nunique() < 2 or y_test.nunique() < 2 or len(test_df) < 20:
        return None  # degenerate fold (e.g. all-one-class or too small); skip rather than report a meaningless metric

    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4, num_leaves=15,
        min_child_samples=10, random_state=42, verbosity=-1,
    )
    model.fit(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES)
    pred_proba = model.predict_proba(X_test)[:, 1]
    pred_label = (pred_proba > 0.5).astype(int)

    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    baseline_proba = baseline.predict_proba(X_test)[:, 1]

    return {
        "n_train": len(train_df), "n_test": len(test_df),
        "test_start": test_df["filing_date"].min(), "test_end": test_df["filing_date"].max(),
        "auc": roc_auc_score(y_test, pred_proba),
        "baseline_auc": 0.5,  # a constant-probability baseline has AUC exactly 0.5 by definition
        "accuracy": accuracy_score(y_test, pred_label),
        "baseline_accuracy": accuracy_score(y_test, baseline.predict(X_test)),
        "precision": precision_score(y_test, pred_label, zero_division=0),
        "recall": recall_score(y_test, pred_label, zero_division=0),
        "model": model,
    }


def walk_forward_cv(df: pd.DataFrame, features: list[str], n_folds: int = N_FOLDS):
    """Expanding-window walk-forward validation: fold k trains on
    everything before a cutoff and tests on the next chunk, cutoffs moving
    forward through time. Gives several independent-ish AUC estimates
    instead of trusting one arbitrary 80/20 split."""
    n = len(df)
    fold_size = n // (n_folds + 1)
    results = []

    for k in range(1, n_folds + 1):
        train_end = fold_size * k
        test_end = min(fold_size * (k + 1), n)
        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]
        res = evaluate_split(train_df, test_df, features)
        if res is None:
            continue
        results.append(res)
        print(f"Fold {k}: train={res['n_train']} test={res['n_test']} "
              f"({res['test_start']} to {res['test_end']})  "
              f"AUC={res['auc']:.3f}  (baseline={res['baseline_auc']:.3f})")

    if not results:
        print("No valid folds (dataset too small or too imbalanced for CV).")
        return

    aucs = [r["auc"] for r in results]
    print(f"\nWalk-forward AUC across {len(results)} folds: "
          f"mean={np.mean(aucs):.3f}  std={np.std(aucs):.3f}  "
          f"min={np.min(aucs):.3f}  max={np.max(aucs):.3f}")
    print("(All folds compared against a 0.5 constant-probability baseline, "
          "since a majority-class-only classifier has AUC=0.5 by construction.)")
    return results


def train():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows.")
    if len(df) < 100:
        print("WARNING: dataset is quite small for a train/test split. "
              "Results below are a pipeline sanity check, not a reliable "
              "estimate of real-world performance - widen the ticker "
              "scope in fetch_prices.py and rerun build_features.py.")

    df = prepare(df)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    print("\n=== Walk-forward cross-validation (multiple time-based folds) ===")
    walk_forward_cv(df, features)

    print("\n=== Single 80/20 holdout (final fold, for feature importance) ===")
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    print(f"Train: {len(train_df)} rows ({train_df['filing_date'].min()} to {train_df['filing_date'].max()})")
    print(f"Test:  {len(test_df)} rows ({test_df['filing_date'].min()} to {test_df['filing_date'].max()})")
    print(f"Train positive rate: {train_df[LABEL_COL].mean():.1%} | "
          f"Test positive rate: {test_df[LABEL_COL].mean():.1%}")

    res = evaluate_split(train_df, test_df, features)
    if res is None:
        print("Final holdout fold was degenerate — skipping.")
        return

    print(f"\n--- Test set performance (time-based holdout) ---")
    print(f"AUC:       {res['auc']:.3f}  (0.5 = random, 1.0 = perfect)")
    print(f"Accuracy:  {res['accuracy']:.3f}  (majority-class baseline: {res['baseline_accuracy']:.3f})")
    print(f"Precision: {res['precision']:.3f}")
    print(f"Recall:    {res['recall']:.3f}")

    importance = pd.Series(res["model"].feature_importances_, index=features).sort_values(ascending=False)
    print(f"\n--- Feature importance ---")
    print(importance.to_string())

    return res["model"], res["auc"]


if __name__ == "__main__":
    train()
