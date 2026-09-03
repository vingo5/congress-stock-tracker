"""
train_model.py — Phase 3 model training.

Time-based train/test split, not random. Randomly shuffling rows before
splitting would let the model train on trades that happened AFTER some of
its test trades - a temporal leakage that would make the reported accuracy
meaningless as an estimate of real-world (out-of-sample, forward-looking)
performance. Instead: sort by filing_date, train on the earliest 80%,
test on the most recent 20% - the same constraint a live trading signal
would actually face.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score

DATA_PATH = Path(__file__).parent.parent / "data" / "ml_features.csv"

CATEGORICAL_FEATURES = ["transaction_type", "owner", "asset_type"]
NUMERIC_FEATURES = ["log_amount_mid", "disclosure_delay_days", "ou_zscore"]
LABEL_COL = "beat_market_30d"


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("filing_date").reset_index(drop=True)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    return df


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

    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    print(f"Train: {len(train_df)} rows ({train_df['filing_date'].min()} to {train_df['filing_date'].max()})")
    print(f"Test:  {len(test_df)} rows ({test_df['filing_date'].min()} to {test_df['filing_date'].max()})")
    print(f"Train positive rate: {train_df[LABEL_COL].mean():.1%} | "
          f"Test positive rate: {test_df[LABEL_COL].mean():.1%}")

    X_train, y_train = train_df[features], train_df[LABEL_COL]
    X_test, y_test = test_df[features], test_df[LABEL_COL]

    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_child_samples=10,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES)

    pred_proba = model.predict_proba(X_test)[:, 1]
    pred_label = (pred_proba > 0.5).astype(int)

    auc = roc_auc_score(y_test, pred_proba)
    acc = accuracy_score(y_test, pred_label)
    prec = precision_score(y_test, pred_label, zero_division=0)
    rec = recall_score(y_test, pred_label, zero_division=0)

    print(f"\n--- Test set performance (time-based holdout) ---")
    print(f"AUC:       {auc:.3f}  (0.5 = random, 1.0 = perfect)")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")

    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print(f"\n--- Feature importance ---")
    print(importance.to_string())

    return model, auc


if __name__ == "__main__":
    train()
