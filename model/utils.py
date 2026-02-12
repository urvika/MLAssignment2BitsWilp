"""
Utility helpers for dataset loading and preprocessing used by train_models.py
"""
import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_dataset(path=None, use_default=False):
    """Load dataset from CSV path or return sklearn iris as DataFrame when use_default=True.
    If path provided, reads CSV and returns DataFrame.
    """
    if path is None and use_default:
        data = load_iris(as_frame=True)
        df = data.frame.copy()
        df.rename(columns={"target": "target"}, inplace=True)
        # sklearn's iris has target already in frame
        return df

    if path is not None:
        df = pd.read_csv(path)
        return df

    raise ValueError("No data path provided and use_default is False")


def preprocess_features_targets(df: pd.DataFrame, target_col: str):
    """Preprocess dataframe: drop missing, one-hot encode categorical features, label-encode target."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe columns: {df.columns.tolist()}")

    df = df.copy()
    df = df.dropna()

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # One-hot encode categorical/object columns
    obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if obj_cols:
        X = pd.get_dummies(X, columns=obj_cols, drop_first=True)

    # Ensure X is numeric
    X = X.select_dtypes(include=["number"]).astype(float)

    # Encode target
    if y.dtype == object or y.dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        # if numeric but not integer labels, try to convert
        try:
            y = y.astype(int)
        except Exception:
            le = LabelEncoder()
            y = le.fit_transform(y)

    return X, y


def save_metrics(df_metrics: pd.DataFrame, output_dir: str):
    ensure_dir(output_dir)
    out_path = os.path.join(output_dir, "metrics_summary.csv")
    df_metrics.to_csv(out_path, index=False)
    print(f"Saved metrics to {out_path}")
