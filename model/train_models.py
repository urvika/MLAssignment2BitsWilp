"""
Train multiple classification models on a user-provided dataset (or default dataset).
Saves trained models and a metrics summary CSV into the output directory.

Usage examples:
python -m model.train_models --use-default --target target --output-dir outputs
python -m model.train_models --data data.csv --target Species --output-dir outputs
"""

import os
import argparse
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, matthews_corrcoef
from sklearn.metrics import roc_auc_score
import joblib

from model.utils import load_dataset, preprocess_features_targets, ensure_dir, save_metrics

warnings.filterwarnings("ignore")

def train_and_evaluate(df, target_col, output_dir, test_size=0.2, random_state=42, scale=True):
    X, y = preprocess_features_targets(df, target_col)
    
    # --- NEW: Get unique class names for the classification report ---
    # We convert to string to ensure the classification_report can label them properly
    class_names = [str(c) for c in np.unique(y)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y))>1 else None
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # ... [Model dictionary definition remains the same] ...
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_state),
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "KNN": KNeighborsClassifier(),
        "GaussianNB": GaussianNB(),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state),
    }

   # XGBoost (may be slower) - include if available
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    except Exception:
        print("XGBoost not available; skipping XGBoost model. Install xgboost to enable it.")

    results = []
    details = {}
    #models_dir = os.path.join(output_dir, "models")
    models_dir = output_dir
    ensure_dir(models_dir)

    
    # Add target_names to the top level of details so Streamlit can find them easily
    details['target_names'] = class_names

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        y_probs = clf.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_probs)
        results.append({
            "model": name,
            "accuracy": acc,
            "mcc-score": mcc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc-score": auc_score,
        })
        # save model
        joblib.dump(clf, os.path.join(models_dir, f"{name}.joblib"))
        details[name] = {
            "y_test": y_test,
            "y_pred": y_pred,
            # Pass the names into the specific model detail too
            "target_names": class_names 
        }
    joblib.dump(scaler, "model/scaler.joblib")
    metrics_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False)
    save_metrics(metrics_df, output_dir)
    return metrics_df, details


def parse_args():
    parser = argparse.ArgumentParser(description="Train several classifiers on a dataset")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset. If omitted, default dataset is used")
    parser.add_argument("--target", type=str, default=None, help="Name of the target column (required if --data given)")
    parser.add_argument("--use-default", action="store_true", help="Use sklearn's iris dataset if no CSV provided")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save models and metrics")
    parser.add_argument("--no-scale", dest="scale", action="store_false", help="Disable scaling of features")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    if args.data is None and not args.use_default:
        print("No dataset provided. Use --data <path> or --use-default to use iris dataset.")
        return

    df = load_dataset(args.data, use_default=args.use_default)

    if args.data is not None and args.target is None:
        print("When using --data you must provide --target column name.")
        return

    # If user provided --target use it; else for default dataset use 'target'
    target_col = args.target if args.target is not None else 'target'

    metrics, details = train_and_evaluate(df, target_col, args.output_dir, test_size=args.test_size,
                                 random_state=args.random_state, scale=args.scale)

    print("\nTraining complete. Metrics summary:")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
