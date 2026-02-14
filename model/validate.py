import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    classification_report, confusion_matrix
)

def validate_model(model_path, data, target_col):
    """
    Loads a joblib model and evaluates performance on provided data.
    """
    try:
        model = joblib.load(model_path)
        
        # Check for ground truth
        has_ground_truth = target_col in data.columns
        if not has_ground_truth:
            X_val = data
            y_true = None
        else:
            X_val = data.drop(columns=[target_col])
            y_true = data[target_col]

        # Predict
        y_pred = model.predict(X_val)
        results = {"predictions": y_pred, "metrics": None, "y_true": y_true}

        if has_ground_truth:
            # Detect classification type
            is_binary = len(set(y_true)) <= 2
            avg = 'binary' if is_binary else 'weighted'

            metrics = {
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred, average=avg),
                "Recall": recall_score(y_true, y_pred, average=avg),
                "F1 Score": f1_score(y_true, y_pred, average=avg),
                "MCC": matthews_corrcoef(y_true, y_pred),
                "Report": classification_report(y_true, y_pred),
                "CM": confusion_matrix(y_true, y_pred),
                "Labels": sorted(list(set(y_true)))
            }

            # Probability metrics (AUC)
            if hasattr(model, "predict_proba"):
                y_probs = model.predict_proba(X_val)
                if is_binary:
                    metrics["AUC"] = roc_auc_score(y_true, y_probs[:, 1])
                else:
                    metrics["AUC"] = roc_auc_score(y_true, y_probs, multi_class='ovr')
            
            results["metrics"] = metrics
            
        return results
    except Exception as e:
        return {"error": str(e)}