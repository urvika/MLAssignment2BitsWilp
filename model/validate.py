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
        scaler = joblib.load("model/scaler.joblib")
# 1. Separate features and target
        X_val = data.drop(columns=[target_col]) if target_col in data.columns else data

        # 2. Handle Categorical Data (Mandatory for Math models like KNN)
        # This converts text to numbers
        for col in X_val.select_dtypes(include=['object']).columns:
            X_val[col] = X_val[col].astype('category').cat.codes

        # 3. Handle Feature Alignment
        if hasattr(model, 'feature_names_in_'):
            # The model knows what it wants!
            expected_features = model.feature_names_in_
            X_val = X_val[expected_features]
        else:
            # The model is "blind" to names. 
            # We must assume the first N columns are correct.
            # Warning: This is risky if your CSV columns moved!
            #st.warning("⚠️ Model has no feature names. Using the first columns found in CSV.")
            # If KNN expects 8 features, take the first 8
            # This is a fallback to prevent the 'expected 8, got 13' error
            num_features_expected = model.n_features_in_ if hasattr(model, 'n_features_in_') else None
            if num_features_expected:
                X_val = X_val.iloc[:, :num_features_expected] 
            has_ground_truth = target_col in data.columns
            y_true = data[target_col]

        """    
        # Check for ground truth
        has_ground_truth = target_col in data.columns
        if not has_ground_truth:
            X_val = data
            #X_val_scaled = scaler.transform(X_val)
            y_true = None
        else:
            X_val = data.drop(columns=[target_col])
            #X_val_scaled = scaler.transform(X_val)
            y_true = data[target_col]
# 2. Convert Categorical strings to numeric codes
        # This handles 'loan_intent', 'person_gender', etc.
        for col in X_val.select_dtypes(include=['object']).columns:
            X_val[col] = X_val[col].astype('category').cat.codes
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
            # Reorder columns and fill missing ones with 0
            X_val = X_val.reindex(columns=expected_features, fill_value=0)    
         """
        # Predict
        #y_pred = model.predict(X_val_scaled)
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
