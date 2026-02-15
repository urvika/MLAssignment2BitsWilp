# 2025AA05394 ML Assignment 2

## Problem Statement
Implement multiple classification models and demonstrate them via Streamlit.

## Dataset
- Source: https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data
- Features: ≥12
- Instances: ≥500

## Models Implemented
- Logistic Regression
- Decision Tree
- KNN
- Naive Bayes
- Random Forest
- XGBoost

## Evaluation Metrics
| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| Logistic Regression | 0.826556 | 0.837266 | 0.81183 | 0.826556 | 0.808616 | 0.432525 |  
| Decision Tree Classifier | 0.817111 | 0.735464 | 0.817111 | 0.817111 | 0.817111 | 0.470929 |
| K-Nearest Neighbor Classifier | 0.836667 | 0.808661 | 0.826518 | 0.836667 | 0.828653 | 0.491635 |
| Naive Bayes Classifier - Gaussian or Multinomial | 0.813 | 0.817976 | 0.807189 | 0.813 | 0.809685 | 0.441389 |
| Ensemble - Random Forest | 0.874778 | 0.899924 | 0.869189 | 0.874778 | 0.869179 | 0.614754 |
| Ensemble - XGBoost | 0.873556 | 0.901198 | 0.869189 |	0.873556 | 0.869123 |	0.614754 |

## Observation
| Model | Observation about model performance |
|:----:|:----:|
| Logistic Regression |  It achieved 82.6% accuracy but its lowest MCC (0.432) suggests it struggles with the non-linear relationships between variables like income and loan-to-value ratio, making it less reliable for catching subtle default patterns. | 
| Decision Tree Classifier | It showed high interpretability, it produced the lowest AUC (0.735) and overall performance. This indicates a high tendency to overfit the training data. |
| K-Nearest Neighbor Classifier | Showed moderate performance with 83.6% accuracy. Its reliance on feature distance means it performed reasonably well after scaling, but it remains computationally heavier than the tree-based models  |
| Naive Bayes Classifier - Gaussian or Multinomial | It got lowest overall accuracy (81.3%). This is due to its core assumption that all features (like credit history and income) are independent, which is rarely true in financial datasets. |
| Ensemble - Random Forest | It got highest Accuracy (87.47%) and MCC (0.61). By averaging multiple decision trees, it effectively eliminated the noise found in the single Decision Tree model, providing a very stable and reliable prediction for loan approvals. |
| Ensemble - XGBoost | From the metrics it is a good performing model overall with the highest AUC Score (0.901). It has a superior ability to distinguish between high-risk and low-risk applicants. Its boosting mechanism successfully captured complex patterns that other models missed. |
