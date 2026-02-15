import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Assuming these are your custom local modules
from model.train_models import train_and_evaluate
from model.utils import load_dataset
from model.validate import validate_model

# Page Config
st.set_page_config(page_title="ML Classifier", layout="wide")

st.title("🚀 ML Assignment 2 - Classification Dashboard")
st.markdown("Upload a dataset and compare multiple classification models instantly.")

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("1. Data Configuration")
    uploaded_file = st.file_uploader("Upload CSV dataset", type="csv")
    use_default = st.checkbox("Use the test dataset", value=False)

# --- Data Loading ---
df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif use_default:
    df = load_dataset("model/loan_test.csv", use_default=True)

if df is not None:
    tab1, tab2 = st.tabs(["📊 Data Exploration", "🤖 Model Validation"])

    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Dataset Shape:**", df.shape)
        with col2:
            target_col = st.selectbox("🎯 Select target column", df.columns, index=len(df.columns)-1)

    with tab2:
        st.subheader("Model Validation & Metrics")
        available_models = [f for f in os.listdir("model") if f.endswith(".joblib")]
        if not available_models:
            st.warning("No models found in /model folder.")
        else:
            selected_model = st.selectbox("Select Model", available_models)
            res = validate_model(os.path.join("model", selected_model), df, target_col)
            if "error" in res:
                st.error(res["error"])
            else:
                m = res["metrics"]
                # Metric Grid
                st.divider()
                r1_c1, r1_c2, r1_c3 = st.columns(3)
                r2_c1, r2_c2, r2_c3 = st.columns(3)
                
                r1_c1.metric("Accuracy", f"{m['Accuracy']:.2%}")
                r1_c2.metric("AUC Score", f"{m.get('AUC', 0):.4f}")
                r1_c3.metric("MCC Score", f"{m['MCC']:.4f}")
                
                r2_c1.metric("Precision", f"{m['Precision']:.2%}")
                r2_c2.metric("Recall", f"{m['Recall']:.2%}")
                r2_c3.metric("F1 Score", f"{m['F1 Score']:.2%}")

                # Report & Matrix
                st.divider()
                col_rep, col_mat = st.columns([1, 1])
                
                with col_rep:
                    st.write("**Classification Report**")
                    st.code(m["Report"])
                
                with col_mat:
                    st.write("**Confusion Matrix**")
                    fig, ax = plt.subplots()
                    sns.heatmap(m["CM"], annot=True, fmt='d', cmap='Blues', 
                                xticklabels=m["Labels"], yticklabels=m["Labels"])
                    st.pyplot(fig)              
else:
    st.info("💡 Please upload a CSV file or check 'Use default dataset' to begin.")
