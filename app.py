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
st.markdown("Upload a dataset to train and compare multiple classification models instantly.")

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("1. Data Configuration")
    uploaded_file = st.file_uploader("Upload CSV dataset", type="csv")
    use_default = st.checkbox("Use default (Iris) dataset", value=False)
    
    st.divider()
    
    st.header("2. Hyperparameters")
    test_size = st.slider("Test set fraction", 0.1, 0.5, 0.2, 0.05)
    scale = st.checkbox("Scale features (StandardScaler)", value=True)
    random_state = st.number_input("Random Seed", value=42)

# --- Data Loading ---
df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif use_default:
    df = load_dataset(None, use_default=True)

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
        if st.button("Run Model Evaluation", type="primary"):
            with st.spinner("Validating..."):
                metrics_df, details = train_and_evaluate(
                    df, target_col, 
                    output_dir="model",
                    test_size=test_size, 
                    random_state=random_state, 
                    scale=scale
                )
                st.session_state['results'] = (metrics_df, details)
                st.success("Validation complete!")

        if 'results' in st.session_state:
            metrics_df, details = st.session_state['results']
            
            st.subheader("Model Comparison")
            st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['accuracy']), use_container_width=True)

            st.divider()

            col_sel, col_space = st.columns([1, 2])
            with col_sel:
                model_names = metrics_df['model'].tolist()
                sel = st.selectbox("🔍 Select model to inspect", model_names)
            info = details.get(sel)
            if info is not None:
                y_test = info['y_test']
                y_pred = info['y_pred']
                target_names = info.get('target_names')
                st.subheader(f"Detailed Analysis: {sel}")
    
           #info = details.get(sel)
           # if info:
           #     y_test, y_pred = info['y_test'], info['y_pred']

                c1, c2 = st.columns(2)
                
                with c1:
                    st.markdown("**Classification Report**")
                    cr = classification_report(y_test, y_pred, target_names=target_names)
                    st.code(cr)
                with c2:
                    st.markdown("**Confusion Matrix**")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(6, 5))
        
                    sns.heatmap(
                        cm, 
                        annot=True, 
                        fmt='d', 
                        cmap='Blues', 
                        ax=ax,
                        )
                    plt.ylabel('Actual Class')
                    plt.xlabel('Predicted Class')
                    st.pyplot(fig)
           
else:
    st.info("💡 Please upload a CSV file or check 'Use default dataset' to begin.")
