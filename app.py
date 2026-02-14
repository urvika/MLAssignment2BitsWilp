import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

from model.train_models import train_and_evaluate
from model.utils import load_dataset


st.title("2025AA05394 - ML Assignment 2 - Classification Models")

uploaded_file = st.file_uploader("Upload CSV dataset", type="csv")
use_default = st.checkbox("Use default (Iris) dataset")

df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif use_default:
    df = load_dataset(None, use_default=True)

if df is not None:
    st.subheader("Dataset Preview")
    st.write(df.head())

    target_default = df.columns[-1]
    target_col = st.selectbox("Select target column", df.columns, index=len(df.columns)-1)

    test_size = st.slider("Test set fraction", 0.1, 0.5, 0.2, 0.05)
    scale = st.checkbox("Scale features (StandardScaler)", value=True)

    if st.button("Train models"):
        with st.spinner("Training models — this may take a moment..."):
            metrics_df, details = train_and_evaluate(df, target_col, output_dir="outputs_streamlit",
                                                     test_size=test_size, random_state=42, scale=scale)

        st.success("Training complete")
        st.subheader("Metrics summary")
        st.dataframe(metrics_df)

        model_names = metrics_df['model'].tolist()
        sel = st.selectbox("Select model to inspect", model_names)
        info = details.get(sel)
        if info is not None:
            y_test = info['y_test']
            y_pred = info['y_pred']

            st.subheader("Classification Report")
            cr = classification_report(y_test, y_pred, output_dict=False)
            st.text(cr)

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)
