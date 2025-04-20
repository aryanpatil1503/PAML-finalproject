#!/usr/bin/env python3
"""
Streamlit app for 30-day hospital readmission prediction.
"""
import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path
import io
import matplotlib.pyplot as plt
from src.data.eda import load_raw, basic_stats, univariate_numeric, univariate_categorical, correlation_matrix, missing_heatmap, pairwise_plot, numeric_scatter_matrix, boxplot_numeric_by_target, countplot_cat_by_target, scatter_numeric_by_target

# project root
def get_root():
    return Path(__file__).parents[1]
root = get_root()
sys.path.insert(0, str(root))

@st.cache_resource
def load_models_and_data():
    preprocessor = joblib.load(root / "data" / "processed" / "preprocessor.joblib")
    logreg = joblib.load(root / "data" / "processed" / "logreg_model.joblib")
    rf = joblib.load(root / "data" / "processed" / "rf_model.joblib")
    raw = pd.read_csv(root / "data" / "raw" / "train_df.csv")
    return preprocessor, logreg, rf, raw

preprocessor, logreg_model, rf_model, raw_df = load_models_and_data()

st.title("Hospital 30-Day Readmission Predictor")

# Sidebar inputs
page = st.sidebar.selectbox("Page", ["Predict", "Data Info"])

if page == "Data Info":
    train, _ = load_raw()
    st.header("Data Information")
    st.subheader("Dataset Shape")
    st.write(f"Rows: {train.shape[0]}, Columns: {train.shape[1]}")
    buf = io.StringIO()
    train.info(buf=buf)
    st.text(buf.getvalue())
    st.subheader("Descriptive Statistics")
    st.dataframe(train.describe(include='all'))
    st.subheader("Missing Values")
    missing = train.isnull().sum()
    st.bar_chart(missing)
    st.subheader("Value Counts per Column")
    for c in train.columns:
        vc = train[c].value_counts(dropna=False)
        st.write(f"**{c}**")
        st.write(vc)
        st.write(f"Sum of value counts: {vc.sum()}")
    st.subheader("Univariate Numeric Plots")
    num_cols = train.select_dtypes(include=['int64','float64']).columns.tolist()
    num_cols = [c for c in num_cols if c!='readmitted']

    def show_plots(func, *args, **kwargs):
        """Run an EDA function and display all generated figures in Streamlit."""
        before = set(plt.get_fignums())
        func(*args, **kwargs)
        after = set(plt.get_fignums())
        new_figs = sorted(after - before)
        for num in new_figs:
            fig = plt.figure(num)
            st.pyplot(fig)

    show_plots(univariate_numeric, train, num_cols)
    st.subheader("Univariate Categorical Plots")
    show_plots(univariate_categorical, train, train.select_dtypes(include=['object','category']).columns.tolist())
    st.subheader("Correlation Matrix")
    show_plots(correlation_matrix, train, num_cols)
    st.subheader("Missing Data Heatmap")
    show_plots(missing_heatmap, train)
    st.subheader("Pairwise Plot")
    show_plots(pairwise_plot, train, num_cols)
    st.subheader("Scatter Matrix")
    show_plots(numeric_scatter_matrix, train, num_cols)
    st.subheader("Boxplot Numeric by Target")
    show_plots(boxplot_numeric_by_target, train, num_cols)
    st.subheader("Countplot Categorical by Target")
    show_plots(countplot_cat_by_target, train, train.select_dtypes(include=['object','category']).columns.tolist())
    st.subheader("Scatter Numeric by Target")
    show_plots(scatter_numeric_by_target, train, num_cols)
    st.stop()

if page == "Predict":
    st.sidebar.header("Input Patient Data")
    # numeric features
    age = st.sidebar.number_input("Age", int(raw_df['age'].min()), int(raw_df['age'].max()), int(raw_df['age'].median()))
    num_procedures = st.sidebar.number_input("Number of Procedures", int(raw_df['num_procedures'].min()), int(raw_df['num_procedures'].max()), int(raw_df['num_procedures'].median()))
    days_in_hospital = st.sidebar.number_input("Days in Hospital", int(raw_df['days_in_hospital'].min()), int(raw_df['days_in_hospital'].max()), int(raw_df['days_in_hospital'].median()))
    comorbidity_score = st.sidebar.number_input("Comorbidity Score", float(raw_df['comorbidity_score'].min()), float(raw_df['comorbidity_score'].max()), float(raw_df['comorbidity_score'].median()))
    # categorical features
    gender = st.sidebar.selectbox("Gender", raw_df['gender'].unique())
    primary_diagnosis = st.sidebar.selectbox("Primary Diagnosis", raw_df['primary_diagnosis'].unique())
    discharge_to = st.sidebar.selectbox("Discharge To", raw_df['discharge_to'].unique())

    # Predict button
    if st.sidebar.button("Predict"):
        input_df = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'primary_diagnosis': [primary_diagnosis],
            'num_procedures': [num_procedures],
            'days_in_hospital': [days_in_hospital],
            'comorbidity_score': [comorbidity_score],
            'discharge_to': [discharge_to]
        })
        X_proc = preprocessor.transform(input_df)
        prob_lr = logreg_model.predict_proba(X_proc)[0]
        pred_lr = logreg_model.predict(X_proc)[0]
        prob_rf = rf_model.predict_proba(X_proc)[0]
        pred_rf = rf_model.predict(X_proc)[0]

        st.subheader("Prediction Results")
        st.write("**Logistic Regression**")
        st.write(f"Readmitted: {'Yes' if pred_lr==1 else 'No'}")
        st.write(f"Probability: {prob_lr:.2f}")
        st.write("**Random Forest**")
        st.write(f"Readmitted: {'Yes' if pred_rf==1 else 'No'}")
        st.write(f"Probability: {prob_rf:.2f}")

# Display hold-out metrics
st.sidebar.header("Hold-out Performance")
st.sidebar.metric("LR ROC-AUC", "0.5447")
st.sidebar.metric("RF ROC-AUC", "0.5920")
