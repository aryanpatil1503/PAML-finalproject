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
import seaborn as sns
from sklearn.decomposition import PCA

def get_root():
    return Path(__file__).parents[1]
root = get_root()
sys.path.insert(0, str(root))

from src.data.eda import load_raw
from src.data.preprocess import load_preprocessor, add_domain_features

@st.cache_resource
def load_models_and_data():
    preprocessor, _ = load_preprocessor(root / "data" / "processed" / "preprocessor.joblib")
    logreg = joblib.load(root / "data" / "processed" / "logreg_model_best.joblib")
    rf = joblib.load(root / "data" / "processed" / "rf_model_best.joblib")
    xgb = joblib.load(root / "data" / "processed" / "xgboost_model_best.joblib")
    raw = pd.read_csv(root / "data" / "raw" / "train_df.csv")
    return preprocessor, logreg, rf, xgb, raw

preprocessor, logreg_model, rf_model, xgb_model, raw_df = load_models_and_data()

st.title("Hospital 30-Day Readmission Predictor")

# Sidebar inputs
page = st.sidebar.selectbox("Page", ["Predict", "Data Info"])

if page == "Data Info":
    train, _ = load_raw()
    st.title("Exploratory Data Analysis")
    # Data Description
    st.header("First 5 Rows")
    st.markdown(train.head().to_markdown(), unsafe_allow_html=True)
    st.header("Column Data Types")
    st.markdown(train.dtypes.to_frame(name="dtype").to_markdown(), unsafe_allow_html=True)
    st.header("Summary Statistics")
    st.markdown(train.describe(include='all').to_markdown(), unsafe_allow_html=True)
    # Missing Value Heatmap
    st.header("Missing Value Heatmap")
    if train.isnull().values.any():
        fig = plt.figure(figsize=(10,6))
        sns.heatmap(train.isnull(), cbar=False)
        st.pyplot(fig)
    else:
        st.write("No missing values in the dataset.")
    # Class Distribution
    st.header("Readmission Class Distribution")
    fig = plt.figure(figsize=(6,4))
    sns.countplot(x='readmitted', data=train, palette='Set2')
    plt.title("Readmission Class Counts")
    st.pyplot(fig)
    # Univariate Analysis – Numeric Features
    st.header("Univariate Analysis – Numeric Features")
    num_cols = train.select_dtypes(include=['number']).columns.tolist()
    if 'readmitted' in num_cols:
        num_cols.remove('readmitted')
    for col in num_cols:
        st.subheader(f"{col} Distribution")
        fig = plt.figure()
        sns.histplot(train[col].dropna(), kde=True, color='skyblue')
        st.pyplot(fig)
        st.subheader(f"{col} Boxplot")
        fig = plt.figure()
        sns.boxplot(x=train[col].dropna(), color='lightcoral')
        st.pyplot(fig)
    # Univariate Analysis – Categorical Features
    st.header("Univariate Analysis – Categorical Features")
    cat_cols = train.select_dtypes(include=['object','category']).columns.tolist()
    for col in cat_cols:
        st.subheader(f"{col} Counts")
        fig = plt.figure()
        sns.countplot(x=col, data=train, palette='Set2')
        st.pyplot(fig)
    # Bivariate Analysis – Correlation Matrix
    st.header("Correlation Matrix")
    corr = train[num_cols].corr()
    fig = plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    st.pyplot(fig)
    # Bivariate Analysis – Numeric vs Readmission
    st.header("Numeric Features by Readmission")
    for col in num_cols:
        st.subheader(f"{col} by Readmission")
        fig = plt.figure()
        sns.boxplot(x='readmitted', y=col, data=train, palette='Set2')
        st.pyplot(fig)
    # PCA – First Two Principal Components
    st.header("PCA – First Two Principal Components")
    pca = PCA(n_components=2)
    comps = pca.fit_transform(train[num_cols].fillna(0))
    fig = plt.figure()
    sns.scatterplot(x=comps[:,0], y=comps[:,1], hue=train['readmitted'], palette='Set1')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    st.pyplot(fig)
    # Feature Importance – Random Forest Model
    st.header("Feature Importance – Random Forest Model")
    try:
        feats = preprocessor.get_feature_names_out()
    except:
        feats = preprocessor.feature_names_in_
    importances = rf_model.feature_importances_
    imp_df = pd.DataFrame({'feature':feats, 'importance':importances})
    imp_df = imp_df.sort_values('importance', ascending=False)
    fig = plt.figure(figsize=(8,len(feats)*0.3))
    sns.barplot(x='importance', y='feature', data=imp_df, palette='viridis')
    st.pyplot(fig)
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
        # add derived features before preprocessing
        input_df = add_domain_features(input_df)
        X_proc = preprocessor.transform(input_df)
        prob_lr = logreg_model.predict_proba(X_proc)[0]
        pred_lr = logreg_model.predict(X_proc)[0]
        prob_rf = rf_model.predict_proba(X_proc)[0]
        pred_rf = rf_model.predict(X_proc)[0]
        prob_xgb = xgb_model.predict_proba(X_proc)[0,1]
        pred_xgb = xgb_model.predict(X_proc)[0]

        st.subheader("Prediction Results")
        st.write("**Logistic Regression**")
        st.write(f"Readmitted: {'Yes' if pred_lr==1 else 'No'}")
        st.write(f"Probability: {prob_lr:.2f}")
        st.write("**Random Forest**")
        st.write(f"Readmitted: {'Yes' if pred_rf==1 else 'No'}")
        st.write(f"Probability: {prob_rf:.2f}")
        st.write("**XGBoost**")
        st.write(f"Readmitted: {'Yes' if pred_xgb==1 else 'No'}")
        st.write(f"Probability: {prob_xgb:.2f}")

# Display hold-out metrics (from evaluation reports if needed)
# st.sidebar.header("Hold-out Performance")
# ... metrics loaded dynamically or view reports/figures
