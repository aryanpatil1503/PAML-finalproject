# PAML-finalproject
PAML final project- ABHT

## Overview
This project predicts 30-day hospital readmissions using both from-scratch implementations and sklearn models (Logistic Regression, Random Forest, XGBoost). It integrates data preprocessing, exploratory analysis, model training & evaluation, and a Streamlit web app for real-time risk prediction.

## Team
- **Aryan Patil**
- **Omkar Garad**
- **Neelraj Patil**

## Pipeline
The `pipeline.py` orchestrates the full workflow:
1. `src/data/run_preprocessing.py` – Missing value imputation, encoding, scaling  
2. `src/data/eda.py` – Exploratory data analysis & visualizations  
3. `src/models/train.py` – Model training & hyperparameter tuning  
4. `src/models/evaluate.py` – Metrics computation & plots

## Installation
```bash
pip install -r requirements.txt
```

## Usage
- **Run full pipeline**: `python pipeline.py`  
- **Run evaluation only**: `python src/models/evaluate.py`  
- **Launch web app**: `streamlit run src/app.py`

## Final Model Performance
Metrics (with optimal thresholds):

| Model               | Accuracy | F1 Score | ROC-AUC |
|---------------------|----------|----------|---------|
| Logistic Regression | 0.5086   | 0.6689   | 0.5279  |
| Random Forest       | 0.8498   | 0.8462   | 0.9121  |
| XGBoost             | 0.8368   | 0.8265   | 0.9015  |
