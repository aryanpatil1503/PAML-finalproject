# Final Project Mid-point Check-in

**Group Members:** Aryan Patil, Omkar Garad, Neelraj Patil  
**Project Title:** Predicting 30-Day Hospital Readmissions Using Scratch ML Models & Streamlit

---
## Overview of Accomplishments So Far

- **Data Pipeline:** Implemented preprocessing (`src/data/run_preprocessing.py`): imputation, encoding, scaling; outputs processed CSVs and saved preprocessor.
- **Exploratory Data Analysis:** Developed `src/data/eda.py` with basic stats, descriptive tables, univariate/bivariate plots (histograms, boxplots, countplots, correlation matrix, pairplots, scatter matrix, heatmaps).
- **Model Implementations:** 
  - `LogisticRegressionScratch` (`src/models/logistic_regression.py`): gradient descent, L2 regularization.
  - `DecisionTreeScratch` & `RandomForestScratch` (`src/models/random_forest.py`).
- **Training & Evaluation Scripts:**
  - `train.py`: 5-fold CV logging (LR avg ROC-AUC ~0.47; RF avg ROC-AUC ~0.59), final model saving.
  - `evaluate.py`: 20% hold‑out evaluation (LR ROC-AUC=0.5447; RF ROC-AUC=0.5920).
- **Streamlit App:** Created multi-page app (`src/app.py`) for:
  - **Predict:** Interactive patient input → real-time LR & RF predictions.
  - **Data Info:** Display raw data stats and full suite of EDA plots.
  - Fixed module import & caching issues; updated to `@st.cache_resource`.

**UI Sketch:**  
*(See attached `assets/ui_sketch.png` for mock-up of sidebar & output layout.)*

## Experiments & Results

- **5-Fold CV:** 
  - Logistic Regression: mean ROC-AUC ≈ 0.4723 ± 0.0092
  - Random Forest:     mean ROC-AUC ≈ 0.5921 ± 0.0065
- **Hold-out Set (20%):**
  - Logistic Regression: ROC-AUC = 0.5447
  - Random Forest:        ROC-AUC = 0.5920

## Website Development

- Built a Streamlit front-end integrating preprocessor and both models.
- Sidebar input widgets + prediction display.
- Multi-page navigation for predictions vs. data insight.

## Scope & Goals Revisions

- Prioritized comprehensive EDA and evaluation pipelines before expanding UI features, reducing initial front-end complexity.


## Remaining Work

1. Implement automated hyperparameter tuning (grid search/CV) and integrate summary into the app.  
2. Add model interpretability (feature importances, SHAP explanations).  
3. Enhance UI: performance dashboard with interactive charts of CV/test metrics.  
4. Containerize & deploy the app (e.g., Heroku/GCP).  
5. Final testing, documentation, & write-up.

---
*This document follows single-column, single-spaced, 1" margins, 12pt font, and fits on one page.*
