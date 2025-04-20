"""
Module for preprocessing: imputation, encoding, and scaling pipeline.
"""
import os
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def build_preprocessing_pipeline(numerical_features, categorical_features):
    """
    Construct a preprocessing pipeline for numeric and categorical features.
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_features),
        ('cat', cat_pipeline, categorical_features)
    ])
    return preprocessor


def preprocess(df: pd.DataFrame,
               numerical_features: list,
               categorical_features: list,
               preprocessor=None,
               fit: bool = True):
    """
    Apply preprocessing to DataFrame. If fit=True, fit and transform; else only transform.
    Returns processed DataFrame and the fitted pipeline.
    """
    if preprocessor is None:
        preprocessor = build_preprocessing_pipeline(numerical_features, categorical_features)

    if fit:
        arr = preprocessor.fit_transform(df)
    else:
        arr = preprocessor.transform(df)

    # get output feature names
    columns = preprocessor.get_feature_names_out()
    processed_df = pd.DataFrame(arr, columns=columns, index=df.index)
    return processed_df, preprocessor


def save_preprocessor(preprocessor, save_path: str):
    """Save fitted preprocessor to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(preprocessor, save_path)


def load_preprocessor(load_path: str):
    """Load preprocessor from disk."""
    return joblib.load(load_path)
