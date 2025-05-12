import os
import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, PowerTransformer, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures


def build_preprocessing_pipeline(numerical_features, categorical_features, 
                                 scaler_type='robust', imputer_type='knn',
                                 categorical_encoder='onehot', feature_selection=None,
                                 selection_k=10, add_interactions=True):
    transformers = []
    
    # Build numerical pipeline
    num_steps = []
    
    # Imputation for numerical features
    if imputer_type == 'median':
        num_steps.append(('imputer', SimpleImputer(strategy='median')))
    elif imputer_type == 'mean':
        num_steps.append(('imputer', SimpleImputer(strategy='mean')))
    elif imputer_type == 'knn':
        num_steps.append(('imputer', KNNImputer(n_neighbors=5)))
    
    # Scaling for numerical features
    if scaler_type == 'standard':
        num_steps.append(('scaler', StandardScaler()))
    elif scaler_type == 'robust':
        num_steps.append(('scaler', RobustScaler()))
    elif scaler_type == 'power':
        num_steps.append(('scaler', PowerTransformer(method='yeo-johnson')))
    
    # Add polynomial features if requested
    if add_interactions:
        num_steps.append(('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)))
    
    if numerical_features:
        num_pipeline = Pipeline(num_steps)
        transformers.append(('num', num_pipeline, numerical_features))
    
    # Build categorical pipeline
    cat_steps = []
    
    # Imputation for categorical features
    cat_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
    
    # Encoding for categorical features
    if categorical_encoder == 'onehot':
        cat_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
    elif categorical_encoder == 'ordinal':
        cat_steps.append(('encoder', OrdinalEncoder()))
    
    if categorical_features:
        cat_pipeline = Pipeline(cat_steps)
        transformers.append(('cat', cat_pipeline, categorical_features))
    
    # Create the column transformer
    preprocessor = ColumnTransformer(transformers, remainder='drop')
    
    # Final pipeline
    steps = [('preprocessor', preprocessor)]
    
    # Add feature selection if requested
    if feature_selection == 'mutual_info':
        steps.append(('selector', SelectKBest(mutual_info_classif, k=selection_k)))
    
    return Pipeline(steps)


def preprocess(df: pd.DataFrame,
               numerical_features: list,
               categorical_features: list,
               target_column=None,
               preprocessor=None,
               fit: bool = True,
               **kwargs):
    numerical_features = list(numerical_features)
    categorical_features = list(categorical_features)
    # Check for missing columns
    all_features = numerical_features + categorical_features
    missing_cols = [col for col in all_features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataframe: {missing_cols}")
    
    # Create domain-specific features for readmission prediction
    df = add_domain_features(df)
    
    # Update numerical_features list if we added derived numerical features
    for col in df.columns:
        if col.startswith('derived_') and col not in all_features and np.issubdtype(df[col].dtype, np.number):
            numerical_features.append(col)
    
    # Update categorical_features list if we added derived categorical features
    for col in df.columns:
        if col.startswith('derived_') and col not in categorical_features and df[col].dtype == object:
            categorical_features.append(col)
    
    # Group rare categories into 'Other' based on frequency threshold
    rare_thresh = kwargs.get('rare_threshold', 0.05)
    for col in categorical_features:
        freq = df[col].value_counts(normalize=True)
        rare_vals = freq[freq < rare_thresh].index.tolist()
        if rare_vals:
            df[col] = df[col].apply(lambda x: x if x not in rare_vals else 'Other')
    
    # Extract target if provided
    y = df[target_column].values if target_column in df.columns else None
    
    # Initialize preprocessor if not provided
    if preprocessor is None:
        preprocessor = build_preprocessing_pipeline(
            numerical_features, 
            categorical_features, 
            **kwargs
        )
    
    # Fit or transform
    if fit:
        # pass target array to pipeline for feature selection if available
        arr = preprocessor.fit_transform(df, y)
    else:
        arr = preprocessor.transform(df)
    
    # Handle feature names
    try:
        columns = preprocessor.get_feature_names_out()
    except (AttributeError, ValueError):
        # Fallback if get_feature_names_out not available
        columns = [f'feature_{i}' for i in range(arr.shape[1])]
    
    # Create output dataframe
    processed_df = pd.DataFrame(arr, columns=columns, index=df.index)
    
    return processed_df, preprocessor


def add_domain_features(df):
    df = df.copy()
    
    # Create comorbidity-related features
    if 'comorbidity_score' in df.columns:
        # High comorbidity is a strong readmission predictor - emphasize with log transform
        df['derived_log_comorbidity'] = np.log1p(df['comorbidity_score'])
        
        # Create risk category based on comorbidity (domain knowledge)
        conditions = [
            df['comorbidity_score'] <= 1,
            df['comorbidity_score'] <= 3,
            df['comorbidity_score'] > 3
        ]
        values = ['low', 'medium', 'high']
        df['derived_risk_category'] = np.select(conditions, values, default='unknown')
    
    # Length of stay vs. procedures ratio (efficiency of care)
    if 'days_in_hospital' in df.columns and 'num_procedures' in df.columns:
        # Avoid division by zero
        df['derived_days_per_procedure'] = df['days_in_hospital'] / df['num_procedures'].clip(lower=1)
    
    # Age-related risk (elderly patients have higher readmission risk)
    if 'age' in df.columns:
        df['derived_age_risk'] = np.where(df['age'] >= 65, 1, 0)
        
        # Age and comorbidity interaction
        if 'comorbidity_score' in df.columns:
            df['derived_age_comorbidity'] = df['age'] * df['comorbidity_score']
    
    # Procedure intensity (many procedures in short time may indicate complexity)
    if 'days_in_hospital' in df.columns and 'num_procedures' in df.columns:
        df['derived_procedure_intensity'] = df['num_procedures'] / df['days_in_hospital'].clip(lower=1)
    
    return df


def save_preprocessor(preprocessor, save_path: str, feature_names=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save preprocessor with metadata
    joblib.dump({
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'timestamp': pd.Timestamp.now().isoformat()
    }, save_path)


def load_preprocessor(load_path: str):
    loaded = joblib.load(load_path)
    
    if isinstance(loaded, dict) and 'preprocessor' in loaded:
        preprocessor = loaded['preprocessor']
        metadata = {k: v for k, v in loaded.items() if k != 'preprocessor'}
        return preprocessor, metadata
    else:
        # Handle legacy format
        return loaded, None