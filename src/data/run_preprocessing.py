import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Ensure project root on path
root = Path(__file__).parents[2]
sys.path.insert(0, str(root))

from src.data.preprocess import preprocess, save_preprocessor, add_domain_features

# Ensure logs directory exists
(root / 'logs').mkdir(parents=True, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(root / 'logs' / 'preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
raw_dir = root / 'data' / 'raw'
processed_dir = root / 'data' / 'processed'
figures_dir = root / 'figures'
for dir_path in [processed_dir, figures_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)

def analyze_data(df, target_col=None):
    logger.info(f"Dataset shape: {df.shape}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing = missing[missing > 0]
        logger.warning(f"Missing values detected:\n{missing}")
    
    # Class distribution if target column provided
    if target_col and target_col in df.columns:
        class_counts = df[target_col].value_counts()
        logger.info(f"Class distribution:\n{class_counts}")
        
        # Calculate class imbalance ratio
        imbalance_ratio = class_counts.max() / class_counts.min()
        logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")
        
        # Create class distribution plot
        plt.figure(figsize=(8, 6))
        sns.countplot(x=target_col, data=df)
        plt.title(f'Class Distribution for {target_col}')
        plt.tight_layout()
        plt.savefig(figures_dir / 'class_distribution.png')
        plt.close()

def main():
    logger.info("Starting preprocessing pipeline")
    
    try:
        # Load raw data
        logger.info("Loading raw data")
        train_df = pd.read_csv(raw_dir / 'train_df.csv')
        test_df = pd.read_csv(raw_dir / 'test_df.csv')
        
        # Analyze the datasets
        logger.info("Analyzing training data")
        analyze_data(train_df, target_col='readmitted')
        logger.info("Analyzing test data")
        analyze_data(test_df)
        
        # Feature lists - expanded with domain knowledge
        numerical_features = ['age', 'num_procedures', 'days_in_hospital', 'comorbidity_score']
        categorical_features = ['gender', 'primary_diagnosis', 'discharge_to']
        
        # Feature correlation analysis
        if all(feat in train_df.columns for feat in numerical_features):
            logger.info("Analyzing feature correlations")
            corr_matrix = train_df[numerical_features].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(figures_dir / 'feature_correlations.png')
            plt.close()
        
        # Preprocess with enhanced options
        logger.info("Preprocessing training data")
        train_proc, preprocessor = preprocess(
            train_df, 
            numerical_features, 
            categorical_features,
            target_column='readmitted',
            fit=True,
            scaler_type='power',                 # Use PowerTransformer (Yeo-Johnson) to reduce skew
            imputer_type='knn',                  # KNN imputation for better missing value handling
            categorical_encoder='onehot',       # One-hot encoding for categorical features
            feature_selection='mutual_info',     # Use mutual information for feature selection
            selection_k=10,                      # Select top 10 features
            add_interactions=True                # Add interaction terms
        )
        
        logger.info("Preprocessing test data")
        test_proc, _ = preprocess(
            test_df, 
            numerical_features, 
            categorical_features, 
            preprocessor=preprocessor, 
            fit=False
        )
        
        # Analyze processed data
        logger.info(f"Processed training data shape: {train_proc.shape}")
        logger.info(f"Processed test data shape: {test_proc.shape}")
        
        # Save outputs
        logger.info("Saving processed data")
        train_proc.to_csv(processed_dir / 'train_processed.csv', index=False)
        test_proc.to_csv(processed_dir / 'test_processed.csv', index=False)
        
        # Balance processed training data and split
        logger.info("Balancing processed training data with SMOTE")
        X_proc = train_proc.values
        y_proc = train_df['readmitted'].values
        sm = SMOTE(random_state=42)
        X_balanced, y_balanced = sm.fit_resample(X_proc, y_proc)
        balanced_df = pd.DataFrame(X_balanced, columns=train_proc.columns)
        balanced_df['readmitted'] = y_balanced
        train_bal, test_bal = train_test_split(
            balanced_df, test_size=0.2, random_state=42, stratify=balanced_df['readmitted']
        )
        train_bal.to_csv(processed_dir / 'balanced_train.csv', index=False)
        test_bal.to_csv(processed_dir / 'balanced_test.csv', index=False)
        logger.info(f"Balanced train shape: {train_bal.shape}, Balanced test shape: {test_bal.shape}")
        
        # Save pipeline with metadata
        logger.info("Saving preprocessor")
        feature_names = {
            'numerical': numerical_features,
            'categorical': categorical_features
        }
        save_preprocessor(preprocessor, processed_dir / 'preprocessor.joblib', feature_names)
        
        # Save a version with original feature mappings for interpretability
        feature_mapping = pd.DataFrame({
            'original_feature': numerical_features + categorical_features,
            'type': ['numerical'] * len(numerical_features) + ['categorical'] * len(categorical_features)
        })
        feature_mapping.to_csv(processed_dir / 'feature_mapping.csv', index=False)
        
        logger.info("Preprocessing completed successfully")
        print(f"Processed train shape: {train_proc.shape}")
        print(f"Processed test shape: {test_proc.shape}")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()