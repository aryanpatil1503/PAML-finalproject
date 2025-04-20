#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd

# ensure project root on path
root = Path(__file__).parents[2]
sys.path.insert(0, str(root))

from src.data.preprocess import preprocess, save_preprocessor

# directories
raw_dir = root / 'data' / 'raw'
processed_dir = root / 'data' / 'processed'
processed_dir.mkdir(parents=True, exist_ok=True)

# load raw data
train_df = pd.read_csv(raw_dir / 'train_df.csv')
test_df = pd.read_csv(raw_dir / 'test_df.csv')

# feature lists
numerical_features = ['age', 'num_procedures', 'days_in_hospital', 'comorbidity_score']
categorical_features = ['gender', 'primary_diagnosis', 'discharge_to']

# preprocess
train_proc, preprocessor = preprocess(train_df, numerical_features, categorical_features, fit=True)
test_proc, _ = preprocess(test_df, numerical_features, categorical_features, preprocessor, fit=False)

# save outputs
train_proc.to_csv(processed_dir / 'train_processed.csv', index=False)
test_proc.to_csv(processed_dir / 'test_processed.csv', index=False)

# save pipeline
save_preprocessor(preprocessor, processed_dir / 'preprocessor.joblib')

print(f"Processed train shape: {train_proc.shape}")
print(f"Processed test shape: {test_proc.shape}")
