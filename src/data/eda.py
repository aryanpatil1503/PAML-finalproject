import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


sns.set_style('whitegrid')
sns.set_palette('Set2')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
FIGURES_DIR = os.path.join(ROOT, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_raw():
    """Load raw train and test data."""
    raw_dir = os.path.join(ROOT, "data", "raw")
    train = pd.read_csv(os.path.join(raw_dir, "train_df.csv"))
    test = pd.read_csv(os.path.join(raw_dir, "test_df.csv"))
    return train, test


def save_figure(name):
    plt.savefig(os.path.join(FIGURES_DIR, f"{name}.png"), dpi=300, bbox_inches='tight')
    plt.close()


def basic_stats(df, save_to=None):
    print("\n=== Dataset Shape ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    print("\n=== Data Types ===")
    dtype_counts = df.dtypes.value_counts()
    print(dtype_counts)
    
    print("\n=== Column Information ===")
    col_info = pd.DataFrame({
        'dtype': df.dtypes,
        'non_null': df.count(),
        'null_count': df.isnull().sum(),
        'null_pct': (df.isnull().sum() / len(df) * 100).round(2),
        'nunique': df.nunique(),
        'unique_pct': (df.nunique() / len(df) * 100).round(2)
    })
    print(col_info)
    
    print("\n=== Numerical Summary ===")
    numeric_summary = df.describe().T
    numeric_summary['skew'] = df.select_dtypes(include=['number']).skew()
    numeric_summary['kurtosis'] = df.select_dtypes(include=['number']).kurtosis()
    print(numeric_summary)
    
    if save_to:
        # Save stats to file
        with open(os.path.join(ROOT, save_to), 'w') as f:
            f.write("=== Dataset Shape ===\n")
            f.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n\n")
            f.write("=== Data Types ===\n")
            f.write(str(dtype_counts) + "\n\n")
            f.write("=== Column Information ===\n")
            f.write(str(col_info) + "\n\n")
            f.write("=== Numerical Summary ===\n")
            f.write(str(numeric_summary))
    
    return col_info, numeric_summary


def class_imbalance_analysis(df, target='readmitted'):
    """Analyze and visualize class imbalance in the target variable."""
    print("\n=== Class Imbalance Analysis ===")
    class_counts = df[target].value_counts()
    class_pcts = df[target].value_counts(normalize=True) * 100
    
    imbalance_ratio = class_counts.max() / class_counts.min()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    
    print(f"Class counts: {class_counts.to_dict()}")
    print(f"Class percentages: {class_pcts.round(2).to_dict()}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
    print(f"Majority class: {majority_class} ({class_counts[majority_class]} samples, {class_pcts[majority_class]:.2f}%)")
    print(f"Minority class: {minority_class} ({class_counts[minority_class]} samples, {class_pcts[minority_class]:.2f}%)")
    
    # Visualize class imbalance
    plt.figure(figsize=(10, 6))
    sns.countplot(x=target, data=df, palette=['lightcoral', 'skyblue'])
    
    # Add count and percentage annotations
    for i, count in enumerate(class_counts):
        plt.text(i, count + 50, f"Count: {count}", ha='center')
        plt.text(i, count/2, f"{class_pcts[i]:.1f}%", ha='center', color='white', fontweight='bold')
    
    plt.title('Class Distribution for Readmission', fontsize=15, fontweight='bold')
    plt.ylabel('Count')
    plt.xlabel('Readmitted')
    plt.tight_layout()
    save_figure('class_imbalance')
    
    # Calculate sufficient sample size for minority class (rule of thumb)
    feature_count = len(df.columns) - 1  # excluding target
    min_samples_rule = 10 * feature_count
    
    print(f"\nFeature count: {feature_count}")
    print(f"Recommended minimum samples per class (10x rule): {min_samples_rule}")
    if class_counts[minority_class] < min_samples_rule:
        print(f"WARNING: Minority class has {class_counts[minority_class]} samples, " 
              f"which is less than the recommended {min_samples_rule}.")
        print("Consider: Oversampling, SMOTE, class weights, or collecting more data.")
    
    return imbalance_ratio


def univariate_analysis(df, cols=None, target='readmitted'):
    if cols is None:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)
    else:
        numeric_cols = [c for c in cols if c in df.columns and c != target]
    
    results = pd.DataFrame(columns=['mean', 'median', 'skew', 'kurtosis', 'shapiro_p', 'transform_suggestion'])
    
    for col in numeric_cols:
        data = df[col].dropna()
        
        # Calculate statistics
        mean = data.mean()
        median = data.median()
        skew = data.skew()
        kurt = data.kurtosis()
        
        # Normality test
        shapiro_stat, shapiro_p = stats.shapiro(data.sample(min(1000, len(data))))
        
        # Transformation suggestion
        transform_suggestion = 'None'
        if abs(skew) > 1:
            if skew > 0:  # Right-skewed
                transform_suggestion = 'Log or Sqrt'
            else:  # Left-skewed
                transform_suggestion = 'Square or Exp'
        
        results.loc[col] = [mean, median, skew, kurt, shapiro_p, transform_suggestion]
        
        # Plot distribution
        plt.figure(figsize=(12, 5))
        
        # Original distribution
        plt.subplot(1, 2, 1)
        sns.histplot(data=df, x=col, hue=target, kde=True, bins=30, palette=['lightcoral', 'skyblue'])
        plt.title(f'Distribution of {col}', fontsize=14)
        plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
        plt.axvline(median, color='green', linestyle='--', label=f'Median: {median:.2f}')
        plt.legend()
        
        # Boxplot by target
        plt.subplot(1, 2, 2)
        sns.boxplot(x=target, y=col, data=df, palette=['lightcoral', 'skyblue'])
        plt.title(f'Boxplot of {col} by Readmission', fontsize=14)
        
        plt.tight_layout()
        save_figure(f'univariate_{col}')
        
        # If highly skewed, show transformed versions
        if abs(skew) > 1:
            plt.figure(figsize=(15, 5))
            
            # Log transformation (for right-skewed)
            plt.subplot(1, 3, 1)
            if all(data > 0):  # Log requires positive values
                sns.histplot(np.log1p(data), kde=True, color='teal')
                plt.title(f'Log Transformation of {col}')
                plt.xlabel(f'Log({col}+1)')
            else:
                plt.text(0.5, 0.5, "Can't apply log (has zero/negative values)", 
                         ha='center', va='center', transform=plt.gca().transAxes)
            
            # Square root transformation (for right-skewed)
            plt.subplot(1, 3, 2)
            if all(data >= 0):  # Sqrt requires non-negative values
                sns.histplot(np.sqrt(data), kde=True, color='purple')
                plt.title(f'Square Root Transformation of {col}')
                plt.xlabel(f'Sqrt({col})')
            else:
                plt.text(0.5, 0.5, "Can't apply sqrt (has negative values)", 
                         ha='center', va='center', transform=plt.gca().transAxes)
            
            # Box-Cox transformation (more flexible)
            plt.subplot(1, 3, 3)
            if all(data > 0):  # Box-Cox requires positive values
                try:
                    transformed_data, lambda_val = stats.boxcox(data)
                    sns.histplot(transformed_data, kde=True, color='orange')
                    plt.title(f'Box-Cox Transformation of {col} (λ={lambda_val:.3f})')
                except:
                    plt.text(0.5, 0.5, "Box-Cox transformation failed", 
                             ha='center', va='center', transform=plt.gca().transAxes)
            else:
                plt.text(0.5, 0.5, "Can't apply Box-Cox (has non-positive values)", 
                         ha='center', va='center', transform=plt.gca().transAxes)
            
            plt.tight_layout()
            save_figure(f'transforms_{col}')
    
    print("\n=== Univariate Analysis Results ===")
    print(results)
    
    # Highlight features that need transformation
    skewed_features = results[abs(results['skew']) > 1].index.tolist()
    if skewed_features:
        print("\nFeatures with significant skew that may benefit from transformation:")
        for feat in skewed_features:
            print(f"  - {feat}: Skew={results.loc[feat, 'skew']:.2f}, " 
                  f"Suggested transform: {results.loc[feat, 'transform_suggestion']}")
    
    return results


def categorical_analysis(df, cols=None, target='readmitted'):
    if cols is None:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        cat_cols = [c for c in cols if c in df.columns]
    
    results = pd.DataFrame(columns=['nunique', 'top_value', 'top_freq', 'chi2_p', 'mi_score'])
    
    for col in cat_cols:
        # Calculate statistics
        nunique = df[col].nunique()
        top_value = df[col].value_counts().index[0]
        top_freq = df[col].value_counts(normalize=True).iloc[0] * 100
        
        # Chi-square test for independence with target
        contingency_table = pd.crosstab(df[col], df[target])
        chi2_stat, chi2_p, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Mutual information with target
        codes = pd.factorize(df[col])[0]
        mi_score = mutual_info_classif(
            codes.reshape(-1, 1),
            df[target].values,
            random_state=42
        )[0]
        
        results.loc[col] = [nunique, top_value, top_freq, chi2_p, mi_score]
        
        # Visualization
        plt.figure(figsize=(12, 10))
        
        # Count by category
        plt.subplot(2, 1, 1)
        value_counts = df[col].value_counts().sort_values(ascending=False)
        ax = sns.barplot(x=value_counts.index, y=value_counts.values, palette='viridis')
        plt.title(f'Value Counts for {col}', fontsize=14)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        
        # Annotate bars with percentages
        for i, p in enumerate(ax.patches):
            percentage = 100 * p.get_height() / len(df)
            ax.annotate(f'{percentage:.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points')
        
        # Distribution by target
        plt.subplot(2, 1, 2)
        cross_tab_pct = pd.crosstab(df[col], df[target], normalize='index') * 100
        cross_tab_pct.plot(kind='bar', stacked=True, colormap='viridis')
        plt.title(f'Percentage of Readmission by {col}', fontsize=14)
        plt.ylabel('Percentage (%)')
        plt.xlabel(col)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Readmitted')
        
        plt.tight_layout()
        save_figure(f'categorical_{col}')
    
    print("\n=== Categorical Analysis Results ===")
    print(results)
    
    # Highlight features with strong target relationship
    significant_cats = results[results['chi2_p'] < 0.05].sort_values('mi_score', ascending=False).index.tolist()
    if significant_cats:
        print("\nCategorical features with significant relationship to readmission:")
        for feat in significant_cats:
            print(f"  - {feat}: Chi2 p-value={results.loc[feat, 'chi2_p']:.4f}, "
                  f"MI score={results.loc[feat, 'mi_score']:.4f}")
    
    return results


def feature_importance_analysis(df, target='readmitted'):
    X = df.drop(columns=[target])
    y = df[target]
    
    # Handle categorical features
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Store results for different methods
    importance_results = pd.DataFrame(index=X.columns)
    
    # 1. Mutual Information
    print("\n=== Feature Importance Analysis ===")
    print("Computing mutual information scores...")
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numeric_cols = X.select_dtypes(include=['number']).columns
    
    mi_scores = []
    for col in X.columns:
        if col in categorical_cols:
            # For categorical features
            codes = pd.factorize(X[col])[0]
            mi = mutual_info_classif(
                codes.reshape(-1, 1),
                y.values,
                random_state=42
            )[0]
        else:
            # For numeric features
            mi = mutual_info_classif(
                X[col].values.reshape(-1, 1),
                y.values,
                random_state=42
            )[0]
        mi_scores.append(mi)
    
    importance_results['mi_score'] = mi_scores
    
    # 2. Random Forest Feature Importance
    print("Computing random forest importance scores...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_encoded, y)
    # Aggregate dummy importances back to original features
    importances = pd.Series(rf.feature_importances_, index=X_encoded.columns)
    # Map each encoded column to its original feature (handle dummies)
    orig_importances = (
        importances
        .groupby(lambda name: name if name in X.columns else name.split('_')[0])
        .sum()
    )
    importance_results['rf_importance'] = orig_importances.reindex(importance_results.index).fillna(0)

    # Normalize scores
    for col in importance_results.columns:
        importance_results[col] = importance_results[col] / importance_results[col].sum()
    
    # Add rank columns
    for col in importance_results.columns:
        importance_results[f'{col}_rank'] = importance_results[col].rank(ascending=False)
    
    # Calculate average importance and rank
    importance_results['avg_importance'] = importance_results.filter(like='importance').mean(axis=1)
    importance_results['avg_rank'] = importance_results.filter(like='rank').mean(axis=1)
    
    # Sort by average importance
    importance_results = importance_results.sort_values('avg_importance', ascending=False)
    
    # Visualize top features
    top_n = min(10, len(importance_results))
    top_features = importance_results.head(top_n).index
    
    plt.figure(figsize=(12, 8))
    
    # Plot MI scores
    plt.subplot(2, 1, 1)
    sns.barplot(x=importance_results.loc[top_features, 'mi_score'], 
                y=top_features, palette='viridis')
    plt.title('Top Features by Mutual Information', fontsize=14)
    plt.xlabel('Mutual Information Score')
    
    # Plot RF importance
    plt.subplot(2, 1, 2)
    sns.barplot(x=importance_results.loc[top_features, 'rf_importance'], 
                y=top_features, palette='viridis')
    plt.title('Top Features by Random Forest Importance', fontsize=14)
    plt.xlabel('Random Forest Importance')
    
    plt.tight_layout()
    save_figure('feature_importance')
    
    print("\nTop 10 Features by Average Importance:")
    print(importance_results[['mi_score', 'rf_importance', 'avg_importance']].head(10))
    
    # Identify features with low importance
    low_importance_threshold = 0.01
    low_importance_features = importance_results[importance_results['avg_importance'] < low_importance_threshold].index.tolist()
    
    if low_importance_features:
        print(f"\nFeatures with low importance (< {low_importance_threshold}):")
        for feat in low_importance_features:
            print(f"  - {feat}: Avg Importance = {importance_results.loc[feat, 'avg_importance']:.4f}")
        print("\nConsider removing these features for model simplification.")
    
    return importance_results


def correlation_analysis(df, target='readmitted'):
    # Select numeric columns excluding the target
    num_feats = [c for c in df.select_dtypes(include=['number']).columns if c != target]
    # Compute Pearson correlation with target
    corrs = df[num_feats].corrwith(df[target])
    # Sort by absolute correlation descending
    corrs = corrs.reindex(corrs.abs().sort_values(ascending=False).index)
    # Print sorted correlations
    print("\nFeature correlation with target (readmitted):")
    for feat, val in corrs.items():
        print(f"  - {feat}: {val:.4f}")
    # Plot bar chart
    plt.figure(figsize=(8, len(corrs)*0.4))
    colors = ['red' if v < 0 else 'blue' for v in corrs]
    sns.barplot(x=corrs.values, y=corrs.index, palette=colors)
    plt.axvline(0, color='black', linestyle='--')
    plt.title('Feature Correlation with Readmission')
    plt.xlabel('Pearson r')
    plt.tight_layout()
    save_figure('target_correlation')
    return corrs


def outlier_analysis(df, numeric_cols=None, target='readmitted', method='iqr'):
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)
    
    print("\n=== Outlier Analysis ===")
    outlier_stats = pd.DataFrame(columns=['outlier_count', 'outlier_percent', 'min', 'max', 'threshold_low', 'threshold_high'])
    
    for col in numeric_cols:
        # Get data without NaNs
        data = df[col].dropna()
        
        if method == 'iqr':
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
        else:  # Z-score method
            mean = data.mean()
            std = data.std()
            
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
        
        # Identify outliers
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_count = len(outliers)
        outlier_percent = outlier_count / len(data) * 100
        
        outlier_stats.loc[col] = [
            outlier_count, 
            outlier_percent, 
            data.min(), 
            data.max(), 
            lower_bound, 
            upper_bound
        ]
        
        # Visualize if there are outliers
        if outlier_count > 0:
            plt.figure(figsize=(12, 5))
            
            # Boxplot with outlier points
            plt.subplot(1, 2, 1)
            sns.boxplot(x=col, data=df, orient='h', color='lightskyblue')
            sns.stripplot(x=col, data=df, orient='h', color='black', alpha=0.3, size=3)
            plt.title(f'Boxplot with Outliers: {col}')
            
            # Distribution by target for outliers vs non-outliers
            plt.subplot(1, 2, 2)
            df['is_outlier'] = ((df[col] < lower_bound) | (df[col] > upper_bound)).astype(int)
            sns.countplot(x='is_outlier', hue=target, data=df, palette=['lightcoral', 'skyblue'])
            plt.title(f'Target Distribution for Outliers vs Non-outliers: {col}')
            plt.xlabel('Is Outlier')
            plt.xticks([0, 1], ['No', 'Yes'])
            
            # Reset the outlier column
            df.drop(columns=['is_outlier'], inplace=True)
            
            plt.tight_layout()
            save_figure(f'outlier_{col}')
    
    # Sort by outlier percentage
    outlier_stats = outlier_stats.sort_values('outlier_percent', ascending=False)
    
    print("Outlier statistics for each feature:")
    print(outlier_stats)
    
    # Highlight features with many outliers
    high_outlier_threshold = 5  # More than 5% outliers
    high_outlier_features = outlier_stats[outlier_stats['outlier_percent'] > high_outlier_threshold].index.tolist()
    
    if high_outlier_features:
        print(f"\nFeatures with high outlier percentage (> {high_outlier_threshold}%):")
        for feat in high_outlier_features:
            print(f"  - {feat}: {outlier_stats.loc[feat, 'outlier_percent']:.2f}% outliers")
        print("\nConsider robust scaling, winsorization, or log transformation for these features.")
    
    return outlier_stats


def pca_analysis(df, target='readmitted'):
    """PCA analysis for dimensionality reduction and clustering tendencies."""
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    
    # Standardize the data
    X = df[numeric_cols]
    X_scaled = StandardScaler().fit_transform(X)
    
    # Apply PCA
    pca = PCA()
    pca_result = pca.fit_transform(X_scaled)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Determine number of components needed for 95% variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    
    print("\n=== PCA Analysis ===")
    print(f"Number of components needed for 95% variance: {n_components_95}")
    print(f"Explained variance by first 3 components: {explained_variance[:3].sum():.4f}")
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, color='skyblue')
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', color='red')
    plt.axhline(y=0.95, color='orange', linestyle='--', label='95% Explained Variance')
    plt.axvline(x=n_components_95, color='green', linestyle='--', label=f'{n_components_95} Components')
    
    plt.title('PCA Explained Variance', fontsize=15, fontweight='bold')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.legend()
    plt.tight_layout()
    save_figure('pca_explained_variance')
    
    # Plot first 2 principal components
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=df[target], cmap='viridis', alpha=0.7)
    plt.colorbar(label='Readmitted')
    plt.title('First Two Principal Components', fontsize=15, fontweight='bold')
    plt.xlabel(f'PC1 ({explained_variance[0]:.2%} Variance)')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2%} Variance)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure('pca_first_two_components')
    
    # Feature loadings for top 2 PCs
    loadings = pd.DataFrame(
        pca.components_.T[:, :2],
        columns=['PC1', 'PC2'],
        index=numeric_cols
    )
    
    # Sort by absolute loading
    loadings['abs_loading'] = np.sqrt(loadings['PC1']**2 + loadings['PC2']**2)
    loadings = loadings.sort_values('abs_loading', ascending=False)
    
    print("\nTop feature loadings for PC1 and PC2:")
    print(loadings[['PC1', 'PC2']].head(10))
    
    # Visualize loadings
    plt.figure(figsize=(10, 8))
    plt.scatter(loadings['PC1'], loadings['PC2'], s=100, alpha=0.7)
    
    # Add feature names as annotations
    for idx, row in loadings.iterrows():
        plt.annotate(idx, (row['PC1'], row['PC2']), fontsize=11)
    
    # Add axis lines
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    
    plt.title('PCA Feature Loadings', fontsize=15, fontweight='bold')
    plt.xlabel('PC1 Loading')
    plt.ylabel('PC2 Loading')
    plt.grid(True, alpha=0.3)
    
    # Add a unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
    plt.gca().add_patch(circle)
    plt.axis('equal')
    
    plt.tight_layout()
    save_figure('pca_feature_loadings')
    
    return explained_variance, loadings


def missing_value_analysis(df):
    """Enhanced missing value analysis with patterns and impact on target."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if missing.empty:
        print("\n=== Missing Value Analysis ===")
        print("No missing values found in the dataset.")
        return None
    
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'count': missing, 'percent': missing_pct})
    
    print("\n=== Missing Value Analysis ===")
    print(missing_df)
    
    # Visualize missing values
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_df.index, y='percent', data=missing_df, palette='viridis')
    plt.title('Percentage of Missing Values by Feature', fontsize=15, fontweight='bold')
    plt.ylabel('Missing Percentage')
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=5, color='red', linestyle='--', label='5% Threshold')
    plt.legend()
    plt.tight_layout()
    save_figure('missing_values_percentage')
    
    # Analyze patterns of missingness
    if len(missing) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[missing_df.index].isnull(), cmap='viridis', cbar=False, yticklabels=False)
        plt.title('Missing Value Patterns', fontsize=15, fontweight='bold')
        plt.xlabel('Features')
        plt.yticks([])
        plt.tight_layout()
        save_figure('missing_value_patterns')
    
    # Analyze impact of missingness on target variable
    if 'readmitted' in df.columns:
        for col in missing_df.index:
            df['is_missing'] = df[col].isnull().astype(int)
            
            # Fisher's exact test for small counts
            contingency_table = pd.crosstab(df['is_missing'], df['readmitted'])
            oddsratio, p_value = stats.fisher_exact(contingency_table)
            
            print(f"\nFeature: {col}")
            print(f"Missing rows: {missing_df.loc[col, 'count']} ({missing_df.loc[col, 'percent']}%)")
            print(f"Fisher's exact test p-value: {p_value:.4f}")
            print(f"Odds ratio: {oddsratio:.4f}")
            
            if p_value < 0.05:
                print(f"SIGNIFICANT: Missingness in {col} is associated with readmission outcome.")
            
            # Visualize
            plt.figure(figsize=(8, 5))
            sns.countplot(x='is_missing', hue='readmitted', data=df, palette=['lightcoral', 'skyblue'])
            plt.title(f'Readmission by Missingness in {col}', fontsize=14)
            plt.xlabel(f'Is {col} Missing')
            plt.xticks([0, 1], ['No', 'Yes'])
            plt.legend(title='Readmitted')
            plt.tight_layout()
            save_figure(f'missing_impact_{col}')
            
            # Remove temporary column
            df.drop(columns=['is_missing'], inplace=True)
    
    return missing_df


def data_leakage_detection(df, target='readmitted'):
    """Detect potential data leakage issues."""
    print("\n=== Data Leakage Detection ===")
    
    # Get data without target
    X = df.drop(columns=[target])
    y = df[target]
    
    # Create dictionaries to store results
    suspiciously_predictive = {}
    perfect_correlations = {}
    
    # Check for perfect correlations with other columns
    numeric_cols = X.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 1:  # Only proceed if there are at least 2 numeric columns
        corr_matrix = X[numeric_cols].corr()
        for i, col1 in enumerate(corr_matrix.columns):
            for col2 in corr_matrix.columns[i+1:]:
                if abs(corr_matrix.loc[col1, col2]) > 0.99:
                    perfect_correlations[(col1, col2)] = corr_matrix.loc[col1, col2]
    
    # Check for suspiciously high predictive power per feature
    for col in X.columns:
        # Skip non-numeric columns for AUC calculation
        if X[col].dtype not in ['int64', 'float64']:
            continue
            
        # Calculate ROC AUC for this single feature
        try:
            auc = roc_auc_score(y, X[col])
            # Adjust AUC to be above 0.5
            auc = max(auc, 1-auc)
            
            # If AUC is suspiciously high, flag it
            if auc > 0.85:
                suspiciously_predictive[col] = auc
        except:
            continue
    
    # Report findings
    if perfect_correlations:
        print("\nPotential data duplication (features with >0.99 correlation):")
        for (col1, col2), corr in perfect_correlations.items():
            print(f"  - {col1} & {col2}: {corr:.4f}")
        print("Warning: These could indicate redundant features or data leakage.")
    
    if suspiciously_predictive:
        print("\nSuspiciously predictive individual features (AUC > 0.85):")
        for col, auc in sorted(suspiciously_predictive.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {col}: AUC={auc:.4f}")
        print("Warning: These features may indicate data leakage or features that shouldn't be available at prediction time.")
    
    if not perfect_correlations and not suspiciously_predictive:
        print("No obvious signs of data leakage detected.")
    
    return {
        'perfect_correlations': perfect_correlations,
        'suspiciously_predictive': suspiciously_predictive
    }


def main():
    """Run the full EDA pipeline with enhanced analysis."""
    print("=== Enhanced EDA for Hospital Readmission Prediction ===")
    
    # Load data
    train, test = load_raw()
    
    # Basic statistics
    col_info, num_summary = basic_stats(train)
    
    # Class imbalance analysis
    imbalance_ratio = class_imbalance_analysis(train)
    
    # Detect datatypes
    num_cols = [c for c in train.select_dtypes(include=['int64', 'float64']).columns.tolist() 
                if c != 'readmitted']
    cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Missing value analysis
    missing_analysis = missing_value_analysis(train)
    
    # Univariate analysis
    univariate_results = univariate_analysis(train, num_cols)
    
    # Categorical analysis
    cat_results = categorical_analysis(train, cat_cols)
    
    # Feature importance analysis
    importance_results = feature_importance_analysis(train)
    
    # Correlation analysis
    corrs = correlation_analysis(train)
    
    # Outlier analysis
    outlier_stats = outlier_analysis(train, num_cols)
    
    # PCA analysis
    explained_variance, loadings = pca_analysis(train)
    
    # Data leakage detection
    leakage_results = data_leakage_detection(train)
    
    print("\n=== EDA Complete ===")
    print(f"Results and visualizations saved to {FIGURES_DIR}")


if __name__ == '__main__':
    import pandas as pd
    # Load raw and processed training data
    raw_train = os.path.join(ROOT, "data", "raw", "train_df.csv")
    processed_train = os.path.join(ROOT, "data", "processed", "train_processed.csv")
    raw_df = pd.read_csv(raw_train)
    proc_df = pd.read_csv(processed_train)
    # Reattach target column
    if 'readmitted' in raw_df.columns:
        proc_df['readmitted'] = raw_df['readmitted']

    print("=== Basic Stats ===")
    basic_stats(proc_df)

    print("\n=== Class Imbalance ===")
    class_imbalance_analysis(proc_df, target='readmitted')

    print("\n=== Univariate Analysis ===")
    univariate_analysis(proc_df, target='readmitted')

    print("\n=== Categorical Analysis ===")
    categorical_analysis(proc_df, target='readmitted')

    print("\n=== Feature Importance ===")
    feature_importance_analysis(proc_df, target='readmitted')

    print("\n=== Correlation Analysis ===")
    correlation_analysis(proc_df, target='readmitted')