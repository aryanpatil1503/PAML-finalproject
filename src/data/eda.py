"""
EDA script: generates a variety of plots and charts for exploratory analysis.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# set global aesthetics
sns.set_style('whitegrid')
sns.set_palette('Set2')

# ensure project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def load_raw():
    """Load raw train and test data."""
    raw_dir = os.path.join(ROOT, "data", "raw")
    train = pd.read_csv(os.path.join(raw_dir, "train_df.csv"))
    test = pd.read_csv(os.path.join(raw_dir, "test_df.csv"))
    return train, test


def basic_stats(df):
    print("\n=== Basic Info ===")
    print(df.info())
    print("\n=== Description ===")
    print(df.describe(include='all'))


def univariate_numeric(df, cols):
    """Plot distributions and boxplots for numeric features."""
    for c in cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df[c], bins=20, kde=True, color='cadetblue')
        plt.title(f'Distribution of {c}', fontsize=14, fontweight='bold')
        plt.xlabel(c.title())
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

    for c in cols:
        plt.figure(figsize=(6,4))
        sns.boxplot(x=df[c], color='lightcoral')
        plt.title(f'Boxplot of {c}', fontsize=14, fontweight='bold')
        plt.xlabel(c.title())
        plt.tight_layout()
        plt.show()


def univariate_categorical(df, cols):
    """Plot countplots for categorical features."""
    for c in cols:
        plt.figure(figsize=(6,4))
        sns.countplot(x=c, data=df, color='skyblue')  # single color
        plt.title(f'Category Counts: {c}', fontsize=14, fontweight='bold')
        plt.xlabel(c.title())
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def correlation_matrix(df, cols):
    """Heatmap of correlation between numeric features."""
    corr = df[cols].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', linewidths=0.5, cbar_kws={'shrink':0.8})
    plt.title('Correlation Matrix of Numeric Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def missing_heatmap(df):
    """Visualize missing data pattern."""
    plt.figure(figsize=(10,6))
    sns.heatmap(df.isnull(), cmap='viridis', cbar=True, yticklabels=False)
    plt.title('Missing Data Heatmap', fontsize=16, fontweight='bold')
    plt.xlabel('Features')
    plt.tight_layout()
    plt.show()


def pairwise_plot(df, cols, target='readmitted'):
    """Pairplot of numeric features colored by readmission."""
    subset = df[cols + [target]]
    sns.pairplot(subset, hue=target, diag_kind='kde', palette='Set1', corner=True, plot_kws={'alpha':0.6})
    plt.suptitle('Pairplot of Numeric Features by Readmission', y=1.02, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def numeric_scatter_matrix(df, cols):
    """Scatter matrix for numeric features."""
    from pandas.plotting import scatter_matrix
    plt.figure(figsize=(12,12))
    scatter_matrix(df[cols], diagonal='kde', color='grey', alpha=0.6)
    plt.suptitle('Scatter Matrix for Numeric Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def boxplot_numeric_by_target(df, numeric_cols, target='readmitted'):
    """Boxplots of numeric features split by readmission."""
    for c in numeric_cols:
        plt.figure(figsize=(6,4))
        sns.boxplot(x=target, y=c, data=df, palette='Set2')
        plt.title(f'{c.title()} by Readmission', fontsize=14, fontweight='bold')
        plt.xlabel('Readmitted')
        plt.ylabel(c.title())
        plt.tight_layout()
        plt.show()


def countplot_cat_by_target(df, categorical_cols, target='readmitted'):
    """Countplots of categorical features by readmission."""
    for c in categorical_cols:
        plt.figure(figsize=(6,4))
        sns.countplot(x=c, hue=target, data=df, palette='Set2')
        plt.title(f'{c.title()} Counts by Readmission', fontsize=14, fontweight='bold')
        plt.xlabel(c.title())
        plt.ylabel('Count')
        plt.legend(title='Readmitted', loc='upper right')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def scatter_numeric_by_target(df, cols, target='readmitted'):
    print("[EDA] scatter_numeric_by_target: start")
    """Scatter plots of numeric feature pairs by readmission."""
    pairs = [(cols[i], cols[j]) for i in range(len(cols)) for j in range(i+1, len(cols))]
    for x, y in pairs[:3]:
        print(f"[EDA] scatter_numeric_by_target plotting {x} vs {y} by {target}")
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=x, y=y, hue=target, data=df, palette='Set1', alpha=0.7, s=50)
        plt.title(f'{x.title()} vs {y.title()} by Readmission', fontsize=14, fontweight='bold')
        plt.xlabel(x.title())
        plt.ylabel(y.title())
        plt.legend(title='Readmitted')
        plt.tight_layout()
        plt.show()
    print("[EDA] scatter_numeric_by_target: done")


if __name__ == '__main__':
    train, test = load_raw()
    print("--- Training Set EDA ---")
    basic_stats(train)

    # detect types
    num = [c for c in train.select_dtypes(include=['int64', 'float64']).columns.tolist() if c != 'readmitted']  # exclude target
    cat = train.select_dtypes(include=['object', 'category']).columns.tolist()

    univariate_numeric(train, num)
    univariate_categorical(train, cat)
    correlation_matrix(train, num)
    pairwise_plot(train, num)
    missing_heatmap(train)

    # Bivariate analysis
    numeric_scatter_matrix(train, num)
    boxplot_numeric_by_target(train, num)
    countplot_cat_by_target(train, cat)
    scatter_numeric_by_target(train, num)

    print("--- Test Set EDA ---")
    basic_stats(test)
    # same plots for test if desired
    # ...
