"""
Module to load raw data files into a pandas DataFrame.
"""
import os
import glob
import pandas as pd

def load_raw_data(raw_data_dir: str = None) -> pd.DataFrame:
    """
    Load all CSV files from the raw data directory and concatenate into one DataFrame.
    Args:
        raw_data_dir: path to the folder containing raw CSVs (defaults to data/raw).
    Returns:
        pd.DataFrame with all rows from each CSV.
    """
    # determine default path if not provided
    if raw_data_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        raw_data_dir = os.path.join(base_dir, "data", "raw")

    csv_files = glob.glob(os.path.join(raw_data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_data_dir}")

    df_list = []
    for fp in csv_files:
        df = pd.read_csv(fp)
        df_list.append(df)

    return pd.concat(df_list, ignore_index=True)
