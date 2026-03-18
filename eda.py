"""
Exploratory Data Analysis for the sales dataset.
Accepts a DataFrame (e.g. from merge.load_and_merge) and prints shape, dtypes,
missing values, duplicates, numerical summary, outliers, SKU/time patterns.
"""
import pandas as pd
import numpy as np

from config import TARGET_COL, DATE_COL, SKU_COL


def run_eda(df):
    """Analyze dataset: shape, dtypes, missing, duplicates, numerical dist, outliers, SKU/time patterns."""
    print("\n" + "=" * 60)
    print("STEP 1: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print("Dataset shape:", df.shape)
    print("\nColumn names and dtypes:")
    print(df.dtypes)
    print("\nMissing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "None")
    n_dup = df.duplicated().sum()
    print(f"\nDuplicate rows: {n_dup}")
    print("\nNumerical columns - summary statistics:")
    num_cols = df.select_dtypes(include=[np.number]).columns
    print(df[num_cols].describe())
    if TARGET_COL in df.columns:
        q1, q3 = df[TARGET_COL].quantile(0.25), df[TARGET_COL].quantile(0.75)
        iqr = q3 - q1
        out = ((df[TARGET_COL] < q1 - 1.5 * iqr) | (df[TARGET_COL] > q3 + 1.5 * iqr)).sum()
        print(f"\nPossible outliers in {TARGET_COL} (IQR method): {out}")
    if SKU_COL in df.columns:
        sku_sales = df.groupby(SKU_COL)[TARGET_COL].agg(["sum", "count", "mean"])
        print("\nSKU-level sales (top 5 by total sales):")
        print(sku_sales.nlargest(5, "sum"))
    print(f"\nSales date column dtype: {df[DATE_COL].dtype}")
    print(f"Date range: {df[DATE_COL].min()} to {df[DATE_COL].max()}")
    daily_vol = df.groupby(df[DATE_COL].dt.date)[TARGET_COL].sum()
    print("\nDaily total sales (first 5 / last 5):")
    print(daily_vol.head(), "\n", daily_vol.tail())
    return df
