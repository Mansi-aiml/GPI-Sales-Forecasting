"""
Data cleaning: drop metadata columns, convert dates, handle missing/negative target,
remove duplicates, fix dtypes, and remove extreme outliers using IQR method.
"""
import pandas as pd
import numpy as np

from config import TARGET_COL, DATE_COL

# Columns to remove (metadata, not useful for forecasting)
COLUMNS_TO_DROP = [
    "id",
    "created_date",
    "created_by",
    "last_updated_date",
    "last_updated",
    "transaction_source",
    "distrcode",
    "gpi_state",
    "zone",
]


def run_cleaning(df):
    """Convert dates, drop metadata columns, remove missing/negative grand_total,
    remove duplicates, fix dtypes, and apply IQR-based outlier removal."""
    print("\n" + "=" * 60)
    print("STEP 2: DATA CLEANING")
    print("=" * 60)
    df = df.copy()
    # Drop metadata columns (only if present)
    to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"Dropped columns: {to_drop}")
    # Convert date column
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    # Remove rows where grand_total is missing or negative
    df = df.dropna(subset=[DATE_COL, TARGET_COL])
    df = df[df[TARGET_COL] >= 0]
    # Fill numeric NaNs with 0 for sales-related columns
    sales_cols = [c for c in df.columns if "sales" in c.lower() or c in [TARGET_COL, "value", "unit_price"]]
    for c in sales_cols:
        if c in df.columns and df[c].dtype in [np.float64, np.float32]:
            df[c] = df[c].fillna(0)
    df = df.drop_duplicates(keep="first")
    # Fix numeric columns that might be object
    for c in ["unit_price", "value", TARGET_COL]:
        if c in df.columns and df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    # Remove extreme outliers using IQR method
    if TARGET_COL in df.columns:
        q1 = df[TARGET_COL].quantile(0.25)
        q3 = df[TARGET_COL].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        before = len(df)
        df = df[(df[TARGET_COL] >= lower) & (df[TARGET_COL] <= upper)]
        print(f"IQR outlier removal: kept {len(df)} rows (dropped {before - len(df)} outside [{lower:.2f}, {upper:.2f}])")
    print(f"Shape after cleaning: {df.shape}")
    return df.reset_index(drop=True)
