"""
Load and merge multiple CSV files into a single DataFrame.
The DataFrame stays in memory; no read from merged_sales_data.csv or external file.
"""
import pandas as pd
import glob
import os

from config import DATE_COL


def load_and_merge(path=None):
    """Read and merge multiple CSV files into a single DataFrame."""
    path = path or os.environ.get("GPI_DATASET_PATH", "/Users/tspl/Downloads/GPI dataset/*.csv")
    if not path.endswith("*.csv"):
        path = os.path.join(path, "*.csv")
    files = sorted(glob.glob(path))
    if not files:
        raise FileNotFoundError(f"No CSV files found at {path}")
    df = pd.DataFrame()
    for f in files:
        print(f"Loading {f}...")
        temp = pd.read_csv(f, low_memory=False)
        df = pd.concat([df, temp], ignore_index=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    print(f"Merged shape: {df.shape}")
    return df


if __name__ == "__main__":
    df = load_and_merge()
    print("Completed. Use this DataFrame in the pipeline or eda.ipynb.")

