"""
Main pipeline orchestrator: load data (from merge), EDA, cleaning, feature engineering,
encoding, time-based split, train, evaluate, then interactive forecast (user enters SKU_ID and days).
Uses in-memory DataFrame only; no read from merged_sales_data.csv.
"""
import numpy as np

from merge import load_and_merge
from eda import run_eda
from data_cleaning import run_cleaning
from feature_engineering import run_feature_engineering
from encoding import run_encoding
from train_evaluate import (
    time_based_split,
    get_feature_columns,
    train_model,
    evaluate_model,
)
from forecast import forecast_future_sales, print_forecast_table
from config import TEST_DAYS, SKU_COL, TARGET_COL


def run_pipeline(path=None):
    """Execute full pipeline using DataFrame from merge.load_and_merge (no external file read)."""
    df = load_and_merge(path)
    run_eda(df)
    df = run_cleaning(df)
    daily = run_feature_engineering(df)
    daily, encoders = run_encoding(daily)
    train_df, test_df = time_based_split(daily, test_days=TEST_DAYS)
    feature_cols = get_feature_columns(daily)
    if not feature_cols:
        raise ValueError("No feature columns found. Check column names.")
    model = train_model(train_df, test_df, feature_cols)
    evaluate_model(model, test_df, feature_cols)
    test_df = test_df.copy()
    # Model predicts log1p(sales); convert back to original scale and clip to non-negative
    test_df["predicted"] = np.clip(np.expm1(model.predict(test_df[feature_cols])), 0, None)
    print("\nSample predictions (test set):")
    print(test_df[["date", SKU_COL, TARGET_COL, "predicted"]].head(10).to_string(index=False))
    return {
        "model": model,
        "daily_df": daily,
        "encoders": encoders,
        "feature_cols": feature_cols,
    }


def run_interactive_forecast(artifacts):
    """Prompt user for SKU_ID and number of days; run forecast and print table."""
    print("\n" + "=" * 60)
    print("INTERACTIVE FORECAST – Enter SKU_ID and number of days")
    print("=" * 60)
    sku_input = input("Enter SKU_ID: ").strip()
    if not sku_input:
        sku_input = str(artifacts["daily_df"][SKU_COL].iloc[0])
        print(f"No input – using example SKU_ID: {sku_input}")
    if artifacts["daily_df"][SKU_COL].dtype in [np.int32, np.int64]:
        try:
            sku_id = int(sku_input)
        except ValueError:
            sku_id = sku_input
    else:
        sku_id = sku_input
    while True:
        days_input = input("Enter number of days to forecast: ").strip()
        if not days_input:
            days_input = "7"
            print(f"No input – using 7 days")
        try:
            horizon_days = int(days_input)
            if horizon_days < 1:
                print("Please enter a positive number.")
                continue
            break
        except ValueError:
            print("Please enter a valid number of days.")
    print(f"\nForecasting for SKU_ID = {sku_id}, next {horizon_days} days:")
    try:
        forecast_df = forecast_future_sales(
            sku_id,
            horizon_days,
            artifacts["daily_df"],
            artifacts["model"],
            artifacts["feature_cols"],
            artifacts["encoders"],
        )
        print_forecast_table(forecast_df)
    except ValueError as e:
        print(f"Error: {e}")
        print("Tip: Use an SKU_ID that exists in the dataset.")


if __name__ == "__main__":
    artifacts = run_pipeline()
    run_interactive_forecast(artifacts)
