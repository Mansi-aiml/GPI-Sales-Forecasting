"""
Main pipeline orchestrator.

Supports:
- SKU forecasting
- Town forecasting

Trains separate models and saves:
- sku_artifacts.pkl
- town_artifacts.pkl
"""

import numpy as np
import pickle

from merge import load_and_merge
from eda import run_eda
from data_cleaning import run_cleaning

from feature_engineering import (
    run_feature_engineering,
)

from encoding import run_encoding

from train_evaluate import (
    time_based_split,
    get_feature_columns,
    train_model,
    evaluate_model,
)

from forecast import (
    forecast_future_sales,
    forecast_town_sales,
    print_forecast_table,
)

from config import (
    TEST_DAYS,
    SKU_COL,
    TOWN_COL,
    TARGET_COL,
)


# TRAIN PIPELINE-----------------------------------------------------

def train_pipeline(
    df,
    forecast_level,
):
    """
    Train one forecasting model.

    forecast_level:
    - sku
    - town
    """

    print("\n" + "=" * 60)
    print(f"TRAINING {forecast_level.upper()} MODEL")
    print("=" * 60)

    
    # FEATURE ENGINEERING----------------------------------------------
    daily = run_feature_engineering(
        df,
        forecast_level=forecast_level,
    )

    
    # ENCODING---------------------------------------------------------
    daily, encoders = run_encoding(daily)

    # SPLIT------------------------------------------------------------
    
    train_df, test_df = time_based_split(
        daily,
        test_days=TEST_DAYS,
    )

    
    # FEATURES-----------------------------------------------
    
    feature_cols = get_feature_columns(daily)

    if not feature_cols:

        raise ValueError(
            "No feature columns found."
        )

    
    # TRAIN----------------------------------------------------
    
    model = train_model(
        train_df,
        test_df,
        feature_cols,
    )

    
    # EVALUATE-------------------------------------------------
    
    evaluate_model(
        model,
        test_df,
        feature_cols,
    )

    # SAMPLE PREDICTIONS----------------------------------------
    
    test_df = test_df.copy()

    test_df["predicted"] = np.clip(

        np.expm1(
            model.predict(
                test_df[feature_cols]
            )
        ),

        0,
        None,
    )

    print("\nSample predictions:")

    cols = ["date", TARGET_COL, "predicted"]

    if forecast_level == "sku":
        cols.insert(1, SKU_COL)

    elif forecast_level == "town":
        cols.insert(1, TOWN_COL)

    elif forecast_level == "sku_town":
     group_cols = [
        "date",
        SKU_COL,
        TOWN_COL,
     ]

    print(
        test_df[cols]
        .head(10)
        .to_string(index=False)
    )

    
    # ARTIFACTS----------------------------------------------
    
    artifacts = {

        "model": model,

        "daily_df": daily,

        "encoders": encoders,

        "feature_cols": feature_cols,

        "forecast_level": forecast_level,
    }

    return artifacts


# SAVE ARTIFACTS----------------------------------------------

def save_artifacts(
    artifacts,
    filename,
):

    with open(filename, "wb") as f:

        pickle.dump(artifacts, f)

    print(f"\n Saved: {filename}")


# MAIN RUNNER------------------------------------------------

def run_pipeline(path=None):


    # LOAD DATA----------------------------------------------
    df = load_and_merge(path)

    run_eda(df)

    df = run_cleaning(df)

    # ========================================================
    # SKU MODEL
    # ========================================================
    sku_artifacts = train_pipeline(
        df,
        forecast_level="sku",
    )

    save_artifacts(
        sku_artifacts,
        "sku_artifacts.pkl",
    )

    # ========================================================
    # TOWN MODEL
    # ========================================================
    town_artifacts = train_pipeline(
        df,
        forecast_level="town",
    )

    save_artifacts(
        town_artifacts,
        "town_artifacts.pkl",
    )

    
    # ========================================================
    # SKU + TOWN MODEL
    # ========================================================
   
    sku_town_artifacts = train_pipeline(
        df,
        forecast_level="sku_town",
    )

    save_artifacts(
        sku_town_artifacts,
        "sku_town_artifacts.pkl",
    )

    return {

    "sku": sku_artifacts,

    "town": town_artifacts,

    "sku_town": sku_town_artifacts,
}

# ============================================================
# INTERACTIVE SKU FORECAST
# ============================================================
def run_interactive_forecast(
    artifacts,
):

    print("\n" + "=" * 60)
    print("INTERACTIVE FORECAST")
    print("=" * 60)

    # ========================================================
    # MODE
    # ========================================================
    mode = input(
        "\nChoose forecast type "
        "(sku/town): "
    ).strip().lower()

    # ========================================================
    # SKU MODE
    # ========================================================
    if mode == "sku":

        sku_input = input(
            "Enter SKU_ID: "
        ).strip()

        sku_id = int(sku_input)

        days = int(
            input(
                "Enter forecast days: "
            )
        )

        forecast_df = forecast_future_sales(

            sku_id,

            days,

            artifacts["sku"]["daily_df"],

            artifacts["sku"]["model"],

            artifacts["sku"]["feature_cols"],

            artifacts["sku"]["encoders"],
        )

        print_forecast_table(
            forecast_df
        )

    # ========================================================
    # TOWN MODE
    # ========================================================
    elif mode == "town":

        town_input = input(
            "Enter TOWN_ID: "
        ).strip()

        town_id = int(town_input)

        days = int(
            input(
                "Enter forecast days: "
            )
        )

        forecast_df = forecast_town_sales(

            town_id,

            days,

            artifacts["town"]["daily_df"],

            artifacts["town"]["model"],

            artifacts["town"]["feature_cols"],

            artifacts["town"]["encoders"],
        )

        print_forecast_table(
            forecast_df
        )

    else:

        print(
            "Invalid mode. "
            "Choose sku or town."
        )


# ============================================================
# ENTRY
# ============================================================
if __name__ == "__main__":

    artifacts = run_pipeline()

    run_interactive_forecast(
        artifacts
    )
# """
# Main pipeline orchestrator: load data (from merge), EDA, cleaning, feature engineering,
# encoding, time-based split, train, evaluate, then interactive forecast (user enters SKU_ID and days).
# Uses in-memory DataFrame only; no read from merged_sales_data.csv.
# """
# import numpy as np
# import pickle
# from merge import load_and_merge
# from eda import run_eda
# from data_cleaning import run_cleaning
# from feature_engineering import run_feature_engineering
# from encoding import run_encoding
# from train_evaluate import (
#     time_based_split,
#     get_feature_columns,
#     train_model,
#     evaluate_model,
# )
# from forecast import forecast_future_sales, print_forecast_table
# from config import TEST_DAYS, SKU_COL, TARGET_COL


# def run_pipeline(path=None):
#     """Execute full pipeline using DataFrame from merge.load_and_merge (no external file read)."""
#     df = load_and_merge(path)
#     run_eda(df)
#     df = run_cleaning(df)
#     daily = run_feature_engineering(df)
#     daily, encoders = run_encoding(daily)
#     train_df, test_df = time_based_split(daily, test_days=TEST_DAYS)
#     feature_cols = get_feature_columns(daily)
#     if not feature_cols:
#         raise ValueError("No feature columns found. Check column names.")
#     model = train_model(train_df, test_df, feature_cols)
#     evaluate_model(model, test_df, feature_cols)
#     test_df = test_df.copy()
#     # Model predicts log1p(sales); convert back to original scale and clip to non-negative
#     test_df["predicted"] = np.clip(np.expm1(model.predict(test_df[feature_cols])), 0, None)
#     print("\nSample predictions (test set):")
#     print(test_df[["date", SKU_COL, TARGET_COL, "predicted"]].head(10).to_string(index=False))


#     artifacts = {
#     "model": model,
#     "daily_df": daily,
#     "encoders": encoders,
#     "feature_cols": feature_cols,
#     }

#     with open("artifacts.pkl", "wb") as f:
#         pickle.dump(artifacts, f)

#     print(" artifacts.pkl saved")

#     return artifacts
#     # return {
#     #     "model": model,
#     #     "daily_df": daily,
#     #     "encoders": encoders,
#     #     "feature_cols": feature_cols,
#     # }


# def run_interactive_forecast(artifacts):
#     """Prompt user for SKU_ID and number of days; run forecast and print table."""
#     print("\n" + "=" * 60)
#     print("INTERACTIVE FORECAST – Enter SKU_ID and number of days")
#     print("=" * 60)
#     sku_input = input("Enter SKU_ID: ").strip()
#     if not sku_input:
#         sku_input = str(artifacts["daily_df"][SKU_COL].iloc[0])
#         print(f"No input – using example SKU_ID: {sku_input}")
#     if artifacts["daily_df"][SKU_COL].dtype in [np.int32, np.int64]:
#         try:
#             sku_id = int(sku_input)
#         except ValueError:
#             sku_id = sku_input
#     else:
#         sku_id = sku_input
#     while True:
#         days_input = input("Enter number of days to forecast: ").strip()
#         if not days_input:
#             days_input = "7"
#             print(f"No input – using 7 days")
#         try:
#             horizon_days = int(days_input)
#             if horizon_days < 1:
#                 print("Please enter a positive number.")
#                 continue
#             break
#         except ValueError:
#             print("Please enter a valid number of days.")
#     print(f"\nForecasting for SKU_ID = {sku_id}, next {horizon_days} days:")
#     try:
#         forecast_df = forecast_future_sales(
#             sku_id,
#             horizon_days,
#             artifacts["daily_df"],
#             artifacts["model"],
#             artifacts["feature_cols"],
#             artifacts["encoders"],
#         )
#         print_forecast_table(forecast_df)
#     except ValueError as e:
#         print(f"Error: {e}")
#         print("Tip: Use an SKU_ID that exists in the dataset.")


# if __name__ == "__main__":
#     artifacts = run_pipeline()
#     run_interactive_forecast(artifacts)
