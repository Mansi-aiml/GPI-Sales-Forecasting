"""
Configuration for the Sales Forecasting Pipeline.
Centralizes column names, split settings, and model hyperparameters.
"""
import os

# Column names
TARGET_COL = "grand_total"
DATE_COL = "sales_date_time"
SKU_COL = "sku_id"

# Time-based train-test split
TEST_DAYS = 30  # Last N days for test set

# Feature engineering: lag and rolling windows
LAG_DAYS = [1, 7, 14, 30]
ROLLING_MEAN_WINDOWS = [7, 14, 30]
ROLLING_STD_WINDOWS = [7, 14]

# LightGBM hyperparameters (higher capacity for spike detection)
LGBM_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "max_depth": 15,
    "num_leaves": 128,
    "min_child_samples": 20,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

# Paths (optional: for saving model/artifacts later)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
