"""
Train-test split (time-based), LightGBM training, and model evaluation.
Predictions are clipped to non-negative; MAPE ignores zero actual values.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb

from config import (
    TARGET_COL,
    SKU_COL,
    TEST_DAYS,
    LGBM_PARAMS,
)


def time_based_split(df, test_days=TEST_DAYS, date_col="date"):
    """Training = historical; Test = last test_days. No shuffling."""
    print("\n" + "=" * 60)
    print("STEP 5: TIME-BASED TRAIN-TEST SPLIT")
    print("=" * 60)
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    cutoff = df[date_col].max() - pd.Timedelta(days=test_days)
    train_df = df[df[date_col] < cutoff].copy()
    test_df = df[df[date_col] >= cutoff].copy()
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)} (last {test_days} days)")
    return train_df, test_df


def get_feature_columns(df):
    """List of feature column names for modeling (excludes target, date, raw sku_id)."""
    exclude = {TARGET_COL, "date", SKU_COL}
    candidates = [
        "year", "month", "day", "day_of_week", "week_of_year", "is_weekend",
        "lag_1", "lag_7", "lag_14", "lag_30",
        "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
        "rolling_std_7", "rolling_std_14",
        "trend_7", "trend_14", "trend_30",
        "velocity_7", "velocity_14",
        "spike_flag",
        "demand_volatility",
        "sku_mean", "sku_std", "lag_1_over_sku_mean",
        "unit_price", "total_local_sales", "total_outstation_sales", "total_other_sales",
        "mt_retail", "mt_horeca", "mt_qcom",
        SKU_COL + "_encoded",
    ]
    return [c for c in candidates if c in df.columns and c not in exclude]


def train_model(train_df, test_df, feature_cols, target_col=TARGET_COL):
    """Train LightGBM regressor. Predictions clipped to >= 0 at predict time."""
    print("\n" + "=" * 60)
    print("STEP 6: MODEL TRAINING (LightGBM)")
    print("=" * 60)
    X_train = train_df[feature_cols]
    # Log-transform target to learn demand patterns and reduce impact of magnitude/spikes
    y_train = np.log1p(train_df[target_col])
    X_test = test_df[feature_cols]
    for c in feature_cols:
        if X_train[c].dtype.name == "object" or (hasattr(X_train[c].dtype, "kind") and X_train[c].dtype.kind == "O"):
            X_train[c] = X_train[c].astype("category")
            X_test[c] = X_test[c].astype("category")
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_train, y_train)
    print("Model trained.")
    return model


def evaluate_model(model, test_df, feature_cols, target_col=TARGET_COL):
    """Compute MAE, RMSE, MAPE. Predictions clipped to non-negative; MAPE ignores zero actuals."""
    print("\n" + "=" * 60)
    print("STEP 7: MODEL EVALUATION")
    print("=" * 60)
    X_test = test_df[feature_cols]
    y_true = test_df[target_col].values
    # Model was trained on log1p(target); convert predictions back to original scale
    y_pred = np.expm1(model.predict(X_test))
    y_pred = np.clip(y_pred, 0, None)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE: ignore zero actual values
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE (%): {mape:.4f}")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
