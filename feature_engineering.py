"""
Feature engineering: daily aggregation per SKU (with sales components), time features,
lag features, rolling statistics, and trend (momentum) features.
"""
import pandas as pd

from config import (
    TARGET_COL,
    DATE_COL,
    SKU_COL,
    LAG_DAYS,
    ROLLING_MEAN_WINDOWS,
    ROLLING_STD_WINDOWS,
)


def aggregate_to_daily(df):
    """Aggregate transaction-level to daily per SKU with informative sales features."""
    df = df.copy()
    df["date"] = df[DATE_COL].dt.date
    agg_dict = {
        TARGET_COL: "sum",
        "unit_price": "mean",
        "total_local_sales": "sum",
        "total_outstation_sales": "sum",
        "total_other_sales": "sum",
        "mt_retail": "sum",
        "mt_horeca": "sum",
        "mt_qcom": "sum",
    }
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    daily = df.groupby([SKU_COL, "date"], as_index=False).agg(agg_dict)
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def add_time_features(df, date_col="date"):
    """Add year, month, day, day_of_week, week_of_year, is_weekend."""
    df = df.copy()
    d = pd.to_datetime(df[date_col])
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["day"] = d.dt.day
    df["day_of_week"] = d.dt.dayofweek
    df["week_of_year"] = d.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_lag_rolling(df, target_col=TARGET_COL, sku_col=SKU_COL):
    """Add lag and rolling features per SKU_ID, ordered by date."""
    df = df.copy()
    df = df.sort_values([sku_col, "date"]).reset_index(drop=True)
    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = df.groupby(sku_col)[target_col].shift(lag)
    for w in ROLLING_MEAN_WINDOWS:
        df[f"rolling_mean_{w}"] = (
            df.groupby(sku_col)[target_col]
            .transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).mean())
        )
    for w in ROLLING_STD_WINDOWS:
        df[f"rolling_std_{w}"] = (
            df.groupby(sku_col)[target_col]
            .transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).std().fillna(0))
        )
    return df


def add_trend_features(df):
    """Add momentum/trend features: trend_7, trend_14, trend_30 (lag_1 - lag_N)."""
    df = df.copy()
    if "lag_1" in df.columns and "lag_7" in df.columns:
        df["trend_7"] = df["lag_1"] - df["lag_7"]
    if "lag_1" in df.columns and "lag_14" in df.columns:
        df["trend_14"] = df["lag_1"] - df["lag_14"]
    if "lag_1" in df.columns and "lag_30" in df.columns:
        df["trend_30"] = df["lag_1"] - df["lag_30"]
    return df


def add_velocity_features(df):
    """Add demand velocity: current demand vs historical average (lag_1 / (rolling_mean + 1))."""
    df = df.copy()
    if "lag_1" in df.columns and "rolling_mean_7" in df.columns:
        df["velocity_7"] = df["lag_1"] / (df["rolling_mean_7"] + 1)
    if "lag_1" in df.columns and "rolling_mean_14" in df.columns:
        df["velocity_14"] = df["lag_1"] / (df["rolling_mean_14"] + 1)
    return df


def add_spike_flag(df):
    """Binary spike indicator: lag_1 > 2 * rolling_mean_14 signals abnormal demand."""
    df = df.copy()
    if "lag_1" in df.columns and "rolling_mean_14" in df.columns:
        df["spike_flag"] = (df["lag_1"] > 2 * df["rolling_mean_14"]).astype(int)
    return df


def add_demand_volatility(df):
    """Add demand volatility: rolling_std_14 / (rolling_mean_14 + 1) to distinguish stable vs volatile SKUs."""
    df = df.copy()
    if "rolling_std_14" in df.columns and "rolling_mean_14" in df.columns:
        df["demand_volatility"] = df["rolling_std_14"] / (df["rolling_mean_14"] + 1)
    return df


def add_sku_scale_features(df, target_col=TARGET_COL, sku_col=SKU_COL):
    """Add SKU-level mean/std and lag_1 relative to SKU mean so the model scales predictions per SKU."""
    df = df.copy()
    sku_stats = df.groupby(sku_col)[target_col].agg(["mean", "std"]).reset_index()
    sku_stats.columns = [sku_col, "sku_mean", "sku_std"]
    sku_stats["sku_std"] = sku_stats["sku_std"].fillna(0)
    df = df.merge(sku_stats, on=sku_col, how="left")
    df["sku_mean"] = df["sku_mean"].fillna(0)
    df["sku_std"] = df["sku_std"].fillna(0)
    # Lag_1 as fraction of SKU mean: helps model use "recent vs typical level" for this SKU
    if "lag_1" in df.columns:
        df["lag_1_over_sku_mean"] = df["lag_1"] / (df["sku_mean"] + 1)
    return df


def run_feature_engineering(df):
    """Aggregate to daily, add time + lag + rolling + trend features."""
    print("\n" + "=" * 60)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 60)
    daily = aggregate_to_daily(df)
    print(f"Daily aggregated shape: {daily.shape}")
    daily = add_time_features(daily)
    daily = add_lag_rolling(daily)
    daily = add_trend_features(daily)
    daily = add_velocity_features(daily)
    daily = add_spike_flag(daily)
    daily = add_demand_volatility(daily)
    daily = add_sku_scale_features(daily)
    lag_cols = [f"lag_{d}" for d in LAG_DAYS]
    daily = daily.dropna(subset=lag_cols, how="any")
    print(f"Feature matrix shape: {daily.shape}")
    return daily
