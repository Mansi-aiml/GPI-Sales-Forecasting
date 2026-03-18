"""
Interactive recursive forecasting: user inputs SKU_ID and forecast horizon (days).
Returns table: Date | SKU_ID | Forecasted_Sales (all values >= 0).
"""
import pandas as pd
import numpy as np

from config import (
    TARGET_COL,
    SKU_COL,
    LAG_DAYS,
    ROLLING_MEAN_WINDOWS,
)


def get_sku_history(daily_df, sku_id, sku_col=SKU_COL, date_col="date"):
    """Get historical daily series for one SKU, sorted by date."""
    mask = daily_df[sku_col] == sku_id
    return daily_df.loc[mask].sort_values(date_col).reset_index(drop=True)


def forecast_future_sales(
    sku_id,
    horizon_days,
    daily_df,
    model,
    feature_cols,
    encoders,
    sku_col=SKU_COL,
    target_col=TARGET_COL,
):
    """
    Recursive forecasting: each predicted value feeds the next step.
    Returns DataFrame with Date | SKU_ID | Forecasted_Sales (all >= 0).
    """
    date_col = "date"
    history = get_sku_history(daily_df, sku_id, sku_col, date_col)
    if history.empty:
        raise ValueError(f"No history found for SKU_ID={sku_id}")
    last_date = pd.to_datetime(history[date_col].iloc[-1])
    start_date = last_date + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=start_date, periods=horizon_days, freq="D")
    max_lag = max(LAG_DAYS)
    sales_list = history[target_col].tolist()
    if len(sales_list) < max_lag:
        sales_list = [0.0] * (max_lag - len(sales_list)) + sales_list
    else:
        sales_list = sales_list[-(max_lag + max(ROLLING_MEAN_WINDOWS)) :]
    last_unit_price = history["unit_price"].iloc[-1] if "unit_price" in history.columns else 0
    last_total_local = history["total_local_sales"].iloc[-1] if "total_local_sales" in history.columns else 0
    last_total_outstation = history["total_outstation_sales"].iloc[-1] if "total_outstation_sales" in history.columns else 0
    last_total_other = history["total_other_sales"].iloc[-1] if "total_other_sales" in history.columns else 0
    last_mt_retail = history["mt_retail"].iloc[-1] if "mt_retail" in history.columns else 0
    last_mt_horeca = history["mt_horeca"].iloc[-1] if "mt_horeca" in history.columns else 0
    last_mt_qcom = history["mt_qcom"].iloc[-1] if "mt_qcom" in history.columns else 0
    # SKU-level scale: use this SKU's historical mean/std so the model predicts at the right scale
    sku_mean = float(history[target_col].mean())
    sku_std = float(history[target_col].std()) if len(history) > 1 else 0.0
    sku_encoded = encoders.get(sku_col, {}).get(str(sku_id), -1)
    if sku_encoded == -1 and sku_col + "_encoded" in feature_cols:
        sku_encoded = 0
    predictions = []
    for fdate in future_dates:
        recent = sales_list
        lag_1 = recent[-1] if len(recent) >= 1 else 0
        lag_7 = recent[-7] if len(recent) >= 7 else lag_1
        lag_14 = recent[-14] if len(recent) >= 14 else lag_1
        lag_30 = recent[-30] if len(recent) >= 30 else lag_1
        trend_7 = lag_1 - lag_7
        trend_14 = lag_1 - lag_14
        trend_30 = lag_1 - lag_30
        r7 = float(np.mean(recent[-7:])) if len(recent) >= 1 else 0
        r14 = float(np.mean(recent[-14:])) if len(recent) >= 1 else 0
        r30 = float(np.mean(recent[-30:])) if len(recent) >= 1 else 0
        s7 = float(np.std(recent[-7:])) if len(recent) >= 2 else 0
        s14 = float(np.std(recent[-14:])) if len(recent) >= 2 else 0
        demand_volatility = s14 / (r14 + 1)
        velocity_7 = lag_1 / (r7 + 1)
        velocity_14 = lag_1 / (r14 + 1)
        spike_flag = 1 if lag_1 > 2 * r14 else 0
        lag_1_over_sku_mean = lag_1 / (sku_mean + 1)
        row = {
            "year": fdate.year,
            "month": fdate.month,
            "day": fdate.day,
            "day_of_week": fdate.dayofweek,
            "week_of_year": int(fdate.isocalendar()[1]),
            "is_weekend": 1 if fdate.dayofweek >= 5 else 0,
            "lag_1": lag_1,
            "lag_7": lag_7,
            "lag_14": lag_14,
            "lag_30": lag_30,
            "rolling_mean_7": r7,
            "rolling_mean_14": r14,
            "rolling_mean_30": r30,
            "rolling_std_7": s7,
            "rolling_std_14": s14,
            "trend_7": trend_7,
            "trend_14": trend_14,
            "trend_30": trend_30,
            "velocity_7": velocity_7,
            "velocity_14": velocity_14,
            "spike_flag": spike_flag,
            "demand_volatility": demand_volatility,
            "sku_mean": sku_mean,
            "sku_std": sku_std,
            "lag_1_over_sku_mean": lag_1_over_sku_mean,
            "unit_price": last_unit_price,
            "total_local_sales": last_total_local,
            "total_outstation_sales": last_total_outstation,
            "total_other_sales": last_total_other,
            "mt_retail": last_mt_retail,
            "mt_horeca": last_mt_horeca,
            "mt_qcom": last_mt_qcom,
            sku_col + "_encoded": sku_encoded,
        }
        X = pd.DataFrame([row])
        X = X[[c for c in feature_cols if c in X.columns]]
        # Model predicts log1p(sales); convert back and ensure non-negative
        pred = np.expm1(model.predict(X)[0])
        pred = max(0.0, float(pred))
        predictions.append(pred)
        sales_list.append(pred)
    result = pd.DataFrame({
        "Date": future_dates,
        "SKU_ID": sku_id,
        "Forecasted_Sales": predictions,
    })
    return result


def print_forecast_table(forecast_df):
    """Print clean forecast table: Date | SKU_ID | Forecasted_Sales."""
    print(forecast_df.to_string(index=False))
