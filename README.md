# Sales Forecasting Pipeline

Modular end-to-end pipeline for retail sales forecasting. Data is loaded in memory from CSV files (no read from merged_sales_data.csv). Run the full pipeline or use individual modules as needed.

## Project structure

| File | Purpose |
|------|---------|
| **config.py** | Column names, TEST_DAYS, LAG_DAYS, rolling windows, LightGBM params |
| **merge.py** | Load and merge CSVs → single DataFrame (in memory) |
| **eda.py** | Exploratory data analysis (shape, dtypes, missing, duplicates, outliers, SKU/time patterns) |
| **eda.ipynb** | Notebook that uses merge + eda modules and adds visualizations |
| **data_cleaning.py** | Clean data: datetime, missing, duplicates, dtypes, outliers |
| **feature_engineering.py** | Daily aggregation per SKU, time features, lag & rolling features |
| **encoding.py** | Label encoding for SKU_ID |
| **train_evaluate.py** | Time-based split, LightGBM training, MAE/RMSE/MAPE evaluation (predictions clipped ≥ 0) |
| **forecast.py** | Recursive forecasting: `forecast_future_sales(sku_id, horizon_days, ...)` → Date \| SKU_ID \| Forecasted_Sales |
| **run_pipeline.py** | Main entry: runs full pipeline then prompts for SKU_ID and number of days |

## How to run

**Full pipeline (load → EDA → clean → features → encode → split → train → evaluate → interactive forecast):**
```bash
python run_pipeline.py
```
You will be prompted to enter **SKU_ID** and **number of days** to get a forecast table.

**EDA only (notebook):**  
Open `eda.ipynb` and run all cells. The first cell loads data via `merge.load_and_merge()` and runs `eda.run_eda(df)`; remaining cells add visualizations.

**Load data only:**
```bash
python merge.py
```
Or in code: `from merge import load_and_merge; df = load_and_merge()`.

## Functionality (unchanged)

- Data loaded from CSV files and kept in memory.
- EDA: shape, dtypes, missing, duplicates, numerical summary, outliers (IQR), SKU-level and time-based patterns.
- Cleaning: datetime, drop missing date/target, fill sales NaNs, drop duplicates, coerce numeric, cap target at 99th percentile.
- Features: daily aggregate per SKU; time (year, month, day, day_of_week, week_of_year, is_weekend); lag_1/7/14/30; rolling_mean_7/14/30; rolling_std_7/14.
- Encoding: label encoding for SKU_ID.
- Split: last 30 days = test, rest = train (time-based, no shuffle).
- LightGBM training; predictions clipped to ≥ 0 for evaluation and forecasting.
- Interactive forecast: user enters SKU_ID and number of days; recursive forecast returns Date \| SKU_ID \| Forecasted_Sales.
