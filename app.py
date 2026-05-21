import streamlit as st
import pandas as pd
import pickle

from forecast import (
    forecast_future_sales,
    forecast_town_sales,
    forecast_sku_town_sales,
)

from config import (
    SKU_COL,
    TOWN_COL,
    TARGET_COL,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="GPI Sales Forecast",
    layout="wide",
)

# ============================================================
# TITLE
# ============================================================
st.title("📊 GPI Sales Forecasting App")

st.write(
    "Forecast future product and town sales."
)

# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():

    with open("sku_artifacts.pkl", "rb") as f:
        sku_artifacts = pickle.load(f)

    with open("town_artifacts.pkl", "rb") as f:
        town_artifacts = pickle.load(f)

    with open("sku_town_artifacts.pkl", "rb") as f:
        sku_town_artifacts = pickle.load(f)

    return (
        sku_artifacts, 
        town_artifacts, 
        sku_town_artifacts,
    )


(
    sku_artifacts,
    town_artifacts,
    sku_town_artifacts,
) = load_artifacts()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("🔧 Forecast Settings")

forecast_type = st.sidebar.radio(

    "Select Forecast Type",

    [
        "SKU Forecast",
        "Town Forecast",
        "SKU + Town Forecast",
    ]
)

days = st.sidebar.slider(

    "Forecast Days",

    1,
    30,
    7
)

# ============================================================
# SKU FORECAST
# ============================================================
if forecast_type == "SKU Forecast":

    daily_df = sku_artifacts["daily_df"]

    model = sku_artifacts["model"]

    feature_cols = sku_artifacts["feature_cols"]

    encoders = sku_artifacts["encoders"]

    sku_list = sorted(
        daily_df[SKU_COL].unique()
    )

    sku_id = st.sidebar.selectbox(
        "Select SKU ID",
        sku_list,
    )

# ============================================================
# TOWN FORECAST
# ============================================================
elif forecast_type == "Town Forecast":

    daily_df = town_artifacts["daily_df"]

    model = town_artifacts["model"]

    feature_cols = town_artifacts["feature_cols"]

    encoders = town_artifacts["encoders"]

    town_list = sorted(
        daily_df[TOWN_COL].unique()
    )

    town_id = st.sidebar.selectbox(
        "Select Town ID",
        town_list,
    )

# ============================================================
# SKU + TOWN FORECAST
# ============================================================
else:

    daily_df = sku_town_artifacts["daily_df"]

    model = sku_town_artifacts["model"]

    feature_cols = sku_town_artifacts["feature_cols"]

    encoders = sku_town_artifacts["encoders"]

    sku_list = sorted(
        daily_df[SKU_COL].unique()
    )

    town_list = sorted(
        daily_df[TOWN_COL].unique()
    )

    sku_id = st.sidebar.selectbox(
        "Select SKU ID",
        sku_list,
    )

    town_id = st.sidebar.selectbox(
        "Select Town ID",
        town_list,
    )

# ============================================================
# FORECAST BUTTON
# ============================================================
if st.sidebar.button("🚀 Generate Forecast"):

    try:

        # ====================================================
        # SKU FORECAST
        # ====================================================
        if forecast_type == "SKU Forecast":

            forecast_df = forecast_future_sales(

                sku_id,

                days,

                daily_df,

                model,

                feature_cols,

                encoders,
            )

            st.success(
                f"Forecast for SKU "
                f"{sku_id} "
                f"for next {days} days"
            )

        # ====================================================
        # TOWN FORECAST
        # ====================================================
        elif forecast_type == "Town Forecast":

            forecast_df = forecast_town_sales(

                town_id,

                days,

                daily_df,

                model,

                feature_cols,

                encoders,
            )

            st.success(
                f"Forecast for Town "
                f"{town_id} "
                f"for next {days} days"
            )

        # ====================================================
        # SKU + TOWN FORECAST
        # ====================================================
        else:

            filtered_df = daily_df[

                (daily_df[SKU_COL] == sku_id) &

                (daily_df[TOWN_COL] == town_id)
            ]

            if filtered_df.empty:

                st.error(
                    "No sales history found for "
                    "selected SKU in this town."
                )

                st.stop()

            forecast_df = forecast_sku_town_sales(

                sku_id,

                town_id,

                days,

                filtered_df,

                model,

                feature_cols,

                encoders,
            )

            st.success(
                f"Forecast for SKU "
                f"{sku_id} in Town "
                f"{town_id} "
                f"for next {days} days"
            )

        # ====================================================
        # TABLE
        # ====================================================
        st.subheader("📄 Forecast Data")

        st.dataframe(
            forecast_df,
            use_container_width=True
        )

        # ====================================================
        # CHART
        # ====================================================
        st.subheader("📈 Forecast Chart")

        st.line_chart(

            forecast_df.set_index("Date")[
                ["Forecasted_Sales"]
            ]
        )

        # ====================================================
        # DOWNLOAD
        # ====================================================
        csv = (
            forecast_df
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(

            label="⬇️ Download Forecast CSV",

            data=csv,

            file_name="forecast.csv",

            mime="text/csv",
        )

    except Exception as e:

        st.error(f"Error: {e}")

# ============================================================
# KPI SECTION
# ============================================================
st.subheader("📊 Overall Dataset Insights")

col1, col2, col3 = st.columns(3)

col1.metric(

    "Avg Sales",

    round(
        daily_df[TARGET_COL].mean(),
        2
    )
)

col2.metric(

    "Max Sales",

    round(
        daily_df[TARGET_COL].max(),
        2
    )
)

col3.metric(

    "Total Records",

    len(daily_df)
)

# ============================================================
# SAMPLE DATA
# ============================================================
if st.checkbox("Show Sample Data"):

    st.subheader("🔍 Sample Data")

    st.dataframe(

        daily_df.head(100),

        use_container_width=True
    )
# import streamlit as st
# import pandas as pd
# import pickle

# from forecast import (
#     forecast_future_sales,
#     forecast_town_sales,
# )

# from config import (
#     SKU_COL,
#     TOWN_COL,
#     TARGET_COL,
# )

# # ============================================================
# # PAGE CONFIG
# # ============================================================
# st.set_page_config(
#     page_title="GPI Sales Forecast",
#     layout="wide",
# )

# # ============================================================
# # TITLE
# # ============================================================
# st.title("📊 GPI Sales Forecasting App")

# st.write(
#     "Forecast future product and town sales."
# )

# # ============================================================
# # LOAD ARTIFACTS
# # ============================================================
# @st.cache_resource
# def load_artifacts():

#     with open("sku_artifacts.pkl", "rb") as f:
#         sku_artifacts = pickle.load(f)

#     with open("town_artifacts.pkl", "rb") as f:
#         town_artifacts = pickle.load(f)

#     return sku_artifacts, town_artifacts


# sku_artifacts, town_artifacts = load_artifacts()

# # ============================================================
# # SIDEBAR
# # ============================================================
# st.sidebar.header("🔧 Forecast Settings")

# forecast_type = st.sidebar.radio(

#     "Select Forecast Type",

#     [
#         "SKU Forecast",
#         "Town Forecast",
#     ]
# )

# days = st.sidebar.slider(

#     "Forecast Days",

#     1,
#     30,
#     7
# )

# # ============================================================
# # SKU FORECAST
# # ============================================================
# if forecast_type == "SKU Forecast":

#     daily_df = sku_artifacts["daily_df"]

#     model = sku_artifacts["model"]

#     feature_cols = sku_artifacts["feature_cols"]

#     encoders = sku_artifacts["encoders"]

#     sku_list = sorted(
#         daily_df[SKU_COL].unique()
#     )

#     sku_id = st.sidebar.selectbox(
#         "Select SKU ID",
#         sku_list,
#     )

# # ============================================================
# # TOWN FORECAST
# # ============================================================
# else:

#     daily_df = town_artifacts["daily_df"]

#     model = town_artifacts["model"]

#     feature_cols = town_artifacts["feature_cols"]

#     encoders = town_artifacts["encoders"]

#     town_list = sorted(
#         daily_df[TOWN_COL].unique()
#     )

#     town_id = st.sidebar.selectbox(
#         "Select Town ID",
#         town_list,
#     )

# # ============================================================
# # FORECAST BUTTON
# # ============================================================
# if st.sidebar.button("🚀 Generate Forecast"):

#     try:

#         # ====================================================
#         # SKU FORECAST
#         # ====================================================
#         if forecast_type == "SKU Forecast":

#             forecast_df = forecast_future_sales(

#                 sku_id,

#                 days,

#                 daily_df,

#                 model,

#                 feature_cols,

#                 encoders,
#             )

#             st.success(
#                 f"Forecast for SKU "
#                 f"{sku_id} "
#                 f"for next {days} days"
#             )

#         # ====================================================
#         # TOWN FORECAST
#         # ====================================================
#         else:

#             forecast_df = forecast_town_sales(

#                 town_id,

#                 days,

#                 daily_df,

#                 model,

#                 feature_cols,

#                 encoders,
#             )

#             st.success(
#                 f"Forecast for Town "
#                 f"{town_id} "
#                 f"for next {days} days"
#             )

#         # ====================================================
#         # TABLE
#         # ====================================================
#         st.subheader("📄 Forecast Data")

#         st.dataframe(
#             forecast_df,
#             use_container_width=True
#         )

#         # ====================================================
#         # CHART
#         # ====================================================
#         st.subheader("📈 Forecast Chart")

#         st.line_chart(

#             forecast_df.set_index("Date")[
#                 ["Forecasted_Sales"]
#             ]
#         )

#         # ====================================================
#         # DOWNLOAD
#         # ====================================================
#         csv = (
#             forecast_df
#             .to_csv(index=False)
#             .encode("utf-8")
#         )

#         st.download_button(

#             label="⬇️ Download Forecast CSV",

#             data=csv,

#             file_name="forecast.csv",

#             mime="text/csv",
#         )

#     except Exception as e:

#         st.error(f"Error: {e}")

# # ============================================================
# # KPI SECTION
# # ============================================================
# st.subheader("📊 Overall Dataset Insights")

# col1, col2, col3 = st.columns(3)

# col1.metric(

#     "Avg Sales",

#     round(
#         daily_df[TARGET_COL].mean(),
#         2
#     )
# )

# col2.metric(

#     "Max Sales",

#     round(
#         daily_df[TARGET_COL].max(),
#         2
#     )
# )

# col3.metric(

#     "Total Records",

#     len(daily_df)
# )

# # ============================================================
# # SAMPLE DATA
# # ============================================================
# if st.checkbox("Show Sample Data"):

#     st.subheader("🔍 Sample Data")

#     st.dataframe(

#         daily_df.head(100),

#         use_container_width=True
#     )
