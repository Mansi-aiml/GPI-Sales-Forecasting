"""
Encoding: label encoding for categoricals
(e.g. SKU_ID, TOWN_ID)
"""

import pandas as pd

from config import SKU_COL, TOWN_COL


def run_encoding(df, fit_encoders=None):
    """
    Label encoding for categoricals.
    Returns:
        df, encoders
    """

    print("\n" + "=" * 60)
    print("STEP 4: ENCODING")
    print("=" * 60)

    df = df.copy()

    encoders = fit_encoders or {}

    
    # SKU ENCODING---------------------------------------------
    
    if SKU_COL in df.columns:

        if SKU_COL not in encoders:
            uniques = (
                df[SKU_COL]
                .astype(str)
                .fillna("__NA__")
                .unique()
            )

            encoders[SKU_COL] = {
                v: i for i, v in enumerate(sorted(uniques))
            }

        mapping = encoders[SKU_COL]

        df[SKU_COL + "_encoded"] = (
            df[SKU_COL]
            .astype(str)
            .fillna("__NA__")
            .map(lambda x: mapping.get(x, -1))
        )

    # TOWN ENCODING---------------------------------------------
    
    if TOWN_COL in df.columns:

        if TOWN_COL not in encoders:
            uniques = (
                df[TOWN_COL]
                .astype(str)
                .fillna("__NA__")
                .unique()
            )

            encoders[TOWN_COL] = {
                v: i for i, v in enumerate(sorted(uniques))
            }

        mapping = encoders[TOWN_COL]

        df[TOWN_COL + "_encoded"] = (
            df[TOWN_COL]
            .astype(str)
            .fillna("__NA__")
            .map(lambda x: mapping.get(x, -1))
        )

    print("Label encoding applied for SKU_ID and TOWN_ID.")

    return df, encoders
# """
# Encoding: label encoding for categoricals (e.g. SKU_ID) to avoid memory-heavy one-hot.
# """
# import pandas as pd

# from config import SKU_COL


# def run_encoding(df, fit_encoders=None):
#     """Label encoding for categoricals (e.g. SKU_ID). Returns (df, encoders)."""
#     print("\n" + "=" * 60)
#     print("STEP 4: ENCODING")
#     print("=" * 60)
#     df = df.copy()
#     encoders = fit_encoders or {}
#     if SKU_COL in df.columns:
#         if SKU_COL not in encoders:
#             uniques = df[SKU_COL].astype(str).fillna("__NA__").unique()
#             encoders[SKU_COL] = {v: i for i, v in enumerate(sorted(uniques))}
#         mapping = encoders[SKU_COL]
#         df[SKU_COL + "_encoded"] = df[SKU_COL].astype(str).fillna("__NA__").map(lambda x: mapping.get(x, -1))
#     print("Label encoding applied for SKU_ID.")
#     return df, encoders
