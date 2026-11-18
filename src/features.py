"""
PaySim feature engineering in pandas: the same silver -> gold logic as the Databricks Fraud Lakehouse
(github.com/ChrishA-tech/fraud-lakehouse), rewritten without Spark so it runs as an Azure ML command job.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

RAW_COLUMNS = {"step": "step", "type": "type", "amount": "amount", "nameOrig": "name_orig",
               "oldbalanceOrg": "old_balance_orig", "newbalanceOrig": "new_balance_orig", "nameDest": "name_dest",
               "oldbalanceDest": "old_balance_dest", "newbalanceDest": "new_balance_dest", "isFraud": "is_fraud",
               "isFlaggedFraud": "is_flagged_by_rule"}
RAW_DTYPES = {"step": "int32", "type": "category", "amount": "float64", "nameOrig": "string",
              "oldbalanceOrg": "float64", "newbalanceOrig": "float64", "nameDest": "string",
              "oldbalanceDest": "float64", "newbalanceDest": "float64", "isFraud": "int8", "isFlaggedFraud": "int8"}
TYPES = ("PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN")
MODEL_TYPES = ("TRANSFER", "CASH_OUT")          # PaySim only contains fraud in these two types
EXPECTED_RAW_ROWS = 6_362_620                     # full PaySim file

# Data-quality rules, same names as the Lakeflow expectations: CRITICAL rows are dropped, WARN rows are counted.
def _critical(df):
    return {"has_account_ids": df.name_orig.notna() & df.name_dest.notna(),
            "amount_valid": df.amount.notna() & (df.amount >= 0),
            "known_type": df["type"].isin(TYPES),
            "label_binary": df.is_fraud.isin([0, 1])}


def _warn(df):
    return {"step_in_month": df.step.between(1, 744),
            "balances_non_negative": (df.old_balance_orig >= 0) & (df.old_balance_dest >= 0)}


def read_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=RAW_DTYPES).rename(columns=RAW_COLUMNS)


def apply_expectations(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    crit, warn = _critical(df), _warn(df)
    counts = {k: int((~v.fillna(False)).sum()) for k, v in {**crit, **warn}.items()}
    keep = np.logical_and.reduce([v.fillna(False).to_numpy() for v in crit.values()])
    return df.loc[keep].reset_index(drop=True), counts


def dest_window_24h(name_dest: pd.Series, step: pd.Series, amount: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Count and sum of transactions received by the same account in the 24 hours before this one
    (steps t-24 .. t-1, never the current or a future hour), vectorised with one global cumulative sum.

    Keys are code * 1000 + step (step <= 744 < 1000), so a range of one account never runs into another's."""
    code = pd.factorize(name_dest)[0].astype(np.int64)
    st = step.to_numpy(np.int64)
    key = code * 1000 + st
    agg = pd.DataFrame({"key": key, "n": 1, "s": amount.to_numpy(float)}).groupby("key", sort=True).sum()
    keys, cn, cs = agg.index.to_numpy(), agg.n.cumsum().to_numpy(), agg.s.cumsum().to_numpy()

    def cum(q):                                     # cumulative total of all keys <= q
        i = np.searchsorted(keys, q, side="right") - 1
        return np.where(i >= 0, cn[np.maximum(i, 0)], 0), np.where(i >= 0, cs[np.maximum(i, 0)], 0.0)

    hi_n, hi_s = cum(code * 1000 + (st - 1))
    lo_n, lo_s = cum(code * 1000 + np.maximum(st - 25, 0))
    return (hi_n - lo_n).astype(np.int64), hi_s - lo_s


def to_gold(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    g["day"] = ((g.step - 1) // 24 + 1).astype("int16")
    g["hour_of_day"] = ((g.step - 1) % 24).astype("int8")
    g["error_balance_orig"] = g.new_balance_orig + g.amount - g.old_balance_orig
    g["error_balance_dest"] = g.old_balance_dest + g.amount - g.new_balance_dest
    g["dest_txn_count_24h"], g["dest_amount_24h"] = dest_window_24h(g.name_dest, g.step, g.amount)
    g["amount_to_orig_balance"] = np.where(g.old_balance_orig > 0, g.amount / g.old_balance_orig.where(g.old_balance_orig > 0), np.nan)
    g["orig_emptied"] = ((g.old_balance_orig > 0) & (g.new_balance_orig == 0)).astype("int8")
    g["is_transfer"] = (g["type"] == "TRANSFER").astype("int8")
    return g
