"""Azure ML step 1: raw PaySim CSV -> data-quality checks -> model features (Parquet)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np

import features as F

MODEL_COLUMNS = ["step", "day", "type", "amount", "is_fraud", "old_balance_orig", "new_balance_orig",
                 "old_balance_dest", "new_balance_dest", "hour_of_day", "is_transfer", "dest_txn_count_24h",
                 "dest_amount_24h", "amount_to_orig_balance", "orig_emptied", "error_balance_orig", "error_balance_dest"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_data", required=True)
    ap.add_argument("--sample_fraction", type=float, default=1.0)
    ap.add_argument("--features", required=True, help="output folder")
    a = ap.parse_args(argv)

    raw = F.read_raw(a.raw_data)
    n_raw = len(raw)
    if a.sample_fraction < 1:
        raw = raw.sample(frac=a.sample_fraction, random_state=7).reset_index(drop=True)
    clean, dq = F.apply_expectations(raw)
    gold = F.to_gold(clean)
    model_rows = gold[gold["type"].isin(F.MODEL_TYPES)][MODEL_COLUMNS].copy()
    model_rows["type"] = model_rows["type"].astype(str)

    out = Path(a.features)
    out.mkdir(parents=True, exist_ok=True)
    model_rows.to_parquet(out / "features.parquet", index=False)
    summary = {"raw_rows": n_raw, "sample_fraction": a.sample_fraction, "full_file": n_raw == F.EXPECTED_RAW_ROWS,
               "rows_after_checks": int(len(clean)), "model_rows": int(len(model_rows)),
               "fraud_cases": int(model_rows.is_fraud.sum()), "expectation_failures": dq}
    (out / "prep_summary.json").write_text(json.dumps(summary, indent=2))

    mlflow.log_metrics({"raw_rows": n_raw, "rows_after_checks": len(clean), "model_rows": len(model_rows),
                        **{f"dq_fail_{k}": v for k, v in dq.items()}})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
