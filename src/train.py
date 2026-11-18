"""
Azure ML step 2: time-based split, XGBoost for two feature sets, threshold chosen on validation to maximise net
savings, evaluation on a held-out later time period, MLflow tracking, and an MLflow model ready for an endpoint.

Same modelling choices as the Databricks Fraud Lakehouse, so the two platforms can be compared like for like.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = ["amount", "old_balance_orig", "new_balance_orig", "old_balance_dest", "new_balance_dest", "hour_of_day",
        "is_transfer", "dest_txn_count_24h", "dest_amount_24h", "amount_to_orig_balance", "orig_emptied"]
FEATURE_SETS = {"full": BASE + ["error_balance_orig", "error_balance_dest"], "no_balance_errors": BASE}
DEPLOY_SET = "no_balance_errors"        # served model: without PaySim's balance-error artefact
TRAIN_SHARE, VALID_SHARE, SEED = 0.70, 0.15, 7
FALSE_DECLINE_COST = 10.0               # illustrative cost of wrongly declining a genuine payment


def split(df: pd.DataFrame) -> pd.DataFrame:
    """Earliest 70% of rows train, next 15% validate, latest 15% test, cut on whole steps (hours) so no future hour
    leaks into training."""
    cut_t = int(df.step.quantile(TRAIN_SHARE))
    cut_v = max(int(df.step.quantile(TRAIN_SHARE + VALID_SHARE)), cut_t + 1)
    df = df.copy()
    df["split"] = np.where(df.step <= cut_t, "train", np.where(df.step <= cut_v, "valid", "test"))
    for s in ("train", "valid", "test"):
        part = df[df.split == s]
        if len(part) == 0 or part.is_fraud.nunique() < 2:
            raise ValueError(f"split {s} has {len(part)} rows and {int(part.is_fraud.sum())} fraud cases")
    df.attrs["cuts"] = (cut_t, cut_v)
    return df


def costs(p, y, amount, thr):
    flag = p >= thr
    caught = float(amount[flag & (y == 1)].sum())
    fd = int((flag & (y == 0)).sum())
    return {"fraud_value_caught": round(caught, 2), "false_declines": fd,
            "net_saving": round(caught - FALSE_DECLINE_COST * fd, 2)}


def choose_threshold(p, y, amount):
    grid = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), np.quantile(p, np.linspace(0.90, 0.9999, 60))]))
    return float(max(grid, key=lambda t: costs(p, y, amount, t)["net_saving"]))


def evaluate(p, y, amount, thr):
    flag = p >= thr
    tp = int((flag & (y == 1)).sum())
    out = {"n": int(len(y)), "fraud_cases": int(y.sum()), "threshold": round(thr, 4),
           "pr_auc": round(float(average_precision_score(y, p)), 4), "roc_auc": round(float(roc_auc_score(y, p)), 4),
           "precision": round(tp / max(int(flag.sum()), 1), 4), "recall": round(tp / max(int(y.sum()), 1), 4),
           "value_recall": round(float(amount[flag & (y == 1)].sum() / max(amount[y == 1].sum(), 1e-9)), 4)}
    out.update(costs(p, y, amount, thr))
    return out


def fit(tr, va, feats, n_estimators):
    from xgboost import XGBClassifier
    pos = max(int(tr.is_fraud.sum()), 1)
    m = XGBClassifier(n_estimators=n_estimators, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=5, scale_pos_weight=(len(tr) - pos) / pos, eval_metric="aucpr",
                      early_stopping_rounds=40, tree_method="hist", random_state=SEED, n_jobs=-1)
    m.fit(tr[feats], tr.is_fraud, eval_set=[(va[feats], va.is_fraud)], verbose=False)
    return m


def save_pyfunc(model, feats, thr, example, path: Path):
    """Write an MLflow pyfunc model (XGBoost booster + threshold) that the endpoint serves without a scoring script."""
    import xgboost
    src = Path(__file__).resolve().parent
    tmp = Path(tempfile.mkdtemp())
    booster = model.get_booster()
    best = getattr(model, "best_iteration", None)
    if best is not None:                       # keep only the trees up to early stopping's best round
        booster = booster[: best + 1]
    booster.save_model(str(tmp / "xgb.json"))
    (tmp / "meta.json").write_text(json.dumps({"features": feats, "threshold": thr}))
    sys.path.insert(0, str(src))
    from fraud_pyfunc import FraudScorer
    from mlflow.models import infer_signature
    sig = infer_signature(example, pd.DataFrame({"fraud_probability": [0.5], "decline": [0]}))
    if path.exists() and any(path.iterdir()):     # Azure ML mounts an empty output folder; only clear leftovers
        shutil.rmtree(path)
    mlflow.pyfunc.save_model(
        path=str(path), python_model=FraudScorer(), code_paths=[str(src / "fraud_pyfunc.py")],
        artifacts={"xgb_model": str(tmp / "xgb.json"), "meta": str(tmp / "meta.json")},
        signature=sig, input_example=example,
        pip_requirements=[f"xgboost=={xgboost.__version__}", f"pandas=={pd.__version__}",
                          f"numpy=={np.__version__}", f"mlflow=={mlflow.__version__}"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="folder with features.parquet")
    ap.add_argument("--n_estimators", type=int, default=600)
    ap.add_argument("--model_output", required=True)
    ap.add_argument("--metrics_output", required=True)
    a = ap.parse_args(argv)

    df = split(pd.read_parquet(Path(a.features) / "features.parquet"))
    tr, va, te = (df[df.split == s] for s in ("train", "valid", "test"))
    results = {"split_sizes": {s: int((df.split == s).sum()) for s in ("train", "valid", "test")},
               "fraud_by_split": {s: int(df.loc[df.split == s, "is_fraud"].sum()) for s in ("train", "valid", "test")},
               "step_cutoffs": dict(zip(("train_max_step", "valid_max_step"), df.attrs["cuts"])),
               "test_days": int(te.day.nunique()), "models": {}}
    summary = Path(a.features) / "prep_summary.json"
    if summary.exists():
        results["prep"] = json.loads(summary.read_text())
    mlflow.log_params({"train_share": TRAIN_SHARE, "valid_share": VALID_SHARE, "n_estimators": a.n_estimators,
                       "false_decline_cost": FALSE_DECLINE_COST, "deployed_feature_set": DEPLOY_SET})
    kept = {}
    for name, feats in FEATURE_SETS.items():
        t0 = time.perf_counter()
        m = fit(tr, va, feats, a.n_estimators)
        train_s = time.perf_counter() - t0
        thr = choose_threshold(m.predict_proba(va[feats])[:, 1], va.is_fraud.to_numpy(), va.amount.to_numpy())
        res = evaluate(m.predict_proba(te[feats])[:, 1], te.is_fraud.to_numpy(), te.amount.to_numpy(), thr)
        res.update({"train_seconds": round(train_s, 1), "best_iteration": int(getattr(m, "best_iteration", -1) or -1)})
        results["models"][name] = {"features": feats, "test": res}
        mlflow.log_metrics({f"{name}_{k}": float(v) for k, v in res.items() if isinstance(v, (int, float))})
        kept[name] = (m, feats, thr)

    m, feats, thr = kept[DEPLOY_SET]
    example = te[feats].head(5).astype(float)
    save_pyfunc(m, feats, thr, example, Path(a.model_output))
    out = Path(a.metrics_output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    # endpoint test request: 3 fraud and 3 genuine payments from the held-out test period, with their true labels
    req = pd.concat([te[te.is_fraud == 1].head(3), te[te.is_fraud == 0].head(3)])
    (out / "sample_request.json").write_text(json.dumps(
        {"input_data": {"columns": feats, "index": list(range(len(req))),
                        "data": req[feats].astype(float).replace({np.nan: None}).values.tolist()}}))
    (out / "sample_labels.json").write_text(json.dumps(req.is_fraud.astype(int).tolist()))
    print(json.dumps({k: v["test"] for k, v in results["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
