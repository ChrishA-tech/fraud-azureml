"""Local checks before spending Azure credit: features match a slow reference, and prep -> train -> serve works."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import features as F  # noqa: E402
import synth  # noqa: E402


def test_dest_window_matches_naive_loop():
    rng = np.random.default_rng(1)
    n = 3000
    df = pd.DataFrame({"name_dest": rng.choice([f"C{i}" for i in range(40)], n),
                       "step": rng.integers(1, 120, n), "amount": rng.uniform(1, 100, n).round(2)})
    cnt, tot = F.dest_window_24h(df.name_dest, df.step, df.amount)
    for i in rng.choice(n, 400, replace=False):
        d, t = df.name_dest[i], df.step[i]
        w = df[(df.name_dest == d) & (df.step >= t - 24) & (df.step <= t - 1)]
        assert cnt[i] == len(w)
        assert abs(tot[i] - w.amount.sum()) < 1e-6


def test_expectations_drop_and_count():
    raw = synth.make(2000).rename(columns=F.RAW_COLUMNS)
    raw.loc[0, "amount"] = -5          # critical -> dropped
    raw.loc[1, "step"] = 900           # warn -> kept, counted
    clean, dq = F.apply_expectations(raw)
    assert dq["amount_valid"] == 1 and dq["step_in_month"] == 1
    assert len(clean) == len(raw) - 1


def test_prep_train_serve(tmp_path, monkeypatch):
    import mlflow
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    csv = tmp_path / "paysim.csv"
    synth.make(40000, seed=3).to_csv(csv, index=False)
    import prep, train
    with mlflow.start_run():
        prep.main(["--raw_data", str(csv), "--features", str(tmp_path / "feat")])
    with mlflow.start_run():
        train.main(["--features", str(tmp_path / "feat"), "--n_estimators", "80",
                    "--model_output", str(tmp_path / "model"), "--metrics_output", str(tmp_path / "metrics")])
    m = json.loads((tmp_path / "metrics" / "metrics.json").read_text())
    assert set(m["models"]) == {"full", "no_balance_errors"}
    assert 0 <= m["models"]["no_balance_errors"]["test"]["pr_auc"] <= 1
    # load the saved MLflow model in a fresh pyfunc, as the endpoint does, and score the sample request
    model = mlflow.pyfunc.load_model(str(tmp_path / "model"))
    req = json.loads((tmp_path / "metrics" / "sample_request.json").read_text())["input_data"]
    out = model.predict(pd.DataFrame(req["data"], columns=req["columns"]))
    assert list(out.columns) == ["fraud_probability", "decline"] and len(out) == 6
    assert out.fraud_probability.between(0, 1).all()
    # served probabilities equal the training-time model's (same trees, same threshold)
    df = train.split(pd.read_parquet(tmp_path / "feat" / "features.parquet"))
    te = df[df.split == "test"]
    feats = train.FEATURE_SETS[train.DEPLOY_SET]
    served = model.predict(te[feats].astype(float))
    assert m["models"]["no_balance_errors"]["test"]["recall"] == round(
        float(served.decline[te.is_fraud.to_numpy() == 1].mean()), 4)
