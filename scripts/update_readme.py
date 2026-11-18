"""Write the real Azure ML results into README.md (between the RESULTS markers). Refuses to run without them."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
m = json.loads((ROOT / "results" / "metrics.json").read_text())
ep_path = ROOT / "results" / "endpoint_test.json"
ep = json.loads(ep_path.read_text()) if ep_path.exists() else None
def kv(path):
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


state = {**kv(ROOT / "config.env"), **kv(ROOT / ".azure_state")}
prep = m.get("prep", {})
s, f = m["split_sizes"], m["fraud_by_split"]

rows = []
for name, label in (("full", "XGBoost, all features"), ("no_balance_errors", "XGBoost, without balance-error artefact (model served by the endpoint)")):
    t = m["models"][name]["test"]
    rows.append(f"| {label} | {t['pr_auc']:.4f} | {t['roc_auc']:.4f} | {t['precision']:.3f} | {t['recall']:.3f} | "
                f"{t['false_declines']:,} | {t['train_seconds']:.0f} s |")
lines = [
    f"Run on Azure ML ({state.get('SUBSCRIPTION', 'Azure for Students')}), pipeline job `{state.get('TRAIN_JOB', '?')}`, "
    f"model `fraud-xgb:{state.get('MODEL_VERSION', '?')}`.",
    "",
    f"Data: {prep.get('raw_rows', 0):,} raw rows, {prep.get('model_rows', 0):,} TRANSFER/CASH_OUT rows modelled; "
    f"time split train / validation / test = {s['train']:,} / {s['valid']:,} / {s['test']:,} "
    f"({f['test']:,} fraud cases in the test period, {m['test_days']} days never used for training or the threshold).",
    "",
    "| Model (held-out test period) | PR-AUC | ROC-AUC | Precision | Recall | False declines | Training time |",
    "|---|---|---|---|---|---|---|", *rows, ""]
if ep:
    lines += [f"**Endpoint** (`{state.get('ENDPOINT_VM', 'Standard_DS2_v2')}`, managed online endpoint, MLflow model with no "
              f"scoring script): {ep['correct']}/{len(ep['labels'])} held-out test payments classified correctly "
              f"(3 fraud, 3 genuine); latency over {ep['requests_timed']} requests of {ep['rows_per_request']} rows: "
              f"p50 {ep['latency_ms_p50']} ms, p95 {ep['latency_ms_p95']} ms (measured from the UK, including network). "
              f"Deployment took about {state.get('DEPLOY_MINUTES', '?')} minutes"
              + ("; the endpoint was deleted after testing." if state.get("ENDPOINT_DELETED") else "."), ""]
dq = prep.get("expectation_failures", {})
if dq:
    lines += ["**Data-quality checks:** " + ", ".join(f"{k} {v:,}" for k, v in dq.items()) + " failing rows.", ""]
lines += ["PaySim is simulated: its balance fields reveal fraud almost perfectly, so treat these scores as an upper bound "
          "for real payments, not an estimate."]
block = "<!-- RESULTS:START -->\n" + "\n".join(lines) + "\n<!-- RESULTS:END -->"
readme = ROOT / "README.md"
readme.write_text(re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", block, readme.read_text(), flags=re.S))
print(block)
