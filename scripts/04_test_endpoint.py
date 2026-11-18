"""Call the live endpoint: check its answers on 3 fraud + 3 genuine held-out payments, then time 50 requests.
Writes results/endpoint_test.json. Uses only the Python standard library and the Azure CLI."""
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ep = sys.argv[1]


def az(*args):
    return subprocess.run(["az", *args, "-o", "tsv", "--only-show-errors"], check=True, capture_output=True,
                          text=True).stdout.strip()


uri = az("ml", "online-endpoint", "show", "-n", ep, "--query", "scoring_uri")
key = az("ml", "online-endpoint", "get-credentials", "-n", ep, "--query", "primaryKey")
body = (ROOT / "results" / "sample_request.json").read_bytes()
labels = json.loads((ROOT / "results" / "sample_labels.json").read_text())


def call():
    req = urllib.request.Request(uri, data=body, method="POST", headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}", "azureml-model-deployment": "blue"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read())
    return out, (time.perf_counter() - t0) * 1000


out, _ = call()
rows = out if isinstance(out, list) else out.get("predictions", out)
if isinstance(rows, dict):                       # column-oriented answer -> list of rows
    rows = [dict(zip(rows, v)) for v in zip(*[list(c.values()) if isinstance(c, dict) else c for c in rows.values()])]
decisions = [int(r["decline"]) for r in rows]
probs = [float(r["fraud_probability"]) for r in rows]
lat = [call()[1] for _ in range(50)]
res = {"endpoint": ep, "rows_per_request": len(rows), "labels": labels, "declines": decisions,
       "probabilities": probs, "correct": sum(d == y for d, y in zip(decisions, labels)),
       "latency_ms_p50": round(statistics.median(lat), 1),
       "latency_ms_p95": round(sorted(lat)[int(0.95 * len(lat)) - 1], 1), "requests_timed": len(lat),
       "tested_at": time.strftime("%Y-%m-%d %H:%M", time.gmtime())}
(ROOT / "results" / "endpoint_test.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
