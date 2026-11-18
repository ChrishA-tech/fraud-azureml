"""Add the package Azure ML's no-code MLflow server needs (azureml-ai-monitoring) to a downloaded MLflow model's
environment files. Used by scripts/fix_model_env.sh; the model weights are not touched."""
import sys
from pathlib import Path

root = Path(sys.argv[1])
mlmodel = next(root.rglob("MLmodel")).parent
PKG = "azureml-ai-monitoring"
req = mlmodel / "requirements.txt"
lines = req.read_text().splitlines()
if PKG not in lines:
    req.write_text("\n".join(lines + [PKG]) + "\n")
conda = mlmodel / "conda.yaml"
if conda.exists() and PKG not in conda.read_text():
    out, done = [], False
    for line in conda.read_text().splitlines():
        out.append(line)
        if not done and line.strip().startswith("- mlflow"):
            out.append(line[: len(line) - len(line.lstrip())] + "- " + PKG)
            done = True
    conda.write_text("\n".join(out) + "\n")
print(mlmodel)
