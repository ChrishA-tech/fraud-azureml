#!/usr/bin/env bash
# Re-register the trained model with azureml-ai-monitoring added to its environment (Azure ML's no-code MLflow
# scoring script imports it). Same trained trees and threshold; only the environment files change. Creates a new
# model version and makes 03_deploy.sh use it.
source "$(dirname "$0")/common.sh"
need_az
OLD=$(state_get MODEL_VERSION)
rm -rf results/model_download && mkdir -p results/model_download
say "Downloading $MODEL_NAME:$OLD"
az ml model download -n "$MODEL_NAME" -v "$OLD" --download-path results/model_download --only-show-errors
DIR=$(python3 scripts/fix_model_env.py results/model_download)
echo "Patched environment in $DIR:"; cat "$DIR/requirements.txt"
say "Registering the patched model as a new version"
VER=$(az ml model create -n "$MODEL_NAME" --type mlflow_model --path "$DIR" \
      --description "fraud-xgb:$OLD with azureml-ai-monitoring added for no-code MLflow serving (same weights)" \
      --query version -o tsv --only-show-errors)
state_set MODEL_VERSION "$VER"
rm -rf results/model_download
echo "Registered $MODEL_NAME:$VER. Next:  bash scripts/03_deploy.sh"
