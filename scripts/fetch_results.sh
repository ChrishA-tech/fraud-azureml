#!/usr/bin/env bash
# Download metrics from a finished pipeline's train step and, for a full run, register the model.
#   bash scripts/fetch_results.sh [pipeline_job_name]     default: the last job started by 02_train.sh
source "$(dirname "$0")/common.sh"
need_az
JOB="${1:-$(state_get TRAIN_JOB)}"
SAMPLE="$(state_get SAMPLE)"
[ -n "$JOB" ] || { echo "No pipeline job recorded. Run scripts/02_train.sh first."; exit 1; }
CHILD=$(az ml job list --parent-job-name "$JOB" --query "[?display_name=='train'].name | [0]" -o tsv --only-show-errors 2>/dev/null)
[ -n "$CHILD" ] || { echo "Could not find the train step of $JOB."; exit 1; }
echo "Pipeline $JOB, train step $CHILD"

say "Downloading metrics"
rm -rf results/azure_job && mkdir -p results/azure_job
az ml job download -n "$CHILD" --output-name metrics_output --download-path results/azure_job --only-show-errors
for f in metrics.json sample_request.json sample_labels.json; do
  src=$(find results/azure_job -name "$f" | head -1)
  [ -n "$src" ] || { echo "$f not found in the download:"; find results/azure_job | head -20; exit 1; }
  cp "$src" "results/$f"
done
python3 -c "import json;m=json.load(open('results/metrics.json'));[print(k, {x:v['test'][x] for x in ('pr_auc','precision','recall','false_declines')}) for k,v in m['models'].items()]"

if [ "$SAMPLE" = "1.0" ] || [ "$SAMPLE" = "1" ]; then
  say "Registering the model in the Azure ML model registry"
  VER=$(az ml model create -n "$MODEL_NAME" --type mlflow_model --path "azureml://jobs/$CHILD/outputs/model_output" \
        --description "XGBoost PaySim fraud model (no balance-error features), threshold chosen on validation" \
        --query version -o tsv --only-show-errors)
  state_set MODEL_VERSION "$VER"
  echo "Registered $MODEL_NAME:$VER"
  echo "Next:  bash scripts/03_deploy.sh"
else
  echo "Sample run only (not registered). For the real result run:  bash scripts/02_train.sh"
fi
