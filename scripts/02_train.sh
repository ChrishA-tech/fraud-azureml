#!/usr/bin/env bash
# Run the training pipeline (prep -> train) on the cluster, register the model, download the metrics.
#   bash scripts/02_train.sh [sample_fraction]     default 1.0 = all 6.36M transactions
source "$(dirname "$0")/common.sh"
need_az
SAMPLE="${1:-1.0}"
say "Submitting the pipeline (sample_fraction=$SAMPLE). The first run builds the environment image (~10-15 min)."
JOB=$(az ml job create -f pipelines/train_pipeline.yml --set inputs.sample_fraction="$SAMPLE" --query name -o tsv --only-show-errors)
echo "Job: $JOB   (watch it in https://ml.azure.com -> Jobs -> fraud-azureml)"
az ml job stream -n "$JOB" --only-show-errors || true
STATUS=$(az ml job show -n "$JOB" --query status -o tsv)
echo "Status: $STATUS"
[ "$STATUS" = "Completed" ] || { echo "The pipeline did not complete. Open the job in ml.azure.com -> Outputs + logs."; exit 1; }
state_set TRAIN_JOB "$JOB"
state_set SAMPLE "$SAMPLE"

say "Downloading metrics"
rm -rf results/azure_job && mkdir -p results/azure_job
az ml job download -n "$JOB" --output-name metrics --download-path results/azure_job --only-show-errors
for f in metrics.json sample_request.json sample_labels.json; do
  cp "$(find results/azure_job -name "$f" | head -1)" "results/$f"
done
python3 -c "import json;m=json.load(open('results/metrics.json'));[print(k, {x:v['test'][x] for x in ('pr_auc','precision','recall','false_declines')}) for k,v in m['models'].items()]"

if [ "$SAMPLE" = "1.0" ] || [ "$SAMPLE" = "1" ]; then
  say "Registering the model in the Azure ML model registry"
  VER=$(az ml model create -n "$MODEL_NAME" --type mlflow_model --path "azureml://jobs/$JOB/outputs/model" \
        --description "XGBoost PaySim fraud model (no balance-error features), threshold chosen on validation" \
        --query version -o tsv --only-show-errors)
  state_set MODEL_VERSION "$VER"
  echo "Registered $MODEL_NAME:$VER"
  echo "Next:  bash scripts/03_deploy.sh"
else
  echo "Sample run only (not registered). For the real result run:  bash scripts/02_train.sh"
fi
