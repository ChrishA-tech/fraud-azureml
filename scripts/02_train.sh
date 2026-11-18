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

bash "$(dirname "$0")/fetch_results.sh" "$JOB"
