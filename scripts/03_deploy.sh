#!/usr/bin/env bash
# Deploy the registered MLflow model to a managed online endpoint (no scoring script), test it, then offer to
# delete it. The endpoint is the only part that costs money every hour, so do not leave it running.
source "$(dirname "$0")/common.sh"
need_az
EP=$(state_get ENDPOINT); VER=$(state_get MODEL_VERSION)
[ -n "$VER" ] || { echo "No registered model yet. Run scripts/02_train.sh first."; exit 1; }

say "Creating endpoint $EP"
az ml online-endpoint show -n "$EP" -o none 2>/dev/null || \
  az ml online-endpoint create -n "$EP" -f deploy/endpoint.yml --only-show-errors -o none

say "Deploying $MODEL_NAME:$VER on $ENDPOINT_VM (about 10-15 minutes)"
START=$(date +%s)
az ml online-deployment create -n blue --endpoint-name "$EP" -f deploy/deployment.yml \
   --set model="azureml:$MODEL_NAME:$VER" instance_type="$ENDPOINT_VM" --all-traffic --only-show-errors -o none
state_set DEPLOY_MINUTES $(( ($(date +%s) - START) / 60 ))

say "Testing the endpoint"
python3 scripts/04_test_endpoint.py "$EP"

echo
read -r -p "Delete the endpoint now to stop charges? Results are saved in results/. [Y/n] " A
if [ "${A:-Y}" != "n" ] && [ "${A:-Y}" != "N" ]; then
  bash scripts/05_cleanup.sh
else
  echo "Endpoint left running. Delete it later with:  bash scripts/05_cleanup.sh"
fi
echo "Next:  python3 scripts/update_readme.py   then   bash scripts/push_to_github.sh"
