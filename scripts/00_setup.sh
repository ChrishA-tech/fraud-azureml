#!/usr/bin/env bash
# One-time setup: Azure CLI ml extension, sign-in, Azure for Students subscription, resource group, workspace,
# and a CPU cluster that scales to zero (no charge while idle).
source "$(dirname "$0")/common.sh"

command -v az >/dev/null || { echo "Azure CLI not found. Install it with:  brew install azure-cli   then re-run."; exit 1; }
say "Adding the Azure ML CLI extension"
az extension add -n ml --upgrade -y --only-show-errors

if ! az account show >/dev/null 2>&1; then
  say "Signing in (a browser window opens; use the account you activated Azure for Students with)"
  az login --only-show-errors >/dev/null
fi

SUB=$(az account list --query "[?contains(name,'Azure for Students')].id | [0]" -o tsv)
if [ -z "$SUB" ]; then
  echo "No 'Azure for Students' subscription found on this login."
  echo "Activate it at https://azure.microsoft.com/free/students (university email, no card), then re-run."
  az account list --query "[].name" -o tsv
  exit 1
fi
az account set --subscription "$SUB"
echo "Using subscription: $(az account show --query name -o tsv)"
OFFER=$(az account show --query name -o tsv)

say "Registering the resource providers Azure ML needs (first time only; can take a few minutes)"
for p in Microsoft.MachineLearningServices Microsoft.Storage Microsoft.KeyVault Microsoft.ContainerRegistry \
         Microsoft.Insights Microsoft.OperationalInsights Microsoft.Network Microsoft.PolicyInsights Microsoft.Cdn; do
  az provider register --namespace "$p" --only-show-errors >/dev/null || true
done
for p in Microsoft.MachineLearningServices Microsoft.Storage Microsoft.KeyVault; do
  for _ in $(seq 1 40); do
    [ "$(az provider show -n "$p" --query registrationState -o tsv)" = "Registered" ] && break; sleep 10
  done
  echo "  $p: $(az provider show -n "$p" --query registrationState -o tsv)"
done

say "Creating resource group $RG in $LOCATION"
if ! az group create -n "$RG" -l "$LOCATION" --only-show-errors -o none; then
  echo "Could not create the group in $LOCATION. Student subscriptions only allow some regions."
  echo "Edit LOCATION in config.env (for example westeurope, northeurope, swedencentral, eastus) and re-run."
  exit 1
fi

say "Creating Azure ML workspace $WS (about 2-4 minutes)"
if ! az ml workspace show -n "$WS" -g "$RG" -o none 2>/dev/null; then
  az ml workspace create -n "$WS" -g "$RG" -l "$LOCATION" --only-show-errors -o none
fi
az configure --defaults group="$RG" workspace="$WS" location="$LOCATION"

say "Creating compute cluster $COMPUTE ($TRAIN_VM, 0-1 nodes, scales down after 5 idle minutes)"
if ! az ml compute show -n "$COMPUTE" -o none 2>/dev/null; then
  az ml compute create -n "$COMPUTE" --type amlcompute --size "$TRAIN_VM" \
     --min-instances 0 --max-instances 1 --idle-time-before-scale-down 300 --only-show-errors -o none
fi

if [ -z "$(state_get ENDPOINT)" ]; then
  state_set ENDPOINT "fraud-xgb-$(LC_ALL=C tr -dc 'a-z0-9' </dev/urandom | head -c 6)"
fi
state_set SUBSCRIPTION "$OFFER"
echo
echo "Setup done. Workspace: https://ml.azure.com  (select $WS)"
echo "Next:  bash scripts/01_upload_data.sh"
