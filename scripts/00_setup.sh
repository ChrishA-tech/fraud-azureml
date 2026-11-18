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

say "Finding the regions this subscription may deploy to"
# Azure for Students limits deployments to a few regions through an 'allowed locations' policy.
ALLOWED=$(az policy assignment list --query "[].parameters.listOfAllowedLocations.value[]" -o tsv 2>/dev/null | sort -u | tr '\n' ' ')
echo "  Allowed by policy: ${ALLOWED:-(not readable; will try common regions)}"
PREFER="$LOCATION uksouth ukwest northeurope westeurope swedencentral francecentral germanywestcentral norwayeast switzerlandnorth italynorth polandcentral spaincentral eastus eastus2 westus2 westus3 centralus southcentralus canadacentral australiaeast centralindia southeastasia japaneast koreacentral"
CANDIDATES=""
if [ -n "$ALLOWED" ]; then
  for r in $PREFER; do case " $ALLOWED " in *" $r "*) case " $CANDIDATES " in *" $r "*) ;; *) CANDIDATES="$CANDIDATES $r";; esac;; esac; done
  for r in $ALLOWED; do case " $CANDIDATES " in *" $r "*) ;; *) CANDIDATES="$CANDIDATES $r";; esac; done
else
  CANDIDATES="$PREFER"
fi

say "Creating resource group $RG"
az group show -n "$RG" -o none 2>/dev/null || az group create -n "$RG" -l "$(echo $CANDIDATES | cut -d' ' -f1)" --only-show-errors -o none

if az ml workspace show -n "$WS" -g "$RG" -o none 2>/dev/null; then
  LOCATION=$(az ml workspace show -n "$WS" -g "$RG" --query location -o tsv)
  echo "Workspace $WS already exists in $LOCATION."
else
  OK=""
  for r in $CANDIDATES; do
    say "Creating Azure ML workspace $WS in $r (about 2-4 minutes)"
    if az ml workspace create -n "$WS" -g "$RG" -l "$r" --only-show-errors -o none 2>/tmp/ws_err.txt; then
      OK=yes; LOCATION="$r"; break
    fi
    if grep -q "RequestDisallowedByAzure" /tmp/ws_err.txt; then
      echo "  $r is not allowed for this subscription; trying the next region."
    else
      cat /tmp/ws_err.txt; exit 1
    fi
  done
  [ -n "$OK" ] || { echo "No allowed region worked. Paste this output to Claude."; exit 1; }
fi
state_set LOCATION "$LOCATION"
echo "Workspace region: $LOCATION"
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
