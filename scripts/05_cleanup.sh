#!/usr/bin/env bash
# Stop charges.  bash scripts/05_cleanup.sh             -> delete the endpoint (the hourly cost)
#                bash scripts/05_cleanup.sh --everything -> delete the whole resource group (workspace, storage,
#                                                           registry, cluster); do this once screenshots are taken
source "$(dirname "$0")/common.sh"
need_az
EP=$(state_get ENDPOINT)
if [ "${1:-}" = "--everything" ]; then
  say "Deleting resource group $RG and everything in it"
  az group delete -n "$RG" --yes --no-wait
  echo "Deletion started; it finishes in the background in a few minutes."
else
  say "Deleting endpoint $EP"
  az ml online-endpoint delete -n "$EP" --yes --only-show-errors || echo "Endpoint not found (already deleted)."
  state_set ENDPOINT_DELETED yes
  echo "The cluster scales to zero by itself. To remove everything later:  bash scripts/05_cleanup.sh --everything"
fi
