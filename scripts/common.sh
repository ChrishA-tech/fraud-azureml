#!/usr/bin/env bash
# Shared helpers (sourced by the other scripts).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/config.env"
STATE="$ROOT/.azure_state"
touch "$STATE"
state_get() { grep "^$1=" "$STATE" 2>/dev/null | tail -1 | cut -d= -f2-; }
state_set() { grep -v "^$1=" "$STATE" > "$STATE.tmp" 2>/dev/null || true; echo "$1=$2" >> "$STATE.tmp"; mv "$STATE.tmp" "$STATE"; }
# a region found to work at setup overrides config.env
if [ -n "$(state_get LOCATION)" ]; then LOCATION="$(state_get LOCATION)"; fi
say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
need_az() {
  command -v az >/dev/null || { echo "Azure CLI not found. Install it with:  brew install azure-cli"; exit 1; }
  az account show >/dev/null 2>&1 || { echo "Not signed in. Run scripts/00_setup.sh first."; exit 1; }
  az ml workspace show -n "$WS" -g "$RG" -o none 2>/dev/null || {
    echo "Workspace $WS does not exist yet. Run  bash scripts/00_setup.sh  and make sure it finishes without errors."; exit 1; }
  az configure --defaults group="$RG" workspace="$WS" >/dev/null
}
