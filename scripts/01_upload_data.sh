#!/usr/bin/env bash
# Upload the PaySim CSV as a versioned Azure ML data asset (paysim-raw:1).
#   bash scripts/01_upload_data.sh [path/to/PS_20174392719_1491204439457_log.csv]
source "$(dirname "$0")/common.sh"
need_az
CSV="${1:-$HOME/Downloads/PS_20174392719_1491204439457_log.csv}"
if [ ! -f "$CSV" ]; then
  CSV=$(find "$HOME/Downloads" "$HOME/Documents" -maxdepth 3 -name 'PS_2017*log.csv' 2>/dev/null | head -1 || true)
fi
[ -f "${CSV:-}" ] || { echo "PaySim CSV not found. Download it from https://www.kaggle.com/datasets/ealaxi/paysim1 and pass its path."; exit 1; }
ROWS=$(($(wc -l < "$CSV") - 1))
echo "File: $CSV  ($ROWS rows)"
if [ "$ROWS" -ne 6362620 ]; then
  echo "Warning: the full PaySim file has 6,362,620 rows. 1,048,575 means it was cut by a spreadsheet app."
fi
if az ml data show -n "$DATA_NAME" -v 1 -o none 2>/dev/null; then
  echo "Data asset $DATA_NAME:1 already exists; skipping upload."
else
  say "Uploading to the workspace datastore (about 470 MB)"
  az ml data create -n "$DATA_NAME" -v 1 --type uri_file --path "$CSV" \
     --description "PaySim mobile-money transactions (Kaggle ealaxi/paysim1), $ROWS rows" --only-show-errors -o none
fi
state_set DATA_ROWS "$ROWS"
echo "Next:  bash scripts/02_train.sh          (or  bash scripts/02_train.sh 0.1  for a quick 10% test run)"
