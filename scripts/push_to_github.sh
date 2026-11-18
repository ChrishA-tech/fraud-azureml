#!/usr/bin/env bash
# Create github.com/ChrishA-tech/fraud-azureml (if needed) and push this folder as Chrish. Asks for a token once and
# never stores it. Refuses to push until the README holds real Azure ML results.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=fraud-azureml
# --code-only pushes the code before the Azure run; the README then says plainly that results are not in yet.
if ! grep -q "Run on Azure ML" README.md && [ "${1:-}" != "--code-only" ]; then
  echo "README has no Azure results yet. Run the pipeline and scripts/update_readme.py first,"
  echo "or push the code now with:  bash scripts/push_to_github.sh --code-only"; exit 1
fi
MSG="Azure ML fraud pipeline: training pipeline, MLflow model, managed online endpoint"
grep -q "Run on Azure ML" README.md && MSG="Add Azure ML run results"

[ -d .git ] || git init -q -b main
git config user.name "Chrish Pep Felix A"
git config user.email "chrishpepfelixa@gmail.com"
git add -A
git commit -q -m "$MSG" || echo "Nothing new to commit."

read -r -s -p "Paste your ChrishA-tech GitHub token: " TOKEN; echo
LOGIN=$(curl -s -H "Authorization: token $TOKEN" https://api.github.com/user | sed -n 's/.*"login": "\(.*\)",/\1/p')
[ "$LOGIN" = "ChrishA-tech" ] || { echo "That token belongs to '$LOGIN', not ChrishA-tech. Stopping."; unset TOKEN; exit 1; }
curl -s -o /dev/null -H "Authorization: token $TOKEN" https://api.github.com/user/repos \
  -d "{\"name\":\"$REPO\",\"description\":\"PaySim fraud model on Azure ML: pipeline, MLflow, model registry, managed online endpoint\",\"private\":false}" || true
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/ChrishA-tech/$REPO.git"
git -c credential.helper= -c http.extraHeader="Authorization: Basic $(printf 'x-access-token:%s' "$TOKEN" | base64)" push -u origin main
unset TOKEN
echo "Pushed: https://github.com/ChrishA-tech/$REPO   (now delete the token at github.com/settings/tokens)"
