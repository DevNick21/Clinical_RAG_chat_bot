#!/usr/bin/env bash
# =====================================================================
# Deploy the React frontend to Azure Static Web Apps.
# =====================================================================
# Independent of infra/deploy.sh — the whole point of going to SWA over
# bundling the frontend into the API container is so UI changes ship
# without rebuilding the Python image. Run this on its own whenever the
# frontend changes; it pulls the ACA backend URL automatically.
#
# Stages:
#   1. terraform apply with deploy_swa=true (Free SKU, ~30s first run)
#   2. Resolve ACA backend FQDN (so REACT_APP_API_URL points at it)
#   3. npm ci + npm run build (with REACT_APP_API_URL/KEY baked in)
#   4. SWA CLI push (build/ → CDN, ~10–20s)
#   5. terraform apply with allowed_origins updated to include SWA URL
#      (CORS update goes through IaC so the next deploy.sh stays clean)
#
# Prereqs:
#   - infra/deploy.sh has run successfully (ACA exists)
#   - Node 18+ + npm on PATH (for building React)
#   - .env at repo root populated with at minimum API_KEY
# =====================================================================
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ---- Config ----------------------------------------------------------
RG="msc_project"
LOCATION_SWA="westeurope"   # Free-tier SWA not in uksouth
BASE_NAME="msc-rag-v2"
SWA_NAME="${BASE_NAME}-web"
ACA_NAME="${BASE_NAME}-api"
ACA_RG="$RG"

# Per-script overrides BEFORE sourcing _load_env.sh.
export TF_VAR_deploy_swa=true
export TF_VAR_swa_location="$LOCATION_SWA"

source infra/tf/_load_env.sh

# Same TF_VAR_api_key_value loaded by helper; alias for the React build
# step further down which expects $API_KEY.
API_KEY="$TF_VAR_api_key_value"

# ---- Stage 1: SWA via terraform --------------------------------------
echo
echo "==> [1/5] terraform apply (creates SWA on first run, ~30s)"
terraform -chdir=infra/tf apply -auto-approve -input=false

SWA_HOST=$(terraform -chdir=infra/tf output -raw swa_hostname)
SWA_URL="https://${SWA_HOST}"
echo "    SWA URL: ${SWA_URL}"

# ---- Stage 2: resolve ACA backend FQDN -------------------------------
echo
echo "==> [2/5] Resolving ACA backend URL"
ACA_FQDN=$(az containerapp show \
  --name "$ACA_NAME" --resource-group "$ACA_RG" \
  --query "properties.configuration.ingress.fqdn" -o tsv 2>/dev/null || true)
if [[ -z "$ACA_FQDN" ]]; then
  echo "ERROR: ACA app '${ACA_NAME}' not found in rg '${ACA_RG}'." >&2
  echo "       Run infra/deploy.sh first to create the backend." >&2
  exit 3
fi
API_BASE="https://${ACA_FQDN}/api"
echo "    REACT_APP_API_URL = ${API_BASE}"

# ---- Stage 3: build React --------------------------------------------
echo
echo "==> [3/5] Building React (CRA bakes envs at build time)"
pushd platform/frontend >/dev/null

if [[ -f package-lock.json ]]; then
  npm ci
else
  npm install
fi

# CRA reads REACT_APP_* envs at build time; they get inlined into the
# minified JS. API_KEY in particular is visible in devtools — acceptable
# for the dissertation footprint, but for a real multi-tenant rollout
# replace with per-user OAuth + server-side session.
REACT_APP_API_URL="$API_BASE" \
REACT_APP_API_KEY="$API_KEY" \
npm run build

popd >/dev/null

# ---- Stage 4: push to SWA --------------------------------------------
echo
echo "==> [4/5] Deploying build/ to SWA"
SWA_TOKEN=$(az staticwebapp secrets list \
  --name "$SWA_NAME" --resource-group "$RG" \
  --query "properties.apiKey" -o tsv)

npx --yes @azure/static-web-apps-cli@latest deploy \
  ./platform/frontend/build \
  --deployment-token "$SWA_TOKEN" \
  --env production

# ---- Stage 5: update ACA CORS allow-list via IaC ---------------------
# Going through Terraform (not `az containerapp update --set-env-vars`)
# so the .tf stays source-of-truth. Otherwise the next deploy.sh run
# would see allowed_origins as drift and try to revert it.
echo
echo "==> [5/5] Adding ${SWA_URL} to allowed_origins via terraform"
CURRENT_ORIGINS="${TF_VAR_allowed_origins:-}"
if [[ ",${CURRENT_ORIGINS}," == *",${SWA_URL},"* ]]; then
  echo "    Already present; no terraform apply needed."
else
  NEW_ORIGINS="${CURRENT_ORIGINS:+${CURRENT_ORIGINS},}${SWA_URL}"
  export TF_VAR_allowed_origins="$NEW_ORIGINS"
  terraform -chdir=infra/tf apply -auto-approve -input=false
  echo "    Set to: ${NEW_ORIGINS}"
  echo "    Now persist this in .env for future deploys:"
  echo "      ALLOWED_ORIGINS=${NEW_ORIGINS}"
  echo "    (ACA rolled a new revision; old one is still serving until the new one is healthy.)"
fi

# ---- Done ------------------------------------------------------------
echo
echo "==> Frontend deployed."
echo "    URL:    ${SWA_URL}"
echo "    API:    ${API_BASE}"
echo "    Test:   open ${SWA_URL} in a browser; UI should be able to reach /api/chat."
