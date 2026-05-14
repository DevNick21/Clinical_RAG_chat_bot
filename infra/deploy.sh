#!/usr/bin/env bash
# =====================================================================
# Clinical RAG v2 - one-shot deploy
# =====================================================================
# Run from repo root after `az login`.
#
# Reads non-secret config from this script and secrets from .env. The
# secret values flow into Terraform via TF_VAR_* env vars (loaded by
# infra/tf/_load_env.sh) and end up in Key Vault as the @secure()
# secrets the Container App references — never baked into the image,
# never appearing on remote command lines.
#
# Idempotent: re-running with the same params updates in-place. The
# script is split into 3 stages so you can comment any out for partial
# runs (e.g., stage 3 only when iterating on the image).
#
#   1. infra      - terraform apply (ACR/MI/KV/LA/AI/ACA-env/ACA-app + secrets)
#   2. image      - docker build + push to ACR
#   3. revision   - point ACA at the new image (creates a new revision)
#
# Terraform state lives on faissprod/tfstate (azurerm backend, see
# infra/tf/backend.tf). On first run from a fresh checkout, you must
# `terraform -chdir=infra/tf init` once before this script will work.
# =====================================================================
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# ---- Config ----------------------------------------------------------

RG="msc_project"
LOCATION="uksouth"
BASE_NAME="msc-rag-v2"
ACR_NAME="mscragv2acr"
KV_NAME="${BASE_NAME}-kv"
APP_NAME="${BASE_NAME}-api"
IMAGE_NAME="clinical-rag"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"
IMAGE_REF="${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG}"

# ---- Load TF_VAR_* from .env -----------------------------------------
# Sourced helper exports api_key_value, model_api_key_value,
# foundry_endpoint, model_deployment_name, fast/audit model vars,
# allowed_origins, reasoning_effort, plus deployer_object_id (live
# via az). Bails loudly on missing required values.
source infra/tf/_load_env.sh

# ---- Auto-discover SWA hostname --------------------------------------
# Append the live SWA hostname to ALLOWED_ORIGINS so a deploy.sh re-run
# doesn't blow away the CORS allow-list that deploy_frontend.sh sets.
# Best-effort: if SWA doesn't exist yet (first deploy), skip silently.
SWA_HOST=$(az staticwebapp show \
  --name "${BASE_NAME}-web" --resource-group "$RG" \
  --query "defaultHostname" -o tsv 2>/dev/null || true)
if [[ -n "$SWA_HOST" ]]; then
  SWA_ORIGIN="https://${SWA_HOST}"
  if [[ -z "${TF_VAR_allowed_origins:-}" ]]; then
    export TF_VAR_allowed_origins="$SWA_ORIGIN"
  elif [[ ",${TF_VAR_allowed_origins}," != *",${SWA_ORIGIN},"* ]]; then
    export TF_VAR_allowed_origins="${TF_VAR_allowed_origins},${SWA_ORIGIN}"
  fi
  echo "==> Auto-detected SWA — adding ${SWA_ORIGIN} to allowed_origins"
fi

# Pin base_name + location overrides (the .tf defaults already match,
# but being explicit here lets you fork the script for a parallel
# environment by changing one block.)
export TF_VAR_base_name="$BASE_NAME"
export TF_VAR_location="$LOCATION"
export TF_VAR_resource_group_name="$RG"

echo "==> Deploying base=${BASE_NAME} image=${IMAGE_REF}"

# ---- Stage 1: infrastructure + secrets ------------------------------
# Secrets flow as @secure()-equivalent terraform vars into KV secret
# resources INSIDE this same apply, so the Container App's secret
# references resolve cleanly when ACA tries to provision.
echo
echo "==> [1/3] terraform apply (idempotent, ~30s when no changes)"
terraform -chdir=infra/tf apply -auto-approve -input=false

# ---- Stage 2: build & push image ------------------------------------
echo
echo "==> [2/3] Building image"
docker build -t "$IMAGE_REF" .

echo "==> Logging into ACR"
az acr login --name "$ACR_NAME"

echo "==> Pushing image to ACR"
docker push "$IMAGE_REF"

# ---- Stage 3: roll the Container App revision -----------------------
# az CLI rather than terraform here because the .tf has
# `lifecycle { ignore_changes = [template.container.image] }` to keep
# it from reverting the image on every plan. The image swap is purely
# operational, not a config change.
echo
echo "==> [3/3] Updating Container App to ${IMAGE_REF}"
az containerapp update \
  --name "$APP_NAME" \
  --resource-group "$RG" \
  --image "$IMAGE_REF" \
  --output none

# ---- Done ------------------------------------------------------------
APP_FQDN=$(az containerapp show \
  --name "$APP_NAME" \
  --resource-group "$RG" \
  --query "properties.configuration.ingress.fqdn" \
  -o tsv)

echo
echo "==> Deployed."
echo "    URL:    https://${APP_FQDN}"
echo "    Health: https://${APP_FQDN}/health"
echo "    Test:   curl -H \"Authorization: Bearer \$API_KEY\" -X POST https://${APP_FQDN}/api/chat -d '{\"message\":\"hi\"}' -H 'Content-Type: application/json'"
