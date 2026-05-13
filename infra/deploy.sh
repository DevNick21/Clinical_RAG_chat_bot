#!/usr/bin/env bash
# =====================================================================
# Clinical RAG v2 - one-shot deploy
# =====================================================================
# Run from repo root after `az login`.
#
# Reads non-secret config from this script and secrets from .env. The
# secret values are pushed into Key Vault as part of the deploy; never
# baked into the image, never appearing in `az` command lines on the
# remote.
#
# Idempotent: re-running with the same params updates in-place. The
# script is split into 4 stages so you can comment any out for partial
# runs (e.g., stage 3 only when iterating on the image).
#
#   1. infra      - Bicep deploys ACR/MI/KV/LA/AI/ACA-env/ACA-app
#   2. secrets    - Push API_KEY + MODEL_API_KEY into Key Vault
#   3. image      - Build, tag, push to ACR
#   4. revision   - Point ACA at the new image (creates a new revision)
# =====================================================================
set -euo pipefail

# ---- Config ----------------------------------------------------------

RG="msc_project"
LOCATION="uksouth"
BASE_NAME="msc-rag-v2"
ACR_NAME="mscragv2acr"           # must match toLower(replace("${BASE_NAME}acr","-","")) in Bicep
KV_NAME="${BASE_NAME}-kv"
APP_NAME="${BASE_NAME}-api"
IMAGE_NAME="clinical-rag"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"
IMAGE_REF="${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG}"

# Pulls TARGET_URL, MODEL_API_KEY, MODEL_DEPLOYMENT_NAME, REASONING_EFFORT,
# API_KEY, ALLOWED_ORIGINS from .env at repo root.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
else
  echo "ERROR: .env not found at repo root. Required for TARGET_URL / API_KEY etc." >&2
  exit 2
fi

# Bail loudly if anything mandatory is unset.
: "${TARGET_URL:?TARGET_URL not set in .env}"
: "${MODEL_API_KEY:?MODEL_API_KEY not set in .env}"
: "${MODEL_DEPLOYMENT_NAME:?MODEL_DEPLOYMENT_NAME not set in .env}"
: "${API_KEY:?API_KEY not set in .env (generate: python -c \"import secrets;print(secrets.token_urlsafe(32))\")}"
REASONING_EFFORT="${REASONING_EFFORT:-low}"
ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-}"

echo "==> Deploying base=${BASE_NAME} image=${IMAGE_REF}"

# ---- Stage 1: infrastructure ----------------------------------------
echo
echo "==> [1/4] Bicep deploy (idempotent, ~3-5 min on first run)"
az deployment group create \
  --resource-group "$RG" \
  --template-file infra/main.bicep \
  --parameters \
      baseName="$BASE_NAME" \
      location="$LOCATION" \
      foundryEndpoint="$TARGET_URL" \
      modelDeploymentName="$MODEL_DEPLOYMENT_NAME" \
      reasoningEffort="$REASONING_EFFORT" \
      allowedOrigins="$ALLOWED_ORIGINS" \
  --output none

# ---- Stage 2: secrets ------------------------------------------------
echo
echo "==> [2/4] Pushing secrets to Key Vault"
# Secrets are passed via stdin instead of --value to avoid showing the
# secret in process listings / shell history.
echo -n "$API_KEY"       | az keyvault secret set --vault-name "$KV_NAME" --name "API-KEY"       --file /dev/stdin --output none
echo -n "$MODEL_API_KEY" | az keyvault secret set --vault-name "$KV_NAME" --name "MODEL-API-KEY" --file /dev/stdin --output none

# ---- Stage 3: build & push image ------------------------------------
echo
echo "==> [3/4] Building image"
docker build -t "$IMAGE_REF" .

echo "==> Logging into ACR"
az acr login --name "$ACR_NAME"

echo "==> Pushing image to ACR"
docker push "$IMAGE_REF"

# ---- Stage 4: roll the Container App revision -----------------------
echo
echo "==> [4/4] Updating Container App to ${IMAGE_REF}"
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
