#!/usr/bin/env bash
# =====================================================================
# Deploy FAST + AUDIT model deployments into an existing Azure AI
# Foundry account.
# =====================================================================
# Separate from infra/deploy.sh because Foundry deployments are gated by
# subscription-level quota — they should be visible and reviewed
# independently of the app infra. Idempotent.
#
# Required env (or set via .env at repo root):
#   FOUNDRY_RESOURCE_GROUP   resource group of the Foundry account
#   FOUNDRY_ACCOUNT_NAME     name of the Azure AI Foundry account
# Optional overrides (with defaults):
#   FAST_MODEL_NAME=gpt-5-mini   FAST_MODEL_DEPLOYMENT_NAME=gpt-5-mini
#   FAST_MODEL_SKU=GlobalStandard FAST_MODEL_CAPACITY=50
#   AUDIT_MODEL_NAME=gpt-5-nano  AUDIT_MODEL_DEPLOYMENT_NAME=gpt-5-nano
#   AUDIT_MODEL_SKU=GlobalStandard AUDIT_MODEL_CAPACITY=20
#   DEPLOY_FAST_MODEL=true       DEPLOY_AUDIT_MODEL=true
#
# Run:
#   bash infra/deploy_foundry_models.sh
# Then copy the deployment names it prints into your .env, set
# FAST_MODEL_DEPLOYMENT_NAME + AUDIT_MODEL_DEPLOYMENT_NAME accordingly,
# and re-run infra/deploy.sh to wire the app to them.
# =====================================================================
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Per-script overrides BEFORE sourcing _load_env.sh so they win over the
# helper's defaults. The model deployment names + foundry account already
# come through .env via the helper, so we only need to enable the toggles.
export TF_VAR_deploy_fast_model="${DEPLOY_FAST_MODEL:-true}"
export TF_VAR_deploy_audit_model="${DEPLOY_AUDIT_MODEL:-true}"

source infra/tf/_load_env.sh

: "${TF_VAR_foundry_account_name:?FOUNDRY_ACCOUNT_NAME missing in .env}"
# foundry_resource_group defaults to the app RG inside the .tf locals,
# so it's optional in .env unless the Foundry account lives in a
# different RG.

# Per-model overrides — only set TF_VAR_* when the .env value differs
# from the .tf defaults, to keep `terraform plan` quiet.
[[ -n "${FAST_MODEL_NAME:-}" ]]                 && export TF_VAR_fast_model_name="$FAST_MODEL_NAME"
[[ -n "${FAST_MODEL_SKU:-}" ]]                  && export TF_VAR_fast_model_sku="$FAST_MODEL_SKU"
[[ -n "${FAST_MODEL_CAPACITY:-}" ]]             && export TF_VAR_fast_model_capacity="$FAST_MODEL_CAPACITY"
[[ -n "${AUDIT_MODEL_NAME:-}" ]]                && export TF_VAR_audit_model_name="$AUDIT_MODEL_NAME"
[[ -n "${AUDIT_MODEL_SKU:-}" ]]                 && export TF_VAR_audit_model_sku="$AUDIT_MODEL_SKU"
[[ -n "${AUDIT_MODEL_CAPACITY:-}" ]]            && export TF_VAR_audit_model_capacity="$AUDIT_MODEL_CAPACITY"

echo "==> Deploying Foundry models to ${TF_VAR_foundry_account_name}"
if [[ "$TF_VAR_deploy_fast_model" == "true" ]]; then
  echo "    FAST:  ${TF_VAR_fast_model_name:-gpt-5-mini} as '${TF_VAR_fast_model_deployment_name:-gpt-5-mini}' SKU=${TF_VAR_fast_model_sku:-GlobalStandard} cap=${TF_VAR_fast_model_capacity:-50}"
else
  echo "    FAST:  (skipped — DEPLOY_FAST_MODEL=false)"
fi
if [[ "$TF_VAR_deploy_audit_model" == "true" ]]; then
  echo "    AUDIT: ${TF_VAR_audit_model_name:-gpt-5-nano} as '${TF_VAR_audit_model_deployment_name:-gpt-5-nano}' SKU=${TF_VAR_audit_model_sku:-GlobalStandard} cap=${TF_VAR_audit_model_capacity:-20}"
else
  echo "    AUDIT: (skipped — DEPLOY_AUDIT_MODEL=false)"
fi

# `terraform apply` against the SHARED state. The Foundry deployments
# are conditional resources (count = var.deploy_*_model && var.foundry_account_name != "")
# so flipping the toggles to true causes Terraform to create them; the
# rest of the stack stays in sync because we sourced _load_env.sh.
terraform -chdir=infra/tf apply -auto-approve -input=false

echo
echo "==> Foundry deployments applied. Next steps:"
echo "    1. Add to .env at repo root if not already:"
echo "         FAST_MODEL_DEPLOYMENT_NAME=${TF_VAR_fast_model_deployment_name:-gpt-5-mini}"
echo "         AUDIT_MODEL_DEPLOYMENT_NAME=${TF_VAR_audit_model_deployment_name:-gpt-5-nano}"
echo "    2. Run infra/deploy.sh to roll the app revision."
