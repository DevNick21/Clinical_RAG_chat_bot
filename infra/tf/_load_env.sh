#!/usr/bin/env bash
# =====================================================================
# Source this from the three deploy*.sh scripts to populate TF_VAR_*
# from .env, so every `terraform apply` in this repo gets the same
# canonical inputs without each script duplicating the load logic.
#
# Why this is necessary: terraform apply by default plans changes for
# the whole state, not just the resources a given script "owns". If
# deploy_frontend.sh runs `terraform apply -var deploy_swa=true` but
# doesn't pass the same secret + foundry vars deploy.sh used, Terraform
# would see those as drift and want to "fix" them — which would
# blank-out KV secrets or revert env vars. Sourcing this file ensures
# every script speaks the same dialect.
#
# Required at repo root: .env (gitignored). Caller must:
#   set -euo pipefail
#   cd "$(git rev-parse --show-toplevel)"
#   source infra/tf/_load_env.sh
#
# Optional caller overrides (set BEFORE sourcing):
#   TF_VAR_allowed_origins
#   TF_VAR_reasoning_effort
#   TF_VAR_deploy_swa  (default: false)
#   TF_VAR_deploy_fast_model / TF_VAR_deploy_audit_model
# =====================================================================

if [[ ! -f .env ]]; then
  echo "ERROR: .env not found at repo root (TF_VAR_api_key_value etc must come from somewhere)." >&2
  return 2 2>/dev/null || exit 2
fi

# Mirror .env -> TF_VAR_*. Uses python-dotenv (same parser the rest
# of the codebase uses) so .env behaves identically regardless of
# who's reading it.
eval "$(python <<'PY'
from dotenv import dotenv_values
import shlex

# .env name -> TF_VAR_* name. Only mirror what the .tf actually consumes.
mapping = {
    "API_KEY":                       "TF_VAR_api_key_value",
    "MODEL_API_KEY":                 "TF_VAR_model_api_key_value",
    "TARGET_URL":                    "TF_VAR_foundry_endpoint",
    "MODEL_DEPLOYMENT_NAME":         "TF_VAR_model_deployment_name",
    "REASONING_EFFORT":              "TF_VAR_reasoning_effort",
    "ALLOWED_ORIGINS":               "TF_VAR_allowed_origins",
    "FAST_MODEL_DEPLOYMENT_NAME":    "TF_VAR_fast_model_deployment_name",
    "FAST_MODEL_BASE_URL":           "TF_VAR_fast_model_base_url",
    "FAST_MODEL_API_KEY":            "TF_VAR_fast_model_api_key_value",
    "AUDIT_MODEL_DEPLOYMENT_NAME":   "TF_VAR_audit_model_deployment_name",
    "AUDIT_MODEL_BASE_URL":          "TF_VAR_audit_model_base_url",
    "AUDIT_MODEL_API_KEY":           "TF_VAR_audit_model_api_key_value",
    # Foundry account vars only used by deploy_foundry_models.sh.
    "FOUNDRY_ACCOUNT_NAME":          "TF_VAR_foundry_account_name",
    "FOUNDRY_RESOURCE_GROUP":        "TF_VAR_foundry_resource_group",
}

import os
for src, dst in mapping.items():
    val = dotenv_values(".env").get(src)
    # Only emit if .env has a value AND the env doesn't already
    # have a higher-priority override (set by the caller).
    if val is None:
        continue
    if os.getenv(dst):
        continue
    print(f"export {dst}={shlex.quote(val)}")
PY
)"

# Required vars — bail loudly if missing.
: "${TF_VAR_api_key_value:?API_KEY missing in .env (generate: python -c 'import secrets;print(secrets.token_urlsafe(32))')}"
: "${TF_VAR_model_api_key_value:?MODEL_API_KEY missing in .env}"
: "${TF_VAR_foundry_endpoint:?TARGET_URL missing in .env}"
: "${TF_VAR_model_deployment_name:?MODEL_DEPLOYMENT_NAME missing in .env}"

# Deployer object id — needed by the kv_secrets_officer_deployer role
# assignment. Always resolved live via az so it tracks whoever's
# actually running the script.
TF_VAR_deployer_object_id=$(az ad signed-in-user show --query id -o tsv 2>/dev/null || true)
if [[ -z "$TF_VAR_deployer_object_id" ]]; then
  echo "ERROR: could not resolve deployer object id via az ad signed-in-user." >&2
  echo "       (CI? Set TF_VAR_deployer_object_id explicitly + TF_VAR_deployer_principal_type=ServicePrincipal.)" >&2
  return 3 2>/dev/null || exit 3
fi
export TF_VAR_deployer_object_id

# Sensible defaults for the optional toggles, only when the caller
# didn't already export them.
: "${TF_VAR_deploy_swa:=false}"
: "${TF_VAR_deploy_fast_model:=false}"
: "${TF_VAR_deploy_audit_model:=false}"
export TF_VAR_deploy_swa TF_VAR_deploy_fast_model TF_VAR_deploy_audit_model
