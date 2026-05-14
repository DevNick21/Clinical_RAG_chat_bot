#!/usr/bin/env bash
# =====================================================================
# One-shot importer: pulls the existing live msc-rag-v2 stack into the
# Terraform state created by `terraform init`. Run from repo root:
#
#   bash infra/tf/import_live_state.sh
#
# Idempotent in the sense that re-running on already-imported resources
# fails the import command but doesn't corrupt state — Terraform refuses
# to import over an existing state entry. To re-import, first run
# `terraform state rm <address>` then re-run this script.
#
# After running, immediately do:
#   terraform -chdir=infra/tf plan
# and iterate on the .tf until plan shows zero changes (drift = the
# .tf file isn't faithful to the live infra).
# =====================================================================
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if [[ ! -f .env ]]; then
  echo "ERROR: .env not found at repo root (need API_KEY + MODEL_API_KEY for plan to be clean)." >&2
  exit 2
fi

# Pass the live secret values via TF_VAR_* env vars so terraform plan
# (run after this script) doesn't try to "rotate" them to empty strings.
eval "$(python <<'PY'
from dotenv import dotenv_values
import shlex
env = dotenv_values(".env")
# Map .env names to TF_VAR_* names. terraform plan picks these up
# automatically.
tf_var_map = {
    "API_KEY":          "TF_VAR_api_key_value",
    "MODEL_API_KEY":    "TF_VAR_model_api_key_value",
    "TARGET_URL":       "TF_VAR_foundry_endpoint",
    "MODEL_DEPLOYMENT_NAME": "TF_VAR_model_deployment_name",
}
for src, dst in tf_var_map.items():
    if env.get(src):
        print(f"export {dst}={shlex.quote(env[src])}")
PY
)"

# Also need the deployer object id.
DEPLOYER_OID=$(az ad signed-in-user show --query id -o tsv)
export TF_VAR_deployer_object_id="$DEPLOYER_OID"

SUB="7041a4d0-9b90-4232-9325-d99881c43114"
RG="msc_project"

# ---- Resource IDs (resolved live, so name changes don't break this) ---
MI_ID="/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.ManagedIdentity/userAssignedIdentities/msc-rag-v2-mi"
ACR_ID="/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.ContainerRegistry/registries/mscragv2acr"
KV_ID="/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.KeyVault/vaults/msc-rag-v2-kv"
LA_ID="/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.OperationalInsights/workspaces/msc-rag-v2-logs"
AI_ID="/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.Insights/components/msc-rag-v2-ai"
ACA_ENV_ID="/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.App/managedEnvironments/msc-rag-v2-env"
ACA_APP_ID="/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.App/containerApps/msc-rag-v2-api"

# Role assignment IDs (queried via az rest above; hardcoded so re-runs
# don't re-query). If you ever recreate any of these role assignments,
# update the GUIDs here.
ACR_PULL_ASSIGN_ID="/subscriptions/$SUB/resourcegroups/$RG/providers/Microsoft.ContainerRegistry/registries/mscragv2acr/providers/Microsoft.Authorization/roleAssignments/d4524a15-642f-5380-96e1-6f8e3d8cafe4"
KV_SECRETS_USER_ASSIGN_ID="/subscriptions/$SUB/resourcegroups/$RG/providers/Microsoft.KeyVault/vaults/msc-rag-v2-kv/providers/Microsoft.Authorization/roleAssignments/ab31cea1-a54c-5b9b-86ca-bfce3aa2b475"
KV_SECRETS_OFFICER_ASSIGN_ID="/subscriptions/$SUB/resourcegroups/$RG/providers/Microsoft.KeyVault/vaults/msc-rag-v2-kv/providers/Microsoft.Authorization/roleAssignments/241bd591-98d2-5c44-8ee9-8a00f3604ea0"
STORAGE_READER_ASSIGN_ID="/subscriptions/$SUB/resourcegroups/$RG/providers/Microsoft.Storage/storageAccounts/faissprod/providers/Microsoft.Authorization/roleAssignments/2ecf4e7c-e682-54e5-9a37-3ddb84fb1d37"

# KV secrets (use the latest version IDs).
API_KEY_SECRET_ID="https://msc-rag-v2-kv.vault.azure.net/secrets/API-KEY"
MODEL_API_KEY_SECRET_ID="https://msc-rag-v2-kv.vault.azure.net/secrets/MODEL-API-KEY"

# ---- Imports (one per address) ---------------------------------------
# Each line is: terraform import <addr> <azure-id>
# Wrapped in a function so a single failure (e.g. already-imported)
# logs and continues, doesn't kill the whole batch.

tf() {
  terraform -chdir=infra/tf "$@"
}

import_or_skip() {
  local addr="$1"
  local id="$2"
  if tf state list 2>/dev/null | grep -qx "$addr"; then
    echo "  [skip] $addr (already in state)"
    return 0
  fi
  echo "  [import] $addr"
  tf import "$addr" "$id"
}

echo "==> Importing 13 resources into Terraform state..."
import_or_skip 'azurerm_user_assigned_identity.mi'                "$MI_ID"
import_or_skip 'azurerm_container_registry.acr'                   "$ACR_ID"
import_or_skip 'azurerm_role_assignment.acr_pull'                 "$ACR_PULL_ASSIGN_ID"
import_or_skip 'azurerm_key_vault.kv'                             "$KV_ID"
import_or_skip 'azurerm_role_assignment.kv_secrets_user_mi'       "$KV_SECRETS_USER_ASSIGN_ID"
import_or_skip 'azurerm_role_assignment.kv_secrets_officer_deployer' "$KV_SECRETS_OFFICER_ASSIGN_ID"
import_or_skip 'azurerm_role_assignment.storage_blob_reader'      "$STORAGE_READER_ASSIGN_ID"
import_or_skip 'azurerm_key_vault_secret.api_key'                 "$API_KEY_SECRET_ID"
import_or_skip 'azurerm_key_vault_secret.model_api_key'           "$MODEL_API_KEY_SECRET_ID"
import_or_skip 'azurerm_log_analytics_workspace.la'               "$LA_ID"
import_or_skip 'azurerm_application_insights.ai'                  "$AI_ID"
import_or_skip 'azurerm_container_app_environment.aca_env'        "$ACA_ENV_ID"
import_or_skip 'azurerm_container_app.api'                        "$ACA_APP_ID"

echo
echo "==> Imports complete. Now run:"
echo "    terraform -chdir=infra/tf plan"
echo
echo "    Iterate on .tf until plan shows 'No changes'. Common drift:"
echo "      - tags (azure adds creator tags)"
echo "      - container app secret ordering (re-sort to match cloud)"
echo "      - log analytics retention_in_days (azure may report different number)"
