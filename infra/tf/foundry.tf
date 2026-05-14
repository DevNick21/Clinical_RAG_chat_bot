# =====================================================================
# Foundry model deployments — fast + audit (port of foundry_deployments.bicep)
# =====================================================================
# Decoupled from the app stack because Foundry capacity / SKUs are
# quota-bound and you may want to deploy them out-of-band via the portal.
#
# What this assumes already exists:
#   - The Azure AI Foundry / Azure OpenAI account (var.foundry_account_name)
#   - Subscription quota for both model SKUs
#
# What this creates (conditional on deploy_fast_model / deploy_audit_model):
#   - <foundry_account_name>/<fast_model_deployment_name>
#   - <foundry_account_name>/<audit_model_deployment_name>
# =====================================================================

# Existing Foundry account.
data "azurerm_cognitive_account" "foundry" {
  count               = var.foundry_account_name != "" ? 1 : 0
  name                = var.foundry_account_name
  resource_group_name = local.foundry_rg
}

# FAST model — small non-reasoning, used for entity extraction + rephrase
# on the TTFP critical path.
resource "azurerm_cognitive_deployment" "fast_model" {
  count                = var.foundry_account_name != "" && var.deploy_fast_model ? 1 : 0
  name                 = var.fast_model_deployment_name != "" ? var.fast_model_deployment_name : "gpt-5-mini"
  cognitive_account_id = data.azurerm_cognitive_account.foundry[0].id

  model {
    format = "OpenAI"
    name   = var.fast_model_name
    # version omitted -> Foundry picks the default stable version.
  }

  sku {
    name     = var.fast_model_sku
    capacity = var.fast_model_capacity
  }

  # OnceNewDefaultVersionAvailable: auto-bump on Foundry's release
  # cadence without surprise breaking changes.
  version_upgrade_option = "OnceNewDefaultVersionAvailable"
  rai_policy_name        = "Microsoft.DefaultV2"
}

# AUDIT model — capable non-reasoning, used for post-stream LLM-as-judge
# faithfulness checks. Lower capacity because audit calls are post-stream
# and rare.
resource "azurerm_cognitive_deployment" "audit_model" {
  count                = var.foundry_account_name != "" && var.deploy_audit_model ? 1 : 0
  name                 = var.audit_model_deployment_name != "" ? var.audit_model_deployment_name : "gpt-5-nano"
  cognitive_account_id = data.azurerm_cognitive_account.foundry[0].id

  model {
    format = "OpenAI"
    name   = var.audit_model_name
  }

  sku {
    name     = var.audit_model_sku
    capacity = var.audit_model_capacity
  }

  version_upgrade_option = "OnceNewDefaultVersionAvailable"
  rai_policy_name        = "Microsoft.DefaultV2"

  # Serial creation: Foundry rejects concurrent deployments under the
  # same account when capacity is tight. Force ordering so quota errors
  # are deterministic instead of racy.
  depends_on = [azurerm_cognitive_deployment.fast_model]
}
