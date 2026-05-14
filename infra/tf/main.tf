# =====================================================================
# Clinical RAG v2 - Azure infrastructure (Terraform port of main.bicep)
# =====================================================================
# Idempotent: safe to re-run with the same vars.
#
# What this creates (all in one resource group):
#   - User-Assigned Managed Identity (one identity for all data-plane auth)
#   - Azure Container Registry (Basic SKU, admin disabled)
#   - Key Vault (RBAC mode, 90-day soft delete, purge protection on)
#   - Log Analytics workspace + workspace-based Application Insights
#   - Container Apps Environment (Consumption, log-analytics sink)
#   - Container App (1-4 replicas, 2 vCPU / 4 GiB, /health probes)
#
# Assumes already exists (created out-of-band):
#   - The resource group itself (data source above)
#   - The 'faissprod' storage account (Phase C, Step 1)
#   - The Foundry project + answer-model deployment
# =====================================================================

# ---- Existing resources -----------------------------------------------

data "azurerm_storage_account" "seed" {
  name                = var.existing_storage_account
  resource_group_name = data.azurerm_resource_group.rg.name
}

# ---- Managed Identity -------------------------------------------------

resource "azurerm_user_assigned_identity" "mi" {
  name                = local.mi_name
  resource_group_name = data.azurerm_resource_group.rg.name
  location            = var.location
}

# ---- Azure Container Registry -----------------------------------------

resource "azurerm_container_registry" "acr" {
  name                = local.acr_name
  resource_group_name = data.azurerm_resource_group.rg.name
  location            = var.location
  sku                 = "Basic"
  admin_enabled       = false
  # publicNetworkAccess = Enabled is the default; left implicit.
}

# MI -> ACR (pull only; this app never pushes from runtime).
# scope: the azurerm provider re-reads the imported assignment from the
# Azure RBAC API, which returns the path with 'resourcegroups' (lowercase)
# but keeps the provider type 'Microsoft.ContainerRegistry' (CamelCase).
# Resource-attribute IDs return 'resourceGroups' (CamelCase) for both.
# Since `scope` is forces_new, the casing mismatch on /resourceGroups/
# would trigger a false replace. Targeted replace() — not lower() —
# because lowercasing the whole ID would also lowercase the provider
# type and re-trigger the same drift in the other direction.
resource "azurerm_role_assignment" "acr_pull" {
  scope                = replace(azurerm_container_registry.acr.id, "resourceGroups", "resourcegroups")
  role_definition_id   = "${data.azurerm_subscription.current.id}/providers/Microsoft.Authorization/roleDefinitions/${local.role_acr_pull}"
  principal_id         = azurerm_user_assigned_identity.mi.principal_id
  principal_type       = "ServicePrincipal"
}

# ---- Key Vault --------------------------------------------------------

resource "azurerm_key_vault" "kv" {
  name                = local.kv_name
  resource_group_name = data.azurerm_resource_group.rg.name
  location            = var.location
  tenant_id           = data.azurerm_client_config.current.tenant_id

  sku_name                    = "standard"
  rbac_authorization_enabled  = true
  soft_delete_retention_days  = 90
  purge_protection_enabled    = true
  # No legacy access policies — RBAC only.
}

# MI -> KV (read secret values only). lower() — see acr_pull comment.
resource "azurerm_role_assignment" "kv_secrets_user_mi" {
  scope                = replace(azurerm_key_vault.kv.id, "resourceGroups", "resourcegroups")
  role_definition_id   = "${data.azurerm_subscription.current.id}/providers/Microsoft.Authorization/roleDefinitions/${local.role_kv_secrets_user}"
  principal_id         = azurerm_user_assigned_identity.mi.principal_id
  principal_type       = "ServicePrincipal"
}

# Deployer -> KV (read + write secret values, so the human running
# deploy.sh can rotate / verify secrets via az keyvault secret commands.)
resource "azurerm_role_assignment" "kv_secrets_officer_deployer" {
  scope              = replace(azurerm_key_vault.kv.id, "resourceGroups", "resourcegroups")
  role_definition_id = "${data.azurerm_subscription.current.id}/providers/Microsoft.Authorization/roleDefinitions/${local.role_kv_secrets_officer}"
  principal_id       = var.deployer_object_id
  principal_type     = var.deployer_principal_type
}

# Secret values themselves. Created BEFORE the Container App is provisioned
# so ACA's secret references resolve cleanly. Sensitive values never appear
# in plan output (sensitive var) or state diffs.
resource "azurerm_key_vault_secret" "api_key" {
  name         = "API-KEY"
  value        = var.api_key_value
  key_vault_id = azurerm_key_vault.kv.id
  content_type = "text/plain"

  # Without this, the deployer's role assignment may not have propagated
  # in time for the secret create call to succeed.
  depends_on = [azurerm_role_assignment.kv_secrets_officer_deployer]
}

resource "azurerm_key_vault_secret" "model_api_key" {
  name         = "MODEL-API-KEY"
  value        = var.model_api_key_value
  key_vault_id = azurerm_key_vault.kv.id
  content_type = "text/plain"
  depends_on   = [azurerm_role_assignment.kv_secrets_officer_deployer]
}

# FAST + AUDIT keys only get their own KV secrets when a separate
# value was passed. Otherwise the runtime reuses MODEL-API-KEY.
resource "azurerm_key_vault_secret" "fast_model_api_key" {
  count        = local.fast_audit_keys_distinct.fast ? 1 : 0
  name         = "FAST-MODEL-API-KEY"
  value        = var.fast_model_api_key_value
  key_vault_id = azurerm_key_vault.kv.id
  content_type = "text/plain"
  depends_on   = [azurerm_role_assignment.kv_secrets_officer_deployer]
}

resource "azurerm_key_vault_secret" "audit_model_api_key" {
  count        = local.fast_audit_keys_distinct.audit ? 1 : 0
  name         = "AUDIT-MODEL-API-KEY"
  value        = var.audit_model_api_key_value
  key_vault_id = azurerm_key_vault.kv.id
  content_type = "text/plain"
  depends_on   = [azurerm_role_assignment.kv_secrets_officer_deployer]
}

# MI -> Storage Account (data-plane read; the seed Parquet + FAISS bytes)
resource "azurerm_role_assignment" "storage_blob_reader" {
  scope                = replace(data.azurerm_storage_account.seed.id, "resourceGroups", "resourcegroups")
  role_definition_id   = "${data.azurerm_subscription.current.id}/providers/Microsoft.Authorization/roleDefinitions/${local.role_blob_data_reader}"
  principal_id         = azurerm_user_assigned_identity.mi.principal_id
  principal_type       = "ServicePrincipal"
}

# ---- Observability ----------------------------------------------------

resource "azurerm_log_analytics_workspace" "la" {
  name                = local.la_name
  resource_group_name = data.azurerm_resource_group.rg.name
  location            = var.location
  sku                 = "PerGB2018"
  retention_in_days   = 30
  # Set explicitly to match the live state (Azure default is true).
  local_authentication_enabled = true
}

resource "azurerm_application_insights" "ai" {
  name                = local.ai_name
  resource_group_name = data.azurerm_resource_group.rg.name
  location            = var.location
  application_type    = "web"
  workspace_id        = azurerm_log_analytics_workspace.la.id

  # Azure stores 0 ("ingest everything") in the live resource; the
  # azurerm provider's default for this field is 100. Set explicitly to
  # match live and avoid a no-op re-apply on every plan.
  sampling_percentage = 0
}

# ---- Container Apps Environment ---------------------------------------

resource "azurerm_container_app_environment" "aca_env" {
  name                       = local.aca_env_name
  resource_group_name        = data.azurerm_resource_group.rg.name
  location                   = var.location
  log_analytics_workspace_id = azurerm_log_analytics_workspace.la.id
}

# ---- Container App ----------------------------------------------------

# Build the env-var list in the SAME order as Azure stores it live.
# The azurerm provider does positional matching on the ACA env block,
# so any reordering shows up as a noisy false-positive plan diff.
# Order is: regular env vars 1-10, then secret refs 11-12 interleaved,
# then more regular env vars 13-18. Mirrors the live state captured
# during the Terraform import. FAST/AUDIT api-key secrets append at
# positions 19-20 only when a distinct key was provided.
locals {
  env_vars_ordered = concat(
    [
      { name = "TARGET_URL",            value = var.foundry_endpoint,             secret_name = null },
      { name = "MODEL_DEPLOYMENT_NAME", value = var.model_deployment_name,        secret_name = null },
      { name = "REASONING_EFFORT",      value = var.reasoning_effort,             secret_name = null },
      { name = "MAX_OUTPUT_TOKENS",     value = tostring(var.max_output_tokens),  secret_name = null },
      { name = "AZURE_STORAGE_ACCOUNT", value = var.existing_storage_account,     secret_name = null },
      { name = "AZURE_BLOB_CONTAINER",  value = var.blob_container,               secret_name = null },
      { name = "USE_BLOB_DATA",         value = "true",                           secret_name = null },
      { name = "USE_BLOB_INDEX",        value = "true",                           secret_name = null },
      { name = "AUDIT_LOG_DIR",         value = "/app/audit",                     secret_name = null },
      { name = "ALLOWED_ORIGINS",       value = var.allowed_origins,              secret_name = null },
      { name = "API_KEY",               value = null,                             secret_name = "api-key" },
      { name = "MODEL_API_KEY",         value = null,                             secret_name = "model-api-key" },
      { name = "FAST_MODEL_DEPLOYMENT_NAME",            value = var.fast_model_deployment_name,            secret_name = null },
      { name = "FAST_MODEL_BASE_URL",                   value = var.fast_model_base_url,                   secret_name = null },
      { name = "AUDIT_MODEL_DEPLOYMENT_NAME",           value = var.audit_model_deployment_name,           secret_name = null },
      { name = "AUDIT_MODEL_BASE_URL",                  value = var.audit_model_base_url,                  secret_name = null },
      { name = "AZURE_CLIENT_ID",                       value = azurerm_user_assigned_identity.mi.client_id,            secret_name = null },
      { name = "APPLICATIONINSIGHTS_CONNECTION_STRING", value = azurerm_application_insights.ai.connection_string,      secret_name = null },
    ],
    local.fast_audit_keys_distinct.fast ? [
      { name = "FAST_MODEL_API_KEY", value = null, secret_name = "fast-model-api-key" }
    ] : [],
    local.fast_audit_keys_distinct.audit ? [
      { name = "AUDIT_MODEL_API_KEY", value = null, secret_name = "audit-model-api-key" }
    ] : [],
  )

  # The ACA secrets list, with KV refs.
  aca_secrets = concat(
    [
      {
        name                = "api-key"
        key_vault_secret_id = azurerm_key_vault_secret.api_key.versionless_id
        identity            = azurerm_user_assigned_identity.mi.id
      },
      {
        name                = "model-api-key"
        key_vault_secret_id = azurerm_key_vault_secret.model_api_key.versionless_id
        identity            = azurerm_user_assigned_identity.mi.id
      },
    ],
    local.fast_audit_keys_distinct.fast ? [
      {
        name                = "fast-model-api-key"
        key_vault_secret_id = azurerm_key_vault_secret.fast_model_api_key[0].versionless_id
        identity            = azurerm_user_assigned_identity.mi.id
      }
    ] : [],
    local.fast_audit_keys_distinct.audit ? [
      {
        name                = "audit-model-api-key"
        key_vault_secret_id = azurerm_key_vault_secret.audit_model_api_key[0].versionless_id
        identity            = azurerm_user_assigned_identity.mi.id
      }
    ] : [],
  )
}

resource "azurerm_container_app" "api" {
  name                         = local.aca_app_name
  resource_group_name          = data.azurerm_resource_group.rg.name
  container_app_environment_id = azurerm_container_app_environment.aca_env.id
  revision_mode                = "Single"

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.mi.id]
  }

  registry {
    server   = azurerm_container_registry.acr.login_server
    identity = azurerm_user_assigned_identity.mi.id
  }

  # Secrets are referenced from KV by URL; the MI authenticates the pull.
  dynamic "secret" {
    for_each = local.aca_secrets
    content {
      name                = secret.value.name
      key_vault_secret_id = secret.value.key_vault_secret_id
      identity            = secret.value.identity
    }
  }

  ingress {
    external_enabled = true
    target_port      = 5000
    transport        = "auto"
    allow_insecure_connections = false

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    min_replicas = var.min_replicas
    max_replicas = var.max_replicas

    container {
      name   = "api"
      image  = var.container_image
      cpu    = 2.0
      memory = "4Gi"

      dynamic "env" {
        # Single ordered loop so secrets stay interleaved at positions
        # 11-12 — matches what Azure stores live, avoids positional drift.
        for_each = local.env_vars_ordered
        content {
          name        = env.value.name
          value       = env.value.value
          secret_name = env.value.secret_name
        }
      }

      # Probe defaults: Terraform's azurerm provider has its own defaults
      # (initial_delay=1, timeout=1) which differ from what Azure stores
      # when the bicep didn't set them. Setting explicitly to match the
      # current live state keeps `terraform plan` clean across runs.
      liveness_probe {
        transport               = "HTTP"
        path                    = "/health"
        port                    = 5000
        interval_seconds        = 30
        failure_count_threshold = 3
        initial_delay           = 1
        timeout                 = 1
      }

      readiness_probe {
        transport               = "HTTP"
        path                    = "/health"
        port                    = 5000
        # Cold start: FAISS download + index load + embedding model load
        # can take ~30-40s. Push the first probe past that so we don't
        # flap during initial startup.
        initial_delay           = 60
        interval_seconds        = 10
        failure_count_threshold = 3
        success_count_threshold = 3
        timeout                 = 1
      }
    }

    http_scale_rule {
      name                = "http-concurrency"
      concurrent_requests = "10"
    }
  }

  # Make sure the role assignments AND the actual secret values are
  # in place before the app tries to resolve its KV secret refs at
  # provisioning time.
  depends_on = [
    azurerm_role_assignment.acr_pull,
    azurerm_role_assignment.kv_secrets_user_mi,
    azurerm_key_vault_secret.api_key,
    azurerm_key_vault_secret.model_api_key,
    azurerm_key_vault_secret.fast_model_api_key,
    azurerm_key_vault_secret.audit_model_api_key,
  ]

  lifecycle {
    # deploy.sh runs `az containerapp update --image` after Terraform
    # applies, to swap the placeholder image for the freshly-built one.
    # Without this, every Terraform run would try to revert the image
    # back to var.container_image.
    ignore_changes = [
      template[0].container[0].image,
    ]
  }
}
