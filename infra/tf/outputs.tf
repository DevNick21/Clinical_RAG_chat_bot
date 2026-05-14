# Outputs mirror the bicep `output` declarations so deploy.sh can read
# them via `terraform output -raw <name>` (analogous to
# `az deployment group show ... --query "properties.outputs.X.value"`).

output "app_url" {
  description = "Public HTTPS URL of the Container App (stable FQDN; doesn't change between revisions)."
  # ingress[0].fqdn is the stable main FQDN. latest_revision_fqdn changes
  # every revision and made `terraform plan` show output drift after every
  # `az containerapp update --image` run from deploy.sh stage 3.
  value       = "https://${azurerm_container_app.api.ingress[0].fqdn}"
}

output "acr_login_server" {
  description = "ACR login server (e.g. msragv2acr.azurecr.io)."
  value       = azurerm_container_registry.acr.login_server
}

output "key_vault_name" {
  description = "Key Vault name."
  value       = azurerm_key_vault.kv.name
}

output "key_vault_uri" {
  description = "Key Vault URI for secret references."
  value       = azurerm_key_vault.kv.vault_uri
}

output "managed_identity_id" {
  description = "Resource ID of the user-assigned MI."
  value       = azurerm_user_assigned_identity.mi.id
}

output "managed_identity_client_id" {
  description = "Client ID of the user-assigned MI (used by DefaultAzureCredential)."
  value       = azurerm_user_assigned_identity.mi.client_id
}

output "container_app_name" {
  description = "Container App name (used by deploy.sh for the post-build image swap)."
  value       = azurerm_container_app.api.name
}

output "app_insights_connection_string" {
  description = "Application Insights connection string."
  value       = azurerm_application_insights.ai.connection_string
  sensitive   = true
}

# ---- Foundry ---- (only set when those deployments are managed here) ---

output "fast_model_deployment_name" {
  description = "FAST model deployment name (or 'skipped')."
  value       = length(azurerm_cognitive_deployment.fast_model) > 0 ? azurerm_cognitive_deployment.fast_model[0].name : "skipped"
}

output "audit_model_deployment_name" {
  description = "AUDIT model deployment name (or 'skipped')."
  value       = length(azurerm_cognitive_deployment.audit_model) > 0 ? azurerm_cognitive_deployment.audit_model[0].name : "skipped"
}

# ---- SWA ---------------------------------------------------------------

output "swa_name" {
  description = "Static Web App resource name (or 'skipped' when not managed)."
  value       = length(azurerm_static_web_app.frontend) > 0 ? azurerm_static_web_app.frontend[0].name : "skipped"
}

output "swa_hostname" {
  description = "SWA default hostname (used by deploy_frontend.sh + ALLOWED_ORIGINS)."
  value       = length(azurerm_static_web_app.frontend) > 0 ? azurerm_static_web_app.frontend[0].default_host_name : ""
}

output "swa_url" {
  description = "Public HTTPS URL of the SWA."
  value       = length(azurerm_static_web_app.frontend) > 0 ? "https://${azurerm_static_web_app.frontend[0].default_host_name}" : ""
}
