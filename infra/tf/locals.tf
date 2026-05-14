# Derived names + role definition GUIDs.
# Kept in one place so renaming base_name cascades cleanly.
locals {
  # ACR names: alphanumeric only, 5-50 chars, globally unique. Mirrors
  # the bicep `toLower(replace('${baseName}acr', '-', ''))` expression.
  acr_name      = lower(replace("${var.base_name}acr", "-", ""))
  kv_name       = "${var.base_name}-kv"
  la_name       = "${var.base_name}-logs"
  ai_name       = "${var.base_name}-ai"
  mi_name       = "${var.base_name}-mi"
  aca_env_name  = "${var.base_name}-env"
  aca_app_name  = "${var.base_name}-api"
  swa_name      = "${var.base_name}-web"

  # Foundry RG defaults to the app RG when not overridden.
  foundry_rg = var.foundry_resource_group != "" ? var.foundry_resource_group : var.resource_group_name

  # Built-in Azure role definition GUIDs (stable across all subscriptions).
  role_acr_pull           = "7f951dda-4ed3-4680-a7ca-43fe172d538d"
  role_kv_secrets_user    = "4633458b-17de-408a-b874-0445c86b69e6"
  role_kv_secrets_officer = "b86a8fe4-44ce-4948-aee5-eccb2c155cd7"
  role_blob_data_reader   = "2a2b9908-6ea1-4ae2-8e65-a410df84e7d1"

  # Conditional FAST/AUDIT secret-ref names.
  fast_audit_keys_distinct = {
    fast  = var.fast_model_api_key_value != ""
    audit = var.audit_model_api_key_value != ""
  }
}
