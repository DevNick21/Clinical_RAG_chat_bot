# Mirror of the Bicep parameters, less the ones we hard-code in locals.tf.
# Keep descriptions in sync with main.bicep so the two files stay
# comparable side-by-side until the .bicep are deleted.

variable "resource_group_name" {
  description = "Existing resource group hosting the v2 stack."
  type        = string
  default     = "msc_project"
}

variable "base_name" {
  description = "Short base name used as prefix for all resources. Lowercase, alphanumeric + hyphens, 3-20 chars."
  type        = string
  default     = "msc-rag-v2"
  validation {
    condition     = length(var.base_name) >= 3 && length(var.base_name) <= 20
    error_message = "base_name must be 3-20 chars."
  }
}

variable "location" {
  description = "Azure region. Default matches existing infra (Foundry, storage account)."
  type        = string
  default     = "uksouth"
}

variable "existing_storage_account" {
  description = "Name of the existing storage account holding the v2-seed Blob container."
  type        = string
  default     = "faissprod"
}

variable "blob_container" {
  description = "Blob container name under the existing storage account."
  type        = string
  default     = "v2-seed"
}

variable "foundry_endpoint" {
  description = "Foundry OpenAI-compatible base URL, e.g. https://<resource>.services.ai.azure.com/openai/v1"
  type        = string
}

variable "model_deployment_name" {
  description = "Exact deployment name of the answer LLM (reasoning model, e.g. gpt-5-nano)."
  type        = string
}

variable "reasoning_effort" {
  description = "Reasoning effort for the answer model: minimal | low | medium | high."
  type        = string
  default     = "low"
  validation {
    condition     = contains(["minimal", "low", "medium", "high"], var.reasoning_effort)
    error_message = "reasoning_effort must be minimal, low, medium, or high."
  }
}

variable "max_output_tokens" {
  description = "Token budget covering reasoning + visible output combined. 16384 is the safe default for medium effort."
  type        = number
  default     = 16384
  validation {
    condition     = var.max_output_tokens >= 2048 && var.max_output_tokens <= 100000
    error_message = "max_output_tokens must be 2048-100000."
  }
}

variable "fast_model_deployment_name" {
  description = "Foundry deployment name of the FAST non-reasoning model. Empty disables fast routing."
  type        = string
  default     = ""
}

variable "fast_model_base_url" {
  description = "Optional override of the Foundry base URL for the FAST model. Empty = reuse foundry_endpoint."
  type        = string
  default     = ""
}

variable "fast_model_api_key_value" {
  description = "Optional override of the Foundry deployment key for the FAST model. Empty = reuse model_api_key_value."
  type        = string
  default     = ""
  sensitive   = true
}

variable "audit_model_deployment_name" {
  description = "Foundry deployment name of the AUDIT model (capable, non-reasoning). Empty disables LLM judge."
  type        = string
  default     = ""
}

variable "audit_model_base_url" {
  description = "Optional override of the Foundry base URL for the AUDIT model. Empty = reuse foundry_endpoint."
  type        = string
  default     = ""
}

variable "audit_model_api_key_value" {
  description = "Optional override of the Foundry deployment key for the AUDIT model. Empty = reuse model_api_key_value."
  type        = string
  default     = ""
  sensitive   = true
}

variable "allowed_origins" {
  description = "Comma-separated list of CORS origins allowed by the API."
  type        = string
  default     = ""
}

variable "container_image" {
  description = "Container image reference. On first deploy this can be the placeholder; deploy.sh updates after build."
  type        = string
  default     = "mcr.microsoft.com/azuredocs/aci-helloworld:latest"
}

variable "min_replicas" {
  description = "Minimum replicas. 1 keeps the API hot."
  type        = number
  default     = 1
  validation {
    condition     = var.min_replicas >= 0 && var.min_replicas <= 10
    error_message = "min_replicas must be 0-10."
  }
}

variable "max_replicas" {
  description = "Maximum replicas."
  type        = number
  default     = 4
  validation {
    condition     = var.max_replicas >= 1 && var.max_replicas <= 30
    error_message = "max_replicas must be 1-30."
  }
}

variable "api_key_value" {
  description = "Bearer token used for /api/* auth. Stored as a Key Vault secret."
  type        = string
  sensitive   = true
}

variable "model_api_key_value" {
  description = "Foundry deployment key. Stored as a Key Vault secret."
  type        = string
  sensitive   = true
}

variable "deployer_object_id" {
  description = "Object ID of the principal running this deploy. Granted Key Vault Secrets Officer."
  type        = string
}

variable "deployer_principal_type" {
  description = "Type of the deployer principal. User for interactive az login, ServicePrincipal for CI."
  type        = string
  default     = "User"
  validation {
    condition     = contains(["User", "ServicePrincipal"], var.deployer_principal_type)
    error_message = "deployer_principal_type must be User or ServicePrincipal."
  }
}

# ---- Foundry model deployments (was foundry_deployments.bicep) ----------

variable "foundry_account_name" {
  description = "Name of the Azure AI Foundry account hosting the model deployments."
  type        = string
  default     = ""
}

variable "foundry_resource_group" {
  description = "Resource group of the Foundry account (may differ from app RG). Empty = same as resource_group_name."
  type        = string
  default     = ""
}

variable "deploy_fast_model" {
  description = "If true, manage the FAST model deployment from Terraform. Set false when the deployment already exists with capacity/SKU you want to keep manually."
  type        = bool
  default     = true
}

variable "deploy_audit_model" {
  description = "If true, manage the AUDIT model deployment from Terraform."
  type        = bool
  default     = true
}

variable "fast_model_name" {
  description = "Foundry model name for the FAST deployment (e.g. gpt-5-mini)."
  type        = string
  default     = "gpt-5-mini"
}

variable "fast_model_sku" {
  description = "Sku name for the FAST model deployment."
  type        = string
  default     = "GlobalStandard"
}

variable "fast_model_capacity" {
  description = "Capacity (TPM in thousands) for the FAST model deployment."
  type        = number
  default     = 50
}

variable "audit_model_name" {
  description = "Foundry model name for the AUDIT deployment (e.g. gpt-5-nano)."
  type        = string
  default     = "gpt-5-nano"
}

variable "audit_model_sku" {
  description = "Sku name for the AUDIT model deployment."
  type        = string
  default     = "GlobalStandard"
}

variable "audit_model_capacity" {
  description = "Capacity (TPM in thousands) for the AUDIT model deployment."
  type        = number
  default     = 20
}

# ---- Static Web App (was static_web_app.bicep) --------------------------

variable "swa_location" {
  description = "Region for the Static Web App. Free tier is not available in uksouth."
  type        = string
  default     = "westeurope"
}

variable "swa_sku" {
  description = "Static Web App SKU."
  type        = string
  default     = "Free"
}

variable "deploy_swa" {
  description = "If true, manage the Static Web App from Terraform. deploy.sh leaves this false; deploy_frontend.sh sets it true so SWA changes happen only in that flow."
  type        = bool
  default     = false
}
