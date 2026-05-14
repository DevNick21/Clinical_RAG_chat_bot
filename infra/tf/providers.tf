# Provider configuration. Subscription is read from the environment via
# `az login` (ARM_USE_AZURECLI / ARM_USE_AZUREAD). For CI, swap to a
# service principal via ARM_CLIENT_ID / ARM_CLIENT_SECRET / ARM_TENANT_ID
# / ARM_SUBSCRIPTION_ID env vars.
provider "azurerm" {
  features {
    key_vault {
      # Default is true. We explicitly DISABLE soft-delete-on-destroy
      # to avoid the 90-day name reservation we already got bitten by
      # on the bicep side. (Re-enable for true production.)
      purge_soft_delete_on_destroy          = false
      purge_soft_deleted_secrets_on_destroy = false
      recover_soft_deleted_key_vaults       = true
    }
  }
}

provider "azapi" {}

# Resource group already exists; everything in this stack lives inside it.
data "azurerm_resource_group" "rg" {
  name = var.resource_group_name
}

data "azurerm_subscription" "current" {}

data "azurerm_client_config" "current" {}
