# Remote state on the existing faissprod storage account. Same account
# already holds the v2-seed Parquet/FAISS artefacts, so nothing new to
# manage operationally. The blob lease provides locking.
#
# To rotate the access key for the state backend, run:
#   az storage account keys renew -g msc_project -n faissprod --key key1
# then re-run `terraform init -reconfigure`.
terraform {
  required_version = ">= 1.5"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
    azapi = {
      # Used for resources the azurerm provider doesn't yet model
      # (notably Azure AI Foundry sub-resource model deployments and a
      # few ACA secret-via-KV-ref nuances).
      source  = "azure/azapi"
      version = "~> 2.0"
    }
  }

  backend "azurerm" {
    resource_group_name  = "msc_project"
    storage_account_name = "faissprod"
    container_name       = "tfstate"
    key                  = "clinical-rag-v2.tfstate"
    use_azuread_auth     = true
  }
}
