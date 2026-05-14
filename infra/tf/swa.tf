# =====================================================================
# Static Web App — Clinical RAG frontend (port of static_web_app.bicep)
# =====================================================================
# Pinned to westeurope because Free-tier SWA isn't available in uksouth
# (the app infra region). The latency from a westeurope CDN edge to the
# user is what matters, not SWA->ACA region affinity (page is fetched
# once; API calls go direct to ACA).
#
# We push the build/ via the SWA CLI from deploy_frontend.sh, not via a
# GitHub Actions integration. Leaving repository_url empty + skipping
# workflow generation keeps this in "deploy via CLI" mode permanently.
# =====================================================================

resource "azurerm_static_web_app" "frontend" {
  count = var.deploy_swa ? 1 : 0

  name                = local.swa_name
  resource_group_name = data.azurerm_resource_group.rg.name
  location            = var.swa_location

  sku_tier = var.swa_sku
  sku_size = var.swa_sku
}
