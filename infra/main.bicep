// =====================================================================
// Clinical RAG v2 - Azure infrastructure
// =====================================================================
// Single-file Bicep so the IaC is easy to read end-to-end and compares
// cleanly with the CloudFormation equivalent we'll write for v3 (AWS).
// Idempotent: safe to re-run with the same parameters.
//
// What this creates (all in one resource group):
//   - User-Assigned Managed Identity (one identity for all data-plane auth)
//   - Azure Container Registry (Basic SKU, admin disabled)
//   - Key Vault (RBAC mode, 90-day soft delete, purge protection on)
//   - Log Analytics workspace + workspace-based Application Insights
//   - Container Apps Environment (Consumption, log-analytics sink)
//   - Container App (1-4 replicas, 2 vCPU / 4 GiB, /health probes)
//
// What this assumes already exists (created out-of-band):
//   - The resource group itself
//   - The 'faissprod' storage account (Phase C, Step 1)
//   - The Foundry project + gpt-5-nano deployment (.env)
//
// Role assignments granted to the MI:
//   - AcrPull on the new ACR
//   - Key Vault Secrets User on the new Key Vault
//   - Storage Blob Data Reader on the existing storage account
//
// Image bootstrap problem: ACA requires a pullable image to come up,
// but our image is built into this same ACR by the next pipeline step.
// We deploy ACA with a placeholder hello-world image; deploy.sh runs
// `az containerapp update --image` once the real image is in ACR.
// =====================================================================

// ---- Parameters -------------------------------------------------------

@description('Short base name used as prefix for all resources. Lowercase, alphanumeric + hyphens, 3-20 chars.')
@minLength(3)
@maxLength(20)
param baseName string = 'msc-rag-v2'

@description('Azure region. Default matches existing infra (Foundry, storage account).')
param location string = 'uksouth'

@description('Name of the existing storage account holding the v2-seed Blob container.')
param existingStorageAccount string = 'faissprod'

@description('Blob container name under the existing storage account.')
param blobContainer string = 'v2-seed'

@description('Foundry OpenAI-compatible base URL, e.g. https://<resource>.services.ai.azure.com/openai/v1')
param foundryEndpoint string

@description('Exact deployment name of the LLM (e.g. gpt-5-nano).')
param modelDeploymentName string

@description('Reasoning effort for gpt-5-nano: minimal | low | medium | high.')
@allowed([
  'minimal'
  'low'
  'medium'
  'high'
])
param reasoningEffort string = 'low'

@description('Token budget covering reasoning + visible output combined. medium effort can spend 5k-10k just on reasoning; 16384 is the safe default. Bump to 32768+ for high effort or complex multi-step queries.')
@minValue(2048)
@maxValue(100000)
param maxOutputTokens int = 16384

@description('Comma-separated list of CORS origins allowed by the API. Empty = no cross-origin browser access.')
param allowedOrigins string = ''

@description('Container image reference. On first deploy this can be the placeholder; deploy.sh updates to the real image once built.')
param containerImage string = 'mcr.microsoft.com/azuredocs/aci-helloworld:latest'

@description('Minimum replicas. 1 keeps the API hot (no cold start in user path) at ~£20-30/month for the always-on instance.')
@minValue(0)
@maxValue(10)
param minReplicas int = 1

@description('Maximum replicas. ACA scales between min and max based on HTTP load.')
@minValue(1)
@maxValue(30)
param maxReplicas int = 4

@description('Bearer token used for /api/* auth. Generate via secrets.token_urlsafe(32). Stored as a Key Vault secret; never logged.')
@secure()
param apiKeyValue string

@description('Foundry deployment key. Stored as a Key Vault secret; never logged.')
@secure()
param modelApiKeyValue string

@description('Object ID (UUID) of the principal running this deploy. Granted Key Vault Secrets Officer on the new vault so the deployer can read/rotate secrets via az keyvault secret commands. deploy.sh fills this in via `az ad signed-in-user show --query id -o tsv`.')
param deployerObjectId string

@description('Type of the deployer principal. "User" for interactive az login, "ServicePrincipal" for CI / SP-based deploys.')
@allowed([
  'User'
  'ServicePrincipal'
])
param deployerPrincipalType string = 'User'

// ---- Derived names ----------------------------------------------------
// ACR names must be alphanumeric only, 5-50 chars, globally unique.
var acrName = toLower(replace('${baseName}acr', '-', ''))
var kvName = '${baseName}-kv'
var laName = '${baseName}-logs'
var aiName = '${baseName}-ai'
var miName = '${baseName}-mi'
var acaEnvName = '${baseName}-env'
var acaAppName = '${baseName}-api'

// Built-in Azure role definition GUIDs (stable across all subscriptions)
var roleAcrPull = '7f951dda-4ed3-4680-a7ca-43fe172d538d'
var roleKvSecretsUser = '4633458b-17de-408a-b874-0445c86b69e6'
var roleKvSecretsOfficer = 'b86a8fe4-44ce-4948-aee5-eccb2c155cd7'
var roleBlobDataReader = '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1'

// ---- Existing resources -----------------------------------------------

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' existing = {
  name: existingStorageAccount
}

// ---- Managed Identity -------------------------------------------------

resource mi 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: miName
  location: location
}

// ---- Azure Container Registry -----------------------------------------

resource acr 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' = {
  name: acrName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: false
    publicNetworkAccess: 'Enabled'
  }
}

// MI -> ACR (pull only; this app never pushes from runtime)
resource acrPullAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: acr
  name: guid(acr.id, mi.id, roleAcrPull)
  properties: {
    principalId: mi.properties.principalId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleAcrPull)
    principalType: 'ServicePrincipal'
  }
}

// ---- Key Vault --------------------------------------------------------

resource kv 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: kvName
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 90
    enablePurgeProtection: true
    accessPolicies: []
  }
}

// MI -> KV (read secret values only)
resource kvSecretsAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: kv
  name: guid(kv.id, mi.id, roleKvSecretsUser)
  properties: {
    principalId: mi.properties.principalId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleKvSecretsUser)
    principalType: 'ServicePrincipal'
  }
}

// Deployer -> KV (read + write secret values, so the human running
// deploy.sh can rotate / verify secrets via `az keyvault secret`
// commands. Without this, Bicep can still write secrets via the
// control plane, but no-one can read them back.)
resource kvSecretsOfficerForDeployer 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: kv
  name: guid(kv.id, deployerObjectId, 'KvSecretsOfficer')
  properties: {
    principalId: deployerObjectId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleKvSecretsOfficer)
    principalType: deployerPrincipalType
  }
}

// Secret values themselves. Declared inside Bicep (with @secure() params)
// so they exist BEFORE the Container App is provisioned. Putting secrets
// in CLI calls after Bicep failed because ACA evaluated its
// secret-references at create time and the names didn't exist yet.
// @secure() values are scrubbed from deployment history by Azure.
resource apiKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: kv
  name: 'API-KEY'
  properties: {
    value: apiKeyValue
    contentType: 'text/plain'
  }
}

resource modelApiKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: kv
  name: 'MODEL-API-KEY'
  properties: {
    value: modelApiKeyValue
    contentType: 'text/plain'
  }
}

// MI -> Storage Account (data-plane read; the seed Parquet + FAISS bytes)
resource storageReadAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: storage
  name: guid(storage.id, mi.id, roleBlobDataReader)
  properties: {
    principalId: mi.properties.principalId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleBlobDataReader)
    principalType: 'ServicePrincipal'
  }
}

// ---- Observability ----------------------------------------------------

resource la 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: laName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

resource ai 'Microsoft.Insights/components@2020-02-02' = {
  name: aiName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: la.id
  }
}

// ---- Container Apps Environment ---------------------------------------

resource acaEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: acaEnvName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: la.properties.customerId
        sharedKey: la.listKeys().primarySharedKey
      }
    }
  }
}

// ---- Container App ----------------------------------------------------

resource acaApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: acaAppName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${mi.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: acaEnv.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 5000
        transport: 'auto'
        allowInsecure: false
      }
      registries: [
        {
          server: acr.properties.loginServer
          identity: mi.id
        }
      ]
      // Secrets are referenced from Key Vault by URL; the MI is what
      // authenticates the pull. The KV secret names use hyphens
      // because KV doesn't allow underscores.
      secrets: [
        {
          name: 'api-key'
          keyVaultUrl: '${kv.properties.vaultUri}secrets/API-KEY'
          identity: mi.id
        }
        {
          name: 'model-api-key'
          keyVaultUrl: '${kv.properties.vaultUri}secrets/MODEL-API-KEY'
          identity: mi.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: containerImage
          resources: {
            cpu: 2
            memory: '4Gi'
          }
          env: [
            { name: 'TARGET_URL', value: foundryEndpoint }
            { name: 'MODEL_DEPLOYMENT_NAME', value: modelDeploymentName }
            { name: 'REASONING_EFFORT', value: reasoningEffort }
            { name: 'MAX_OUTPUT_TOKENS', value: string(maxOutputTokens) }
            { name: 'AZURE_STORAGE_ACCOUNT', value: existingStorageAccount }
            { name: 'AZURE_BLOB_CONTAINER', value: blobContainer }
            { name: 'USE_BLOB_DATA', value: 'true' }
            { name: 'USE_BLOB_INDEX', value: 'true' }
            { name: 'AUDIT_LOG_DIR', value: '/app/audit' }
            { name: 'ALLOWED_ORIGINS', value: allowedOrigins }
            { name: 'API_KEY', secretRef: 'api-key' }
            { name: 'MODEL_API_KEY', secretRef: 'model-api-key' }
            // DefaultAzureCredential needs to know which user-assigned
            // MI to use when several are present.
            { name: 'AZURE_CLIENT_ID', value: mi.properties.clientId }
            // Wired but not consumed yet — adding OTel instrumentation
            // is a follow-up commit.
            { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: ai.properties.ConnectionString }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: 5000
              }
              periodSeconds: 30
              failureThreshold: 3
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/health'
                port: 5000
              }
              // Cold start: FAISS download + index load + embedding model load
              // can take ~30-40s. Push the first probe past that so we don't
              // flap during initial startup.
              initialDelaySeconds: 60
              periodSeconds: 10
              failureThreshold: 3
            }
          ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        rules: [
          {
            name: 'http-concurrency'
            http: {
              metadata: {
                concurrentRequests: '10'
              }
            }
          }
        ]
      }
    }
  }
  dependsOn: [
    // Make sure the role assignments AND the actual secret values are
    // in place before the app tries to resolve its KV secret refs at
    // provisioning time.
    acrPullAssignment
    kvSecretsAssignment
    apiKeySecret
    modelApiKeySecret
  ]
}

// ---- Outputs ----------------------------------------------------------

output appUrl string = 'https://${acaApp.properties.configuration.ingress.fqdn}'
output acrLoginServer string = acr.properties.loginServer
output keyVaultName string = kv.name
output keyVaultUri string = kv.properties.vaultUri
output managedIdentityId string = mi.id
output managedIdentityClientId string = mi.properties.clientId
output containerAppName string = acaApp.name
output appInsightsConnectionString string = ai.properties.ConnectionString
