# Clinical RAG v2 — Azure infrastructure

Hand-written Terraform (HCL) + a thin shell orchestrator. State lives on
the existing `faissprod` storage account (azurerm backend, container
`tfstate`, key `clinical-rag-v2.tfstate`). Originally Bicep — converted
2026-05; the Bicep files were deleted in the same commit that flipped
`deploy.sh` to call `terraform apply`.

## What gets created

In the `msc_project` resource group, region `uksouth`:

| Resource | Name | Purpose |
|---|---|---|
| User-Assigned Managed Identity | `msc-rag-v2-mi` | Single identity for ACR pull, KV secret read, Blob data read |
| Azure Container Registry | `mscragv2acr` | Private image registry; ACA pulls via MI |
| Key Vault | `msc-rag-v2-kv` | RBAC mode, soft-delete 90d, purge protection on |
| Log Analytics workspace | `msc-rag-v2-logs` | Container stdout/stderr sink |
| Application Insights | `msc-rag-v2-ai` | Workspace-based, connection string wired into the container env |
| Container Apps Environment | `msc-rag-v2-env` | Consumption plan, log-analytics linked |
| Container App | `msc-rag-v2-api` | 1–4 replicas, 2 vCPU / 4 GiB, `/health` probes |

The existing `faissprod` storage account (Phase C) is **referenced**, not
recreated. The MI gets `Storage Blob Data Reader` on it.

Conditional on opt-in flags:

| Resource | Toggle | Set by |
|---|---|---|
| FAST + AUDIT model deployments | `var.deploy_fast_model` / `var.deploy_audit_model` (false by default) | `deploy_foundry_models.sh` |
| Static Web App (`msc-rag-v2-web`, westeurope) | `var.deploy_swa` (false by default) | `deploy_frontend.sh` |

Splitting via opt-in vars — instead of separate Terraform configs — keeps
the IaC in one state file (one source-of-truth, one `terraform plan`)
while still letting the three deploy scripts manage their own resources.

## Prereqs

- `terraform` ≥ 1.5 on PATH (`winget install HashiCorp.Terraform`)
- `az` CLI installed and `az login` done
- Logged into the right subscription: `az account show`
- Docker daemon running (the script builds + pushes the image)
- `.env` at repo root populated with at minimum:

  ```dotenv
  # ---- Clinical answer model (reasoning) -----------------------------
  TARGET_URL=https://<resource>.services.ai.azure.com/openai/v1
  MODEL_API_KEY=<foundry deployment key>
  MODEL_DEPLOYMENT_NAME=gpt-5.4-nano
  REASONING_EFFORT=medium

  # ---- Auth + CORS ----------------------------------------------------
  API_KEY=<long random secret — generate: python -c "import secrets;print(secrets.token_urlsafe(32))">
  ALLOWED_ORIGINS=https://your-frontend-host

  # ---- FAST model (entity-extract + rephrase). Optional. -------------
  FAST_MODEL_DEPLOYMENT_NAME=gpt-5.4-mini

  # ---- AUDIT model (LLM-as-judge). Optional. -------------------------
  AUDIT_MODEL_DEPLOYMENT_NAME=gpt-5-nano

  # ---- Foundry account (for deploy_foundry_models.sh only) -----------
  FOUNDRY_ACCOUNT_NAME=<your-foundry-account-name>
  # FOUNDRY_RESOURCE_GROUP=  # only if Foundry lives in a different RG
  ```

## First-time setup

On a fresh checkout you need to initialise Terraform once before any
script will work — it downloads providers and connects to the azurerm
backend:

```bash
terraform -chdir=infra/tf init
```

If the state container `tfstate` doesn't exist on `faissprod` yet
(brand-new fork), create it once:

```bash
az storage container create --account-name faissprod --name tfstate --auth-mode login
```

## Deploy

The IaC lives in one shared Terraform state. Three bash orchestrators
flip the right vars and call `terraform apply`:

| Script | What it touches | When to run |
|---|---|---|
| `deploy.sh` | App stack: ACR, KV, ACA, Log Analytics, App Insights, MI + secrets + image push + revision roll | Every backend revision |
| `deploy_foundry_models.sh` | FAST + AUDIT model deployments inside an existing AI Foundry resource (`var.deploy_*_model=true`) | Once per model rotation (rare) |
| `deploy_frontend.sh` | Static Web App (`var.deploy_swa=true`) + React build + SWA push + ALLOWED_ORIGINS update | Every frontend revision |

```bash
# 1. Create the FAST + AUDIT model deployments (one-off).
bash infra/deploy_foundry_models.sh

# 2. App infra + image (idempotent, the usual workflow).
bash infra/deploy.sh

# 3. Frontend (independent of #2 — UI tweaks ship without rebuilding
#    the Python image). Run any time after #2 has succeeded once.
bash infra/deploy_frontend.sh
```

### How env values flow into Terraform

Every script sources `infra/tf/_load_env.sh`, which mirrors `.env` →
`TF_VAR_*` via python-dotenv (the same parser the rest of the codebase
uses). This matters because all three scripts share the same Terraform
state — without consistent vars, one script's `apply` could revert
another's changes (e.g. wiping a KV secret to empty). The helper bails
loudly on missing required values.

Secrets stay in `.env` only; never on the command line, never in plan
output (vars marked `sensitive`), never in state diffs.

### How the SWA + ACA CORS dance works

`deploy_frontend.sh` discovers the ACA backend FQDN via az, bakes it
into the React build as `REACT_APP_API_URL`, pushes the bundle to SWA,
then re-runs `terraform apply` with the SWA URL appended to
`var.allowed_origins`. The ALLOWED_ORIGINS update goes through IaC
rather than `az containerapp update --set-env-vars`, so the next
`deploy.sh` run doesn't see drift trying to revert the value.

`deploy.sh` also auto-discovers the SWA hostname via `az staticwebapp
show` at the top and appends it to `TF_VAR_allowed_origins` before
applying — so a backend redeploy stays in sync with whatever the
frontend deploy last set. Both scripts converge on the same state
regardless of order, as long as `deploy.sh` has run at least once
before `deploy_frontend.sh`.

### Stages of `deploy.sh`

1. **`terraform apply`** (~30s when no infra changes) — creates/updates every resource above.
2. **Build + push image** — `docker build`, then `az acr login` + `docker push`. Image tag defaults to current git short SHA (`IMAGE_TAG=...` to override).
3. **Roll ACA revision** — `az containerapp update --image` creates a new revision and shifts traffic. Done outside Terraform because the .tf has `lifecycle { ignore_changes = [template.container.image] }` to keep IaC from reverting deploy-time image swaps.

The script is idempotent; safe to re-run. Comment out stage 1 when iterating just on the image.

## How auth flows in cloud

```
┌──────────────────┐
│ Container App    │   has user-assigned MI
└─────┬────────────┘
      │ token
      ├──────────► ACR  (AcrPull role)              -> pulls clinical-rag:<sha>
      ├──────────► Key Vault (Secrets User role)    -> resolves API_KEY, MODEL_API_KEY
      ├──────────► Storage Account (Blob Data Reader) -> reads silver/, gold/, prod-index/faiss_winner/
      └──────────► Foundry (OpenAI-compat endpoint) -> uses MODEL_API_KEY (Foundry serverless doesn't support MI yet)
```

`AZURE_CLIENT_ID` env var on the container points `DefaultAzureCredential` at the right MI when there are multiple identities present.

## Working with Terraform directly

The bash scripts are the supported entry points, but you can run
`terraform` against the state directly for inspection or one-off ops:

```bash
# Plan (against current .env values)
source infra/tf/_load_env.sh
terraform -chdir=infra/tf plan

# Inspect state
terraform -chdir=infra/tf state list
terraform -chdir=infra/tf show

# Read a single output
terraform -chdir=infra/tf output -raw app_url
```

`infra/tf/import_live_state.sh` documents the original one-shot import
of the live Bicep-deployed stack into Terraform state — kept for
reference and replay (state-rebuild scenarios).

## What this deliberately does NOT include

| Feature | Why deferred |
|---|---|
| GitHub Actions CI/CD workflow | Local scripts cover everything for now. Add OIDC-based GHA when there's a team pushing to main |
| OpenTelemetry instrumentation | App Insights connection string is wired in, but `RAG_chat_pipeline` doesn't emit telemetry yet. Adding `azure-monitor-opentelemetry` is a focused follow-up |
| Custom domain / TLS cert | Default `*.azurecontainerapps.io` URL is fine for v2 |
| Redis-backed rate limiter | flask-limiter in-memory works with `--workers 1`. Multi-worker scale-out needs Redis (Azure Cache for Redis) |
| ACR geo-replication | Single-region. Multi-region would need Premium SKU |
| Foundry Managed Identity auth | Serverless deployments accept API key only at the moment; revisit when Foundry MI lands |

## Costs (rough, monthly)

| Resource | Cost |
|---|---|
| ACR Basic | ~£4 |
| Key Vault standard | ~£0.03 per 10k ops |
| Log Analytics (PerGB2018, ~5 GB/mo) | ~£10 |
| Application Insights | free at low volume |
| ACA (1 always-on replica, 2 vCPU / 4 GiB) | ~£25–35 |
| Storage account (~6.4 GB Hot) | ~£0.20 |
| Foundry per-token | depends on usage |
| **Total floor (no traffic)** | **~£40–50/month** |

The always-on replica is the biggest line. Drop `var.min_replicas` to 0 to halve it, at the cost of ~30s cold-start latency on the first request after idle.

## Common gotchas

- **Key Vault name conflict on re-create**: KV has 90-day soft-delete + purge protection. If you destroy the KV via Terraform (or the RG), the name is reserved for 90 days. Either wait, recover via `az keyvault recover`, or change `var.base_name`.
- **ACR name globally unique**: `mscragv2acr` is currently unclaimed. If someone takes it, change `var.base_name` so the derived ACR name shifts.
- **First-deploy image bootstrap**: the .tf declares ACA with a placeholder hello-world image because ACA needs a pullable image to come up. `deploy.sh` stage 3 swaps in the real image.
- **MI propagation lag**: role assignments can take 1–5 min to propagate. If a stage fails with `AuthorizationFailed`, wait a couple of minutes and re-run — the scripts are idempotent.
- **ACA env block order**: the azurerm provider does positional matching on the ACA env list. If you reorder env vars in the portal, `terraform plan` will show false drift. Stay in `main.tf` for env changes.
- **Role-assignment scope casing**: scope expressions in `main.tf` use `replace(...id, "resourceGroups", "resourcegroups")` because the azurerm RBAC API returns lowercase `resourcegroups` while resource-attribute IDs use camelCase. Without this, `forces_new` triggers a destroy + recreate on every plan.
