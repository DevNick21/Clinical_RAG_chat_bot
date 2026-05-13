# Clinical RAG v2 — Azure infrastructure

Hand-written Bicep + a thin shell orchestrator. Single-file template by
design so it reads end-to-end and compares cleanly with the
CloudFormation/CDK we'll write for v3 (AWS).

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

## Prereqs

- `az` CLI installed and `az login` done
- Logged into the right subscription: `az account show`
- Docker daemon running (the script builds + pushes the image)
- `.env` at repo root populated with at minimum:

  ```
  TARGET_URL=https://<resource>.services.ai.azure.com/openai/v1
  MODEL_API_KEY=<foundry deployment key>
  MODEL_DEPLOYMENT_NAME=gpt-5-nano
  REASONING_EFFORT=low
  API_KEY=<long random secret — generate: python -c "import secrets;print(secrets.token_urlsafe(32))">
  ALLOWED_ORIGINS=https://your-frontend-host
  ```

## Deploy

```bash
bash infra/deploy.sh
```

Stages, in order:

1. **Bicep deploy** (3–5 min on first run, ~30s on re-runs) — creates/updates every resource above.
2. **Push secrets to Key Vault** — `API_KEY` and `MODEL_API_KEY` go into KV; ACA reads them via secret references at startup.
3. **Build + push image** — `docker build`, then `az acr login` + `docker push`. Image tag defaults to current git short SHA (`IMAGE_TAG=...` to override).
4. **Roll ACA revision** — `az containerapp update --image` creates a new revision and shifts traffic.

The script is idempotent; safe to re-run. Comment out stages 1–2 when iterating just on the image.

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

## What this deliberately does NOT include

| Feature | Why deferred |
|---|---|
| GitHub Actions CI/CD workflow | Local `deploy.sh` covers everything for now. Add OIDC-based GHA when there's actually a team pushing to main |
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

The always-on replica is the biggest line. Drop `minReplicas` to 0 in the Bicep parameters to halve it, at the cost of ~30s cold-start latency on the first request after idle.

## Common gotchas

- **Key Vault name conflict on re-create**: KV has 90-day soft-delete + purge protection. If you delete the RG and try to redeploy, the KV name is reserved. Either wait 90 days, recover via `az keyvault recover`, or change `baseName`.
- **ACR name globally unique**: `mscragv2acr` is currently unclaimed. If someone takes it, change `baseName` so the derived ACR name shifts.
- **First-deploy image bootstrap**: the Bicep declares ACA with a placeholder hello-world image because ACA needs a pullable image to come up. `deploy.sh` updates to the real image in stage 4.
- **MI propagation lag**: role assignments can take 1–5 min to propagate. If stage 4 fails with `AuthorizationFailed`, wait a couple of minutes and re-run — the bash script is idempotent.
