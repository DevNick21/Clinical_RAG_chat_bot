"""Upload v2-seed artefacts to Azure Blob Storage.

Lift-and-shift of the irreplaceable local artefacts (raw CSV samples,
joined link-table pickles, gold chunked docs, the winning FAISS index,
dev FAISS indexes, HF model snapshots) into a versioned, soft-delete-
protected container. Idempotent: an existing blob is skipped, never
overwritten, so re-running is safe and resumable.

Layers, in upload order (most precious first):
    gold          chunked_docs.pkl                                ~50 MB
    silver        mimic_sample_1000/exports/*.pkl                ~50 MB
    bronze        mimic_sample_1000/*.csv                       ~135 MB
    prod-index    vector_stores/<winner>/                        ~370 MB
    winner-model  models/<winner>/                               ~420 MB
    dev-indexes   vector_stores/<other 8>/                       ~2.7 GB
    other-models  models/<other 8>/                              ~2.6 GB

Usage (run from repo root):
    python -m data_engineering.seed_upload                  # all layers
    python -m data_engineering.seed_upload --layers gold silver bronze
    python -m data_engineering.seed_upload --dry-run

Auth: DefaultAzureCredential. Locally that's `az login`; in ACA it'll
be the container's Managed Identity.
"""
import argparse
import hashlib
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from azure.core.exceptions import ResourceExistsError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent

# The embedding that fell out of the 54-combo evaluation as the winner.
# Used to identify the prod-index and the winner-model when slicing the
# vector_stores/ and models/ directories into "winner" vs "rest".
WINNER_EMBEDDING = "biomedbert"
WINNER_VECTORSTORE_DIR = "faiss_mimic_sample1000_biomedbert"
WINNER_MODEL_DIR = "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# Read in 4 MB chunks for hashing — keeps memory flat for the 400 MB
# model weight files.
HASH_CHUNK_BYTES = 4 * 1024 * 1024


@dataclass
class BlobPlan:
    """A single file to upload, paired with its destination blob path."""
    local_path: Path
    blob_path: str
    layer: str


@dataclass
class UploadResult:
    blob_path: str
    layer: str
    size_bytes: int
    sha256: str
    status: str  # "uploaded" | "skipped" | "error"
    error: Optional[str] = None


def _files_under(root: Path) -> List[Path]:
    """Return all regular files under `root`, skipping noise."""
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and not p.name.startswith(".")
    )


def build_plan(layers: Iterable[str]) -> List[BlobPlan]:
    """Map local files to blob paths, layered.

    Layer order is preserved in the output list so the upload loop puts
    the most precious data on the wire first.
    """
    plan: List[BlobPlan] = []
    sample_dir = REPO_ROOT / "mimic_sample_1000"
    exports_dir = sample_dir / "exports"
    vector_stores = REPO_ROOT / "vector_stores"
    models_dir = REPO_ROOT / "models"

    if "gold" in layers:
        gold_file = sample_dir / "chunked_docs.pkl"
        if gold_file.exists():
            plan.append(BlobPlan(gold_file, "gold/chunked_docs.pkl", "gold"))

    if "silver" in layers:
        for f in _files_under(exports_dir):
            plan.append(BlobPlan(f, f"silver/{f.name}", "silver"))

    if "bronze" in layers:
        for f in sorted(sample_dir.glob("*.csv")):
            plan.append(BlobPlan(f, f"bronze/{f.name}", "bronze"))

    if "prod-index" in layers:
        winner_dir = vector_stores / WINNER_VECTORSTORE_DIR
        for f in _files_under(winner_dir):
            rel = f.relative_to(winner_dir)
            plan.append(BlobPlan(f, f"prod-index/faiss_winner/{rel.as_posix()}", "prod-index"))

    if "winner-model" in layers:
        winner_model = models_dir / WINNER_MODEL_DIR
        for f in _files_under(winner_model):
            rel = f.relative_to(winner_model)
            plan.append(BlobPlan(f, f"models/winner/{rel.as_posix()}", "winner-model"))

    if "dev-indexes" in layers and vector_stores.exists():
        for d in sorted(vector_stores.iterdir()):
            if not d.is_dir() or d.name == WINNER_VECTORSTORE_DIR:
                continue
            for f in _files_under(d):
                rel = f.relative_to(d)
                plan.append(BlobPlan(f, f"dev-indexes/{d.name}/{rel.as_posix()}", "dev-indexes"))

    if "other-models" in layers and models_dir.exists():
        for d in sorted(models_dir.iterdir()):
            if not d.is_dir() or d.name == WINNER_MODEL_DIR:
                continue
            for f in _files_under(d):
                rel = f.relative_to(d)
                plan.append(BlobPlan(f, f"models/{d.name}/{rel.as_posix()}", "other-models"))

    return plan


def sha256_streaming(path: Path) -> str:
    """Hex-digest SHA256 of `path`, read in chunks to bound memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def upload_one(container_client, item: BlobPlan) -> UploadResult:
    """Upload a single file. Skip cleanly if the blob already exists."""
    size = item.local_path.stat().st_size
    digest = sha256_streaming(item.local_path)
    blob_client = container_client.get_blob_client(item.blob_path)

    try:
        with item.local_path.open("rb") as f:
            blob_client.upload_blob(
                f,
                overwrite=False,
                length=size,
                metadata={"sha256": digest, "layer": item.layer},
            )
        return UploadResult(item.blob_path, item.layer, size, digest, "uploaded")
    except ResourceExistsError:
        return UploadResult(item.blob_path, item.layer, size, digest, "skipped")
    except Exception as exc:
        return UploadResult(item.blob_path, item.layer, size, digest, "error", str(exc))


def write_manifest(results: List[UploadResult], output_path: Path, account: str, container: str) -> None:
    """Write SEED_MANIFEST.md grouped by layer with size and hash columns."""
    by_layer: dict[str, List[UploadResult]] = {}
    for r in results:
        by_layer.setdefault(r.layer, []).append(r)

    total_files = len(results)
    total_bytes = sum(r.size_bytes for r in results)
    uploaded = sum(1 for r in results if r.status == "uploaded")
    skipped = sum(1 for r in results if r.status == "skipped")
    errors = sum(1 for r in results if r.status == "error")

    lines = [
        "# SEED_MANIFEST",
        "",
        f"**Frozen:** {datetime.now(timezone.utc).isoformat()}",
        f"**Storage:** `{account}` / container `{container}`",
        "**Encryption:** Microsoft-managed keys (default for Standard_LRS)",
        "**Versioning:** Enabled",
        "**Soft-delete:** 30 days (blob + container)",
        "",
        "## Why this exists",
        "",
        "Upstream MIMIC-IV access was lost. The 1000-row CSV samples and",
        "everything derived from them (joined link tables, gold chunked",
        "documents, winning FAISS index) are the only surviving copies of",
        "research-time work. This manifest is the data-lineage record",
        "until a managed catalog is in place.",
        "",
        f"## Summary",
        "",
        f"- Total files: **{total_files}**",
        f"- Total size: **{total_bytes / (1024 * 1024):.1f} MB**",
        f"- Uploaded this run: {uploaded}",
        f"- Skipped (already present): {skipped}",
        f"- Errors: {errors}",
        "",
        "## Provenance notes",
        "",
        f"- `gold/chunked_docs.pkl` was produced with embedding model `{WINNER_EMBEDDING}`.",
        f"- `prod-index/faiss_winner/` is the FAISS index produced by `{WINNER_EMBEDDING}` —",
        "  selected as the winner from the 54-combo (9 emb x 6 LLM) evaluation grid.",
        "- `dev-indexes/` holds the other 8 FAISS indexes, kept only to reproduce the",
        "  benchmark grid. Production retrieval uses `prod-index/` only.",
        "- `models/` holds HF checkpoints. Re-downloadable from HuggingFace if lost,",
        "  but mirrored for offline / air-gapped runs.",
        "",
    ]

    layer_order = ["gold", "silver", "bronze", "prod-index", "winner-model", "dev-indexes", "other-models"]
    for layer in layer_order:
        items = by_layer.get(layer, [])
        if not items:
            continue
        layer_total = sum(r.size_bytes for r in items)
        lines += [
            f"## {layer} ({len(items)} files, {layer_total / (1024 * 1024):.1f} MB)",
            "",
            "| blob | size (MB) | sha256 |",
            "|---|---|---|",
        ]
        for r in sorted(items, key=lambda x: x.blob_path):
            lines.append(f"| `{r.blob_path}` | {r.size_bytes / (1024 * 1024):.2f} | `{r.sha256[:16]}…` |")
        lines.append("")

    if errors:
        lines += ["## Errors", ""]
        for r in results:
            if r.status == "error":
                lines.append(f"- `{r.blob_path}` — {r.error}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["gold", "silver", "bronze", "prod-index", "winner-model", "dev-indexes", "other-models"],
        choices=["gold", "silver", "bronze", "prod-index", "winner-model", "dev-indexes", "other-models"],
        help="Layers to upload (default: all).",
    )
    parser.add_argument("--dry-run", action="store_true", help="List files that would upload, don't transfer.")
    args = parser.parse_args()

    account = os.getenv("AZURE_STORAGE_ACCOUNT")
    container = os.getenv("AZURE_BLOB_CONTAINER")
    if not account or not container:
        print("ERROR: AZURE_STORAGE_ACCOUNT and AZURE_BLOB_CONTAINER must be set in .env", file=sys.stderr)
        return 2

    plan = build_plan(args.layers)
    if not plan:
        print("Nothing to upload (no matching files in the requested layers).")
        return 0

    total_bytes = sum(p.local_path.stat().st_size for p in plan)
    print(f"Plan: {len(plan)} files, {total_bytes / (1024 * 1024):.1f} MB across layers: {sorted(set(p.layer for p in plan))}")

    if args.dry_run:
        for p in plan:
            print(f"  [{p.layer}] {p.local_path} -> {p.blob_path}")
        return 0

    blob_service = BlobServiceClient(
        account_url=f"https://{account}.blob.core.windows.net",
        credential=DefaultAzureCredential(),
    )
    container_client = blob_service.get_container_client(container)

    results: List[UploadResult] = []
    for i, item in enumerate(plan, start=1):
        size_mb = item.local_path.stat().st_size / (1024 * 1024)
        print(f"[{i:>4}/{len(plan)}] [{item.layer:>13}] {item.blob_path}  ({size_mb:.1f} MB) ... ", end="", flush=True)
        result = upload_one(container_client, item)
        results.append(result)
        print(result.status if result.status != "error" else f"ERROR: {result.error}")

    manifest_path = REPO_ROOT / "data_engineering" / "SEED_MANIFEST.md"
    write_manifest(results, manifest_path, account, container)
    print(f"\nManifest written: {manifest_path}")

    # Also upload the manifest itself so the data is self-describing
    try:
        with manifest_path.open("rb") as f:
            container_client.get_blob_client("SEED_MANIFEST.md").upload_blob(f, overwrite=True)
        print(f"Manifest mirrored to blob: SEED_MANIFEST.md")
    except Exception as exc:
        print(f"WARNING: could not mirror manifest to blob: {exc}")

    errors = [r for r in results if r.status == "error"]
    if errors:
        print(f"\n{len(errors)} file(s) failed. Re-run to retry — uploads are idempotent.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
