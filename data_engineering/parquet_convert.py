"""Convert silver/gold pickles into Parquet, upload to Blob.

Step 2 of the v2 priming plan. The .pkl files lifted in Step 1 are
preserved in `v2-seed/silver/` and `v2-seed/gold/` as the source of
truth; this script writes a Parquet form alongside them. Once
DataProvider reads Parquet (Step 4), the .pkl blobs become a frozen
safety net and the .parquet blobs are the working surface.

Why Parquet
- Schema-stable: pickle is Python-version- and class-coupled
  (LangChain Document layout can change between versions).
- Portable: readable from pandas, DuckDB, Spark, Trino, Polars,
  Azure AI Search ingestion, without LangChain installed.
- Columnar + compressed: ~3-10x smaller on disk than pickle for
  these tables.

What's converted
    silver/admissions_df.pkl          -> silver/admissions.parquet
    silver/link_tables.pkl (dict)     -> silver/<table>.parquet (one per dict entry)
    gold/chunked_docs.pkl (list[Doc]) -> gold/chunked_docs.parquet

What's intentionally dropped
    silver/grouped_tables.pkl -> not converted. It's
        link_tables['<x>'].groupby('hadm_id') wearing a costume; re-derive
        at load time. One less artefact to maintain.

Run from repo root:
    python -m data_engineering.parquet_convert            # all
    python -m data_engineering.parquet_convert --dry-run  # plan only
"""
import argparse
import io
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = REPO_ROOT / "mimic_sample_1000"
EXPORTS_DIR = SAMPLE_DIR / "exports"


def _doc_to_record(doc: Any) -> Dict[str, Any]:
    """Flatten a LangChain Document to a flat dict for DataFrame ingestion.

    Trust the metadata layout observed in this corpus (hadm_id, subject_id,
    section, admittime, dischtime — all scalar). If a future ingestion
    introduces nested metadata, this assertion will fail loud and we'll
    handle it then rather than silently coerce.
    """
    content = getattr(doc, "page_content", None)
    metadata = getattr(doc, "metadata", {}) or {}
    if content is None:
        raise ValueError(f"Document has no page_content: {doc!r}")
    for k, v in metadata.items():
        if isinstance(v, (dict, list, set, tuple)):
            raise ValueError(
                f"Nested metadata value at key {k!r} ({type(v).__name__}); "
                "Parquet write would need a schema strategy. Refusing to coerce."
            )
    return {"content": content, **metadata}


def load_conversions() -> List[Tuple[str, pd.DataFrame]]:
    """Build the (blob_path, dataframe) pairs we'll upload.

    Order matters only for human-readable log output.
    """
    pairs: List[Tuple[str, pd.DataFrame]] = []

    # 1. admissions_df
    with (EXPORTS_DIR / "admissions_df.pkl").open("rb") as f:
        admissions = pickle.load(f)
    if not isinstance(admissions, pd.DataFrame):
        raise TypeError(f"admissions_df.pkl: expected DataFrame, got {type(admissions).__name__}")
    pairs.append(("silver/admissions.parquet", admissions))

    # 2. link_tables (dict[str, DataFrame])
    with (EXPORTS_DIR / "link_tables.pkl").open("rb") as f:
        link_tables = pickle.load(f)
    if not isinstance(link_tables, dict):
        raise TypeError(f"link_tables.pkl: expected dict, got {type(link_tables).__name__}")
    for name, df in link_tables.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"link_tables[{name!r}]: expected DataFrame, got {type(df).__name__}")
        # Sanitise table name to a safe blob segment (lowercase, no spaces)
        safe = name.strip().lower().replace(" ", "_")
        pairs.append((f"silver/{safe}.parquet", df))

    # 3. chunked_docs (list[Document])
    with (SAMPLE_DIR / "chunked_docs.pkl").open("rb") as f:
        chunked_docs = pickle.load(f)
    if not isinstance(chunked_docs, list):
        raise TypeError(f"chunked_docs.pkl: expected list, got {type(chunked_docs).__name__}")
    records = [_doc_to_record(d) for d in chunked_docs]
    chunked_df = pd.DataFrame(records)
    pairs.append(("gold/chunked_docs.parquet", chunked_df))

    return pairs


def df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    """Serialise to Parquet in memory. Snappy compression is the pandas default."""
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", compression="snappy", index=False)
    return buf.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="List what would be uploaded and exit.")
    args = parser.parse_args()

    account = os.getenv("AZURE_STORAGE_ACCOUNT")
    container = os.getenv("AZURE_BLOB_CONTAINER")
    if not account or not container:
        print("ERROR: AZURE_STORAGE_ACCOUNT and AZURE_BLOB_CONTAINER must be set in .env", file=sys.stderr)
        return 2

    print("Loading and converting...")
    pairs = load_conversions()

    if args.dry_run:
        print(f"\nWould upload {len(pairs)} Parquet blob(s):")
        for path, df in pairs:
            print(f"  {path}  ({df.shape[0]:>7} rows x {df.shape[1]:>3} cols)")
        return 0

    blob_service = BlobServiceClient(
        account_url=f"https://{account}.blob.core.windows.net",
        credential=DefaultAzureCredential(),
    )
    container_client = blob_service.get_container_client(container)

    print()
    for path, df in pairs:
        rows, cols = df.shape
        parquet_bytes = df_to_parquet_bytes(df)
        size_mb = len(parquet_bytes) / (1024 * 1024)
        print(f"  {path}  ({rows:>7} rows x {cols:>3} cols, {size_mb:.2f} MB) ... ", end="", flush=True)
        container_client.get_blob_client(path).upload_blob(
            parquet_bytes,
            overwrite=True,
            metadata={"source_format": "pickle", "rows": str(rows), "cols": str(cols)},
        )
        print("uploaded")

    # Sanity check: read each Parquet back from Blob and confirm row count
    print("\nVerifying round-trip...")
    for path, expected in pairs:
        blob_bytes = container_client.get_blob_client(path).download_blob().readall()
        roundtrip = pd.read_parquet(io.BytesIO(blob_bytes))
        ok = roundtrip.shape[0] == expected.shape[0]
        status = "OK" if ok else "MISMATCH"
        print(f"  {path}: read back {roundtrip.shape[0]:>7} rows ({status})")
        if not ok:
            return 1

    print("\nDone.")
    print("Note: grouped_tables.pkl was intentionally NOT converted.")
    print("      Re-derive at load time: link_tables[<name>].groupby('hadm_id').")
    return 0


if __name__ == "__main__":
    sys.exit(main())
