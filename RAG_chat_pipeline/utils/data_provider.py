"""
Data provider module that automatically selects a data source.

Source priority (highest first):
  1. Azure Blob (`v2-seed/`) — when USE_BLOB_DATA=true and the storage env
     vars are set. Reads Parquet for silver/gold; FAISS bytes are handled
     separately by embeddings_manager.
  2. Local real MIMIC-IV pickles + CSVs (mimic_sample_1000/).
  3. Local synthetic data (synthetic_data/), generating it if necessary.

The Blob path uses fsspec/adlfs with DefaultAzureCredential, so the same
code runs locally (`az login`) and in Azure Container Apps (Managed
Identity). The Parquet form was produced by data_engineering.parquet_convert.
"""

import os
import pickle
import pandas as pd
from pathlib import Path

from dotenv import load_dotenv
from RAG_chat_pipeline.utils.logger import ClinicalLogger

# Idempotent: pulls AZURE_*, USE_BLOB_DATA from .env so callers don't have to.
load_dotenv()


class DataProvider:
    """
    Provides data loading capabilities that automatically selects between
    Azure Blob (Parquet), real MIMIC-IV local data, and synthetic data.
    """

    # Mapping from logical link-table name to the silver/<name>.parquet blob.
    # Mirrors the dict structure in the original link_tables.pkl.
    _LINK_TABLE_NAMES = (
        "diagnoses_icd",
        "procedures_icd",
        "labevents",
        "microbiologyevents",
        "prescriptions",
        "transfers",
    )

    def __init__(self, real_data_path="mimic_sample_1000", synthetic_data_path="synthetic_data", verbose: bool = True):
        """
        Initialize the data provider.

        Args:
            real_data_path: Path to the real MIMIC-IV data
            synthetic_data_path: Path to the synthetic data
            verbose: If True, print informative messages; if False, stay quiet
        """
        # Get the project root directory
        current_file = Path(__file__).resolve()
        project_root = current_file
        while project_root.name != "msc_project" and project_root.parent != project_root:
            project_root = project_root.parent

        # Set absolute paths
        self.real_data_path = project_root / real_data_path
        self.synthetic_data_path = project_root / synthetic_data_path
        self.using_synthetic = False
        self.verbose = verbose

        # Blob mode — env-driven. Falls back to local if env vars are absent
        # so callers don't need to know which mode they're in.
        self.use_blob = os.getenv("USE_BLOB_DATA", "").lower() in ("true", "1", "yes")
        self.azure_account = os.getenv("AZURE_STORAGE_ACCOUNT")
        self.azure_container = os.getenv("AZURE_BLOB_CONTAINER")
        if self.use_blob and not (self.azure_account and self.azure_container):
            if self.verbose:
                ClinicalLogger.warning(
                    "USE_BLOB_DATA=true but AZURE_STORAGE_ACCOUNT/CONTAINER missing; falling back to local data."
                )
            self.use_blob = False

        # Lazy-init credential (DefaultAzureCredential does network probes
        # on construction, so don't pay that cost if we never read from Blob).
        self._azure_cred = None

        # Local source detection only matters if we're not using Blob
        self.data_source_path = None if self.use_blob else self._determine_data_source()

        if self.use_blob:
            if self.verbose:
                ClinicalLogger.info(
                    "Loading from Azure Blob",
                    account=self.azure_account,
                    container=self.azure_container,
                )
        elif self.using_synthetic:
            if self.verbose:
                ClinicalLogger.info(
                    "Using synthetic data. For research with real data, please obtain MIMIC-IV access."
                )
        else:
            if self.verbose:
                ClinicalLogger.info("Using real MIMIC-IV data.")

    # ---------- Blob helpers ----------

    def _credential(self):
        if self._azure_cred is None:
            from azure.identity import DefaultAzureCredential
            self._azure_cred = DefaultAzureCredential()
        return self._azure_cred

    def _read_blob_parquet(self, blob_path: str) -> pd.DataFrame:
        """Read a Parquet blob into a DataFrame.

        The container itself (typically `v2-seed`) is the medallion root,
        so `blob_path` looks like `gold/chunked_docs.parquet` or
        `silver/admissions.parquet`.
        """
        return pd.read_parquet(
            f"abfs://{self.azure_container}/{blob_path}",
            storage_options={
                "account_name": self.azure_account,
                "credential": self._credential(),
            },
        )

    @staticmethod
    def _df_to_documents(df: pd.DataFrame):
        """Convert a chunked-docs DataFrame back to LangChain Documents.

        Drops NaN/NaT metadata values so consumers don't see them. Done lazily
        in a list comprehension to avoid materialising 105k records dict twice.
        """
        from langchain.schema import Document
        records = df.to_dict(orient="records")
        docs = []
        for rec in records:
            content = rec.pop("content")
            metadata = {k: v for k, v in rec.items() if pd.notna(v)}
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def _determine_data_source(self):
        """
        Determine which data source to use based on availability.
        Returns the path to the data source.
        """
        # Check if real data exists
        real_chunked_docs_path = self.real_data_path / "chunked_docs.pkl"

        if real_chunked_docs_path.exists():
            self.using_synthetic = False
            return self.real_data_path

        # Check if synthetic data exists
        synth_chunked_docs_path = self.synthetic_data_path / "chunked_docs_synthetic.pkl"

        if synth_chunked_docs_path.exists():
            self.using_synthetic = True
            return self.synthetic_data_path

        # If synthetic data doesn't exist, try to create it
        try:
            synthetic_generator_path = self.synthetic_data_path / "synthetic_data_generator.py"
            if synthetic_generator_path.exists():
                if self.verbose:
                    ClinicalLogger.info("Real data not found. Generating synthetic data...")
                from synthetic_data.synthetic_data_generator import create_synthetic_data
                create_synthetic_data()
                self.using_synthetic = True
                return self.synthetic_data_path
        except Exception as e:
            if self.verbose:
                ClinicalLogger.warning("Error generating synthetic data", error=str(e))

        # If neither exists and synthetic data can't be created, raise error
        raise FileNotFoundError(
            "No data found. Either provide real MIMIC-IV data in the 'mimic_sample_1000' directory "
            "or run the synthetic data generator in the 'synthetic_data' directory."
        )

    def load_chunked_docs(self):
        """
        Load the chunked documents for the RAG system.
        """
        if self.use_blob:
            df = self._read_blob_parquet("gold/chunked_docs.parquet")
            return self._df_to_documents(df)
        if self.using_synthetic:
            with open(self.data_source_path / "chunked_docs_synthetic.pkl", "rb") as f:
                return pickle.load(f)
        else:
            with open(self.data_source_path / "chunked_docs.pkl", "rb") as f:
                return pickle.load(f)

    def load_admissions(self):
        """
        Load the admissions data.
        """
        if self.use_blob:
            return self._read_blob_parquet("silver/admissions.parquet")
        if self.using_synthetic:
            return pd.read_csv(self.data_source_path / "admissions_synthetic.csv")
        else:
            return pd.read_csv(self.data_source_path / "admissions.csv_sample1000.csv")

    def load_diagnoses(self):
        """
        Load the diagnoses data.
        """
        if self.use_blob:
            return self._read_blob_parquet("silver/diagnoses_icd.parquet")
        if self.using_synthetic:
            return pd.read_csv(self.data_source_path / "diagnoses_synthetic.csv")
        else:
            return pd.read_csv(self.data_source_path / "diagnoses_icd.csv_sample1000.csv")

    def load_procedures(self):
        """
        Load the procedures data.
        """
        if self.use_blob:
            return self._read_blob_parquet("silver/procedures_icd.parquet")
        if self.using_synthetic:
            return pd.read_csv(self.data_source_path / "procedures_synthetic.csv")
        else:
            return pd.read_csv(self.data_source_path / "procedures_icd.csv_sample1000.csv")

    def load_lab_events(self):
        """
        Load the lab events data.
        """
        if self.use_blob:
            return self._read_blob_parquet("silver/labevents.parquet")
        if self.using_synthetic:
            return pd.read_csv(self.data_source_path / "labevents_synthetic.csv")
        else:
            return pd.read_csv(self.data_source_path / "labevents.csv_sample1000.csv")

    def load_medications(self):
        """
        Load the medications data.
        """
        if self.use_blob:
            return self._read_blob_parquet("silver/prescriptions.parquet")
        if self.using_synthetic:
            return pd.read_csv(self.data_source_path / "medications_synthetic.csv")
        else:
            return pd.read_csv(self.data_source_path / "prescriptions.csv_sample1000.csv")

    def get_data_source_type(self):
        """
        Return a string indicating the type of data being used.
        """
        if self.use_blob:
            return "blob"
        return "synthetic" if self.using_synthetic else "real"

    def load_test_data(self):
        """Load exported data for testing - equivalent to data_loader.load_test_data()"""
        if self.use_blob:
            # silver/admissions.parquet + 6 link-table parquets, with grouped
            # derived from link_tables on-the-fly (grouped_tables was dropped
            # in Step 2 of the priming plan as redundant).
            try:
                admissions_df = self._read_blob_parquet("silver/admissions.parquet")
                link_tables = {
                    name: self._read_blob_parquet(f"silver/{name}.parquet")
                    for name in self._LINK_TABLE_NAMES
                }
                grouped = {
                    name: df.groupby("hadm_id")
                    for name, df in link_tables.items()
                    if "hadm_id" in df.columns
                }
                return admissions_df, link_tables, grouped
            except Exception as e:
                if self.verbose:
                    ClinicalLogger.warning("Error reading silver layer from Blob", error=str(e))
                return None, None, None

        if self.using_synthetic:
            # For synthetic data, return basic structures
            try:
                admissions_df = self.load_admissions()

                # Create mock link_tables structure
                link_tables = {
                    'diagnoses_icd': self.load_diagnoses(),
                    'procedures_icd': self.load_procedures(),
                    'labevents': self.load_lab_events(),
                    'prescriptions': self.load_medications()
                }

                # Create mock grouped structure (basic grouping by hadm_id)
                grouped = {}
                for key, df in link_tables.items():
                    if 'hadm_id' in df.columns:
                        grouped[key] = df.groupby('hadm_id')

                return admissions_df, link_tables, grouped

            except FileNotFoundError as e:
                if self.verbose:
                    ClinicalLogger.warning("Synthetic data files not found", error=str(e))
                return None, None, None
        else:
            # Use real MIMIC data exports
            export_dir = self.real_data_path / "exports"
            try:
                # Load admissions_df
                with open(export_dir / "admissions_df.pkl", "rb") as f:
                    admissions_df = pickle.load(f)

                # Load link_tables
                with open(export_dir / "link_tables.pkl", "rb") as f:
                    link_tables = pickle.load(f)

                # Load grouped tables
                with open(export_dir / "grouped_tables.pkl", "rb") as f:
                    grouped = pickle.load(f)

                return admissions_df, link_tables, grouped

            except FileNotFoundError as e:
                if self.verbose:
                    ClinicalLogger.warning("MIMIC data export files not found", error=str(e))
                    ClinicalLogger.info(
                        "Please run the data processing notebook to export data first"
                    )
                return None, None, None

    def get_sample_data(self):
        """Get sample data for quick testing - equivalent to data_loader.get_sample_data()"""
        admissions_df, link_tables, grouped = self.load_test_data()

        if admissions_df is None:
            return None

        # Get sample admission IDs
        sample_hadm_ids = admissions_df['hadm_id'].head(10).tolist()

        # Get common diagnoses (handle both synthetic and real data column names)
        try:
            if 'long_title' in link_tables["diagnoses_icd"].columns:
                common_diagnoses = link_tables["diagnoses_icd"]['long_title'].value_counts(
                ).head(5).index.tolist()
            else:
                # Fallback for synthetic data
                diag_col = link_tables["diagnoses_icd"].columns[1] if len(
                    link_tables["diagnoses_icd"].columns) > 1 else link_tables["diagnoses_icd"].columns[0]
                common_diagnoses = link_tables["diagnoses_icd"][diag_col].value_counts().head(
                    5).index.tolist()
        except (KeyError, IndexError):
            common_diagnoses = []

        # Get common lab tests
        try:
            if 'label' in link_tables["labevents"].columns:
                common_labs = link_tables["labevents"]['label'].value_counts().head(
                    5).index.tolist()
            else:
                # Fallback for synthetic data
                lab_col = link_tables["labevents"].columns[1] if len(
                    link_tables["labevents"].columns) > 1 else link_tables["labevents"].columns[0]
                common_labs = link_tables["labevents"][lab_col].value_counts().head(
                    5).index.tolist()
        except (KeyError, IndexError):
            common_labs = []

        # Get common medications
        try:
            if 'drug' in link_tables["prescriptions"].columns:
                common_meds = link_tables["prescriptions"]['drug'].value_counts().head(
                    5).index.tolist()
            else:
                # Fallback for synthetic data
                med_col = link_tables["prescriptions"].columns[1] if len(
                    link_tables["prescriptions"].columns) > 1 else link_tables["prescriptions"].columns[0]
                common_meds = link_tables["prescriptions"][med_col].value_counts().head(
                    5).index.tolist()
        except (KeyError, IndexError):
            common_meds = []

        return {
            "admissions_df": admissions_df,
            "link_tables": link_tables,
            "grouped": grouped,
            "hadm_ids": sample_hadm_ids,
            "diagnoses": common_diagnoses,
            "labs": common_labs,
            "meds": common_meds
        }


# Global instance for backward compatibility
_default_provider = None


def get_default_provider():
    """Get or create the default data provider instance"""
    global _default_provider
    if _default_provider is None:
        _default_provider = DataProvider()
    return _default_provider


# Convenience functions for backward compatibility with data_loader.py
def load_test_data():
    """Load exported data for testing - backward compatibility function"""
    return get_default_provider().load_test_data()


def get_sample_data():
    """Get sample data for quick testing - backward compatibility function"""
    return get_default_provider().get_sample_data()


if __name__ == "__main__":
    # Test the provider
    provider = DataProvider()
    ClinicalLogger.info("Using data source", source=provider.get_data_source_type())

    sample_data = provider.get_sample_data()
    if sample_data:
        ClinicalLogger.info("Sample data loaded successfully")
    else:
        ClinicalLogger.warning("Failed to load sample data")
