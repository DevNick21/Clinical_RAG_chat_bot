"""
Data provider module that automatically selects between real MIMIC-IV data
and synthetic data based on availability.
"""

import os
import pickle
import pandas as pd
from pathlib import Path


class DataProvider:
    """
    Provides data loading capabilities that automatically selects between
    real MIMIC-IV data and synthetic data based on availability.
    """

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

        # Check what data is available
        self.data_source_path = self._determine_data_source()

        if self.using_synthetic:
            if self.verbose:
                print(
                    "📌 Using synthetic data. For research with real data, please obtain MIMIC-IV access.")
        else:
            if self.verbose:
                print("📌 Using real MIMIC-IV data.")

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
                    print("🔄 Real data not found. Generating synthetic data...")
                import sys
                sys.path.append(str(self.synthetic_data_path.parent))
                from synthetic_data.synthetic_data_generator import create_synthetic_data
                create_synthetic_data()
                self.using_synthetic = True
                return self.synthetic_data_path
        except Exception as e:
            if self.verbose:
                print(f" Error generating synthetic data: {e}")

        # If neither exists and synthetic data can't be created, raise error
        raise FileNotFoundError(
            "No data found. Either provide real MIMIC-IV data in the 'mimic_sample_1000' directory "
            "or run the synthetic data generator in the 'synthetic_data' directory."
        )

    def load_chunked_docs(self):
        """
        Load the chunked documents for the RAG system.
        """
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
        if self.using_synthetic:
            return pd.read_csv(self.data_source_path / "admissions_synthetic.csv")
        else:
            return pd.read_csv(self.data_source_path / "admissions.csv_sample1000.csv")

    def load_diagnoses(self):
        """
        Load the diagnoses data.
        """
        if self.using_synthetic:
            return pd.read_csv(self.data_source_path / "diagnoses_synthetic.csv")
        else:
            return pd.read_csv(self.data_source_path / "diagnoses_icd.csv_sample1000.csv")

    def load_procedures(self):
        """
        Load the procedures data.
        """
        if self.using_synthetic:
            return pd.read_csv(self.data_source_path / "procedures_synthetic.csv")
        else:
            return pd.read_csv(self.data_source_path / "procedures_icd.csv_sample1000.csv")

    def load_lab_events(self):
        """
        Load the lab events data.
        """
        if self.using_synthetic:
            return pd.read_csv(self.data_source_path / "labevents_synthetic.csv")
        else:
            return pd.read_csv(self.data_source_path / "labevents.csv_sample1000.csv")

    def load_medications(self):
        """
        Load the medications data.
        """
        if self.using_synthetic:
            return pd.read_csv(self.data_source_path / "medications_synthetic.csv")
        else:
            return pd.read_csv(self.data_source_path / "prescriptions.csv_sample1000.csv")

    def get_data_source_type(self):
        """
        Return a string indicating the type of data being used.
        """
        return "synthetic" if self.using_synthetic else "real"

    def load_test_data(self):
        """Load exported data for testing - equivalent to data_loader.load_test_data()"""
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
                    print(f" Synthetic data files not found: {e}")
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
                    print(f" MIMIC data export files not found: {e}")
                    print(
                        "Please run the data processing notebook to export data first")
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
    print(f"Using {provider.get_data_source_type()} data")

    sample_data = provider.get_sample_data()
    if sample_data:
        print("\nSample data loaded successfully!")
        print(f"Sample admission IDs: {sample_data['hadm_ids'][:3]}")
        print(f"Common diagnoses: {sample_data['diagnoses'][:2]}")
        print(f"Common labs: {sample_data['labs'][:2]}")
        print(f"Common medications: {sample_data['meds'][:2]}")
    else:
        print("Failed to load sample data")
