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

    def __init__(self, real_data_path="mimic_sample_1000", synthetic_data_path="synthetic_data"):
        """
        Initialize the data provider.

        Args:
            real_data_path: Path to the real MIMIC-IV data
            synthetic_data_path: Path to the synthetic data
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

        # Check what data is available
        self.data_source_path = self._determine_data_source()

        if self.using_synthetic:
            print(
                "📌 Using synthetic data. For research with real data, please obtain MIMIC-IV access.")
        else:
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
                print("🔄 Real data not found. Generating synthetic data...")
                import sys
                sys.path.append(str(self.synthetic_data_path.parent))
                from synthetic_data.synthetic_data_generator import create_synthetic_data
                create_synthetic_data()
                self.using_synthetic = True
                return self.synthetic_data_path
        except Exception as e:
            print(f"⚠️ Error generating synthetic data: {e}")

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
