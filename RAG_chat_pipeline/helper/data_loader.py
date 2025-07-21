"""Data loader for test modules"""
import pickle
import os
import pandas as pd
from pathlib import Path
from RAG_chat_pipeline.config.config import BASE
from RAG_chat_pipeline.utils.data_provider import DataProvider


# Initialize the data provider
data_provider = DataProvider()


def load_test_data():
    """
    Load exported data for testing.

    Uses either real MIMIC-IV data or synthetic data depending on availability.
    """
    try:
        if data_provider.using_synthetic:
            # Create synthetic dataframes that match expected structure
            admissions_df = data_provider.load_admissions()

            # Create link_tables dictionary with required tables
            link_tables = {
                "diagnoses_icd": data_provider.load_diagnoses(),
                "procedures_icd": data_provider.load_procedures(),
                "labevents": data_provider.load_lab_events(),
                "prescriptions": data_provider.load_medications()
            }

            # Handle potential column name differences
            if 'diagnosis' in link_tables["diagnoses_icd"].columns:
                link_tables["diagnoses_icd"]['long_title'] = link_tables["diagnoses_icd"]['diagnosis']

            if 'test_name' in link_tables["labevents"].columns:
                link_tables["labevents"]['label'] = link_tables["labevents"]['test_name']

            if 'drug' not in link_tables["prescriptions"].columns and 'medication' in link_tables["prescriptions"].columns:
                link_tables["prescriptions"]['drug'] = link_tables["prescriptions"]['medication']

            # Create a simple grouped tables dict with just the basics
            grouped = {
                "admissions_by_patient": admissions_df.groupby('subject_id'),
                "diagnoses_by_admission": link_tables["diagnoses_icd"].groupby('hadm_id'),
                "procedures_by_admission": link_tables["procedures_icd"].groupby('hadm_id'),
                "labs_by_admission": link_tables["labevents"].groupby('hadm_id'),
                "meds_by_admission": link_tables["prescriptions"].groupby('hadm_id')
            }

            return admissions_df, link_tables, grouped
        else:
            # Use real data
            export_dir = BASE / "mimic_sample_1000" / "exports"

            # Load admissions_df
            with open(export_dir / "admissions_df.pkl", "rb") as f:
                admissions_df = pickle.load(f)

            # Load link_tables
            # Contains:
            # ['diagnoses_icd', 'procedures_icd', 'labevents',
            #  'microbiologyevents', 'prescriptions', 'transfers']
            with open(export_dir / "link_tables.pkl", "rb") as f:
                link_tables = pickle.load(f)

            # Load grouped tables
            with open(export_dir / "grouped_tables.pkl", "rb") as f:
                grouped = pickle.load(f)

            return admissions_df, link_tables, grouped

    except Exception as e:
        print(f"❌ Data loading error: {e}")
        print("Please ensure either MIMIC-IV data or synthetic data is available")
        return None, None, None


def get_sample_data():
    """Get sample data for quick testing"""
    admissions_df, link_tables, grouped = load_test_data()

    if admissions_df is None:
        return None

    # Get sample admission IDs
    sample_hadm_ids = admissions_df['hadm_id'].head(10).tolist()

    # Get common diagnoses
    common_diagnoses = link_tables["diagnoses_icd"]['long_title'].value_counts(
    ).head(5).index.tolist()

    # Get common lab tests
    common_labs = link_tables["labevents"]['label'].value_counts().head(
        5).index.tolist()

    # Get common medications
    common_meds = link_tables["prescriptions"]['drug'].value_counts().head(
        5).index.tolist()

    return {
        "admissions_df": admissions_df,
        "link_tables": link_tables,
        "grouped": grouped,
        "hadm_ids": sample_hadm_ids,
        "diagnoses": common_diagnoses,
        "labs": common_labs,
        "meds": common_meds
    }


if __name__ == "__main__":
    # Test the loader
    sample_data = get_sample_data()
    if sample_data:
        print("\nSample data loaded successfully!")
        print(f"Sample admission IDs: {sample_data['sample_hadm_ids'][:3]}")
        print(f"Common diagnoses: {sample_data['common_diagnoses'][:2]}")
