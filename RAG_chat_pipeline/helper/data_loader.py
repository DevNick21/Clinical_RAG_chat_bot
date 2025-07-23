"""Data loader for test modules"""
import pickle
from RAG_chat_pipeline.config.config import BASE


def load_test_data():
    """Load exported data for testing"""
    export_dir = BASE / "mimic_sample_1000" / "exports"

    try:
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

    except FileNotFoundError as e:
        print(f"❌ Data files not found: {e}")
        print("Please run the notebook to export data first")
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
