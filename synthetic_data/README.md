# Synthetic Medical Data Generator

This module generates synthetic medical data that mimics the structure of MIMIC-IV, but with entirely fictitious patient information suitable for public sharing.

## Purpose

The synthetic data generator serves several purposes:

1. **Enable Public Distribution**: Real MIMIC-IV data requires credentialing and cannot be publicly shared. This synthetic data can be distributed with the project.

2. **Demo Capabilities**: Allow new users to try out the RAG system without requiring MIMIC-IV access.

3. **Development Testing**: Provide a consistent dataset for development and testing of the system.

## Generated Data

The generator creates the following synthetic datasets:

- **patients_synthetic.csv**: Basic patient demographic information
- **admissions_synthetic.csv**: Hospital admission records
- **diagnoses_synthetic.csv**: Patient diagnoses with ICD codes
- **procedures_synthetic.csv**: Medical procedures performed
- **labevents_synthetic.csv**: Laboratory test results
- **medications_synthetic.csv**: Medication administration records
- **chunked_docs_synthetic.pkl**: Pre-processed document chunks for RAG system

## How to Use

1. **Generate the data**:

```bash
python -m synthetic_data.synthetic_data_generator
```

2. **Automatic Integration**:

The system will automatically use this synthetic data if real MIMIC-IV data is not available. No configuration changes are needed.

## Data Characteristics

- 100 synthetic patients
- 150 hospital admissions
- Realistic medical conditions and treatments
- Statistically plausible laboratory values
- Properly structured for the RAG pipeline

## Important Note

This data is completely fictitious and should NOT be used for any clinical decision-making or real medical research. It is intended solely for demonstration and development of the RAG system.
