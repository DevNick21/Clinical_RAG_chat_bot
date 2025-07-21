"""
synthetic_data_generator.py

Generates synthetic medical data that mimics the structure of MIMIC-IV,
but with entirely fictitious patient information suitable for public sharing.
"""

import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import pickle
from pathlib import Path

# Ensure reproducibility
np.random.seed(42)
random.seed(42)

# Constants for data generation
NUM_PATIENTS = 100
NUM_ADMISSIONS = 150
START_DATE = datetime(2019, 1, 1)
END_DATE = datetime(2022, 12, 31)

# Lists for synthetic data generation
GENDERS = ['M', 'F']
ADMISSION_TYPES = ['EMERGENCY', 'ELECTIVE',
                   'URGENT', 'NEWBORN', 'DIRECT ADMISSION']
ADMISSION_LOCATIONS = ['EMERGENCY ROOM', 'CLINIC REFERRAL',
                       'TRANSFER FROM HOSPITAL', 'PHYSICIAN REFERRAL']
DISCHARGE_LOCATIONS = ['HOME', 'SKILLED NURSING FACILITY',
                       'REHABILITATION', 'LONG TERM CARE HOSPITAL', 'EXPIRED']
INSURANCE = ['Medicare', 'Medicaid', 'Private', 'Self Pay']
LANGUAGES = ['ENGLISH', 'SPANISH', 'CHINESE', 'FRENCH', 'ARABIC', 'OTHER']
MARITAL_STATUS = ['SINGLE', 'MARRIED', 'DIVORCED', 'WIDOWED', 'SEPARATED']
ETHNICITIES = ['WHITE', 'BLACK', 'HISPANIC',
               'ASIAN', 'NATIVE AMERICAN', 'OTHER']
RELIGIONS = ['CATHOLIC', 'PROTESTANT', 'JEWISH',
             'MUSLIM', 'HINDU', 'BUDDHIST', 'ATHEIST', 'OTHER']
SERVICES = ['MED', 'SURG', 'EMED', 'NMED', 'NSURG', 'TSURG', 'PSYCH', 'OBS']
PROCEDURES = [
    'Chest X-ray', 'ECG', 'CT Scan', 'MRI', 'Endoscopy', 'Colonoscopy',
    'Ultrasound', 'Blood Transfusion', 'Mechanical Ventilation', 'Hemodialysis'
]
DIAGNOSES = [
    'Hypertension', 'Diabetes Type 2', 'Asthma', 'COPD', 'Pneumonia',
    'Heart Failure', 'Atrial Fibrillation', 'Acute Kidney Injury',
    'Urinary Tract Infection', 'Sepsis', 'Stroke', 'Myocardial Infarction',
    'Major Depressive Disorder', 'Anxiety Disorder', 'Alzheimer\'s Disease'
]

LAB_TESTS = [
    {'name': 'Hemoglobin', 'unit': 'g/dL', 'range': (8, 18)},
    {'name': 'White Blood Cell Count', 'unit': 'K/uL', 'range': (3, 15)},
    {'name': 'Platelets', 'unit': 'K/uL', 'range': (100, 400)},
    {'name': 'Sodium', 'unit': 'mEq/L', 'range': (130, 150)},
    {'name': 'Potassium', 'unit': 'mEq/L', 'range': (3.0, 5.5)},
    {'name': 'Chloride', 'unit': 'mEq/L', 'range': (95, 110)},
    {'name': 'Bicarbonate', 'unit': 'mEq/L', 'range': (20, 30)},
    {'name': 'BUN', 'unit': 'mg/dL', 'range': (5, 30)},
    {'name': 'Creatinine', 'unit': 'mg/dL', 'range': (0.5, 2.5)},
    {'name': 'Glucose', 'unit': 'mg/dL', 'range': (60, 200)}
]

MEDICATIONS = [
    'Lisinopril', 'Metoprolol', 'Atorvastatin', 'Metformin', 'Albuterol',
    'Furosemide', 'Insulin', 'Aspirin', 'Levothyroxine', 'Amlodipine',
    'Omeprazole', 'Hydrochlorothiazide', 'Ibuprofen', 'Acetaminophen', 'Gabapentin'
]


def random_date(start_date, end_date):
    """Generate a random date between start_date and end_date"""
    time_between = end_date - start_date
    days_between = time_between.days
    random_days = random.randint(0, days_between)
    return start_date + timedelta(days=random_days)


def create_synthetic_data():
    """Generate synthetic data mimicking MIMIC-IV structure"""
    output_dir = Path("synthetic_data")
    output_dir.mkdir(exist_ok=True)

    # Generate patient data
    patient_ids = [f"P{i:06d}" for i in range(1, NUM_PATIENTS + 1)]
    patients = pd.DataFrame({
        'subject_id': patient_ids,
        'gender': [random.choice(GENDERS) for _ in range(NUM_PATIENTS)],
        'anchor_age': [random.randint(18, 90) for _ in range(NUM_PATIENTS)],
        'anchor_year': [random.randint(2019, 2022) for _ in range(NUM_PATIENTS)],
        'dod': [random_date(START_DATE, END_DATE) if random.random() < 0.1 else None for _ in range(NUM_PATIENTS)]
    })

    # Generate admissions data
    admission_ids = [f"A{i:06d}" for i in range(1, NUM_ADMISSIONS + 1)]
    admissions = []

    for i in range(NUM_ADMISSIONS):
        patient_id = random.choice(patient_ids)
        admit_time = random_date(START_DATE, END_DATE)
        los_days = random.randint(1, 30)  # Length of stay
        discharge_time = admit_time + timedelta(days=los_days)

        admissions.append({
            'hadm_id': admission_ids[i],
            'subject_id': patient_id,
            'admittime': admit_time,
            'dischtime': discharge_time,
            'deathtime': discharge_time if random.random() < 0.05 else None,
            'admission_type': random.choice(ADMISSION_TYPES),
            'admission_location': random.choice(ADMISSION_LOCATIONS),
            'discharge_location': random.choice(DISCHARGE_LOCATIONS),
            'insurance': random.choice(INSURANCE),
            'language': random.choice(LANGUAGES),
            'marital_status': random.choice(MARITAL_STATUS),
            'ethnicity': random.choice(ETHNICITIES),
            'edregtime': admit_time - timedelta(hours=random.randint(1, 12)) if random.random() < 0.7 else None,
            'edouttime': admit_time - timedelta(minutes=random.randint(30, 300)) if random.random() < 0.7 else None,
            'hospital_expire_flag': 1 if random.random() < 0.05 else 0,
            'religion': random.choice(RELIGIONS)
        })

    admissions_df = pd.DataFrame(admissions)

    # Generate diagnoses data
    diagnoses = []
    for i, hadm_id in enumerate(admissions_df['hadm_id']):
        num_diagnoses = random.randint(1, 7)
        for j in range(num_diagnoses):
            diagnoses.append({
                'subject_id': admissions_df.loc[i, 'subject_id'],
                'hadm_id': hadm_id,
                'seq_num': j + 1,
                'icd_code': f"D{random.randint(1, 99):02d}.{random.randint(1, 9)}",
                'icd_version': 10,
                'diagnosis': random.choice(DIAGNOSES)
            })

    diagnoses_df = pd.DataFrame(diagnoses)

    # Generate procedures data
    procedures = []
    for i, hadm_id in enumerate(admissions_df['hadm_id']):
        num_procedures = random.randint(0, 5)
        for j in range(num_procedures):
            procedures.append({
                'subject_id': admissions_df.loc[i, 'subject_id'],
                'hadm_id': hadm_id,
                'proc_seq_num': j + 1,
                'chartdate': admissions_df.loc[i, 'admittime'] + timedelta(days=random.randint(0, 5)),
                'procedure': random.choice(PROCEDURES),
                'icd_code': f"P{random.randint(1, 99):02d}.{random.randint(1, 9)}",
                'icd_version': 10
            })

    procedures_df = pd.DataFrame(procedures)

    # Generate lab events data
    lab_events = []
    lab_ids = [f"L{i:06d}" for i in range(1, len(LAB_TESTS) + 1)]

    for i, hadm_id in enumerate(admissions_df['hadm_id']):
        admit_time = admissions_df.loc[i, 'admittime']
        discharge_time = admissions_df.loc[i, 'dischtime']
        num_labs = random.randint(3, 10)

        for j in range(num_labs):
            lab_test = random.choice(LAB_TESTS)
            min_val, max_val = lab_test['range']
            abnormal = random.random() < 0.2

            # Generate more extreme values for abnormal results
            if abnormal:
                # Either too low or too high
                factor = random.choice([0.7, 1.3])
                value = min_val * factor if factor < 1 else max_val * factor
            else:
                value = random.uniform(min_val, max_val)

            lab_events.append({
                'subject_id': admissions_df.loc[i, 'subject_id'],
                'hadm_id': hadm_id,
                'lab_id': lab_ids[LAB_TESTS.index(lab_test)],
                'charttime': admit_time + timedelta(hours=random.randint(1, (discharge_time - admit_time).days * 24)),
                'test_name': lab_test['name'],
                'value': round(value, 2),
                'valuenum': round(value, 2),
                'valueuom': lab_test['unit'],
                'flag': 'abnormal' if abnormal else 'normal'
            })

    lab_events_df = pd.DataFrame(lab_events)

    # Generate medication data
    medication_data = []
    for i, hadm_id in enumerate(admissions_df['hadm_id']):
        admit_time = admissions_df.loc[i, 'admittime']
        discharge_time = admissions_df.loc[i, 'dischtime']
        num_meds = random.randint(2, 8)

        for j in range(num_meds):
            med_name = random.choice(MEDICATIONS)
            start_time = admit_time + timedelta(hours=random.randint(0, 24))
            end_time = min(
                start_time + timedelta(days=random.randint(1, 10)), discharge_time)

            medication_data.append({
                'subject_id': admissions_df.loc[i, 'subject_id'],
                'hadm_id': hadm_id,
                'med_id': f"M{MEDICATIONS.index(med_name):03d}",
                'starttime': start_time,
                'endtime': end_time,
                'drug': med_name,
                'route': random.choice(['IV', 'PO', 'IM', 'SC', 'ORAL']),
                'dose': f"{random.randint(1, 1000)} {random.choice(['mg', 'mcg', 'g', 'mL'])}"
            })

    medications_df = pd.DataFrame(medication_data)

    # Save all dataframes to CSV
    patients.to_csv(output_dir / "patients_synthetic.csv", index=False)
    admissions_df.to_csv(output_dir / "admissions_synthetic.csv", index=False)
    diagnoses_df.to_csv(output_dir / "diagnoses_synthetic.csv", index=False)
    procedures_df.to_csv(output_dir / "procedures_synthetic.csv", index=False)
    lab_events_df.to_csv(output_dir / "labevents_synthetic.csv", index=False)
    medications_df.to_csv(
        output_dir / "medications_synthetic.csv", index=False)

    # Create a preprocessed "chunked docs" pickle similar to what the real pipeline might create
    chunked_docs = create_synthetic_chunked_docs(patients, admissions_df, diagnoses_df, procedures_df,
                                                 lab_events_df, medications_df)

    with open(output_dir / "chunked_docs_synthetic.pkl", 'wb') as f:
        pickle.dump(chunked_docs, f)

    print(f"Synthetic data created successfully in {output_dir}!")
    return {
        "patients": patients,
        "admissions": admissions_df,
        "diagnoses": diagnoses_df,
        "procedures": procedures_df,
        "lab_events": lab_events_df,
        "medications": medications_df,
        "chunked_docs": chunked_docs
    }


def create_synthetic_chunked_docs(patients, admissions, diagnoses, procedures, lab_events, medications):
    """Create synthetic document chunks for RAG processing"""
    chunked_docs = []

    # For each admission, create several document chunks
    for idx, admission in admissions.iterrows():
        hadm_id = admission['hadm_id']
        patient_id = admission['subject_id']

        # Get patient info
        patient = patients[patients['subject_id'] == patient_id].iloc[0]

        # Get related data
        patient_diagnoses = diagnoses[diagnoses['hadm_id'] == hadm_id]
        patient_procedures = procedures[procedures['hadm_id'] == hadm_id]
        patient_labs = lab_events[lab_events['hadm_id'] == hadm_id]
        patient_meds = medications[medications['hadm_id'] == hadm_id]

        # Create demographic chunk
        demographic_text = f"""
        Patient ID: {patient_id}
        Admission ID: {hadm_id}
        Gender: {patient['gender']}
        Age at admission: {patient['anchor_age']}
        Ethnicity: {admission['ethnicity']}
        Marital Status: {admission['marital_status']}
        Religion: {admission['religion']}
        Language: {admission['language']}
        Insurance: {admission['insurance']}
        Admission Type: {admission['admission_type']}
        Admission Location: {admission['admission_location']}
        Admission Date: {admission['admittime']}
        Discharge Date: {admission['dischtime']}
        Length of Stay: {(admission['dischtime'] - admission['admittime']).days} days
        """

        chunked_docs.append({
            'content': demographic_text.strip(),
            'metadata': {
                'hadm_id': hadm_id,
                'subject_id': patient_id,
                'section': 'demographics'
            }
        })

        # Create diagnoses chunk
        if not patient_diagnoses.empty:
            diagnoses_text = f"Patient ID: {patient_id}\nAdmission ID: {hadm_id}\n\nDiagnoses:\n"
            for _, diag in patient_diagnoses.iterrows():
                diagnoses_text += f"- {diag['diagnosis']} (ICD-{diag['icd_version']} Code: {diag['icd_code']})\n"

            chunked_docs.append({
                'content': diagnoses_text.strip(),
                'metadata': {
                    'hadm_id': hadm_id,
                    'subject_id': patient_id,
                    'section': 'diagnoses'
                }
            })

        # Create procedures chunk
        if not patient_procedures.empty:
            procedures_text = f"Patient ID: {patient_id}\nAdmission ID: {hadm_id}\n\nProcedures:\n"
            for _, proc in patient_procedures.iterrows():
                procedures_text += f"- {proc['procedure']} (Date: {proc['chartdate']}, ICD-{proc['icd_version']} Code: {proc['icd_code']})\n"

            chunked_docs.append({
                'content': procedures_text.strip(),
                'metadata': {
                    'hadm_id': hadm_id,
                    'subject_id': patient_id,
                    'section': 'procedures'
                }
            })

        # Create lab results chunk
        if not patient_labs.empty:
            labs_text = f"Patient ID: {patient_id}\nAdmission ID: {hadm_id}\n\nLab Results:\n"
            for _, lab in patient_labs.iterrows():
                labs_text += f"- {lab['test_name']}: {lab['value']} {lab['valueuom']} (Date: {lab['charttime']}, Status: {lab['flag']})\n"

            chunked_docs.append({
                'content': labs_text.strip(),
                'metadata': {
                    'hadm_id': hadm_id,
                    'subject_id': patient_id,
                    'section': 'lab_results'
                }
            })

        # Create medications chunk
        if not patient_meds.empty:
            meds_text = f"Patient ID: {patient_id}\nAdmission ID: {hadm_id}\n\nMedications:\n"
            for _, med in patient_meds.iterrows():
                meds_text += f"- {med['drug']} {med['dose']} ({med['route']}, Started: {med['starttime']}, Ended: {med['endtime']})\n"

            chunked_docs.append({
                'content': meds_text.strip(),
                'metadata': {
                    'hadm_id': hadm_id,
                    'subject_id': patient_id,
                    'section': 'medications'
                }
            })

        # Create summary note chunk
        summary_text = f"""
        Patient ID: {patient_id}
        Admission ID: {hadm_id}

        CLINICAL SUMMARY:
        {patient['anchor_age']} year old {patient['gender']} patient admitted for {random.choice(DIAGNOSES).lower()}.
        Patient was admitted on {admission['admittime']} via {admission['admission_location'].lower()}.

        Past Medical History:
        {', '.join(random.sample(DIAGNOSES, random.randint(1, 3)))}

        Hospital Course:
        Patient was treated with {', '.join(random.sample(MEDICATIONS, random.randint(2, 4)))}.
        {random.choice(['Patient responded well to treatment.', 'Patient showed moderate improvement with treatment.', 'Patient had complications during treatment but eventually stabilized.'])}

        Discharge Disposition:
        Patient was discharged to {admission['discharge_location'].lower()} after {(admission['dischtime'] - admission['admittime']).days} days of hospitalization.

        Discharge Medications:
        {', '.join(random.sample(MEDICATIONS, random.randint(2, 5)))}
        """

        chunked_docs.append({
            'content': summary_text.strip(),
            'metadata': {
                'hadm_id': hadm_id,
                'subject_id': patient_id,
                'section': 'summary'
            }
        })

    return chunked_docs


if __name__ == "__main__":
    create_synthetic_data()
