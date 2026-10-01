"""Build the demo hospital database (SQLite by default, or MySQL via --url).

Creates 6 related tables with deterministic synthetic data so evaluation
results are reproducible:

    departments, doctors, patients, admissions, diagnoses, prescriptions

Usage:
    python data/build_database.py                      # -> data/hospital.db (SQLite)
    python data/build_database.py --url mysql+pymysql://user:pw@localhost:3306/hospital
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from sqlalchemy import (
    Column, Date, ForeignKey, Integer, MetaData, Numeric, String, Table,
    create_engine, insert,
)

DEFAULT_SQLITE = Path(__file__).resolve().parent / "hospital.db"
DESCRIPTIONS_FILE = Path(__file__).resolve().parent / "schema_descriptions.json"

metadata = MetaData()

departments = Table(
    "departments", metadata,
    Column("department_id", Integer, primary_key=True),
    Column("name", String(60), nullable=False),
    Column("floor", Integer, nullable=False),
    Column("bed_capacity", Integer, nullable=False),
    comment="Hospital clinical departments and their bed capacity.",
)

doctors = Table(
    "doctors", metadata,
    Column("doctor_id", Integer, primary_key=True),
    Column("first_name", String(40), nullable=False),
    Column("last_name", String(40), nullable=False),
    Column("specialty", String(60), nullable=False),
    Column("department_id", Integer, ForeignKey("departments.department_id"), nullable=False),
    Column("years_experience", Integer, nullable=False),
    comment="Attending physicians, their specialty and home department.",
)

patients = Table(
    "patients", metadata,
    Column("patient_id", Integer, primary_key=True),
    Column("first_name", String(40), nullable=False),
    Column("last_name", String(40), nullable=False),
    Column("gender", String(1), nullable=False, comment="'F' or 'M'"),
    Column("birth_date", Date, nullable=False),
    Column("city", String(40), nullable=False),
    Column("state", String(2), nullable=False),
    Column("insurance_type", String(20), nullable=False,
           comment="Medicare, Medicaid, Private, or Uninsured"),
    comment="Patient demographics and insurance type.",
)

admissions = Table(
    "admissions", metadata,
    Column("admission_id", Integer, primary_key=True),
    Column("patient_id", Integer, ForeignKey("patients.patient_id"), nullable=False),
    Column("doctor_id", Integer, ForeignKey("doctors.doctor_id"), nullable=False),
    Column("department_id", Integer, ForeignKey("departments.department_id"), nullable=False),
    Column("admit_date", Date, nullable=False),
    Column("discharge_date", Date, nullable=False),
    Column("length_of_stay", Integer, nullable=False, comment="Days between admit and discharge"),
    Column("admission_type", String(20), nullable=False, comment="Emergency, Elective, or Urgent"),
    Column("outcome", String(20), nullable=False, comment="Recovered, Improved, Transferred, or Deceased"),
    Column("total_charges", Numeric(10, 2), nullable=False, comment="Billed charges in USD"),
    Column("readmitted_30d", Integer, nullable=False, comment="1 if readmitted within 30 days, else 0"),
    comment="Inpatient hospital stays, one row per admission.",
)

diagnoses = Table(
    "diagnoses", metadata,
    Column("diagnosis_id", Integer, primary_key=True),
    Column("admission_id", Integer, ForeignKey("admissions.admission_id"), nullable=False),
    Column("icd10_code", String(10), nullable=False),
    Column("description", String(120), nullable=False),
    Column("is_primary", Integer, nullable=False, comment="1 for the principal diagnosis of the stay"),
    comment="ICD-10 diagnoses recorded during an admission.",
)

prescriptions = Table(
    "prescriptions", metadata,
    Column("prescription_id", Integer, primary_key=True),
    Column("admission_id", Integer, ForeignKey("admissions.admission_id"), nullable=False),
    Column("drug_name", String(60), nullable=False),
    Column("dose_mg", Integer, nullable=False),
    Column("days_supply", Integer, nullable=False),
    Column("cost", Numeric(8, 2), nullable=False, comment="Medication cost in USD"),
    comment="Medications prescribed during an admission.",
)

# --------------------------------------------------------------------------- data
DEPTS = [
    ("Cardiology", 3, 40), ("Oncology", 4, 35), ("Neurology", 5, 25),
    ("Orthopedics", 2, 30), ("Pulmonology", 3, 20), ("General Surgery", 2, 45),
    ("Pediatrics", 1, 30), ("Emergency Medicine", 1, 50),
]
SPECIALTY = {
    "Cardiology": "Cardiologist", "Oncology": "Oncologist", "Neurology": "Neurologist",
    "Orthopedics": "Orthopedic Surgeon", "Pulmonology": "Pulmonologist",
    "General Surgery": "General Surgeon", "Pediatrics": "Pediatrician",
    "Emergency Medicine": "Emergency Physician",
}
DX = {
    "Cardiology": [("I21.9", "Acute myocardial infarction"), ("I50.9", "Heart failure"), ("I48.91", "Atrial fibrillation")],
    "Oncology": [("C50.919", "Breast cancer"), ("C34.90", "Lung cancer"), ("C56.9", "Ovarian cancer")],
    "Neurology": [("I63.9", "Ischemic stroke"), ("G40.909", "Epilepsy"), ("G35", "Multiple sclerosis")],
    "Orthopedics": [("S72.001A", "Hip fracture"), ("M17.11", "Knee osteoarthritis")],
    "Pulmonology": [("J44.1", "COPD exacerbation"), ("J18.9", "Pneumonia")],
    "General Surgery": [("K35.80", "Acute appendicitis"), ("K80.20", "Gallstones")],
    "Pediatrics": [("J45.909", "Asthma"), ("J21.9", "Bronchiolitis")],
    "Emergency Medicine": [("T78.40XA", "Allergic reaction"), ("R07.9", "Chest pain"), ("E11.65", "Type 2 diabetes with hyperglycemia")],
}
SECONDARY_DX = [("I10", "Essential hypertension"), ("E78.5", "Hyperlipidemia"),
                ("E11.9", "Type 2 diabetes"), ("N18.3", "Chronic kidney disease stage 3")]
DRUGS = [("Metoprolol", 50, 0.4), ("Lisinopril", 10, 0.3), ("Atorvastatin", 40, 0.5),
         ("Metformin", 500, 0.2), ("Amoxicillin", 500, 0.6), ("Heparin", 5000, 2.5),
         ("Ondansetron", 4, 1.2), ("Albuterol", 2, 0.9), ("Prednisone", 20, 0.3),
         ("Morphine", 4, 3.1), ("Levetiracetam", 500, 1.8), ("Carboplatin", 300, 45.0)]
FIRST_F = ["Aisha", "Maria", "Emily", "Priya", "Grace", "Sofia", "Hannah", "Olivia", "Lakshmi", "Chloe", "Zoe", "Ava"]
FIRST_M = ["James", "Rahul", "Michael", "David", "Carlos", "Ethan", "Noah", "Daniel", "Arjun", "Lucas", "Omar", "Ben"]
LAST = ["Smith", "Johnson", "Patel", "Garcia", "Williams", "Brown", "Reddy", "Davis", "Martinez",
        "Wilson", "Anderson", "Thomas", "Lee", "Nguyen", "Clark", "Lewis"]
CITIES = [("Birmingham", "AL"), ("Hoover", "AL"), ("Tuscaloosa", "AL"), ("Huntsville", "AL"),
          ("Montgomery", "AL"), ("Atlanta", "GA"), ("Nashville", "TN")]
INSURANCE = ["Medicare", "Medicaid", "Private", "Private", "Uninsured"]


def generate(seed: int = 42) -> dict[str, list[dict]]:
    rnd = random.Random(seed)
    rows: dict[str, list[dict]] = {t: [] for t in metadata.tables}

    for i, (name, floor, beds) in enumerate(DEPTS, start=1):
        rows["departments"].append(dict(department_id=i, name=name, floor=floor, bed_capacity=beds))

    doc_id = 0
    doc_by_dept: dict[int, list[int]] = {}
    for dept_id, (name, *_rest) in enumerate(DEPTS, start=1):
        for _ in range(rnd.randint(2, 4)):
            doc_id += 1
            doc_by_dept.setdefault(dept_id, []).append(doc_id)
            rows["doctors"].append(dict(
                doctor_id=doc_id,
                first_name=rnd.choice(FIRST_F + FIRST_M), last_name=rnd.choice(LAST),
                specialty=SPECIALTY[name], department_id=dept_id,
                years_experience=rnd.randint(2, 35)))

    for pid in range(1, 301):
        g = rnd.choice("FM")
        city, state = rnd.choice(CITIES)
        rows["patients"].append(dict(
            patient_id=pid, first_name=rnd.choice(FIRST_F if g == "F" else FIRST_M),
            last_name=rnd.choice(LAST), gender=g,
            birth_date=date(1935, 1, 1) + timedelta(days=rnd.randint(0, 365 * 85)),
            city=city, state=state, insurance_type=rnd.choice(INSURANCE)))

    adm_id = dx_id = rx_id = 0
    for _ in range(900):
        adm_id += 1
        dept_id = rnd.randint(1, len(DEPTS))
        dept_name = DEPTS[dept_id - 1][0]
        admit = date(2024, 1, 1) + timedelta(days=rnd.randint(0, 729))
        los = max(1, int(rnd.gammavariate(2.0, 2.2 if dept_name != "Oncology" else 3.5)))
        outcome = rnd.choices(["Recovered", "Improved", "Transferred", "Deceased"], [55, 32, 9, 4])[0]
        rows["admissions"].append(dict(
            admission_id=adm_id, patient_id=rnd.randint(1, 300),
            doctor_id=rnd.choice(doc_by_dept[dept_id]), department_id=dept_id,
            admit_date=admit, discharge_date=admit + timedelta(days=los), length_of_stay=los,
            admission_type=rnd.choices(["Emergency", "Elective", "Urgent"], [50, 30, 20])[0],
            outcome=outcome, total_charges=round(rnd.uniform(2500, 6500) * los, 2),
            readmitted_30d=int(rnd.random() < 0.14)))

        code, desc = rnd.choice(DX[dept_name])
        dx_id += 1
        rows["diagnoses"].append(dict(diagnosis_id=dx_id, admission_id=adm_id,
                                      icd10_code=code, description=desc, is_primary=1))
        for code, desc in rnd.sample(SECONDARY_DX, rnd.randint(0, 2)):
            dx_id += 1
            rows["diagnoses"].append(dict(diagnosis_id=dx_id, admission_id=adm_id,
                                          icd10_code=code, description=desc, is_primary=0))

        for drug, dose, unit_cost in rnd.sample(DRUGS, rnd.randint(1, 4)):
            rx_id += 1
            days = rnd.randint(1, 30)
            rows["prescriptions"].append(dict(
                prescription_id=rx_id, admission_id=adm_id, drug_name=drug, dose_mg=dose,
                days_supply=days, cost=round(unit_cost * days * rnd.uniform(0.8, 1.2), 2)))
    return rows


def build(url: str) -> None:
    engine = create_engine(url)
    metadata.drop_all(engine)
    metadata.create_all(engine)
    data = generate()
    with engine.begin() as conn:
        for table in metadata.sorted_tables:  # parent tables first
            conn.execute(insert(table), data[table.name])
    export_descriptions()
    counts = ", ".join(f"{t}={len(r)}" for t, r in data.items())
    print(f"Built {engine.url.render_as_string(hide_password=True)}  ({counts})")


def export_descriptions(path: Path = DESCRIPTIONS_FILE) -> None:
    """Save table/column comments as JSON. SQLite drops COMMENTs, so the engine
    reads this file to give the LLM business meaning for each table and column."""
    out = {
        t.name: {
            "description": t.comment or "",
            "columns": {c.name: c.comment for c in t.columns if c.comment},
        }
        for t in metadata.sorted_tables
    }
    path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=f"sqlite:///{DEFAULT_SQLITE}",
                   help="SQLAlchemy URL (default: SQLite file in data/)")
    build(p.parse_args().url)
