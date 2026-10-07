#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent
REPORT_ROOT = PROJECT_ROOT / "reports"


def describe(filename: str) -> str:
    name = filename.lower()
    descriptions = {
        "adni1_": "ADNI MRI image metadata, demographics, visit, acquisition, and diagnostic group",
        "harp_data": "HarP hippocampal-label image metadata",
        "adsxlist": "reported diagnoses, symptoms, and medical conditions",
        "faq": "Functional Activities Questionnaire items and totals",
        "fnihbc": "plasma/blood biomarker assay results and laboratory metadata",
        "labdata": "clinical laboratory measurements",
        "labtests": "laboratory test definitions and result metadata",
        "medhist": "medical history and comorbidities",
        "mmse": "Mini-Mental State Examination items and total score",
        "mrifind": "clinical MRI findings",
        "neuropath": "neuropathology and post-mortem findings",
        "physical": "physical examination findings",
        "study_entry": "study enrollment and entry records",
        "vitals": "vital signs and anthropometrics",
        "adverse_": "ADNI adverse-event reports, severity, outcomes, seriousness, and treatment relationship",
        "npstatus": "ADNI neuropathology consent and decision status by visit",
        "datadic": "ADNI data dictionary with field definitions, codes, units, and mapping notes",
        "ptdemog_06oct": "ADNI participant demographics, education, race/ethnicity, language, and living situation",
        "recadv": "ADNI adverse-event history and hospitalization records",
        "registry_07oct": "ADNI longitudinal visit registry, participant status, visit type, and rescreening fields",
        "roster": "ADNI participant roster and site/phase provenance",
        "studysum": "ADNI study inclusion, exclusion, disposition, and primary/secondary reason summary",
        "treatdis": "ADNI treatment discontinuation and follow-up status",
        "all_subjects_cdr": "ADNI Clinical Dementia Rating domains, global score, and sum of boxes",
        "dxsum": "ADNI visit-level diagnostic summary and dementia etiology fields",
        "ucsffsx7": "ADNI UCSF FreeSurfer 7 regional morphometry and quality-control measures",
        "apoeres": "APOE genotype results",
        "av45meta": "florbetapir/AV45 PET acquisition metadata",
        "bslcheck": "AIBL baseline eligibility/check fields",
        "aibl_cdr": "Clinical Dementia Rating global score by visit",
        "flutemeta": "flutemetamol PET acquisition metadata",
        "aibl_labdata": "AIBL laboratory and blood measurements",
        "aibl_medhist": "AIBL medical history",
        "aibl_mmse": "AIBL MMSE score by visit",
        "mri3meta": "AIBL 3T MRI visit/acquisition metadata",
        "mrimeta": "AIBL MRI visit/acquisition metadata",
        "navmeta": "NAV4694 PET acquisition metadata",
        "neurobat": "AIBL neuropsychological battery summary",
        "pdxconv": "AIBL current diagnosis and diagnosis indicator fields",
        "pibmeta": "PiB PET acquisition metadata",
        "ptdemog": "AIBL participant demographics",
        "registry": "AIBL registry scaffold; currently empty",
        "visits": "AIBL visit-code lookup",
        "oasis_2": "OASIS-2 demographics, diagnosis group, MMSE, CDR, and MRI-derived measures",
        "oas3": "OASIS-2 demographics, diagnosis group, MMSE, CDR, and MRI-derived measures",
        "oasis3_demographics": "OASIS-3 subject-level demographics, education, race/ethnicity, handedness, and APOE",
        "participant_demo": "OASIS-3 visit-level participant demographics and living situation",
        "cs_demo": "OASIS-3 informant demographics",
        "udsa3": "OASIS-3 family history",
        "med_codes": "OASIS-3 detailed medication codes",
        "med_names": "OASIS-3 medication names/classes",
        "health_history": "OASIS-3 health history and comorbidities",
        "physical_eval": "OASIS-3 physical evaluation",
        "his_cvd": "OASIS-3 Hachinski/vascular disease evaluation",
        "udsb3": "OASIS-3 UPDRS/Parkinsonian evaluation",
        "udsb4_cdr": "OASIS-3 CDR domains, CDR sum/global, MMSE, and diagnosis codes",
        "npiq": "OASIS-3 Neuropsychiatric Inventory Questionnaire",
        "gds": "OASIS-3 Geriatric Depression Scale",
        "faq_fas": "OASIS-3 functional activities questionnaire items",
        "neuro_exam": "OASIS-3 neurological examination findings",
        "symptoms": "OASIS-3 clinician judgment of symptoms",
        "udsd1_diagnoses": "OASIS-3 cognitive status, MCI/dementia etiologies, biomarkers, and psychiatric diagnoses",
        "med_conditions": "OASIS-3 medical conditions",
        "cognitive_assessments": "OASIS-3 psychometric and cognitive assessment scores",
        "braak_tauopathy": "OASIS-3 tau PET Braak-region summary",
        "amyloid_centiloid": "OASIS-3 amyloid PET Centiloid values",
        "oasis_master_dataset": "existing local OASIS MRI-clinical joined subset",
        "dictionary": "OASIS-3 clinical and psychometrics data dictionary workbook",
    }
    for token, description in descriptions.items():
        if token in name:
            return description
    return "clinical or imaging metadata table"


def load_csv(path: Path) -> tuple[pd.DataFrame, str]:
    try:
        return pd.read_csv(path, low_memory=False), ""
    except pd.errors.ParserError as error:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
            rows = list(csv.reader(stream))
        header = rows[0]
        data_rows = rows[1:]
        maximum_width = max([len(header), *(len(row) for row in data_rows)])
        extra_count = maximum_width - len(header)
        columns = header + [f"__extra_field_{number}" for number in range(1, extra_count + 1)]
        normalized = [row[:maximum_width] + [""] * (maximum_width - len(row)) for row in data_rows]
        frame = pd.DataFrame(normalized, columns=columns)
        affected = sum(len(row) != len(header) for row in data_rows)
        warning = (
            f"strict CSV parse failed; preserved {affected} nonstandard-width rows in "
            f"{extra_count} synthetic extra-field column(s): {str(error).splitlines()[0]}"
        )
        return frame, warning


file_rows: list[dict[str, object]] = []
column_rows: list[dict[str, object]] = []
dataset_bases = {"adni": "ADNI", "aibl": "AIBL", "oasis": "OASIS"}

for dataset, raw_base in dataset_bases.items():
    clinical_root = DATA_ROOT / "Processed" / "Clinical" / dataset
    manifest_path = clinical_root / f"{dataset}_clinical_manifest.csv"
    manifest = pd.read_csv(manifest_path, low_memory=False).drop_duplicates("sha256", keep="first")
    for record in manifest.itertuples(index=False):
        copied_path = clinical_root / str(record.copied_relative_path)
        raw_source = DATA_ROOT / raw_base / str(record.source_relative_path)
        common = {
            "dataset": dataset.upper(),
            "raw_source_path": str(raw_source.relative_to(DATA_ROOT)),
            "processed_copy_path": str(copied_path.relative_to(DATA_ROOT)),
            "description": describe(copied_path.name),
            "file_type": str(record.file_type),
            "sha256": str(record.sha256),
        }
        if str(record.file_type).lower() != "csv":
            file_rows.append({
                **common,
                "manifest_row_count": record.row_count,
                "parsed_row_count": pd.NA,
                "column_count": record.column_count,
                "total_cells": pd.NA,
                "blank_na_cells": pd.NA,
                "blank_na_percent": pd.NA,
                "sentinel_minus4_cells": pd.NA,
                "parse_warning": "workbook retained as a source data dictionary; column missingness not computed",
                "column_names": "",
            })
            continue

        frame, warning = load_csv(copied_path)
        total_cells = int(frame.size)
        blank_count = int(frame.isna().sum().sum())
        as_text = frame.astype(str)
        minus4_count = int(as_text.isin(["-4", "-4.0"]).sum().sum())
        if pd.notna(record.row_count) and int(record.row_count) != len(frame):
            difference = int(record.row_count) - len(frame)
            warning = (warning + "; " if warning else "") + f"manifest has {int(record.row_count)} rows; parser retained {len(frame)} ({difference} difference)"
        file_rows.append({
            **common,
            "manifest_row_count": int(record.row_count) if pd.notna(record.row_count) else pd.NA,
            "parsed_row_count": len(frame),
            "column_count": len(frame.columns),
            "total_cells": total_cells,
            "blank_na_cells": blank_count,
            "blank_na_percent": round(100.0 * blank_count / total_cells, 4) if total_cells else 0.0,
            "sentinel_minus4_cells": minus4_count,
            "parse_warning": warning,
            "column_names": " | ".join(map(str, frame.columns)),
        })
        for position, column in enumerate(frame.columns, start=1):
            missing = int(frame[column].isna().sum())
            non_missing = int(frame[column].notna().sum())
            minus4 = int(frame[column].astype(str).isin(["-4", "-4.0"]).sum())
            column_rows.append({
                **common,
                "parsed_row_count": len(frame),
                "column_position": position,
                "column_name": str(column),
                "inferred_dtype": str(frame[column].dtype),
                "non_missing_count": non_missing,
                "blank_na_count": missing,
                "blank_na_percent": round(100.0 * missing / len(frame), 4) if len(frame) else pd.NA,
                "sentinel_minus4_count": minus4,
                "parse_warning": warning,
            })

REPORT_ROOT.mkdir(parents=True, exist_ok=True)
for frame, filename in (
    (pd.DataFrame(file_rows), "clinical_file_summary.csv"),
    (pd.DataFrame(column_rows), "clinical_column_missingness.csv"),
):
    destination = REPORT_ROOT / filename
    temporary = destination.with_name(f".{destination.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    print(destination, len(frame))
