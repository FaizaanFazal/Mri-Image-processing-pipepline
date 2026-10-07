#!/usr/bin/env python3
"""Build auditable image-level manifests without moving source data."""

from __future__ import annotations

import bisect
import csv
import os
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Manifests"
OUT.mkdir(parents=True, exist_ok=True)

FIELDS = [
    "dataset",
    "record_kind",
    "subject_id",
    "session_id",
    "visit_code",
    "clinical_visit_id",
    "acquisition_date",
    "image_id",
    "scan_id",
    "modality",
    "sequence_description",
    "source_format",
    "dimension",
    "raw_path",
    "source_json_path",
    "source_file_count",
    "existing_3d_path",
    "conversion_status",
    "source_group",
    "group_class",
    "diagnosis",
    "diagnosis_code",
    "cdr_global",
    "cdr_sum",
    "cdr_days_from_scan",
    "mmse",
    "faq_total",
    "sex",
    "age",
    "education",
    "apoe",
    "diagnosis_days_from_scan",
    "clinical_match",
    "clinical_sources",
    "notes",
]


def clean(value) -> str:
    if value is None:
        return ""
    value = str(value).strip().strip("\ufeff")
    return "" if value.lower() in {"nan", "na", "n/a", "none"} else value


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        return [{clean(k): clean(v) for k, v in row.items()} for row in csv.DictReader(handle)]


def empty_row() -> dict[str, str]:
    return {field: "" for field in FIELDS}


def write_rows(filename: str, rows: list[dict[str, str]]) -> None:
    with (OUT / filename).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def valid_rating(value: str) -> bool:
    return clean(value) not in {"", "-4"}


def index_visit_rows(rows, subject_key, visit_keys, value_key=None):
    result = {}
    for row in rows:
        subject = clean(row.get(subject_key))
        if not subject:
            continue
        if value_key and not valid_rating(row.get(value_key, "")):
            continue
        for key in visit_keys:
            visit = clean(row.get(key))
            if visit:
                result[(subject, visit)] = row
    return result


def add_clinical_source(sources: list[str], path: Path) -> None:
    value = rel(path)
    if value not in sources:
        sources.append(value)


def build_adni() -> list[dict[str, str]]:
    base = ROOT / "ADNI"
    image_sources = [
        ("primary_mri", base / "ADNI1_3Yr_1.5T.csv", "nifti_source"),
        ("primary_mri", base / "CustomIds" / "CstomIds_MPRAGE_9_30_2025.csv", "dicom_source"),
        ("label_mask", base / "HARP" / "HarP" / "HarP_data_5_11_2025.csv", "harp_mask"),
    ]

    tables = base / "Tables"
    mmse_path = tables / "All_Subjects_MMSE_21Aug2025.csv"
    faq_path = tables / "All_Subjects_FAQ_21Aug2025.csv"
    entry_path = tables / "All_Subjects_Study_Entry_21Aug2025.csv"
    mmse = index_visit_rows(read_csv(mmse_path), "PTID", ("VISCODE", "VISCODE2"), "MMSCORE")
    faq = index_visit_rows(read_csv(faq_path), "PTID", ("VISCODE", "VISCODE2"), "FAQTOTAL")
    entry = {row["subject_id"]: row for row in read_csv(entry_path) if row.get("subject_id")}

    nifti_by_id = {}
    for root in (base / "ADNI1 3yr 1.5T", base / "HARP"):
        for path in root.rglob("*"):
            if not path.is_file() or not path.name.lower().endswith((".nii", ".nii.gz")):
                continue
            match = re.search(r"(?<![A-Za-z0-9])(I\d+)(?!\d)", str(path))
            if match:
                nifti_by_id[match.group(1)] = path

    dicom_by_id = {}
    for dirpath, _, filenames in os.walk(base / "CustomIds" / "ADNI"):
        count = sum(name.lower().endswith(".dcm") for name in filenames)
        image_id = Path(dirpath).name
        if count and re.fullmatch(r"I\d+", image_id):
            dicom_by_id[image_id] = (Path(dirpath), count)

    final_by_id = {}
    for path in (base / "final").glob("*.nii.gz"):
        match = re.search(r"_(I\d+)_", path.name)
        if match:
            final_by_id[match.group(1)] = path

    rows = []
    harp_manifest_path = base / "HARP" / "HarP" / "HarP_data_5_11_2025.csv"
    harp_manifest_rows = read_csv(harp_manifest_path)
    harp_subject_meta = {
        row["Subject"]: row for row in harp_manifest_rows if row.get("Subject")
    }
    for record_kind, manifest_path, source_kind in image_sources:
        for source in read_csv(manifest_path):
            image_id = clean(source.get("Image Data ID"))
            subject = clean(source.get("Subject"))
            visit = clean(source.get("Visit"))
            row = empty_row()
            row.update(
                dataset="ADNI",
                record_kind=record_kind,
                subject_id=subject,
                session_id=visit,
                visit_code=visit,
                acquisition_date=clean(source.get("Acq Date")),
                image_id=image_id,
                scan_id=image_id,
                modality=clean(source.get("Modality")),
                sequence_description=clean(source.get("Description")),
                group_class=clean(source.get("Group")),
                diagnosis=clean(source.get("Group")),
                sex=clean(source.get("Sex")),
                age=clean(source.get("Age")),
            )
            sources = [rel(manifest_path)]
            mmse_row = mmse.get((subject, visit), {})
            faq_row = faq.get((subject, visit), {})
            entry_row = entry.get(subject, {})
            row["mmse"] = clean(mmse_row.get("MMSCORE"))
            row["faq_total"] = clean(faq_row.get("FAQTOTAL"))
            if not row["group_class"]:
                row["group_class"] = clean(entry_row.get("entry_research_group"))
                row["diagnosis"] = row["group_class"]
            if not row["age"]:
                row["age"] = clean(entry_row.get("entry_age"))
            if mmse_row:
                add_clinical_source(sources, mmse_path)
            if faq_row:
                add_clinical_source(sources, faq_path)
            if entry_row:
                add_clinical_source(sources, entry_path)

            if source_kind == "dicom_source":
                series_path, count = dicom_by_id.get(image_id, (Path(), 0))
                final_path = final_by_id.get(image_id)
                row.update(
                    source_format="DICOM series",
                    dimension="2D slices with 3D counterpart" if final_path else "2D uncombined",
                    raw_path=rel(series_path) if count else "",
                    source_file_count=str(count),
                existing_3d_path=rel(final_path) if final_path else "",
                    conversion_status="existing_3d_preprocessed" if final_path else "uncombined_single_image",
                )
            else:
                path = nifti_by_id.get(image_id)
                row.update(
                    source_format="NIfTI",
                    dimension="3D",
                    raw_path=rel(path) if path else "",
                    source_file_count="1" if path else "0",
                    existing_3d_path=rel(path) if path else "",
                    conversion_status="existing_3d" if path else "source_file_not_found",
                )
                if record_kind == "label_mask":
                    row["notes"] = "Hippocampal label mask; not an MRI intensity input."
            row["clinical_match"] = "image manifest" + (" + visit" if mmse_row or faq_row else "")
            row["source_group"] = clean(source.get("Group"))
            row["clinical_sources"] = ";".join(sources)
            rows.append(row)

    # The supplied HARP image CSV covers only a subset of the masks on disk.
    # Add every remaining mask using the identifiers encoded in its path.
    covered_harp_ids = {
        row["image_id"] for row in rows if row["record_kind"] == "label_mask"
    }
    subject_pattern = re.compile(r"\d{3}_S_\d{4}")
    date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
    for path in sorted((base / "HARP").rglob("*")):
        if not path.is_file() or not path.name.lower().endswith((".nii", ".nii.gz")):
            continue
        image_match = re.search(r"(?<![A-Za-z0-9])(I\d+)(?!\d)", str(path))
        subject_match = subject_pattern.search(str(path))
        date_match = date_pattern.search(str(path))
        if not image_match or not subject_match:
            continue
        image_id = image_match.group(1)
        if image_id in covered_harp_ids:
            continue
        subject = subject_match.group(0)
        acquisition_date = date_match.group(0) if date_match else ""
        source = harp_subject_meta.get(subject, {})
        entry_row = entry.get(subject, {})
        sources = [rel(harp_manifest_path)]
        if entry_row:
            add_clinical_source(sources, entry_path)
        row = empty_row()
        row.update(
            dataset="ADNI",
            record_kind="label_mask",
            subject_id=subject,
            session_id=acquisition_date,
            acquisition_date=acquisition_date,
            image_id=image_id,
            scan_id=image_id,
            modality="MRI label",
            sequence_description="Hippocampal Mask",
            source_format="NIfTI",
            dimension="3D",
            raw_path=rel(path),
            source_file_count="1",
            existing_3d_path=rel(path),
            conversion_status="existing_3d",
            group_class=clean(source.get("Group")) or clean(entry_row.get("entry_research_group")),
            source_group=clean(source.get("Group")),
            diagnosis=clean(source.get("Group")) or clean(entry_row.get("entry_research_group")),
            sex=clean(source.get("Sex")),
            age=clean(source.get("Age")) or clean(entry_row.get("entry_age")),
            clinical_match="subject only",
            clinical_sources=";".join(sources),
            notes="Hippocampal label mask discovered from disk; not an MRI intensity input. The source HARP CSV does not contain this image ID.",
        )
        rows.append(row)
    return rows


def aibl_diagnosis(row: dict[str, str]) -> str:
    checks = [
        ("DXAD", "AD"),
        ("DXMCI", "MCI"),
        ("DXNORM", "Cognitively normal"),
        ("DXPARK", "Parkinson's disease"),
        ("DXOTHDEM", "Other dementia"),
    ]
    return next((label for field, label in checks if clean(row.get(field)) == "1"), "")


def build_aibl() -> list[dict[str, str]]:
    base = ROOT / "AIBL"
    manifest_path = base / "AIBL_3D_2671_9_18_2026.csv"
    clinical = base / "Clinical" / "aibl_19Sep2019" / "Data_extract_3.3.0"
    cdr_path = clinical / "aibl_cdr_01-Jun-2018.csv"
    dx_path = clinical / "aibl_pdxconv_01-Jun-2018.csv"
    mmse_path = clinical / "aibl_mmse_01-Jun-2018.csv"
    demo_path = clinical / "aibl_ptdemog_01-Jun-2018.csv"
    apoe_path = clinical / "aibl_apoeres_01-Jun-2018.csv"

    cdr = index_visit_rows(read_csv(cdr_path), "RID", ("VISCODE",), "CDGLOBAL")
    diagnosis = index_visit_rows(read_csv(dx_path), "RID", ("VISCODE",), "DXCURREN")
    mmse = index_visit_rows(read_csv(mmse_path), "RID", ("VISCODE",), "MMSCORE")
    demo_rows = read_csv(demo_path)
    demo = index_visit_rows(demo_rows, "RID", ("VISCODE",))
    demo_subject = {row["RID"]: row for row in demo_rows if row.get("RID")}
    apoe_rows = read_csv(apoe_path)
    apoe = index_visit_rows(apoe_rows, "RID", ("VISCODE",))
    apoe_subject = {row["RID"]: row for row in apoe_rows if row.get("RID")}

    visit_map = {
        "Base": "bl",
        "18 Mo": "m18",
        "36 Mo": "m36",
        "54 Mo": "m54",
        "72 Mo": "m72",
        "90 Mo": "m90",
        "108 Mo": "m108",
    }

    series_by_id = {}
    roots = [base / "AIBL 3D 2671" / "AIBL", base / "AIBL 3D 2671_dataset" / "AIBL"]
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            count = sum(name.lower().endswith(".dcm") for name in filenames)
            image_id = Path(dirpath).name
            if count and re.fullmatch(r"I\d+", image_id):
                series_by_id[image_id] = (Path(dirpath), count)

    rows = []
    for source in read_csv(manifest_path):
        subject = clean(source.get("Subject"))
        image_id = clean(source.get("Image Data ID"))
        visit_name = clean(source.get("Visit"))
        visit = visit_map.get(visit_name, visit_name)
        cdr_row = cdr.get((subject, visit), {})
        dx_row = diagnosis.get((subject, visit), {})
        mmse_row = mmse.get((subject, visit), {})
        demo_row = demo.get((subject, visit), demo_subject.get(subject, {}))
        apoe_row = apoe.get((subject, visit), apoe_subject.get(subject, {}))
        path, count = series_by_id.get(image_id, (Path(), 0))
        specific_dx = aibl_diagnosis(dx_row)
        sources = [rel(manifest_path)]
        for matched, source_path in (
            (cdr_row, cdr_path),
            (dx_row, dx_path),
            (mmse_row, mmse_path),
            (demo_row, demo_path),
            (apoe_row, apoe_path),
        ):
            if matched:
                add_clinical_source(sources, source_path)
        row = empty_row()
        row.update(
            dataset="AIBL",
            record_kind="primary_mri",
            subject_id=subject,
            session_id=visit,
            visit_code=visit,
            clinical_visit_id=visit,
            acquisition_date=clean(source.get("Acq Date")),
            image_id=image_id,
            scan_id=image_id,
            modality=clean(source.get("Modality")),
            sequence_description=clean(source.get("Description")),
            source_format="DICOM series",
            dimension="2D uncombined; expected 3D after conversion",
            raw_path=rel(path) if count else "",
            source_file_count=str(count),
            conversion_status="needs_dicom_to_nifti",
            source_group=clean(source.get("Group")),
            group_class=specific_dx,
            diagnosis=specific_dx,
            diagnosis_code=clean(dx_row.get("DXCURREN")),
            cdr_global=clean(cdr_row.get("CDGLOBAL")),
            mmse=clean(mmse_row.get("MMSCORE")),
            sex=clean(source.get("Sex")),
            age=clean(source.get("Age")),
            apoe="/".join(filter(None, (clean(apoe_row.get("APGEN1")), clean(apoe_row.get("APGEN2"))))),
            clinical_match="subject + visit" if any((cdr_row, dx_row, mmse_row)) else "subject only",
            clinical_sources=";".join(sources),
            notes="Image manifest Group is 'Patient'; group_class prefers derived diagnosis when available.",
        )
        rows.append(row)
    return rows


def build_oasis2() -> list[dict[str, str]]:
    base = ROOT / "OASIS" / "OASIS2"
    clinical_path = base / "OAsis_2.csv"
    clinical = {row["MRI ID"]: row for row in read_csv(clinical_path) if row.get("MRI ID")}
    rows = []
    for path in sorted((base / "Raw").rglob("*.img")):
        match = re.search(r"(OAS2_\d{4}_MR\d+)", str(path))
        if not match:
            continue
        session = match.group(1)
        source = clinical.get(session, {})
        subject_match = re.search(r"OAS2_\d{4}", session)
        subject = subject_match.group(0) if subject_match else clean(source.get("Subject ID"))
        scan = path.name.removesuffix(".img").removesuffix(".nifti")
        header_path = path.with_suffix(".hdr")
        row = empty_row()
        row.update(
            dataset="OASIS2",
            record_kind="primary_mri",
            subject_id=subject,
            session_id=session,
            visit_code=clean(source.get("Visit")),
            clinical_visit_id=session,
            image_id=session,
            scan_id=scan,
            modality="MRI",
            sequence_description="MPR",
            source_format="Analyze 7.5 (.img/.hdr)",
            dimension="3D",
            raw_path=rel(path),
            source_file_count="2" if header_path.exists() else "1",
            existing_3d_path=rel(path),
            conversion_status="existing_3d_analyze; optional_nifti_conversion",
            group_class=clean(source.get("Group")),
            source_group=clean(source.get("Group")),
            diagnosis=clean(source.get("Group")),
            cdr_global=clean(source.get("CDR")),
            mmse=clean(source.get("MMSE")),
            sex=clean(source.get("Gender")),
            age=clean(source.get("Age")),
            education=clean(source.get("EDUC")),
            clinical_match="direct MRI ID",
            clinical_sources=rel(clinical_path),
            notes="One manifest row per 3D MPR run; .img and .hdr together form one volume.",
        )
        rows.append(row)
    return rows


def sorted_visit_index(rows: list[dict[str, str]], subject_key: str, day_key: str):
    result = defaultdict(list)
    for row in rows:
        subject = clean(row.get(subject_key))
        day = clean(row.get(day_key))
        try:
            numeric_day = int(float(day))
        except ValueError:
            continue
        result[subject].append((numeric_day, row))
    for subject in result:
        result[subject].sort(key=lambda item: item[0])
    return result


def nearest(index, subject: str, day: int):
    visits = index.get(subject, [])
    if not visits:
        return {}, ""
    days = [item[0] for item in visits]
    position = bisect.bisect_left(days, day)
    candidates = []
    if position < len(visits):
        candidates.append(visits[position])
    if position:
        candidates.append(visits[position - 1])
    visit_day, row = min(candidates, key=lambda item: abs(item[0] - day))
    return row, str(abs(visit_day - day))


def oasis3_class(row: dict[str, str]) -> str:
    if clean(row.get("DEMENTED")) == "1":
        return "Dementia"
    if any(clean(value) == "1" for key, value in row.items() if key.startswith("MCI")):
        return "MCI"
    if clean(row.get("NORMCOG")) == "1":
        return "Cognitively normal"
    if clean(row.get("IMPNOMCI")) == "1":
        return "Impaired, not MCI"
    return ""


def build_oasis3() -> list[dict[str, str]]:
    base = ROOT / "OASIS" / "oasis_datta" / "oasis_data"
    manifest_path = base / "oasis_manifest_long.csv"
    canonical = ROOT / "OASIS" / "OASIS3" / "OASIS3_data_files" / "scans"
    cdr_path = canonical / "UDSb4-Form_B4__Global_Staging__CDR__Standard_and_Supplemental" / "resources" / "csv" / "files" / "OASIS3_UDSb4_cdr.csv"
    diagnosis_path = canonical / "UDSd1-Form_D1__Clinician_Diagnosis___Cognitive_Status_and_Dementia" / "resources" / "csv" / "files" / "OASIS3_UDSd1_diagnoses.csv"
    demographics_path = canonical / "demo-demographics" / "resources" / "csv" / "files" / "OASIS3_demographics.csv"
    cdr_index = sorted_visit_index(read_csv(cdr_path), "OASISID", "days_to_visit")
    diagnosis_index = sorted_visit_index(read_csv(diagnosis_path), "OASISID", "days_to_visit")
    demographics = {row["OASISID"]: row for row in read_csv(demographics_path) if row.get("OASISID")}

    def make_row(source, record_kind="primary_mri"):
        subject = clean(source.get("subject_id"))
        session = clean(source.get("session_id"))
        day_text = clean(source.get("day")).lstrip("d")
        try:
            day = int(day_text)
        except ValueError:
            match = re.search(r"_d(\d+)", session)
            day = int(match.group(1)) if match else 0
        cdr_row, cdr_diff = nearest(cdr_index, subject, day)
        dx_row, dx_diff = nearest(diagnosis_index, subject, day)
        demo = demographics.get(subject, {})
        modality = clean(source.get("modality"))
        nifti_value = clean(source.get("nii_path"))
        json_value = clean(source.get("json_path"))
        if nifti_value.startswith("/data/"):
            nifti_path = base / nifti_value.removeprefix("/data/")
        else:
            nifti_path = Path(nifti_value)
        if json_value.startswith("/data/"):
            json_path = base / json_value.removeprefix("/data/")
        else:
            json_path = Path(json_value) if json_value else Path()
        group = oasis3_class(dx_row)
        diagnosis = clean(cdr_row.get("dx1")) or group
        gender_code = clean(demo.get("GENDER"))
        sex = {"1": "M", "2": "F"}.get(gender_code, gender_code)
        is_4d = modality.lower() == "dwi" or nifti_path.name.lower().endswith("_dwi.nii.gz")
        row = empty_row()
        row.update(
            dataset="OASIS3",
            record_kind=record_kind,
            subject_id=subject,
            session_id=session,
            visit_code=clean(source.get("day")),
            clinical_visit_id=clean(cdr_row.get("OASIS_session_label")),
            acquisition_date=clean(source.get("day")),
            image_id=session,
            scan_id="/".join(filter(None, (clean(source.get("anat_dir")), clean(source.get("run")), clean(source.get("acq"))))),
            modality=modality,
            sequence_description=modality,
            source_format="NIfTI",
            dimension="4D" if is_4d else "3D",
            raw_path=rel(nifti_path),
            source_json_path=rel(json_path) if json_value else "",
            source_file_count="1",
            existing_3d_path=rel(nifti_path),
            conversion_status="existing_4d" if is_4d else "existing_3d",
            group_class=group,
            diagnosis=diagnosis,
            diagnosis_code=clean(cdr_row.get("dx1_code")),
            cdr_global=clean(cdr_row.get("CDRTOT")),
            cdr_sum=clean(cdr_row.get("CDRSUM")),
            cdr_days_from_scan=cdr_diff,
            mmse=clean(cdr_row.get("MMSE")),
            sex=sex,
            age=clean(cdr_row.get("age at visit")) or clean(dx_row.get("age at visit")) or clean(demo.get("AgeatEntry")),
            education=clean(demo.get("EDUC")),
            apoe=clean(demo.get("APOE")),
            diagnosis_days_from_scan=dx_diff,
            clinical_match="nearest clinical visit by subject and day" if cdr_row or dx_row else "subject demographics only",
            clinical_sources=";".join((rel(manifest_path), rel(cdr_path), rel(diagnosis_path), rel(demographics_path))),
            notes="CDR and diagnosis are nearest-visit matches; inspect day-difference columns before analysis.",
        )
        return row

    rows = [make_row(source) for source in read_csv(manifest_path)]
    listed = {row["raw_path"] for row in rows}
    subject_pattern = re.compile(r"OAS3\d{4}")
    session_pattern = re.compile(r"OAS3\d{4}_MR_d\d+")
    derivatives = base / "derivatives"
    for path in sorted(derivatives.rglob("*.nii.gz")):
        if rel(path) in listed:
            continue
        text = str(path)
        subject_match = subject_pattern.search(text)
        session_match = session_pattern.search(text)
        if not subject_match or not session_match:
            continue
        session = session_match.group(0)
        day_match = re.search(r"_d(\d+)", session)
        source = {
            "subject_id": subject_match.group(0),
            "session_id": session,
            "day": f"d{day_match.group(1)}" if day_match else "",
            "modality": "derived_dti_map",
            "anat_dir": rel(path.parent),
            "run": path.stem,
            "nii_path": str(path),
        }
        row = make_row(source, "derived_map")
        row["raw_path"] = rel(path)
        row["existing_3d_path"] = rel(path)
        row["notes"] = "Existing derivative map; not a raw MRI input. " + row["notes"]
        rows.append(row)
    return rows


def write_summary(groups: dict[str, list[dict[str, str]]]) -> None:
    fields = [
        "dataset",
        "manifest_rows",
        "unique_subjects",
        "primary_mri_rows",
        "label_or_derived_rows",
        "rows_with_group_class",
        "rows_with_diagnosis",
        "rows_with_cdr",
        "rows_with_mmse",
        "rows_needing_conversion",
        "rows_missing_raw_path",
    ]
    with (OUT / "manifest_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset, rows in groups.items():
            writer.writerow(
                {
                    "dataset": dataset,
                    "manifest_rows": len(rows),
                    "unique_subjects": len({row["subject_id"] for row in rows if row["subject_id"]}),
                    "primary_mri_rows": sum(row["record_kind"] == "primary_mri" for row in rows),
                    "label_or_derived_rows": sum(row["record_kind"] != "primary_mri" for row in rows),
                    "rows_with_group_class": sum(bool(row["group_class"]) for row in rows),
                    "rows_with_diagnosis": sum(bool(row["diagnosis"]) for row in rows),
                    "rows_with_cdr": sum(bool(row["cdr_global"]) for row in rows),
                    "rows_with_mmse": sum(bool(row["mmse"]) for row in rows),
                    "rows_needing_conversion": sum("needs_" in row["conversion_status"] for row in rows),
                    "rows_missing_raw_path": sum(not row["raw_path"] for row in rows),
                }
            )


def write_class_summary(groups: dict[str, list[dict[str, str]]]) -> None:
    fields = ["dataset", "class_label", "scan_record_count", "unique_subject_count"]
    with (OUT / "class_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset, rows in groups.items():
            counts = defaultdict(int)
            subjects = defaultdict(set)
            for row in rows:
                if row["record_kind"] != "primary_mri":
                    continue
                label = row["group_class"] or "Unclassified"
                counts[label] += 1
                if row["subject_id"]:
                    subjects[label].add(row["subject_id"])
            for label in sorted(counts, key=lambda value: (value == "Unclassified", value)):
                writer.writerow(
                    {
                        "dataset": dataset,
                        "class_label": label,
                        "scan_record_count": counts[label],
                        "unique_subject_count": len(subjects[label]),
                    }
                )


def main() -> None:
    groups = {
        "ADNI": build_adni(),
        "AIBL": build_aibl(),
        "OASIS2": build_oasis2(),
        "OASIS3": build_oasis3(),
    }
    filenames = {
        "ADNI": "adni_detailed_manifest.csv",
        "AIBL": "aibl_detailed_manifest.csv",
        "OASIS2": "oasis2_detailed_manifest.csv",
        "OASIS3": "oasis3_detailed_manifest.csv",
    }
    for dataset, rows in groups.items():
        write_rows(filenames[dataset], rows)
    combined = []
    for rows in groups.values():
        combined.extend(rows)
    write_rows("all_datasets_detailed_manifest.csv", combined)
    write_summary(groups)
    write_class_summary(groups)
    for dataset, rows in groups.items():
        print(f"{dataset}: rows={len(rows)}, subjects={len({r['subject_id'] for r in rows if r['subject_id']})}")
    print(f"combined: rows={len(combined)}")


if __name__ == "__main__":
    main()
