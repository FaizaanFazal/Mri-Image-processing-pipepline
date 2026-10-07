#!/usr/bin/env python3
"""Inventory OASIS-3 recursively and build a deduplicated clinical-only master."""

from __future__ import annotations

import hashlib
import os
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

from build_clinical_masters import Source, atomic_csv, merge_sources, read_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent
RAW_OASIS = DATA_ROOT / "OASIS"
CANONICAL_ROOT = RAW_OASIS / "OASIS3"
DUPLICATE_ROOT = RAW_OASIS / "oasis_datta" / "oasis_data"
CLINICAL_ROOT = DATA_ROOT / "Processed" / "Clinical" / "oasis"
OASIS3_CLINICAL_ROOT = CLINICAL_ROOT / "OASIS3"
MRI_MANIFEST_ROOT = DATA_ROOT / "Processed" / "MRI" / "OASIS3" / "manifests"
TABULAR_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls"}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def tabular_paths() -> list[Path]:
    paths: set[Path] = set()
    data_root = CANONICAL_ROOT / "OASIS3_data_files"
    for path in data_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in TABULAR_SUFFIXES:
            paths.add(path)
    for path in (CANONICAL_ROOT / "MR Session").glob("*.csv"):
        paths.add(path)
    duplicate_data_root = DUPLICATE_ROOT / "OASIS3_data_files"
    if duplicate_data_root.exists():
        for path in duplicate_data_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in TABULAR_SUFFIXES:
                paths.add(path)
    for path in DUPLICATE_ROOT.iterdir():
        if path.is_file() and path.suffix.lower() in TABULAR_SUFFIXES:
            paths.add(path)
    return sorted(paths)


def classify(path: Path) -> tuple[str, str]:
    lowered = str(path).lower()
    name = path.name.lower()
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return "dictionary", "clinical/imaging data dictionary"
    if "upenn" in lowered or "csf" in name:
        return "clinical", "CSF biomarker table"
    if "oasis3subjects.csv" in name:
        return "imaging", "subject/session registry containing MR, PET, and CT session counts"
    clinical_tokens = (
        "_uds",
        "demographics.csv",
        "amyloid_centiloid",
        "braak_tauopathy",
    )
    if any(token in name for token in clinical_tokens):
        if "centiloid" in name or "braak" in name:
            return "clinical", "derived PET biomarker summary retained as a clinical biomarker"
        return "clinical", "clinical, demographic, behavioral, or psychometric table"
    imaging_tokens = (
        "mr session",
        "mr_json",
        "ct_json",
        "datasetdescription",
        "pet_json",
        "freesurfer",
        "pup",
        "oasis_manifest",
        "oasis_master_dataset",
        "t1_list",
    )
    if any(token in lowered for token in imaging_tokens):
        return "imaging", "MRI/PET/CT acquisition, processing, or session summary"
    return "review", "unclassified tabular file; excluded pending review"


def preferred(paths: list[Path]) -> Path:
    return sorted(
        paths,
        key=lambda path: (
            0 if str(path).startswith(str(CANONICAL_ROOT)) else 1,
            len(path.parts),
            str(path),
        ),
    )[0]


def row_and_column_count(path: Path) -> tuple[object, object, str]:
    if path.suffix.lower() != ".csv":
        return pd.NA, pd.NA, "workbook not parsed"
    try:
        frame, warning, _ = read_csv(path)
        return len(frame), len(frame.columns), warning
    except Exception as error:  # retained in the inventory for manual review
        return pd.NA, pd.NA, f"parse failed: {type(error).__name__}: {error}"


def processed_destination(path: Path) -> Path:
    relative = path.relative_to(CANONICAL_ROOT)
    return OASIS3_CLINICAL_ROOT / relative


def build_inventory() -> tuple[pd.DataFrame, list[Path]]:
    candidates = tabular_paths()
    by_hash: dict[str, list[Path]] = defaultdict(list)
    sizes: dict[Path, int] = {}
    for path in candidates:
        file_hash = digest(path)
        by_hash[file_hash].append(path)
        sizes[path] = path.stat().st_size

    inventory_rows: list[dict[str, object]] = []
    selected_clinical: list[Path] = []
    for file_hash, paths in sorted(by_hash.items(), key=lambda item: str(preferred(item[1]))):
        canonical = preferred(paths)
        classification, reason = classify(canonical)
        selected = classification == "clinical"
        rows, columns, warning = row_and_column_count(canonical)
        if selected and canonical.suffix.lower() == ".csv":
            selected_clinical.append(canonical)
        for path in sorted(paths):
            inventory_rows.append(
                {
                    "path": str(path.relative_to(DATA_ROOT)),
                    "sha256": file_hash,
                    "size_bytes": sizes[path],
                    "classification": classification,
                    "classification_reason": reason,
                    "canonical_path": str(canonical.relative_to(DATA_ROOT)),
                    "duplicate_group_size": len(paths),
                    "is_canonical_copy": path == canonical,
                    "selected_for_clinical_master": selected and path == canonical and path.suffix.lower() == ".csv",
                    "row_count": rows,
                    "column_count": columns,
                    "parse_warning": warning,
                }
            )
    inventory = pd.DataFrame(inventory_rows)
    return inventory, sorted(selected_clinical)


def copy_clinical_sources(paths: list[Path]) -> list[Path]:
    copied: list[Path] = []
    for source in paths:
        if not str(source).startswith(str(CANONICAL_ROOT)):
            raise RuntimeError(f"Selected source is not under canonical OASIS3 root: {source}")
        destination = processed_destination(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or digest(destination) != digest(source):
            temporary = destination.with_name(f".{destination.name}.tmp")
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        copied.append(destination)
    return copied


def rebuild_oasis_manifest(selected_paths: list[Path], inventory: pd.DataFrame) -> None:
    rows: list[dict[str, object]] = []
    oasis2 = CLINICAL_ROOT / "OASIS2" / "OAsis_2.csv"
    sources = [(RAW_OASIS / "OASIS2" / "OAsis_2.csv", oasis2)]
    sources.extend((source, processed_destination(source)) for source in selected_paths)
    dictionary_rows = inventory[
        (inventory["classification"] == "dictionary") & (inventory["is_canonical_copy"])
    ]
    for record in dictionary_rows.itertuples(index=False):
        source = DATA_ROOT / record.path
        destination = processed_destination(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or digest(destination) != digest(source):
            temporary = destination.with_name(f".{destination.name}.tmp")
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        sources.append((source, destination))

    for source, copied in sources:
        if copied.suffix.lower() == ".csv":
            frame, _, _ = read_csv(copied)
            row_count, column_count = len(frame), len(frame.columns)
        else:
            row_count = column_count = pd.NA
        rows.append(
            {
                "dataset": "oasis",
                "source_relative_path": str(source.relative_to(RAW_OASIS)),
                "copied_relative_path": str(copied.relative_to(CLINICAL_ROOT)),
                "file_type": copied.suffix.lower().lstrip("."),
                "size_bytes": copied.stat().st_size,
                "sha256": digest(copied),
                "row_count": row_count,
                "column_count": column_count,
            }
        )
    manifest = pd.DataFrame(rows).sort_values("copied_relative_path", kind="stable")
    atomic_csv(manifest, CLINICAL_ROOT / "oasis_clinical_manifest.csv")


def cohort_summary(master: pd.DataFrame, sources: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    cdr_column = "udsb4_cdr__CDRTOT"
    days = pd.to_numeric(master["visit_id"], errors="coerce")
    cdr = pd.to_numeric(master[cdr_column], errors="coerce") if cdr_column in master else pd.Series(pd.NA, index=master.index)
    cdr_rows = master.loc[cdr.notna() & days.notna(), ["subject_id"]].copy()
    cdr_rows["days"] = days[cdr.notna() & days.notna()]
    cdr_rows["cdr"] = cdr[cdr.notna() & days.notna()]
    baseline_ids = set(cdr_rows.loc[(cdr_rows["days"] == 0) & (cdr_rows["cdr"] == 0.5), "subject_id"])
    followup_ids = set(cdr_rows.loc[cdr_rows["days"] > 0, "subject_id"])
    at_risk_followup = len(baseline_ids & followup_ids)
    age_death = pd.to_numeric(master.get("demographics__AgeatDeath", pd.Series(dtype=str)), errors="coerce")
    cdr_source = sources.loc[sources["source_table"] == "udsb4_cdr", "parsed_rows"]
    clinical_visit_rows = int(cdr_source.iloc[0]) if not cdr_source.empty else 0
    return pd.DataFrame(
        [
            {
                "dataset": "OASIS3",
                "clinical_source_tables": len(sources),
                "master_rows": len(master),
                "master_columns": len(master.columns),
                "unique_subjects": master["subject_id"].nunique(),
                "clinical_visit_rows_cdr_table": clinical_visit_rows,
                "cdr_global_scores_nonmissing": int(cdr.notna().sum()),
                "subjects_with_age_at_death": int(master.loc[age_death.notna(), "subject_id"].nunique()),
                "baseline_cdr_0_5_with_later_cdr_followup": at_risk_followup,
                "duplicate_master_keys": int(master.duplicated(["cohort", "subject_id", "visit_id"]).sum()),
                "tabular_file_instances_inventoried": len(inventory),
                "unique_tabular_hashes": inventory["sha256"].nunique(),
                "duplicate_file_instances_ignored": len(inventory) - inventory["sha256"].nunique(),
                "upenn_or_csf_files_found": int(inventory["path"].str.contains("upenn|csf", case=False, regex=True).sum()),
            }
        ]
    )


def main() -> None:
    inventory, selected = build_inventory()
    copied = copy_clinical_sources(selected)
    sources = [Source(cohort="OASIS3", path=path) for path in copied]
    master, source_audit = merge_sources("oasis3", sources)

    master_path = OASIS3_CLINICAL_ROOT / "oasis3_clinical_master.csv"
    source_path = OASIS3_CLINICAL_ROOT / "oasis3_clinical_master_sources.csv"
    summary_path = OASIS3_CLINICAL_ROOT / "oasis3_clinical_master_summary.csv"
    clinical_inventory_path = OASIS3_CLINICAL_ROOT / "oasis3_clinical_source_inventory.csv"
    full_inventory_path = MRI_MANIFEST_ROOT / "oasis3_recursive_tabular_inventory.csv"

    atomic_csv(master, master_path)
    atomic_csv(source_audit, source_path)
    atomic_csv(inventory[inventory["classification"] == "clinical"], clinical_inventory_path)
    atomic_csv(inventory, full_inventory_path)
    summary = cohort_summary(master, source_audit, inventory)
    atomic_csv(summary, summary_path)
    rebuild_oasis_manifest(selected, inventory)

    print(summary.to_json(orient="records"))
    print(master_path)
    print(source_path)
    print(full_inventory_path)


if __name__ == "__main__":
    main()
