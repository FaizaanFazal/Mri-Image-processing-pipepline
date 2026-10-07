#!/usr/bin/env python3
"""Build traceable subject-visit clinical master tables for ADNI, AIBL, and OASIS."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent
CLINICAL_ROOT = DATA_ROOT / "Processed" / "Clinical"
KEY_COLUMNS = ["cohort", "subject_id", "visit_id"]
MISSING_TEXT = {"", "nan", "none", "null", "nat", "<na>"}


@dataclass(frozen=True)
class Source:
    cohort: str
    path: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def read_csv(path: Path) -> tuple[pd.DataFrame, str, int]:
    """Read text exactly where possible and report permissively skipped rows."""
    try:
        frame = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            low_memory=False,
            encoding="utf-8-sig",
        )
        return frame, "", 0
    except pd.errors.ParserError as error:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
            rows = list(csv.reader(stream))
        header = rows[0]
        data_rows = rows[1:]
        maximum_width = max([len(header), *(len(row) for row in data_rows)])
        extra_count = maximum_width - len(header)
        columns = header + [f"__extra_field_{number}" for number in range(1, extra_count + 1)]
        normalized = [row[:maximum_width] + [""] * (maximum_width - len(row)) for row in data_rows]
        frame = pd.DataFrame(normalized, columns=columns, dtype=str)
        affected = sum(len(row) != len(header) for row in data_rows)
        warning = (
            f"strict parse failed; preserved {affected} nonstandard-width rows in "
            f"{extra_count} synthetic extra-field column(s): {str(error).splitlines()[0]}"
        )
        return frame, warning, 0


def choose_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    by_lower = {str(column).strip().lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def clean_key(series: pd.Series) -> pd.Series:
    result = series.astype(str).str.strip()
    return result.mask(result.str.lower().isin(MISSING_TEXT), "")


def clean_visit_key(series: pd.Series, column: str) -> pd.Series:
    result = clean_key(series)
    if column.lower() == "days_to_visit":
        numeric = pd.to_numeric(result, errors="coerce")
        integer_mask = numeric.notna() & (numeric % 1 == 0)
        result = result.mask(integer_mask, numeric.loc[integer_mask].astype("int64").astype(str))
    return result


def source_slug(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"^all_subjects_", "", stem)
    stem = re.sub(r"^oasis3_", "", stem)
    stem = re.sub(r"^aibl_", "", stem)
    stem = re.sub(r"_\d{1,2}[-_]?[a-z]{3}[-_]?(?:19|20)\d{2}$", "", stem)
    stem = re.sub(r"_\d{1,2}[a-z]{3}(?:19|20)\d{2}$", "", stem)
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_")


def combine_unique(values: pd.Series) -> str:
    unique: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        if value.lower() in MISSING_TEXT or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    return json.dumps(unique, ensure_ascii=False, separators=(",", ":"))


def representative_value(values: pd.Series) -> str:
    cleaned = values.fillna("").astype(str).str.strip()
    cleaned = cleaned.loc[~cleaned.str.lower().isin(MISSING_TEXT | {"-4", "-4.0"})]
    if cleaned.empty:
        return ""
    counts = cleaned.value_counts(sort=False)
    return str(counts.idxmax())


def discover_sources(dataset: str) -> list[Source]:
    root = CLINICAL_ROOT / dataset
    derived_tokens = (
        "clinical_manifest",
        "clinical_master",
        "master_sources",
        "source_inventory",
        "duplicate_audit",
        "recursive_tabular_inventory",
    )
    paths = []
    for path in root.rglob("*.csv"):
        lowered = path.name.lower()
        if any(token in lowered for token in derived_tokens):
            continue
        if dataset == "oasis" and "oasis_datta" in path.parts:
            continue
        paths.append(path)

    sources: list[Source] = []
    for path in sorted(paths):
        if dataset == "oasis":
            cohort = "OASIS2" if "OASIS2" in path.parts else "OASIS3"
        else:
            cohort = dataset.upper()
        sources.append(Source(cohort=cohort, path=path))
    return sources


def key_candidates(dataset: str, cohort: str) -> tuple[list[str], list[str], list[str]]:
    if dataset == "adni":
        return (
            ["PTID", "subject_id", "ParticipantID", "Subject", "RID"],
            ["VISCODE2", "VISCODE", "entry_visit", "Visit"],
            ["EXAMDATE", "VISDATE", "entry_date", "StudyDate", "APTESTDT", "Acq Date"],
        )
    if dataset == "aibl":
        return (["RID"], ["VISCODE"], ["EXAMDATE", "APTESTDT", "RGCONDCT"])
    if cohort == "OASIS2":
        return (["Subject ID"], ["MRI ID"], ["Visit", "MR Delay"])
    return (
        ["OASISID", "OASIS_ID", "subject_id", "Subject"],
        ["days_to_visit", "OASIS_session_label", "oasis_session_id"],
        ["days_to_visit"],
    )


def prepare_source(dataset: str, source: Source) -> tuple[pd.DataFrame | None, dict[str, object]]:
    frame, warning, skipped_rows = read_csv(source.path)
    subject_candidates, visit_candidates, date_candidates = key_candidates(dataset, source.cohort)
    subject_column = choose_column(frame, subject_candidates)
    visit_column = choose_column(frame, visit_candidates)
    date_column = choose_column(frame, date_candidates)
    slug = source_slug(source.path)
    audit: dict[str, object] = {
        "dataset": dataset.upper(),
        "cohort": source.cohort,
        "source_table": slug,
        "source_path": str(source.path.relative_to(DATA_ROOT)),
        "sha256": sha256(source.path),
        "parsed_rows": len(frame),
        "source_columns": len(frame.columns),
        "subject_key_column": subject_column or "",
        "visit_key_column": visit_column or "",
        "date_column": date_column or "",
        "malformed_rows_skipped": skipped_rows,
        "parse_warning": warning,
    }

    if frame.empty:
        audit.update({"rows_missing_subject_key": 0, "master_key_count": 0, "rows_in_repeated_keys": 0, "status": "empty source retained in audit"})
        return None, audit
    if subject_column is None:
        audit.update({"rows_missing_subject_key": len(frame), "master_key_count": 0, "rows_in_repeated_keys": 0, "status": "lookup/keyless source retained in audit only"})
        return None, audit

    original_columns = [str(column) for column in frame.columns]
    prefixed = {column: f"{slug}__{column}" for column in original_columns}
    working = frame.rename(columns=prefixed).copy()
    working["cohort"] = source.cohort
    working["subject_id"] = clean_key(working[prefixed[subject_column]])
    working["visit_id"] = clean_visit_key(working[prefixed[visit_column]], visit_column) if visit_column else ""
    if date_column:
        working["canonical_event_time"] = clean_key(working[prefixed[date_column]])
    else:
        working["canonical_event_time"] = ""

    missing_subject = int((working["subject_id"] == "").sum())
    working = working.loc[working["subject_id"] != ""].copy()
    subject_level = visit_column is None
    group_keys = ["cohort", "subject_id"] if subject_level else KEY_COLUMNS
    repeated = int(working.duplicated(group_keys, keep=False).sum())
    key_count = int(working[group_keys].drop_duplicates().shape[0])
    audit.update(
        {
            "rows_missing_subject_key": missing_subject,
            "master_key_count": key_count,
            "rows_in_repeated_keys": repeated,
            "status": "subject-level broadcast" if subject_level else "subject-visit merge",
        }
    )

    value_columns = [prefixed[column] for column in original_columns]
    event_column = f"{slug}__event_time"
    working[event_column] = working["canonical_event_time"]
    value_columns.append(event_column)

    if working.empty:
        return None, audit
    sizes = working.groupby(group_keys, sort=False, dropna=False).size().rename(f"{slug}__source_row_count")
    if repeated:
        reduced = working.groupby(group_keys, sort=False, dropna=False)[value_columns].agg(combine_unique)
    else:
        reduced = working.set_index(group_keys)[value_columns]
    reduced = reduced.join(sizes).reset_index()
    reduced[f"{slug}__present"] = 1
    return reduced, audit


def merge_sources(dataset: str, sources: list[Source]) -> tuple[pd.DataFrame, pd.DataFrame]:
    visit_frames: list[pd.DataFrame] = []
    subject_frames: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for source in sources:
        prepared, audit = prepare_source(dataset, source)
        audits.append(audit)
        if prepared is None:
            continue
        if audit["status"] == "subject-level broadcast":
            subject_frames.append(prepared)
        else:
            visit_frames.append(prepared)

    master: pd.DataFrame | None = None
    for frame in visit_frames:
        master = frame if master is None else master.merge(frame, on=KEY_COLUMNS, how="outer", validate="one_to_one")
    if master is None:
        master = pd.DataFrame(columns=KEY_COLUMNS)

    for frame in subject_frames:
        subject_keys = ["cohort", "subject_id"]
        existing = set(zip(master["cohort"], master["subject_id"]))
        unseen = frame.loc[
            ~frame.apply(lambda row: (row["cohort"], row["subject_id"]) in existing, axis=1),
            subject_keys,
        ].copy()
        if not unseen.empty:
            unseen["visit_id"] = ""
            master = pd.concat([master, unseen[KEY_COLUMNS]], ignore_index=True)
        master = master.merge(frame, on=subject_keys, how="left", validate="many_to_one")

    master = add_canonical_columns(dataset, master)
    master = master.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)
    front = [
        "dataset",
        "cohort",
        "subject_id",
        "visit_id",
        "event_time",
        "age",
        "sex",
        "education",
        "race",
        "ethnicity",
        "handedness",
        "diagnosis_or_group",
        "cdr_global",
        "cdr_sum",
        "mmse",
    ]
    master.insert(0, "dataset", dataset.upper())
    front = [column for column in front if column in master.columns]
    master = master[front + [column for column in master.columns if column not in front]]
    return master, pd.DataFrame(audits)


def first_available(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    result = pd.Series("", index=frame.index, dtype=str)
    for column in candidates:
        if column not in frame.columns:
            continue
        values = frame[column].fillna("").astype(str).str.strip()
        valid = ~values.str.lower().isin(MISSING_TEXT)
        result = result.mask((result == "") & valid, values)
    return result


def add_canonical_columns(dataset: str, frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    event_candidates = [column for column in frame.columns if column.endswith("__event_time")]
    if dataset == "adni":
        mappings = {
            "age": ["study_entry__entry_age", "adni1_3yr_1_5t__Age", "adni1_complete_3yr_1_5t_4_06_2025__Age", "harp_data_5_11_2025__Age"],
            "sex": ["ptdemog__PTGENDER", "adni1_3yr_1_5t__Sex", "adni1_complete_3yr_1_5t_4_06_2025__Sex", "harp_data_5_11_2025__Sex"],
            "education": ["ptdemog__PTEDUCAT"],
            "race": ["ptdemog__PTRACCAT"],
            "ethnicity": ["ptdemog__PTETHCAT"],
            "handedness": ["ptdemog__PTHAND"],
            "diagnosis_or_group": ["dxsum__DIAGNOSIS", "study_entry__entry_research_group", "adni1_3yr_1_5t__Group", "adni1_complete_3yr_1_5t_4_06_2025__Group", "harp_data_5_11_2025__Group"],
            "cdr_global": ["cdr__CDGLOBAL"],
            "cdr_sum": ["cdr__CDRSB"],
            "mmse": ["mmse__MMSCORE"],
        }
    elif dataset == "aibl":
        mappings = {
            "age": [],
            "sex": ["ptdemog__PTGENDER"],
            "education": [],
            "race": [],
            "ethnicity": [],
            "handedness": [],
            "diagnosis_or_group": ["pdxconv__DXCURREN"],
            "cdr_global": ["cdr__CDGLOBAL"],
            "cdr_sum": [],
            "mmse": ["mmse__MMSCORE"],
        }
    else:
        mappings = {
            "age": ["udsa1_participant_demo__age at visit", "oasis_2__Age"],
            "sex": ["demographics__GENDER", "oasis3subjects__M/F", "oasis_2__Gender"],
            "education": ["demographics__EDUC", "oasis_2__EDUC"],
            "race": ["demographics__race"],
            "ethnicity": ["demographics__ETHNIC"],
            "handedness": ["demographics__HAND", "oasis_2__Hand"],
            "diagnosis_or_group": ["oasis_2__Group", "oasis3subjects__Group", "udsd1_diagnoses__NORMCOG", "udsd1_diagnoses__DEMENTED"],
            "cdr_global": ["udsb4_cdr__CDRTOT", "oasis_2__CDR"],
            "cdr_sum": ["udsb4_cdr__CDRSUM"],
            "mmse": ["udsb4_cdr__MMSE", "oasis_2__MMSE"],
        }
    frame["event_time"] = first_available(frame, event_candidates)
    for target, candidates in mappings.items():
        frame[target] = first_available(frame, candidates)
    if dataset == "adni":
        decode_maps = {
            "sex": ({"1": "Male", "2": "Female"}, "ptdemog__PTGENDER"),
            "race": (
                {
                    "1": "American Indian or Alaska Native",
                    "2": "Asian",
                    "3": "Native Hawaiian or Other Pacific Islander",
                    "4": "Black or African American",
                    "5": "White",
                    "6": "More than one race",
                    "7": "Unknown",
                    "8": "Native Hawaiian",
                    "9": "Other Pacific Islander",
                },
                "ptdemog__PTRACCAT",
            ),
            "ethnicity": (
                {"1": "Hispanic or Latino", "2": "Not Hispanic or Latino", "3": "Unknown"},
                "ptdemog__PTETHCAT",
            ),
            "handedness": ({"1": "Right", "2": "Left"}, "ptdemog__PTHAND"),
            "diagnosis_or_group": (
                {"1": "CN", "2": "MCI", "3": "Dementia", "10": "TEAM_NODX"},
                "dxsum__DIAGNOSIS",
            ),
        }
        for target, (codes, source_column) in decode_maps.items():
            if source_column not in frame:
                continue
            raw = frame[source_column].fillna("").astype(str).str.strip()
            def decode_value(value: str) -> str:
                if value.lower() in MISSING_TEXT or value in {"-4", "-4.0"}:
                    return ""
                labels = [codes[part] for part in value.split("|") if part in codes]
                if not labels:
                    return ""
                return " | ".join(dict.fromkeys(labels))
            decoded = raw.map(decode_value)
            fallback = frame[target].fillna("").astype(str).str.strip()
            fallback = fallback.replace({"-4": "", "-4.0": ""})
            if target == "sex":
                fallback = fallback.replace({"M": "Male", "F": "Female", "m": "Male", "f": "Female"})
            frame[target] = decoded.mask(decoded == "", fallback)
        education_numeric = pd.to_numeric(frame["education"], errors="coerce")
        frame["education"] = frame["education"].mask(~education_numeric.between(0, 20), "")
        for target in ("sex", "education", "race", "ethnicity", "handedness"):
            frame[target] = frame.groupby(["cohort", "subject_id"], sort=False)[target].transform(representative_value)
        birth_year = first_available(frame, ["ptdemog__PTDOBYY", "ptdemog__PTDOB"])
        birth_year = birth_year.str.extract(r"((?:19|20)\d{2})", expand=False)
        birth_year = birth_year.groupby([frame["cohort"], frame["subject_id"]], sort=False).transform(combine_unique)
        event_year = frame["event_time"].str.extract(r"((?:19|20)\d{2})", expand=False)
        derived_age = pd.to_numeric(event_year, errors="coerce") - pd.to_numeric(birth_year, errors="coerce")
        derived_age = derived_age.astype("Int64").astype(str).replace("<NA>", "")
        frame["age"] = frame["age"].mask(frame["age"].str.strip() == "", derived_age)
    if dataset == "aibl":
        frame["sex"] = frame.groupby(["cohort", "subject_id"], sort=False)["sex"].transform(combine_unique)
        birth_year = first_available(frame, ["ptdemog__PTDOB"]).str.extract(r"((?:19|20)\d{2})", expand=False)
        birth_year = birth_year.groupby([frame["cohort"], frame["subject_id"]], sort=False).transform(combine_unique)
        event_year = frame["event_time"].str.extract(r"((?:19|20)\d{2})", expand=False)
        age = pd.to_numeric(event_year, errors="coerce") - pd.to_numeric(birth_year, errors="coerce")
        frame["age"] = age.astype("Int64").astype(str).replace("<NA>", "")
    return frame


def refresh_adni_manifest() -> None:
    root = CLINICAL_ROOT / "adni"
    destination = root / "adni_clinical_manifest.csv"
    existing = pd.read_csv(destination, dtype=str, keep_default_na=False) if destination.exists() else pd.DataFrame()
    source_by_copy = dict(zip(existing.get("copied_relative_path", []), existing.get("source_relative_path", [])))
    derived_tokens = ("clinical_manifest", "clinical_master", "source_inventory", "demographics_summary", "demographics_counts")
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            continue
        if any(token in path.name.lower() for token in derived_tokens):
            continue
        relative = str(path.relative_to(root))
        if path.suffix.lower() == ".csv":
            frame, _, _ = read_csv(path)
            row_count, column_count = len(frame), len(frame.columns)
        else:
            row_count = column_count = pd.NA
        rows.append(
            {
                "dataset": "adni",
                "source_relative_path": source_by_copy.get(relative, relative),
                "copied_relative_path": relative,
                "file_type": path.suffix.lower().lstrip("."),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "row_count": row_count,
                "column_count": column_count,
            }
        )
    manifest = pd.DataFrame(rows).sort_values("copied_relative_path", kind="stable").reset_index(drop=True)
    atomic_csv(manifest, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("adni", "aibl", "oasis", "all"),
        default="all",
        help="Generate one cohort or all cohorts (default: all).",
    )
    arguments = parser.parse_args()
    refresh_adni_manifest()
    summaries: list[dict[str, object]] = []
    datasets = ("adni", "aibl", "oasis") if arguments.dataset == "all" else (arguments.dataset,)
    for dataset in datasets:
        sources = discover_sources(dataset)
        master, audit = merge_sources(dataset, sources)
        root = CLINICAL_ROOT / dataset
        master_path = root / f"{dataset}_clinical_master.csv"
        audit_path = root / f"{dataset}_clinical_master_sources.csv"
        atomic_csv(master, master_path)
        atomic_csv(audit, audit_path)
        summary = {
            "dataset": dataset.upper(),
            "source_tables": len(sources),
            "master_rows": len(master),
            "master_columns": len(master.columns),
            "unique_subjects": int(master[["cohort", "subject_id"]].drop_duplicates().shape[0]),
            "duplicate_master_keys": int(master.duplicated(KEY_COLUMNS).sum()),
            "source_rows_parsed": int(audit["parsed_rows"].sum()),
            "malformed_rows_skipped": int(audit["malformed_rows_skipped"].sum()),
            "master_path": str(master_path.relative_to(DATA_ROOT)),
            "source_audit_path": str(audit_path.relative_to(DATA_ROOT)),
        }
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))
    summary_path = CLINICAL_ROOT / "clinical_master_summary.csv"
    if summary_path.exists() and arguments.dataset != "all":
        previous = pd.read_csv(summary_path, dtype=str, keep_default_na=False)
        previous = previous.loc[~previous["dataset"].isin([row["dataset"] for row in summaries])]
        summary_frame = pd.concat([previous, pd.DataFrame(summaries)], ignore_index=True)
        summary_frame = summary_frame.sort_values("dataset", kind="stable")
    else:
        summary_frame = pd.DataFrame(summaries)
    atomic_csv(summary_frame, summary_path)


if __name__ == "__main__":
    main()
