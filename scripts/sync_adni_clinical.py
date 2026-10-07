#!/usr/bin/env python3
"""Deduplicate and copy ADNI raw clinical tables into processed clinical storage."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from build_clinical_masters import atomic_csv, read_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent
RAW_ROOT = DATA_ROOT / "ADNI" / "Tables"
PROCESSED_ROOT = DATA_ROOT / "Processed" / "Clinical" / "adni"
PROCESSED_TABLES = PROCESSED_ROOT / "Tables"
INVENTORY_PATH = PROCESSED_ROOT / "adni_source_inventory.csv"
TABULAR_SUFFIXES = {".csv", ".xlsx", ".xls"}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def export_date(path: Path) -> datetime:
    match = re.search(r"(\d{1,2}[A-Za-z]{3}\d{4})", path.stem)
    if not match:
        return datetime.min
    try:
        return datetime.strptime(match.group(1), "%d%b%Y")
    except ValueError:
        return datetime.min


def classify(path: Path) -> str:
    name = path.name.lower()
    if "datadic" in name:
        return "data_dictionary"
    if "ptdemog" in name:
        return "demographics"
    if "adverse" in name or "recadv" in name:
        return "adverse_events"
    if "npstatus" in name or "neuropath" in name:
        return "neuropathology"
    if "registry" in name or "roster" in name or "studysum" in name:
        return "enrollment_registry"
    if "treatdis" in name:
        return "treatment_discontinuation"
    if "biomarker" in name or "lab" in name:
        return "laboratory_biomarker"
    return "clinical_assessment"


def main() -> None:
    raw_files = sorted(path for path in RAW_ROOT.iterdir() if path.is_file() and path.suffix.lower() in TABULAR_SUFFIXES)
    by_hash: dict[str, list[Path]] = defaultdict(list)
    metadata: dict[Path, tuple[int | object, int | object, str]] = {}
    for path in raw_files:
        file_hash = digest(path)
        by_hash[file_hash].append(path)
        if path.suffix.lower() == ".csv":
            frame, warning, _ = read_csv(path)
            metadata[path] = (len(frame), len(frame.columns), warning)
        else:
            metadata[path] = (pd.NA, pd.NA, "workbook not parsed")

    canonical_by_hash = {
        file_hash: sorted(paths, key=lambda path: (export_date(path), path.name), reverse=True)[0]
        for file_hash, paths in by_hash.items()
    }
    PROCESSED_TABLES.mkdir(parents=True, exist_ok=True)
    processed_hashes = {
        digest(path): path
        for path in PROCESSED_TABLES.iterdir()
        if path.is_file() and path.suffix.lower() in TABULAR_SUFFIXES
    }

    copied_count = 0
    inventory_rows: list[dict[str, object]] = []
    for file_hash, paths in sorted(by_hash.items(), key=lambda item: canonical_by_hash[item[0]].name):
        canonical = canonical_by_hash[file_hash]
        destination = processed_hashes.get(file_hash, PROCESSED_TABLES / canonical.name)
        if file_hash not in processed_hashes:
            temporary = destination.with_name(f".{destination.name}.tmp")
            shutil.copy2(canonical, temporary)
            os.replace(temporary, destination)
            processed_hashes[file_hash] = destination
            copied_count += 1
        for path in sorted(paths):
            row_count, column_count, warning = metadata[path]
            inventory_rows.append(
                {
                    "raw_path": str(path.relative_to(DATA_ROOT)),
                    "sha256": file_hash,
                    "size_bytes": path.stat().st_size,
                    "classification": classify(path),
                    "row_count": row_count,
                    "column_count": column_count,
                    "duplicate_group_size": len(paths),
                    "is_canonical_export": path == canonical,
                    "processed_relative_path": str(destination.relative_to(PROCESSED_ROOT)),
                    "parse_warning": warning,
                }
            )

    inventory = pd.DataFrame(inventory_rows)
    atomic_csv(inventory, INVENTORY_PATH)
    print(
        {
            "raw_file_instances": len(inventory),
            "unique_hashes": inventory["sha256"].nunique(),
            "duplicate_instances_ignored": len(inventory) - inventory["sha256"].nunique(),
            "new_files_copied": copied_count,
            "inventory": str(INVENTORY_PATH),
        }
    )


if __name__ == "__main__":
    main()
