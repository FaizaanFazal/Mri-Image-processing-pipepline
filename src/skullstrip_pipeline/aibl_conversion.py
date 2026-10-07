from __future__ import annotations

import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .io_utils import absolute_from_data, anonymous_key, atomic_write_csv, relative_to_data
from .paths import (
    AIBL_CONVERSION_MANIFEST,
    AIBL_RAW3D_ROOT,
    AIBL_RAW_ROOT,
    DATA_ROOT,
    MANIFEST_ROOT,
)
from .qc import volume_qc


CONVERSION_COLUMNS = [
    "dataset", "record_key", "subject_id", "image_id", "scan_id", "source_2d_path",
    "sequence_description", "modality_family", "group_class", "cdr_global", "source_file_count",
    "converted_3d_path", "status", "qc_pass", "shape",
    "voxel_sizes_mm", "finite", "intensity_range", "error_type", "error_message", "updated_utc",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_source_rows() -> pd.DataFrame:
    frame = pd.read_csv(MANIFEST_ROOT / "aibl_detailed_manifest.csv", low_memory=False)
    frame = frame[(frame["record_kind"] == "primary_mri") & frame["dimension"].astype(str).str.startswith("2D")].copy()
    frame["record_key"] = [anonymous_key("AIBL", image, scan, path) for image, scan, path in zip(frame.image_id, frame.scan_id, frame.raw_path)]
    return frame.sort_values("record_key").reset_index(drop=True)


def _modality_family(description: object) -> str:
    value = str(description).upper()
    if any(token in value for token in ("SWI", "MIP_IMAGES(SW)")):
        return "SWI"
    if "FLAIR" in value:
        return "T2_FLAIR"
    if "T2" in value:
        return "T2"
    if any(token in value for token in ("MPR", "T1", "SPGR")):
        return "T1"
    return "OTHER"


def _initial_manifest(rows: pd.DataFrame) -> pd.DataFrame:
    output_root = AIBL_RAW3D_ROOT
    records = []
    for row in rows.itertuples(index=False):
        key = row.record_key
        modality_family = _modality_family(row.sequence_description)
        records.append({
            "dataset": "AIBL", "record_key": key, "subject_id": row.subject_id,
            "image_id": row.image_id, "scan_id": row.scan_id,
            "source_2d_path": row.raw_path, "sequence_description": row.sequence_description,
            "modality_family": modality_family,
            "group_class": row.group_class, "cdr_global": row.cdr_global,
            "source_file_count": row.source_file_count,
            "converted_3d_path": relative_to_data(
                output_root / modality_family / f"aibl_{modality_family}_{key}.nii.gz", DATA_ROOT
            ),
            "status": "pending", "qc_pass": False, "shape": "", "voxel_sizes_mm": "",
            "finite": "", "intensity_range": "", "error_type": "", "error_message": "", "updated_utc": "",
        })
    return pd.DataFrame(records, columns=CONVERSION_COLUMNS).astype(object)


def _convert_one(record: dict[str, object], replace: bool, dcm2niix_bin: Path) -> dict[str, object]:
    result = dict(record)
    source_dir = absolute_from_data(record["source_2d_path"], DATA_ROOT)
    output_path = absolute_from_data(record["converted_3d_path"], DATA_ROOT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not replace:
        try:
            metrics = volume_qc(output_path)
            result.update(
                metrics,
                status="success" if metrics["qc_pass"] else "qc_failed",
                error_type="" if metrics["qc_pass"] else "VolumeQCFailure",
                error_message="" if metrics["qc_pass"] else "Existing 3D volume failed automated QC",
                updated_utc=_now(),
            )
            return result
        except Exception as exc:
            result.update(
                status="failed", qc_pass=False, error_type=type(exc).__name__,
                error_message=str(exc)[:500], updated_utc=_now(),
            )
            return result
    if not source_dir.is_dir():
        result.update(
            status="failed", qc_pass=False, error_type="MissingSourceDirectory",
            error_message=f"Source DICOM directory is missing: {source_dir}", updated_utc=_now(),
        )
        return result
    try:
        with tempfile.TemporaryDirectory(prefix="aibl_dcm2niix_") as tmp_name:
            temporary = Path(tmp_name)
            command = [str(dcm2niix_bin), "-b", "y", "-z", "y", "-f", "converted", "-o", str(temporary), str(source_dir)]
            completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=900)
            if completed.returncode != 0:
                raise RuntimeError("dcm2niix returned a non-zero status")
            candidates = []
            for candidate in temporary.glob("*.nii*"):
                try:
                    image = nib.load(str(candidate))
                    if len(image.shape) == 3:
                        candidates.append((int(image.shape[0] * image.shape[1] * image.shape[2]), candidate))
                except Exception:
                    continue
            if not candidates:
                raise RuntimeError("dcm2niix produced no valid 3D NIfTI")
            selected = max(candidates, key=lambda item: item[0])[1]
            temporary_target = output_path.with_name(f".{output_path.name}.tmp.nii.gz")
            shutil.copy2(selected, temporary_target)
            temporary_target.replace(output_path)
            selected_json = selected.with_suffix("").with_suffix(".json") if selected.name.endswith(".nii.gz") else selected.with_suffix(".json")
            if selected_json.exists():
                shutil.copy2(selected_json, output_path.with_suffix("").with_suffix(".json"))
        metrics = volume_qc(output_path)
        result.update(
            metrics,
            status="success" if metrics["qc_pass"] else "qc_failed",
            error_type="" if metrics["qc_pass"] else "VolumeQCFailure",
            error_message="" if metrics["qc_pass"] else "Converted 3D volume failed automated QC",
            updated_utc=_now(),
        )
    except Exception as exc:
        result.update(
            status="failed", qc_pass=False, error_type=type(exc).__name__,
            error_message=str(exc)[:500], updated_utc=_now(),
        )
    return result


def run_aibl_conversion(
    max_files: int | None = 5,
    replace: bool = False,
    workers: int = 2,
    stop_on_error: bool = False,
) -> pd.DataFrame:
    rows = _load_source_rows()
    manifest_path = AIBL_CONVERSION_MANIFEST
    current = _initial_manifest(rows)
    if manifest_path.exists():
        old = pd.read_csv(manifest_path, low_memory=False).astype(object)
        retained = old.set_index("record_key")
        # Migrate earlier anonymous flat outputs into modality-labelled paths.
        for row in current.itertuples(index=False):
            if row.record_key not in retained.index:
                continue
            previous_value = retained.at[row.record_key, "converted_3d_path"] if "converted_3d_path" in retained.columns else ""
            if not isinstance(previous_value, str) or not previous_value:
                continue
            previous_path = absolute_from_data(previous_value, DATA_ROOT)
            current_path = absolute_from_data(row.converted_3d_path, DATA_ROOT)
            if previous_path != current_path and previous_path.exists() and not current_path.exists():
                current_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(previous_path, current_path)
                previous_json = previous_path.with_suffix("").with_suffix(".json")
                current_json = current_path.with_suffix("").with_suffix(".json")
                if previous_json.exists():
                    shutil.move(previous_json, current_json)
        for column in [c for c in CONVERSION_COLUMNS if c not in {"dataset", "record_key", "subject_id", "image_id", "scan_id", "source_2d_path", "sequence_description", "modality_family", "group_class", "cdr_global", "source_file_count", "converted_3d_path"}]:
            current[column] = current["record_key"].map(retained[column]) if column in retained.columns else current[column]
    candidates = current[(current["status"] != "success") | bool(replace)].copy()
    candidates["t1_priority"] = ~candidates["sequence_description"].astype(str).str.contains("MPR|T1|SPGR", case=False, na=False)
    candidates = candidates.sort_values(["t1_priority", "record_key"]).drop(columns="t1_priority")
    if max_files is not None:
        candidates = candidates.head(int(max_files))
    dcm2niix_bin = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "dcm2niix"
    indexed = current.set_index("record_key", drop=False)
    pool = ThreadPoolExecutor(max_workers=max(1, int(workers)))
    failure: dict[str, object] | None = None
    try:
        futures = {
            pool.submit(_convert_one, row._asdict(), replace, dcm2niix_bin): row.record_key
            for row in candidates.itertuples(index=False)
        }
        for future in as_completed(futures):
            record = future.result()
            key = str(record["record_key"])
            for column in CONVERSION_COLUMNS:
                indexed.at[key, column] = record.get(column, "")

            # Persist every completed series so an interrupted job resumes cleanly.
            current = indexed.reset_index(drop=True).astype(object)
            atomic_write_csv(current[CONVERSION_COLUMNS], manifest_path)

            if stop_on_error and str(record.get("status")) != "success":
                failure = record
                for queued in futures:
                    if not queued.done():
                        queued.cancel()
                break
    finally:
        pool.shutdown(wait=True, cancel_futures=bool(failure))

    current = indexed.reset_index(drop=True).astype(object)
    atomic_write_csv(current[CONVERSION_COLUMNS], manifest_path)
    if failure is not None:
        raise RuntimeError(
            "AIBL conversion stopped after "
            f"{failure.get('record_key')} entered {failure.get('status')}: "
            f"{failure.get('error_type')} {failure.get('error_message')}"
        )
    return current


def save_conversion_qc_examples(manifest: pd.DataFrame, max_plots: int = 3) -> list[Path]:
    output_dir = AIBL_RAW_ROOT / "raw3DAll_qc_examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    passing = manifest[(manifest["status"] == "success") & manifest["qc_pass"].astype(str).str.lower().eq("true")].head(int(max_plots))
    paths: list[Path] = []
    for number, row in enumerate(passing.itertuples(index=False), start=1):
        data = np.asarray(nib.load(str(absolute_from_data(row.converted_3d_path, DATA_ROOT))).dataobj, dtype=np.float32)
        centers = tuple(int(size // 2) for size in data.shape)
        views = [data[centers[0], :, :], data[:, centers[1], :], data[:, :, centers[2]]]
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
        for axis, view in zip(axes, views):
            axis.imshow(np.rot90(view), cmap="gray")
            axis.axis("off")
        fig.suptitle(f"anonymous converted volume {number:03d}")
        fig.tight_layout()
        path = output_dir / f"converted_qc_sample_{number:03d}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
