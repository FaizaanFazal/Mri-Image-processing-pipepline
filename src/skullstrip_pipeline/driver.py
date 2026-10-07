from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from .io_utils import absolute_from_data, anonymous_key, atomic_write_csv, relative_to_data
from .paths import AIBL_CONVERSION_MANIFEST, DATA_ROOT, MNI_TEMPLATE, PROJECT_ROOT, SOURCE_MANIFEST, dataset_output_root
from .qc import (
    n4_qc,
    normalization_qc,
    registration_qc,
    skullstrip_qc,
    synthstrip_mask_qc,
    volume_qc,
)
from .reporting import save_anonymous_qc_examples


PIPELINE_VERSION = "synthstrip-mask-n4-affine-mni-zscore-v1"
SYNTHSTRIP_SCRIPT = PROJECT_ROOT / "resources" / "synthstrip" / "mri_synthstrip"
SYNTHSTRIP_MODEL = PROJECT_ROOT / "resources" / "synthstrip" / "synthstrip.1.pt"
SYNTHSTRIP_SOURCE = "https://github.com/freesurfer/freesurfer/tree/dev/mri_synthstrip"
SYNTHSTRIP_MODEL_SOURCE = "https://ftp.nmr.mgh.harvard.edu/pub/dist/freesurfer/synthstrip/models/synthstrip.1.pt"
N4_PARAMETERS = {
    "shrink_factor": 4,
    "convergence": {"iters": [50, 50, 30, 20], "tol": 1e-6},
    "spline_param": 200,
    "rescale_intensities": False,
}
REGISTRATION_PARAMETERS = {
    "type_of_transform": "Affine",
    "aff_metric": "mattes",
    "aff_sampling": 32,
    "aff_random_sampling_rate": 0.2,
    "aff_iterations": [2100, 1200, 1200, 10],
    "aff_shrink_factors": [6, 4, 2, 1],
    "aff_smoothing_sigmas": [3, 2, 1, 0],
    "random_seed": 42,
    "singleprecision": True,
    "histogram_matching": False,
    "image_interpolation": "linear",
    "mask_interpolation": "nearestNeighbor",
}

STATUS_COLUMNS = [
    "dataset", "record_key", "subject_id", "image_id", "scan_id", "source_path",
    "prepared_3d_path", "synthstrip_mask_path", "bias_corrected_path", "brain_path", "mask_path",
    "registered_brain_path", "registered_mask_path", "normalized_path", "transform_path", "template_path",
    "pipeline_version", "status", "stage_completed", "qc_pass", "synthstrip_qc_pass", "n4_qc_pass",
    "native_qc_pass", "registration_qc_pass", "normalization_qc_pass", "source_shape",
    "synthstrip_shape_match", "synthstrip_affine_match", "synthstrip_metadata_explicit", "synthstrip_finite", "synthstrip_binary",
    "native_mask_fraction", "native_mask_volume_ml", "native_largest_component_fraction",
    "n4_shape_match", "n4_affine_match", "n4_finite", "n4_brain_mean", "n4_brain_std",
    "n4_brain_p01", "n4_brain_p99", "native_shape_match", "native_affine_match", "native_finite",
    "native_background_abs_max", "registered_shape", "registered_spacing_mm", "registration_shape_match",
    "registration_spacing_match", "registration_affine_match", "registration_finite",
    "registered_mask_fraction", "registered_largest_component_fraction", "template_overlap_fraction",
    "centroid_distance_mm", "registered_background_abs_max", "normalization_geometry_match",
    "normalization_finite", "normalized_brain_mean", "normalized_brain_std",
    "normalized_background_abs_max", "registration_retry_status", "error_type", "error_message",
    "attempts", "updated_utc",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _dataset_rows(dataset: str) -> pd.DataFrame:
    dataset = dataset.upper()
    if dataset == "AIBL":
        if not AIBL_CONVERSION_MANIFEST.exists():
            raise FileNotFoundError("Run the AIBL 2D-to-3D conversion notebook first")
        converted = pd.read_csv(AIBL_CONVERSION_MANIFEST, low_memory=False)
        converted = converted[
            (converted["status"] == "success")
            & converted["qc_pass"].astype(str).str.lower().eq("true")
        ].copy()
        if "modality_family" in converted.columns:
            converted = converted[converted["modality_family"].astype(str).eq("T1")]
        else:
            converted = converted[
                converted["sequence_description"].astype(str).str.contains("MPR|T1|SPGR", case=False, na=False)
            ]
        converted["raw_path"] = converted["converted_3d_path"]
        return converted

    frame = pd.read_csv(SOURCE_MANIFEST, low_memory=False)
    frame = frame[(frame["dataset"] == dataset) & (frame["record_kind"] == "primary_mri")].copy()
    frame = frame[frame["dimension"].astype(str).str.startswith("3D")]
    if dataset == "ADNI":
        frame = frame[frame["sequence_description"].astype(str).str.contains("MPR|T1", case=False, na=False)]
    elif dataset == "OASIS2":
        frame = frame[frame["sequence_description"].astype(str).str.contains("MPR", case=False, na=False)]
    elif dataset == "OASIS3":
        frame = frame[frame["modality"].astype(str).str.upper().eq("T1W")]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return frame


def _prepare_nifti(source_path: Path, cache_path: Path, replace: bool) -> Path:
    if source_path.name.endswith(".nii.gz"):
        metrics = volume_qc(source_path)
        if not metrics["qc_pass"]:
            raise ValueError("Source NIfTI failed 3D sanity checks")
        image = nib.load(str(source_path))
        qform, qform_code = image.get_qform(coded=True)
        sform, sform_code = image.get_sform(coded=True)
        if bool(
            qform_code > 0 and sform_code > 0 and qform is not None and sform is not None
            and np.allclose(qform, image.affine, atol=1e-3)
            and np.allclose(sform, image.affine, atol=1e-3)
        ):
            return source_path
    if cache_path.exists() and not replace and volume_qc(cache_path)["qc_pass"]:
        return cache_path
    image = nib.load(str(source_path))
    if len(image.shape) == 4 and image.shape[-1] == 1:
        data = np.asarray(image.dataobj)[..., 0]
    elif len(image.shape) == 3:
        data = np.asarray(image.dataobj)
    else:
        raise ValueError("SynthStrip input must be one 3D structural volume")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.tmp.nii.gz")
    prepared = nib.Nifti1Image(data, image.affine, image.header)
    prepared.set_qform(image.affine, code=1)
    prepared.set_sform(image.affine, code=1)
    nib.save(prepared, str(temporary))
    os.replace(temporary, cache_path)
    if not volume_qc(cache_path)["qc_pass"]:
        raise ValueError("Prepared NIfTI failed sanity checks")
    return cache_path


def _run_synthstrip(
    source_path: Path,
    mask_path: Path,
    log_path: Path,
    gpu_physical_id: int,
    replace: bool,
) -> dict[str, object]:
    if mask_path.exists() and not replace:
        metrics = synthstrip_mask_qc(source_path, mask_path)
        if metrics["synthstrip_qc_pass"]:
            return metrics
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="synthstrip_") as temporary_name:
        temporary_mask = Path(temporary_name) / "mask.nii.gz"
        environment = os.environ.copy()
        environment.update({
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(int(gpu_physical_id)),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
        })
        command = [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"), str(SYNTHSTRIP_SCRIPT),
            "-i", str(source_path), "-m", str(temporary_mask), "-g", "-b", "1",
            "--model", str(SYNTHSTRIP_MODEL),
        ]
        completed = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=1800)
        log_path.write_text(
            "command: " + " ".join(command) + "\n"
            + f"return_code: {completed.returncode}\n"
            + "stdout:\n" + completed.stdout + "\nstderr:\n" + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0 or not temporary_mask.exists():
            message = (completed.stderr or completed.stdout or "SynthStrip produced no mask").strip().splitlines()[-1]
            raise RuntimeError(f"SynthStrip failed: {message[:300]}")
        source_image = nib.load(str(source_path))
        mask_data = (np.asarray(nib.load(str(temporary_mask)).dataobj) > 0.5).astype(np.uint8)
        staged = mask_path.with_name(f".{mask_path.name}.tmp.nii.gz")
        standardized_mask = nib.Nifti1Image(mask_data, source_image.affine, source_image.header)
        standardized_mask.set_data_dtype(np.uint8)
        standardized_mask.set_qform(source_image.affine, code=1)
        standardized_mask.set_sform(source_image.affine, code=1)
        nib.save(standardized_mask, str(staged))
        os.replace(staged, mask_path)
    metrics = synthstrip_mask_qc(source_path, mask_path)
    if not metrics["synthstrip_qc_pass"]:
        raise ValueError("SynthStrip mask failed automated QC")
    return metrics


def _run_n4(source_path: Path, mask_path: Path, output_path: Path, replace: bool) -> dict[str, object]:
    if output_path.exists() and not replace:
        metrics = n4_qc(source_path, output_path, mask_path)
        if metrics["n4_qc_pass"]:
            return metrics
    import ants

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ants_n4_") as temporary_name:
        temporary_output = Path(temporary_name) / "n4_corrected.nii.gz"
        image = ants.image_read(str(source_path), pixeltype="float")
        loaded_mask = ants.image_read(str(mask_path), pixeltype="float")
        if tuple(loaded_mask.shape) != tuple(image.shape):
            raise ValueError("SynthStrip mask and source shapes differ before N4")
        # NIfTI qform quaternion rounding can produce micrometre-scale spacing
        # differences. The arrays already passed affine QC; force the binary mask
        # onto the exact source geometry before passing both images to ITK/ANTs.
        mask = ants.copy_image_info(image, loaded_mask)
        corrected = ants.n4_bias_field_correction(
            image,
            mask=mask,
            rescale_intensities=N4_PARAMETERS["rescale_intensities"],
            shrink_factor=N4_PARAMETERS["shrink_factor"],
            convergence=N4_PARAMETERS["convergence"],
            spline_param=N4_PARAMETERS["spline_param"],
            verbose=False,
        )
        ants.image_write(corrected, str(temporary_output))
        staged = output_path.with_name(f".{output_path.name}.tmp.nii.gz")
        shutil.copy2(temporary_output, staged)
        os.replace(staged, output_path)
    metrics = n4_qc(source_path, output_path, mask_path)
    if not metrics["n4_qc_pass"]:
        raise ValueError("Mask-guided ANTs N4 output failed automated QC")
    return metrics


def _apply_mask(corrected_path: Path, mask_path: Path, brain_path: Path, replace: bool) -> dict[str, object]:
    if brain_path.exists() and not replace:
        metrics = skullstrip_qc(corrected_path, brain_path, mask_path)
        if metrics["native_qc_pass"]:
            return metrics
    corrected = nib.load(str(corrected_path))
    mask_image = nib.load(str(mask_path))
    mask = np.asarray(mask_image.dataobj) > 0.5
    data = np.asarray(corrected.dataobj, dtype=np.float32)
    if data.shape != mask.shape:
        raise ValueError("N4 image and SynthStrip mask shapes differ")
    data[~mask] = 0.0
    brain_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = brain_path.with_name(f".{brain_path.name}.tmp.nii.gz")
    output = nib.Nifti1Image(data, corrected.affine, corrected.header)
    output.set_data_dtype(np.float32)
    output.set_qform(corrected.affine, code=1)
    output.set_sform(corrected.affine, code=1)
    nib.save(output, str(temporary))
    os.replace(temporary, brain_path)
    metrics = skullstrip_qc(corrected_path, brain_path, mask_path)
    if not metrics["native_qc_pass"]:
        raise ValueError("Mask application/skull stripping failed automated QC")
    return metrics


def _run_registration(
    brain_path: Path,
    mask_path: Path,
    registered_path: Path,
    registered_mask_path: Path,
    transform_path: Path,
    robust_initialization: bool = False,
) -> None:
    import ants

    registered_path.parent.mkdir(parents=True, exist_ok=True)
    registered_mask_path.parent.mkdir(parents=True, exist_ok=True)
    transform_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ants_affine_") as temporary_name:
        temporary_dir = Path(temporary_name)
        fixed = ants.image_read(str(MNI_TEMPLATE), pixeltype="float")
        moving = ants.image_read(str(brain_path), pixeltype="float")
        loaded_moving_mask = ants.image_read(str(mask_path), pixeltype="float")
        if tuple(loaded_moving_mask.shape) != tuple(moving.shape):
            raise ValueError("SynthStrip mask and skull-stripped image shapes differ before registration")
        moving_mask = ants.copy_image_info(moving, loaded_moving_mask)
        fixed_mask = fixed.new_image_like((fixed.numpy() > 0).astype(np.uint8))
        initial_transform = None
        if robust_initialization:
            initial_transform = ants.affine_initializer(
                fixed, moving, search_factor=20, radian_fraction=0.25,
                use_principal_axis=True, local_search_iterations=10,
                txfn=str(temporary_dir / "initializer.mat"),
            )
        registration = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="Affine",
            initial_transform=initial_transform,
            outprefix=str(temporary_dir / "transform_"),
            mask=fixed_mask,
            moving_mask=moving_mask,
            mask_all_stages=True,
            aff_metric="mattes",
            aff_sampling=32,
            aff_random_sampling_rate=0.2,
            aff_iterations=(2100, 1200, 1200, 10),
            aff_shrink_factors=(6, 4, 2, 1),
            aff_smoothing_sigmas=(3, 2, 1, 0),
            random_seed=42,
            singleprecision=True,
            use_legacy_histogram_matching=False,
            write_composite_transform=True,
            verbose=False,
        )
        raw_transforms = registration["fwdtransforms"]
        transform_list = [raw_transforms] if isinstance(raw_transforms, str) else list(raw_transforms)
        warped_brain = ants.apply_transforms(
            fixed=fixed, moving=moving, transformlist=transform_list,
            interpolator="linear", singleprecision=True,
        )
        warped_mask = ants.apply_transforms(
            fixed=fixed, moving=moving_mask, transformlist=transform_list,
            interpolator="nearestNeighbor", singleprecision=True,
        )
        mask_array = (warped_mask.numpy() > 0.5).astype(np.uint8)
        warped_mask = warped_mask.new_image_like(mask_array)
        brain_array = warped_brain.numpy().astype(np.float32)
        brain_array[mask_array == 0] = 0.0
        warped_brain = warped_brain.new_image_like(brain_array)
        temporary_brain = temporary_dir / "registered_brain.nii.gz"
        temporary_mask = temporary_dir / "registered_mask.nii.gz"
        ants.image_write(warped_brain, str(temporary_brain))
        ants.image_write(warped_mask, str(temporary_mask))
        generated_transform = Path(transform_list[0])
        staged_brain = registered_path.with_name(f".{registered_path.name}.tmp.nii.gz")
        staged_mask = registered_mask_path.with_name(f".{registered_mask_path.name}.tmp.nii.gz")
        staged_transform = transform_path.with_name(f".{transform_path.name}.tmp")
        shutil.copy2(temporary_brain, staged_brain)
        shutil.copy2(temporary_mask, staged_mask)
        shutil.copy2(generated_transform, staged_transform)
        os.replace(staged_brain, registered_path)
        os.replace(staged_mask, registered_mask_path)
        os.replace(staged_transform, transform_path)


def _normalize_registered(
    registered_path: Path,
    registered_mask_path: Path,
    normalized_path: Path,
    replace: bool,
) -> dict[str, object]:
    if normalized_path.exists() and not replace:
        metrics = normalization_qc(normalized_path, registered_mask_path, MNI_TEMPLATE)
        if metrics["normalization_qc_pass"]:
            return metrics
    registered = nib.load(str(registered_path))
    mask_image = nib.load(str(registered_mask_path))
    data = np.asarray(registered.dataobj, dtype=np.float32)
    mask = np.asarray(mask_image.dataobj) > 0.5
    if not mask.any():
        raise ValueError("Registered brain mask is empty")
    brain = data[mask].astype(np.float64)
    mean = float(brain.mean())
    std = float(brain.std())
    if not np.isfinite([mean, std]).all() or std <= 1e-6:
        raise ValueError("Registered brain intensity standard deviation is near zero")
    normalized = np.zeros(data.shape, dtype=np.float32)
    normalized[mask] = ((data[mask].astype(np.float64) - mean) / std).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Normalized output contains NaN or Inf")
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = normalized_path.with_name(f".{normalized_path.name}.tmp.nii.gz")
    output = nib.Nifti1Image(normalized, registered.affine, registered.header)
    output.set_data_dtype(np.float32)
    output.set_qform(registered.affine, code=1)
    output.set_sform(registered.affine, code=1)
    nib.save(output, str(temporary))
    os.replace(temporary, normalized_path)
    metrics = normalization_qc(normalized_path, registered_mask_path, MNI_TEMPLATE)
    if not metrics["normalization_qc_pass"]:
        raise ValueError("Brain-masked Z-score output failed automated QC")
    return metrics


def _initial_status(dataset: str, rows: pd.DataFrame) -> pd.DataFrame:
    root = dataset_output_root(dataset)
    records: list[dict[str, object]] = []
    for row in rows.itertuples(index=False):
        supplied_key = getattr(row, "record_key", "") if dataset == "AIBL" else ""
        key = str(supplied_key) if supplied_key and str(supplied_key) != "nan" else anonymous_key(
            dataset, getattr(row, "image_id", ""), getattr(row, "scan_id", ""), row.raw_path
        )
        cache_path = root / "source_cache" / f"{dataset.lower()}_{key}.nii.gz"
        source_path = str(row.raw_path)
        prepared_path = relative_to_data(cache_path, DATA_ROOT)
        stem = f"{dataset.lower()}_{key}"
        record: dict[str, object] = {column: "" for column in STATUS_COLUMNS}
        record.update({
            "dataset": dataset,
            "record_key": key,
            "subject_id": getattr(row, "subject_id", ""),
            "image_id": getattr(row, "image_id", ""),
            "scan_id": getattr(row, "scan_id", ""),
            "source_path": source_path,
            "prepared_3d_path": prepared_path,
            "synthstrip_mask_path": relative_to_data(root / "synthstrip" / "masks" / f"{stem}_synthstrip_mask.nii.gz", DATA_ROOT),
            "bias_corrected_path": relative_to_data(root / "bias_corrected" / "images" / f"{stem}_synthstrip_n4.nii.gz", DATA_ROOT),
            "brain_path": relative_to_data(root / "skullstripped" / "images" / f"{stem}_synthstrip_n4_brain.nii.gz", DATA_ROOT),
            "mask_path": relative_to_data(root / "synthstrip" / "masks" / f"{stem}_synthstrip_mask.nii.gz", DATA_ROOT),
            "registered_brain_path": relative_to_data(root / "registered" / "images" / f"{stem}_synthstrip_n4_mni_affine.nii.gz", DATA_ROOT),
            "registered_mask_path": relative_to_data(root / "registered" / "masks" / f"{stem}_synthstrip_mni_affine_mask.nii.gz", DATA_ROOT),
            "normalized_path": relative_to_data(root / "normalized" / "images" / f"{stem}_synthstrip_n4_mni_affine_zscore.nii.gz", DATA_ROOT),
            "transform_path": relative_to_data(root / "registered" / "transforms" / f"{stem}_mni_affine_composite.h5", DATA_ROOT),
            "template_path": relative_to_data(MNI_TEMPLATE, DATA_ROOT),
            "pipeline_version": PIPELINE_VERSION,
            "status": "pending",
            "stage_completed": "none",
            "qc_pass": False,
            "synthstrip_qc_pass": False,
            "n4_qc_pass": False,
            "native_qc_pass": False,
            "registration_qc_pass": False,
            "normalization_qc_pass": False,
            "registration_retry_status": "not_run",
            "attempts": 0,
        })
        records.append(record)
    return pd.DataFrame(records, columns=STATUS_COLUMNS).sort_values("record_key").reset_index(drop=True).astype(object)


def _merge_previous(current: pd.DataFrame, status_path: Path) -> pd.DataFrame:
    if not status_path.exists():
        return current
    previous_frame = pd.read_csv(status_path, low_memory=False).astype(object)
    if "pipeline_version" not in previous_frame.columns:
        return current
    previous_frame = previous_frame[previous_frame["pipeline_version"].astype(str).eq(PIPELINE_VERSION)]
    if previous_frame.empty:
        return current
    previous = previous_frame.drop_duplicates("record_key", keep="last").set_index("record_key")
    immutable = {
        "dataset", "record_key", "subject_id", "image_id", "scan_id", "source_path",
        "synthstrip_mask_path", "bias_corrected_path", "brain_path", "mask_path",
        "registered_brain_path", "registered_mask_path", "normalized_path", "transform_path",
        "template_path", "pipeline_version",
    }
    if "prepared_3d_path" in previous.columns:
        prior = current["record_key"].map(previous["prepared_3d_path"])
        valid = prior.map(lambda value: bool(pd.notna(value) and absolute_from_data(value, DATA_ROOT).exists()))
        current.loc[valid, "prepared_3d_path"] = prior[valid]
    for column in [name for name in STATUS_COLUMNS if name not in immutable | {"prepared_3d_path"}]:
        if column in previous.columns:
            current[column] = current["record_key"].map(previous[column]).fillna(current[column])
    return current.astype(object)


def _apply_metrics(indexed: pd.DataFrame, key: str, metrics: dict[str, object]) -> None:
    for name, value in metrics.items():
        if name in indexed.columns:
            indexed.at[key, name] = value


def _checkpoint(indexed: pd.DataFrame, status_path: Path) -> None:
    atomic_write_csv(indexed.reset_index(drop=True)[STATUS_COLUMNS], status_path)


def _write_reproducibility_log(dataset: str, gpu_physical_id: int) -> None:
    root = dataset_output_root(dataset)
    template = nib.load(str(MNI_TEMPLATE))
    packages = {}
    for name in ("numpy", "nibabel", "pandas", "scipy", "surfa", "torch", "antspyx"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    payload = {
        "generated_utc": _now(),
        "pipeline_version": PIPELINE_VERSION,
        "dataset": dataset,
        "python": platform.python_version(),
        "packages": packages,
        "gpu_policy": {"physical_gpu_id": int(gpu_physical_id), "cuda_visible_devices_inside_worker": "0"},
        "synthstrip": {
            "script": relative_to_data(SYNTHSTRIP_SCRIPT, DATA_ROOT),
            "script_sha256": _sha256(SYNTHSTRIP_SCRIPT),
            "source": SYNTHSTRIP_SOURCE,
            "model": relative_to_data(SYNTHSTRIP_MODEL, DATA_ROOT),
            "model_sha256": _sha256(SYNTHSTRIP_MODEL),
            "model_source": SYNTHSTRIP_MODEL_SOURCE,
            "parameters": {"gpu": True, "border_mm": 1, "no_csf": False},
        },
        "n4": N4_PARAMETERS,
        "registration": REGISTRATION_PARAMETERS,
        "template": {
            "path": relative_to_data(MNI_TEMPLATE, DATA_ROOT),
            "sha256": _sha256(MNI_TEMPLATE),
            "shape": list(map(int, template.shape)),
            "spacing_mm": list(map(float, template.header.get_zooms()[:3])),
            "affine": np.asarray(template.affine).tolist(),
        },
        "normalization": {
            "scope": "per-volume brain voxels only",
            "formula": "(x - brain_mean) / brain_std",
            "background_value": 0,
        },
    }
    _atomic_json(payload, root / "logs" / "preprocessing_run_config.json")


def run_dataset(
    dataset: str,
    max_files: int | None = 3,
    replace: bool = False,
    save_qc_plots: bool = True,
    max_qc_plots: int = 3,
    gpu_physical_id: int = 2,
    progress: bool = True,
) -> pd.DataFrame:
    dataset = dataset.upper()
    for required in (MNI_TEMPLATE, SYNTHSTRIP_SCRIPT, SYNTHSTRIP_MODEL):
        if not required.exists():
            raise FileNotFoundError(f"Required pipeline resource is missing: {required}")
    rows = _dataset_rows(dataset)
    root = dataset_output_root(dataset)
    status_path = root / "manifests" / f"{dataset.lower()}_preprocessing_manifest.csv"
    status = _merge_previous(_initial_status(dataset, rows), status_path)
    eligible = status[(status["status"] != "success") | bool(replace)].copy()
    if max_files is not None:
        eligible = eligible.head(int(max_files))
    _write_reproducibility_log(dataset, gpu_physical_id)

    total_selected = len(eligible)
    if progress:
        existing_success = int(status["status"].eq("success").sum())
        print(
            f"{dataset}: selected {total_selected} scans; "
            f"already successful {existing_success}; gpu {gpu_physical_id}",
            flush=True,
        )

    indexed = status.set_index("record_key", drop=False)
    for position, row in enumerate(eligible.itertuples(index=False), start=1):
        key = str(row.record_key)
        started = time.monotonic()
        if progress:
            print(f"{dataset}: scan {position}/{total_selected} started", flush=True)
        indexed.at[key, "attempts"] = int(float(row.attempts or 0)) + 1
        indexed.at[key, "qc_pass"] = False
        indexed.at[key, "error_type"] = ""
        indexed.at[key, "error_message"] = ""
        try:
            paths = {
                "source": absolute_from_data(row.source_path, DATA_ROOT),
                "cache": absolute_from_data(row.prepared_3d_path, DATA_ROOT),
                "mask": absolute_from_data(row.synthstrip_mask_path, DATA_ROOT),
                "n4": absolute_from_data(row.bias_corrected_path, DATA_ROOT),
                "brain": absolute_from_data(row.brain_path, DATA_ROOT),
                "registered": absolute_from_data(row.registered_brain_path, DATA_ROOT),
                "registered_mask": absolute_from_data(row.registered_mask_path, DATA_ROOT),
                "normalized": absolute_from_data(row.normalized_path, DATA_ROOT),
                "transform": absolute_from_data(row.transform_path, DATA_ROOT),
            }
            if not paths["source"].exists():
                raise FileNotFoundError("Source MRI does not exist")
            cache_path = paths["cache"]
            if cache_path.resolve() == paths["source"].resolve():
                cache_path = root / "source_cache" / f"{dataset.lower()}_{key}.nii.gz"
            prepared = _prepare_nifti(paths["source"], cache_path, replace)
            indexed.at[key, "prepared_3d_path"] = relative_to_data(prepared, DATA_ROOT)
            indexed.at[key, "status"] = "running_synthstrip"
            indexed.at[key, "stage_completed"] = "source_prepared"
            indexed.at[key, "updated_utc"] = _now()
            _checkpoint(indexed, status_path)

            mask_metrics = _run_synthstrip(
                prepared, paths["mask"], root / "logs" / "synthstrip" / f"{key}.log",
                gpu_physical_id, replace,
            )
            _apply_metrics(indexed, key, mask_metrics)
            indexed.at[key, "status"] = "running_n4"
            indexed.at[key, "stage_completed"] = "synthstrip_mask"
            indexed.at[key, "updated_utc"] = _now()
            _checkpoint(indexed, status_path)

            n4_metrics = _run_n4(prepared, paths["mask"], paths["n4"], replace)
            _apply_metrics(indexed, key, n4_metrics)
            indexed.at[key, "status"] = "running_skullstrip"
            indexed.at[key, "stage_completed"] = "mask_guided_n4"
            indexed.at[key, "updated_utc"] = _now()
            _checkpoint(indexed, status_path)

            native_metrics = _apply_mask(paths["n4"], paths["mask"], paths["brain"], replace)
            _apply_metrics(indexed, key, native_metrics)
            indexed.at[key, "status"] = "running_registration"
            indexed.at[key, "stage_completed"] = "skullstripped"
            indexed.at[key, "updated_utc"] = _now()
            _checkpoint(indexed, status_path)

            registration_valid = False
            if all(paths[name].exists() for name in ("registered", "registered_mask", "transform")) and not replace:
                reg_metrics = registration_qc(MNI_TEMPLATE, paths["registered"], paths["registered_mask"])
                registration_valid = bool(reg_metrics["registration_qc_pass"])
            if not registration_valid:
                _run_registration(
                    paths["brain"], paths["mask"], paths["registered"],
                    paths["registered_mask"], paths["transform"], robust_initialization=False,
                )
                reg_metrics = registration_qc(MNI_TEMPLATE, paths["registered"], paths["registered_mask"])
                indexed.at[key, "registration_retry_status"] = "not_needed"
                if not reg_metrics["registration_qc_pass"]:
                    indexed.at[key, "registration_retry_status"] = "robust_initialization_running"
                    _checkpoint(indexed, status_path)
                    _run_registration(
                        paths["brain"], paths["mask"], paths["registered"],
                        paths["registered_mask"], paths["transform"], robust_initialization=True,
                    )
                    reg_metrics = registration_qc(MNI_TEMPLATE, paths["registered"], paths["registered_mask"])
                    indexed.at[key, "registration_retry_status"] = "robust_initialization_success"
            else:
                indexed.at[key, "registration_retry_status"] = "valid_existing_output"
            _apply_metrics(indexed, key, reg_metrics)
            if not reg_metrics["registration_qc_pass"]:
                indexed.at[key, "registration_retry_status"] = "failed"
                raise ValueError("Affine registration output failed automated QC")
            indexed.at[key, "status"] = "running_normalization"
            indexed.at[key, "stage_completed"] = "affine_mni_registration"
            indexed.at[key, "updated_utc"] = _now()
            _checkpoint(indexed, status_path)

            normalization_metrics = _normalize_registered(
                paths["registered"], paths["registered_mask"], paths["normalized"], replace
            )
            _apply_metrics(indexed, key, normalization_metrics)
            indexed.at[key, "status"] = "success"
            indexed.at[key, "stage_completed"] = "brain_masked_zscore"
            indexed.at[key, "qc_pass"] = True
        except Exception as exc:
            indexed.at[key, "status"] = "qc_failed" if isinstance(exc, ValueError) else "failed"
            indexed.at[key, "qc_pass"] = False
            indexed.at[key, "error_type"] = type(exc).__name__
            indexed.at[key, "error_message"] = str(exc)[:500]
        indexed.at[key, "updated_utc"] = _now()
        _checkpoint(indexed, status_path)
        if progress:
            result = str(indexed.at[key, "status"])
            elapsed = time.monotonic() - started
            print(
                f"{dataset}: scan {position}/{total_selected} {result} "
                f"in {elapsed:.1f}s",
                flush=True,
            )

    status = indexed.reset_index(drop=True)
    if save_qc_plots:
        save_anonymous_qc_examples(status, dataset, int(max_qc_plots))
    return status


def public_status_summary(status: pd.DataFrame) -> pd.DataFrame:
    return status.groupby(["dataset", "status"], dropna=False).size().rename("count").reset_index()
