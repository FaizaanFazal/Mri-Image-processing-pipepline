from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage


def _largest_component_fraction(mask: np.ndarray) -> float:
    labels, count = ndimage.label(mask)
    if not count:
        return 0.0
    sizes = np.bincount(labels.ravel())[1:]
    return float(sizes.max() / max(sizes.sum(), 1))


def volume_qc(path: Path) -> dict[str, object]:
    image = nib.load(str(path))
    shape = tuple(int(v) for v in image.shape)
    zooms = tuple(float(v) for v in image.header.get_zooms()[: len(shape)])
    data = np.asarray(image.dataobj, dtype=np.float32)
    finite = bool(np.isfinite(data).all())
    dynamic = float(np.nanmax(data) - np.nanmin(data)) if finite and data.size else np.nan
    is_3d = len(shape) == 3
    pass_qc = bool(is_3d and min(shape) >= 16 and finite and dynamic > 0 and all(z > 0 for z in zooms))
    return {
        "shape": "x".join(map(str, shape)),
        "voxel_sizes_mm": "x".join(f"{v:.4g}" for v in zooms),
        "is_3d": is_3d,
        "finite": finite,
        "intensity_range": dynamic,
        "qc_pass": pass_qc,
    }


def synthstrip_mask_qc(source_path: Path, mask_path: Path) -> dict[str, object]:
    source = nib.load(str(source_path))
    mask_image = nib.load(str(mask_path))
    raw_mask = np.asarray(mask_image.dataobj, dtype=np.float32)
    mask = raw_mask > 0.5
    shape_match = tuple(source.shape) == tuple(mask_image.shape)
    affine_match = bool(np.allclose(source.affine, mask_image.affine, atol=1e-3))
    _, source_qform_code = source.get_qform(coded=True)
    _, source_sform_code = source.get_sform(coded=True)
    _, mask_qform_code = mask_image.get_qform(coded=True)
    _, mask_sform_code = mask_image.get_sform(coded=True)
    metadata_explicit = bool(
        source_qform_code > 0 and source_sform_code > 0
        and mask_qform_code > 0 and mask_sform_code > 0
    )
    finite = bool(np.isfinite(raw_mask).all())
    binary = bool(finite and np.all((raw_mask == 0) | (raw_mask == 1)))
    fraction = float(mask.mean()) if mask.size else 0.0
    voxel_volume = float(np.prod(mask_image.header.get_zooms()[:3]))
    volume_ml = float(mask.sum() * voxel_volume / 1000.0)
    component_fraction = _largest_component_fraction(mask)
    passed = bool(
        shape_match and affine_match and metadata_explicit and finite and binary and mask.any()
        and 0.03 <= fraction <= 0.80
        and 150.0 <= volume_ml <= 2500.0
        and component_fraction >= 0.90
    )
    return {
        "synthstrip_shape_match": shape_match,
        "synthstrip_affine_match": affine_match,
        "synthstrip_metadata_explicit": metadata_explicit,
        "synthstrip_finite": finite,
        "synthstrip_binary": binary,
        "native_mask_fraction": fraction,
        "native_mask_volume_ml": volume_ml,
        "native_largest_component_fraction": component_fraction,
        "synthstrip_qc_pass": passed,
    }


def n4_qc(source_path: Path, corrected_path: Path, mask_path: Path) -> dict[str, object]:
    source = nib.load(str(source_path))
    corrected = nib.load(str(corrected_path))
    mask_image = nib.load(str(mask_path))
    data = np.asarray(corrected.dataobj, dtype=np.float32)
    mask = np.asarray(mask_image.dataobj) > 0.5
    shape_match = tuple(source.shape) == tuple(corrected.shape) == tuple(mask_image.shape)
    affine_match = bool(
        np.allclose(source.affine, corrected.affine, atol=1e-3)
        and np.allclose(source.affine, mask_image.affine, atol=1e-3)
    )
    finite = bool(np.isfinite(data).all())
    brain = data[mask] if shape_match and mask.any() else np.asarray([], dtype=np.float32)
    mean = float(np.mean(brain)) if brain.size else np.nan
    std = float(np.std(brain)) if brain.size else np.nan
    p01 = float(np.percentile(brain, 1)) if brain.size else np.nan
    p99 = float(np.percentile(brain, 99)) if brain.size else np.nan
    passed = bool(
        len(corrected.shape) == 3 and shape_match and affine_match and finite
        and brain.size > 0 and np.isfinite([mean, std, p01, p99]).all()
        and std > 1e-6 and p99 > p01
    )
    return {
        "n4_shape_match": shape_match,
        "n4_affine_match": affine_match,
        "n4_finite": finite,
        "n4_brain_mean": mean,
        "n4_brain_std": std,
        "n4_brain_p01": p01,
        "n4_brain_p99": p99,
        "n4_qc_pass": passed,
    }


def skullstrip_qc(source_path: Path, brain_path: Path, mask_path: Path) -> dict[str, object]:
    source = nib.load(str(source_path))
    brain = nib.load(str(brain_path))
    mask_image = nib.load(str(mask_path))
    source_shape = tuple(int(v) for v in source.shape)
    brain_data = np.asarray(brain.dataobj, dtype=np.float32)
    mask = np.asarray(mask_image.dataobj) > 0.5
    shape_match = source_shape == tuple(brain.shape) == tuple(mask_image.shape)
    affine_match = bool(
        np.allclose(source.affine, brain.affine, atol=1e-3)
        and np.allclose(source.affine, mask_image.affine, atol=1e-3)
    )
    finite = bool(np.isfinite(brain_data).all())
    background_abs_max = float(np.max(np.abs(brain_data[~mask]))) if shape_match and (~mask).any() else 0.0
    passed = bool(
        len(source_shape) == 3 and shape_match and affine_match and finite and mask.any()
        and background_abs_max <= 1e-6 and np.std(brain_data[mask]) > 1e-6
    )
    return {
        "source_shape": "x".join(map(str, source_shape)),
        "native_shape_match": shape_match,
        "native_affine_match": affine_match,
        "native_finite": finite,
        "native_background_abs_max": background_abs_max,
        "native_qc_pass": passed,
    }


def registration_qc(template_path: Path, registered_path: Path, registered_mask_path: Path) -> dict[str, object]:
    template = nib.load(str(template_path))
    registered = nib.load(str(registered_path))
    mask_image = nib.load(str(registered_mask_path))
    template_data = np.asarray(template.dataobj, dtype=np.float32)
    registered_data = np.asarray(registered.dataobj, dtype=np.float32)
    mask = np.asarray(mask_image.dataobj) > 0.5
    template_mask = template_data > 0
    shape_match = tuple(template.shape) == tuple(registered.shape) == tuple(mask_image.shape)
    spacing_match = bool(
        np.allclose(template.header.get_zooms()[:3], registered.header.get_zooms()[:3], atol=1e-4)
        and np.allclose(template.header.get_zooms()[:3], mask_image.header.get_zooms()[:3], atol=1e-4)
    )
    affine_match = bool(
        np.allclose(template.affine, registered.affine, atol=1e-3)
        and np.allclose(template.affine, mask_image.affine, atol=1e-3)
    )
    finite = bool(np.isfinite(registered_data).all())
    mask_voxels = int(mask.sum())
    mask_fraction = float(mask.mean()) if mask.size else 0.0
    overlap = float((mask & template_mask).sum() / max(mask_voxels, 1))
    component_fraction = _largest_component_fraction(mask)
    if mask_voxels and template_mask.any():
        subject_center = np.asarray(ndimage.center_of_mass(mask), dtype=float)
        template_center = np.asarray(ndimage.center_of_mass(template_mask), dtype=float)
        subject_world = nib.affines.apply_affine(template.affine, subject_center)
        template_world = nib.affines.apply_affine(template.affine, template_center)
        centroid_distance = float(np.linalg.norm(subject_world - template_world))
    else:
        centroid_distance = np.inf
    background_abs_max = float(np.max(np.abs(registered_data[~mask]))) if (~mask).any() else 0.0
    passed = bool(
        shape_match and spacing_match and affine_match and finite and mask_voxels > 0
        and 0.02 <= mask_fraction <= 0.60 and overlap >= 0.85
        and component_fraction >= 0.90 and centroid_distance <= 30.0
        and background_abs_max <= 1e-5
    )
    return {
        "registered_shape": "x".join(map(str, registered.shape)),
        "registered_spacing_mm": "x".join(f"{v:.4g}" for v in registered.header.get_zooms()[:3]),
        "registration_shape_match": shape_match,
        "registration_spacing_match": spacing_match,
        "registration_affine_match": affine_match,
        "registration_finite": finite,
        "registered_mask_fraction": mask_fraction,
        "registered_largest_component_fraction": component_fraction,
        "template_overlap_fraction": overlap,
        "centroid_distance_mm": centroid_distance,
        "registered_background_abs_max": background_abs_max,
        "registration_qc_pass": passed,
    }


def normalization_qc(normalized_path: Path, registered_mask_path: Path, template_path: Path) -> dict[str, object]:
    normalized = nib.load(str(normalized_path))
    mask_image = nib.load(str(registered_mask_path))
    template = nib.load(str(template_path))
    data = np.asarray(normalized.dataobj, dtype=np.float32)
    mask = np.asarray(mask_image.dataobj) > 0.5
    geometry_match = bool(
        tuple(normalized.shape) == tuple(mask_image.shape) == tuple(template.shape)
        and np.allclose(normalized.affine, template.affine, atol=1e-3)
        and np.allclose(normalized.header.get_zooms()[:3], template.header.get_zooms()[:3], atol=1e-4)
    )
    finite = bool(np.isfinite(data).all())
    brain = data[mask] if mask.any() else np.asarray([], dtype=np.float32)
    mean = float(np.mean(brain)) if brain.size else np.nan
    std = float(np.std(brain)) if brain.size else np.nan
    background_abs_max = float(np.max(np.abs(data[~mask]))) if (~mask).any() else 0.0
    passed = bool(
        geometry_match and finite and brain.size > 0
        and abs(mean) <= 1e-3 and 0.99 <= std <= 1.01
        and background_abs_max <= 1e-7
    )
    return {
        "normalization_geometry_match": geometry_match,
        "normalization_finite": finite,
        "normalized_brain_mean": mean,
        "normalized_brain_std": std,
        "normalized_background_abs_max": background_abs_max,
        "normalization_qc_pass": passed,
    }


def save_pipeline_qc_plot(
    native_source_path: Path,
    native_mask_path: Path,
    template_path: Path,
    normalized_path: Path,
    registered_mask_path: Path,
    output_path: Path,
    sample_number: int,
    list_position: int | None = None,
) -> None:
    native = np.asarray(nib.load(str(native_source_path)).dataobj, dtype=np.float32)
    native_mask = np.asarray(nib.load(str(native_mask_path)).dataobj) > 0.5
    native_brain = np.zeros_like(native, dtype=np.float32)
    native_brain[native_mask] = native[native_mask]
    normalized = np.asarray(nib.load(str(normalized_path)).dataobj, dtype=np.float32)
    registered_mask = np.asarray(nib.load(str(registered_mask_path)).dataobj) > 0.5
    template = np.asarray(nib.load(str(template_path)).dataobj, dtype=np.float32)

    def middle_views(data: np.ndarray) -> list[np.ndarray]:
        center = tuple(int(size // 2) for size in data.shape)
        return [data[center[0], :, :], data[:, center[1], :], data[:, :, center[2]]]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    for axis, view, overlay in zip(axes[0], middle_views(native_brain), middle_views(native_mask)):
        axis.imshow(np.rot90(view), cmap="gray")
        axis.contour(np.rot90(overlay), levels=[0.5], colors="lime", linewidths=0.6)
        axis.axis("off")
    for axis, view, overlay, template_overlay in zip(
        axes[1], middle_views(normalized), middle_views(registered_mask), middle_views(template > 0)
    ):
        axis.imshow(np.rot90(view), cmap="gray")
        axis.contour(np.rot90(template_overlay), levels=[0.5], colors="deepskyblue", linewidths=0.5, linestyles="--")
        axis.contour(np.rot90(overlay), levels=[0.5], colors="lime", linewidths=0.6)
        axis.axis("off")
    axes[0, 0].set_title("native skull-stripped + synthstrip mask", loc="left", fontsize=9)
    axes[1, 0].set_title("affine mni + z-score", loc="left", fontsize=9)
    title = f"anonymous pipeline qc sample {sample_number:03d}"
    if list_position is not None:
        title += f" | manifest position {int(list_position):05d}"
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
