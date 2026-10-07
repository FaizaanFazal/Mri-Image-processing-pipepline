from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io_utils import absolute_from_data, atomic_write_csv, relative_to_data
from .paths import DATA_ROOT, dataset_output_root
from .qc import save_pipeline_qc_plot


QC_STAGES = (
    ("SynthStrip brain mask", "synthstrip_qc_pass"),
    ("mask-guided N4", "n4_qc_pass"),
    ("native skull stripping", "native_qc_pass"),
    ("affine MNI registration", "registration_qc_pass"),
    ("brain-masked Z-score", "normalization_qc_pass"),
)

QC_METRICS = (
    ("brain-mask volume", "native_mask_volume_ml", "mL", "150 to 2500"),
    ("template overlap", "template_overlap_fraction", "fraction", ">= 0.85"),
    ("centroid distance", "centroid_distance_mm", "mm", "<= 30"),
    ("normalized brain mean", "normalized_brain_mean", "Z", "about 0"),
    ("normalized brain standard deviation", "normalized_brain_std", "Z", "about 1"),
    ("normalized background maximum", "normalized_background_abs_max", "absolute Z", "0"),
)


def _boolean(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result.loc[normalized.isin({"true", "1", "yes"})] = True
    result.loc[normalized.isin({"false", "0", "no"})] = False
    return result


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def build_dataset_qc_report(status: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build anonymous aggregate status and QC tables from a preprocessing manifest."""
    if status.empty:
        empty = pd.DataFrame()
        return {
            "status": empty,
            "stage_qc": empty,
            "sanity_checks": empty,
            "metric_summary": empty,
            "geometry": empty,
            "failures": empty,
        }

    total = len(status)
    status_counts = status["status"].fillna("missing").astype(str).value_counts()
    status_table = status_counts.rename_axis("status").reset_index(name="count")
    status_table["percent"] = (100.0 * status_table["count"] / total).round(2)

    stage_rows = []
    for label, column in QC_STAGES:
        values = _boolean(status[column]) if column in status.columns else pd.Series(pd.NA, index=status.index, dtype="boolean")
        evaluated = int(values.notna().sum())
        passed = int(values.eq(True).sum())
        failed = int(values.eq(False).sum())
        stage_rows.append({
            "stage": label,
            "passed": passed,
            "failed": failed,
            "not_evaluated": total - evaluated,
            "pass_rate_percent": round(100.0 * passed / evaluated, 2) if evaluated else np.nan,
        })
    stage_table = pd.DataFrame(stage_rows)

    successful = status[status["status"].astype(str).eq("success")].copy()
    final_paths = successful.get("normalized_path", pd.Series(index=successful.index, dtype=object))
    final_exists = final_paths.map(
        lambda value: bool(pd.notna(value) and str(value).strip() and absolute_from_data(value, DATA_ROOT).is_file())
    )

    def passed_count(column: str, predicate=None) -> tuple[int, int]:
        if column not in successful.columns:
            return 0, len(successful)
        if predicate is None:
            values = _boolean(successful[column])
            return int(values.eq(True).sum()), len(successful)
        values = pd.to_numeric(successful[column], errors="coerce")
        return int(predicate(values).fillna(False).sum()), len(successful)

    checks: list[dict[str, object]] = []

    def add_check(name: str, passed: int, expected: int, requirement: str) -> None:
        checks.append({
            "check": name,
            "passed": passed,
            "expected": expected,
            "result": "PASS" if passed == expected else "CHECK",
            "requirement": requirement,
        })

    add_check("dataset rows completed", len(successful), total, "status is success")
    add_check("final normalized files exist", int(final_exists.sum()), len(successful), "one final file per successful row")
    for label, column in (
        ("overall automated QC", "qc_pass"),
        ("final geometry matches MNI", "normalization_geometry_match"),
        ("final values are finite", "normalization_finite"),
    ):
        passed, expected = passed_count(column)
        add_check(label, passed, expected, "all successful rows")
    passed, expected = passed_count("normalized_brain_mean", lambda value: value.abs() <= 1e-3)
    add_check("brain-only normalized mean", passed, expected, "absolute mean <= 0.001")
    passed, expected = passed_count("normalized_brain_std", lambda value: value.between(0.99, 1.01))
    add_check("brain-only normalized standard deviation", passed, expected, "0.99 to 1.01")
    passed, expected = passed_count("normalized_background_abs_max", lambda value: value <= 1e-7)
    add_check("normalized background is zero", passed, expected, "absolute maximum <= 1e-7")
    sanity_table = pd.DataFrame(checks)

    metric_rows = []
    for label, column, unit, target in QC_METRICS:
        values = _numeric(successful, column)
        metric_rows.append({
            "metric": label,
            "unit": unit,
            "count": len(values),
            "median": values.median() if len(values) else np.nan,
            "p05": values.quantile(0.05) if len(values) else np.nan,
            "p95": values.quantile(0.95) if len(values) else np.nan,
            "minimum": values.min() if len(values) else np.nan,
            "maximum": values.max() if len(values) else np.nan,
            "target": target,
        })
    metric_table = pd.DataFrame(metric_rows).round(6)

    geometry_rows = []
    for label, column in (("final shape", "registered_shape"), ("final spacing (mm)", "registered_spacing_mm")):
        if column not in successful.columns:
            continue
        counts = successful[column].dropna().astype(str).value_counts()
        for value, count in counts.items():
            geometry_rows.append({"property": label, "value": value, "count": int(count)})
    geometry_table = pd.DataFrame(geometry_rows)

    failed = status[status["status"].astype(str).isin({"failed", "qc_failed"})].copy()
    if failed.empty:
        failure_table = pd.DataFrame(columns=["status", "error_type", "reason", "count"])
    else:
        failed["reason"] = failed.get("error_message", "").fillna("").astype(str).str.slice(0, 180)
        failure_table = (
            failed.groupby(["status", "error_type", "reason"], dropna=False)
            .size().rename("count").reset_index().sort_values("count", ascending=False).reset_index(drop=True)
        )

    return {
        "status": status_table,
        "stage_qc": stage_table,
        "sanity_checks": sanity_table,
        "metric_summary": metric_table,
        "geometry": geometry_table,
        "failures": failure_table,
    }


def save_dataset_qc_summary_plot(status: pd.DataFrame, dataset: str) -> Path:
    """Save one anonymous aggregate QC dashboard for a dataset."""
    dataset = dataset.upper()
    report = build_dataset_qc_report(status)
    successful = status[status["status"].astype(str).eq("success")]
    output = dataset_output_root(dataset) / "qc_examples" / "qc_summary.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))

    status_table = report["status"]
    axes[0, 0].bar(status_table["status"], status_table["count"], color="#4472C4")
    axes[0, 0].set_title("manifest status")
    axes[0, 0].tick_params(axis="x", rotation=25)
    axes[0, 0].set_ylabel("scans")

    stage_table = report["stage_qc"]
    axes[0, 1].barh(stage_table["stage"], stage_table["pass_rate_percent"], color="#70AD47")
    axes[0, 1].set_xlim(0, 101)
    axes[0, 1].set_title("QC pass rate among evaluated scans")
    axes[0, 1].set_xlabel("percent")

    histogram_specs = (
        ("native_mask_volume_ml", "brain-mask volume (mL)", "#5B9BD5"),
        ("template_overlap_fraction", "template overlap", "#ED7D31"),
        ("centroid_distance_mm", "centroid distance (mm)", "#A5A5A5"),
    )
    for axis, (column, title, color) in zip((axes[0, 2], axes[1, 0], axes[1, 1]), histogram_specs):
        values = _numeric(successful, column)
        if len(values):
            axis.hist(values, bins=min(30, max(5, int(np.sqrt(len(values))))), color=color, edgecolor="white")
            axis.axvline(values.median(), color="black", linestyle="--", linewidth=1, label=f"median {values.median():.3g}")
            axis.legend(fontsize=8)
        else:
            axis.text(0.5, 0.5, "not available", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(title)
        axis.set_ylabel("scans")

    mean = _numeric(successful, "normalized_brain_mean")
    std = _numeric(successful, "normalized_brain_std")
    background = _numeric(successful, "normalized_background_abs_max")
    axis = axes[1, 2]
    if len(mean) and len(std) and len(background):
        tolerance_ratios = [
            float(mean.abs().max() / 1e-3),
            float((std - 1.0).abs().max() / 0.01),
            float(background.abs().max() / 1e-7),
        ]
        labels = ["|mean|", "|std-1|", "background"]
        axis.bar(labels, tolerance_ratios, color=["#5B9BD5", "#ED7D31", "#70AD47"])
        axis.axhline(1.0, color="red", linestyle="--", linewidth=1, label="QC limit")
        axis.set_ylabel("maximum / QC limit")
        axis.tick_params(axis="x", rotation=20)
        axis.legend(fontsize=8)
    else:
        axis.text(0.5, 0.5, "not available", ha="center", va="center", transform=axis.transAxes)
    axis.set_title("final Z-score tolerance ratios")

    fig.suptitle(f"{dataset} anonymous preprocessing QC summary")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def anonymous_qc_plot_paths(dataset: str, limit: int = 3) -> list[Path]:
    qc_root = dataset_output_root(dataset) / "qc_examples"
    return sorted(qc_root.glob("qc_sample_*.png"))[: int(limit)]


def save_anonymous_qc_examples(
    status: pd.DataFrame,
    dataset: str,
    limit: int = 3,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Save deterministic-random visual QC examples identified only by manifest position."""
    dataset = dataset.upper()
    frame = status.reset_index(drop=True)
    successful_positions = np.flatnonzero(
        frame["status"].astype(str).eq("success").to_numpy()
        & _boolean(frame["qc_pass"]).fillna(False).to_numpy(dtype=bool)
    )
    columns = ["sample_number", "list_position", "plot_path"]
    if not len(successful_positions) or int(limit) <= 0:
        return pd.DataFrame(columns=columns)

    stable_seed = int(random_seed) + sum(ord(character) for character in dataset)
    generator = np.random.default_rng(stable_seed)
    selected = np.sort(
        generator.choice(successful_positions, size=min(int(limit), len(successful_positions)), replace=False)
    )
    qc_root = dataset_output_root(dataset) / "qc_examples"
    qc_root.mkdir(parents=True, exist_ok=True)
    records = []
    for sample_number, zero_based_position in enumerate(selected, start=1):
        row = frame.iloc[int(zero_based_position)]
        list_position = int(zero_based_position) + 1
        output_path = qc_root / f"qc_sample_{sample_number:03d}.png"
        save_pipeline_qc_plot(
            absolute_from_data(row["prepared_3d_path"], DATA_ROOT),
            absolute_from_data(row["synthstrip_mask_path"], DATA_ROOT),
            absolute_from_data(row["template_path"], DATA_ROOT),
            absolute_from_data(row["normalized_path"], DATA_ROOT),
            absolute_from_data(row["registered_mask_path"], DATA_ROOT),
            output_path,
            sample_number,
            list_position=list_position,
        )
        records.append({
            "sample_number": sample_number,
            "list_position": list_position,
            "plot_path": relative_to_data(output_path, DATA_ROOT),
        })

    selection = pd.DataFrame(records, columns=columns)
    atomic_write_csv(selection, qc_root / "qc_selection.csv")
    return selection
