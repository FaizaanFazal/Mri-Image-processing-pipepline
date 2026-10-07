from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .paths import PROJECT_ROOT, SOURCE_MANIFEST


DATASET_ORDER = ["ADNI", "AIBL", "OASIS2", "OASIS3"]


def harmonize_class(value: object) -> str:
    label = "" if pd.isna(value) else str(value).strip().lower()
    if not label:
        return "Unknown"
    if "normal" in label or label in {"cn", "hc", "control", "nondemented"}:
        return "Cognitively normal"
    if "mci" in label or "mild cognitive" in label or "impairment" in label or "memory only" in label:
        return "MCI"
    if (
        "dement" in label or "demt" in label or "dat" in label or "dlbd" in label
        or "frontotemporal" in label or label.startswith("ad ")
        or label in {"ad", "alzheimers disease", "alzheimer's disease"}
    ):
        return "Dementia"
    if "convert" in label:
        return "Converted"
    if "significant memory concern" in label or label == "smc":
        return "SMC"
    return str(value).strip()


def load_public_demographics() -> tuple[pd.DataFrame, pd.DataFrame]:
    scans = pd.read_csv(SOURCE_MANIFEST, low_memory=False)
    scans = scans[(scans["record_kind"] == "primary_mri") & scans["dataset"].isin(DATASET_ORDER)].copy()
    scans["age"] = pd.to_numeric(scans["age"], errors="coerce")
    scans.loc[~scans["age"].between(18, 110), "age"] = np.nan
    scans["class"] = scans["group_class"].where(scans["group_class"].notna(), scans["diagnosis"])
    scans["class"] = scans["class"].map(harmonize_class)
    scans["sex"] = scans["sex"].astype("string").str.upper().replace({"MALE": "M", "FEMALE": "F", "X": pd.NA, "NAN": pd.NA, "<NA>": pd.NA})
    scans = scans.sort_values(["dataset", "subject_id", "acquisition_date"], na_position="last")

    def first_present(series: pd.Series):
        values = series.dropna()
        return values.iloc[0] if len(values) else np.nan

    subjects = scans.groupby(["dataset", "subject_id"], as_index=False).agg(
        age=("age", first_present), sex=("sex", first_present), class_label=("class", first_present)
    )
    return scans, subjects


def aggregate_tables(scans: pd.DataFrame, subjects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for dataset in DATASET_ORDER:
        scan_part = scans[scans["dataset"] == dataset]
        subject_part = subjects[subjects["dataset"] == dataset]
        ages = subject_part["age"].dropna()
        rows.append({
            "dataset": dataset,
            "unique_patients": int(len(subject_part)),
            "mri_records": int(len(scan_part)),
            "age_available": int(ages.notna().sum()),
            "mean_age": round(float(ages.mean()), 2) if len(ages) else np.nan,
            "median_age": round(float(ages.median()), 2) if len(ages) else np.nan,
            "minimum_age": round(float(ages.min()), 2) if len(ages) else np.nan,
            "maximum_age": round(float(ages.max()), 2) if len(ages) else np.nan,
            "female": int(subject_part["sex"].eq("F").sum()),
            "male": int(subject_part["sex"].eq("M").sum()),
            "number_of_classes": int(subject_part.loc[subject_part["class_label"] != "Unknown", "class_label"].nunique()),
        })
    total_ages = subjects["age"].dropna()
    rows.append({
        "dataset": "TOTAL", "unique_patients": int(len(subjects)), "mri_records": int(len(scans)),
        "age_available": int(len(total_ages)), "mean_age": round(float(total_ages.mean()), 2),
        "median_age": round(float(total_ages.median()), 2), "minimum_age": round(float(total_ages.min()), 2),
        "maximum_age": round(float(total_ages.max()), 2), "female": int(subjects["sex"].eq("F").sum()),
        "male": int(subjects["sex"].eq("M").sum()),
        "number_of_classes": int(subjects.loc[subjects["class_label"] != "Unknown", "class_label"].nunique()),
    })
    overview = pd.DataFrame(rows)
    class_counts = subjects.groupby(["dataset", "class_label"], dropna=False).size().rename("unique_patients").reset_index()
    sex_counts = subjects.assign(sex=subjects["sex"].fillna("Unknown")).groupby(["dataset", "sex"]).size().rename("unique_patients").reset_index()
    return overview, class_counts, sex_counts


def save_public_plots(subjects: pd.DataFrame, output_dir: Path | None = None) -> list[Path]:
    output_dir = output_dir or PROJECT_ROOT / "results" / "demographics"
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    paths = []

    class_counts = subjects.groupby(["dataset", "class_label"]).size().rename("count").reset_index()
    fig, axis = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=class_counts, x="dataset", y="count", hue="class_label", order=DATASET_ORDER, ax=axis)
    axis.set(title="class distribution by dataset (unique patients)", xlabel="dataset", ylabel="unique patients")
    axis.legend(title="harmonized class", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    path = output_dir / "class_distribution.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    age_data = subjects.dropna(subset=["age"])
    fig, axis = plt.subplots(figsize=(10, 5.5))
    sns.violinplot(data=age_data, x="dataset", y="age", order=DATASET_ORDER, inner="quartile", cut=0, ax=axis)
    means = age_data.groupby("dataset")["age"].mean()
    for position, dataset in enumerate(DATASET_ORDER):
        if dataset in means:
            axis.scatter(position, means[dataset], color="red", marker="D", s=42, zorder=5)
            axis.text(position, means[dataset] + 1.2, f"mean {means[dataset]:.1f}", ha="center", fontsize=9)
    axis.set(title="age distribution by dataset (unique patients)", xlabel="dataset", ylabel="age (years)")
    fig.tight_layout()
    path = output_dir / "age_distribution.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    sex_counts = subjects.assign(sex=subjects["sex"].fillna("Unknown")).groupby(["dataset", "sex"]).size().rename("count").reset_index()
    fig, axis = plt.subplots(figsize=(9, 5))
    sns.barplot(data=sex_counts, x="dataset", y="count", hue="sex", order=DATASET_ORDER, ax=axis)
    axis.set(title="sex distribution by dataset (unique patients)", xlabel="dataset", ylabel="unique patients")
    fig.tight_layout()
    path = output_dir / "sex_distribution.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths
