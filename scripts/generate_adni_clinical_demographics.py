#!/usr/bin/env python3
"""Create aggregate, identifier-free demographics from the ADNI clinical master."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from build_clinical_masters import atomic_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent
CLINICAL_ROOT = DATA_ROOT / "Processed" / "Clinical" / "adni"
MASTER_PATH = CLINICAL_ROOT / "adni_clinical_master.csv"
OVERVIEW_PATH = CLINICAL_ROOT / "adni_clinical_demographics_summary.csv"
COUNTS_PATH = CLINICAL_ROOT / "adni_clinical_demographics_counts.csv"
PLOT_PATH = PROJECT_ROOT / "results" / "demographics" / "adni_clinical_demographics.png"
FIELDS = ["age", "sex", "education", "race", "ethnicity", "handedness", "diagnosis_or_group"]


def first_present(series: pd.Series) -> str:
    values = series.fillna("").astype(str).str.strip()
    values = values.loc[values != ""]
    return str(values.iloc[0]) if not values.empty else ""


def main() -> None:
    usecols = ["subject_id", "event_time", *FIELDS]
    records = pd.read_csv(MASTER_PATH, usecols=usecols, dtype=str, keep_default_na=False)
    records["event_date"] = pd.to_datetime(records["event_time"], format="mixed", errors="coerce")
    records["age_numeric"] = pd.to_numeric(records["age"], errors="coerce")
    records.loc[~records["age_numeric"].between(18, 110), "age_numeric"] = np.nan
    records["education_numeric"] = pd.to_numeric(records["education"], errors="coerce")
    records.loc[~records["education_numeric"].between(0, 20), "education_numeric"] = np.nan
    records = records.sort_values(["subject_id", "event_date"], kind="stable", na_position="last")

    subjects = records.groupby("subject_id", as_index=False).agg(
        age=("age_numeric", lambda values: values.dropna().iloc[0] if values.notna().any() else np.nan),
        education=("education_numeric", lambda values: values.dropna().iloc[0] if values.notna().any() else np.nan),
        sex=("sex", first_present),
        race=("race", first_present),
        ethnicity=("ethnicity", first_present),
        handedness=("handedness", first_present),
        baseline_diagnosis=("diagnosis_or_group", first_present),
    )
    subjects["baseline_diagnosis"] = subjects["baseline_diagnosis"].replace(
        {"AD": "Dementia", "EMCI": "MCI", "LMCI": "MCI", "Patient": ""}
    )

    ages = subjects["age"].dropna()
    education = subjects["education"].dropna()
    total = len(subjects)
    overview = pd.DataFrame(
        [
            {
                "master_rows": len(records),
                "unique_subjects": total,
                "age_available": len(ages),
                "age_coverage_percent": round(100 * len(ages) / total, 2),
                "mean_baseline_age": round(float(ages.mean()), 2),
                "median_baseline_age": round(float(ages.median()), 2),
                "minimum_baseline_age": round(float(ages.min()), 2),
                "maximum_baseline_age": round(float(ages.max()), 2),
                "sex_available": int(subjects["sex"].ne("").sum()),
                "education_available": len(education),
                "education_coverage_percent": round(100 * len(education) / total, 2),
                "mean_education_years": round(float(education.mean()), 2),
                "median_education_years": round(float(education.median()), 2),
                "race_available": int(subjects["race"].ne("").sum()),
                "ethnicity_available": int(subjects["ethnicity"].ne("").sum()),
                "handedness_available": int(subjects["handedness"].ne("").sum()),
                "baseline_diagnosis_available": int(subjects["baseline_diagnosis"].ne("").sum()),
            }
        ]
    )

    count_rows: list[dict[str, object]] = []
    for field in ("sex", "race", "ethnicity", "handedness", "baseline_diagnosis"):
        values = subjects[field].replace("", "Missing")
        value_counts = values.value_counts(dropna=False)
        rare_categories = set(value_counts[value_counts < 10].index)
        values = values.map(lambda value: "Other/combined (categories <10)" if value in rare_categories else value)
        for category, count in values.value_counts(dropna=False).items():
            count_rows.append(
                {
                    "dimension": field,
                    "category": category,
                    "unique_subjects": int(count),
                    "percent": round(100 * count / total, 2),
                }
            )
    counts = pd.DataFrame(count_rows)
    atomic_csv(overview, OVERVIEW_PATH)
    atomic_csv(counts, COUNTS_PATH)

    plot_subjects = subjects.copy()
    race_counts = plot_subjects["race"].replace("", "Missing").value_counts()
    rare_races = set(race_counts[race_counts < 25].index)
    plot_subjects["race_plot"] = plot_subjects["race"].replace("", "Missing").map(
        lambda value: "Other/multiple" if value in rare_races else value
    )
    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(data=plot_subjects, x="age", bins=24, ax=axes[0, 0])
    axes[0, 0].axvline(ages.mean(), color="red", linestyle="--", label=f"mean {ages.mean():.1f}")
    axes[0, 0].set(title="ADNI baseline age", xlabel="age (years)", ylabel="subjects")
    axes[0, 0].legend()
    sns.histplot(data=plot_subjects, x="education", discrete=True, ax=axes[0, 1])
    axes[0, 1].set(title="ADNI education", xlabel="years", ylabel="subjects")
    sex_counts = plot_subjects["sex"].replace("", "Missing").value_counts().rename_axis("category").reset_index(name="subjects")
    sns.barplot(data=sex_counts, x="category", y="subjects", ax=axes[1, 0])
    axes[1, 0].set(title="ADNI sex", xlabel="", ylabel="subjects")
    race_plot_counts = plot_subjects["race_plot"].value_counts().rename_axis("category").reset_index(name="subjects")
    sns.barplot(data=race_plot_counts, y="category", x="subjects", ax=axes[1, 1])
    axes[1, 1].set(title="ADNI race", xlabel="subjects", ylabel="")
    figure.suptitle("ADNI full clinical cohort demographics", fontsize=15)
    figure.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(PLOT_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)

    print(overview.to_json(orient="records"))
    print(f"category rows: {len(counts)}")
    print(PLOT_PATH)


if __name__ == "__main__":
    main()
