import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test


# ============================================================
# PERSON B - COX PROPORTIONAL HAZARDS ANALYSIS
# Clinical Survival Model
#
# Covariates:
#   - Age
#   - Sex
#   - Tumor Grade
#
# Outcome:
#   - survival_time_days
#   - event
#
# Tumor Stage is excluded because it is "Not Reported"
# for all 138 patients in the current clinical dataset.
# ============================================================


# ============================================================
# PATH SETUP
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "..")
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "personB",
    "Survival_Analysis"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# INPUT FILE
# ============================================================

SURVIVAL_FILE = os.path.join(
    DATA_DIR,
    "clinical_survival.csv"
)


print("=" * 75)
print("PERSON B - COX PROPORTIONAL HAZARDS ANALYSIS")
print("=" * 75)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nInput file:")
print(SURVIVAL_FILE)

print("\nResults directory:")
print(RESULTS_DIR)


if not os.path.exists(SURVIVAL_FILE):
    raise FileNotFoundError(
        "Clinical survival file not found:\n"
        + SURVIVAL_FILE
    )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    SURVIVAL_FILE
)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Case Submitter ID",
    "survival_time_days",
    "event",
    "age_years",
    "Sex",
    "Tumor Grade"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + str(missing_columns)
    )


# ============================================================
# SELECT VARIABLES
# ============================================================

cox_df = df[
    [
        "Case Submitter ID",
        "survival_time_days",
        "event",
        "age_years",
        "Sex",
        "Tumor Grade"
    ]
].copy()


# ============================================================
# BASIC CLEANING
# ============================================================

cox_df["survival_time_days"] = pd.to_numeric(
    cox_df["survival_time_days"],
    errors="coerce"
)

cox_df["event"] = pd.to_numeric(
    cox_df["event"],
    errors="coerce"
)

cox_df["age_years"] = pd.to_numeric(
    cox_df["age_years"],
    errors="coerce"
)


# Remove invalid survival records

cox_df = cox_df.dropna(
    subset=[
        "survival_time_days",
        "event",
        "age_years",
        "Sex",
        "Tumor Grade"
    ]
).copy()


cox_df = cox_df[
    cox_df["survival_time_days"] >= 0
].copy()


cox_df = cox_df[
    cox_df["event"].isin([0, 1])
].copy()


# ============================================================
# CLEAN TEXT VARIABLES
# ============================================================

cox_df["Sex"] = (
    cox_df["Sex"]
    .astype(str)
    .str.strip()
    .str.lower()
)

cox_df["Tumor Grade"] = (
    cox_df["Tumor Grade"]
    .astype(str)
    .str.strip()
    .str.upper()
)


# ============================================================
# CHECK CATEGORIES
# ============================================================

print("\n" + "-" * 75)
print("CATEGORIES BEFORE ENCODING")
print("-" * 75)

print("\nSex:")
print(
    cox_df["Sex"].value_counts(
        dropna=False
    )
)

print("\nTumor Grade:")
print(
    cox_df["Tumor Grade"].value_counts(
        dropna=False
    )
)


# ============================================================
# ENCODE SEX
# ============================================================
#
# Female = 0
# Male   = 1
# ============================================================

sex_mapping = {
    "female": 0,
    "male": 1
}

cox_df["sex_male"] = (
    cox_df["Sex"]
    .map(sex_mapping)
)


# ============================================================
# ENCODE TUMOR GRADE
# ============================================================
#
# G1 = 1
# G2 = 2
# G3 = 3
# G4 = 4
#
# This treats grade as an ordered variable.
# ============================================================

grade_mapping = {
    "G1": 1,
    "G2": 2,
    "G3": 3,
    "G4": 4
}

cox_df["tumor_grade_numeric"] = (
    cox_df["Tumor Grade"]
    .map(grade_mapping)
)


# ============================================================
# CHECK ENCODING
# ============================================================

print("\n" + "-" * 75)
print("ENCODING CHECK")
print("-" * 75)

print("\nSex encoding:")
print(
    cox_df[
        ["Sex", "sex_male"]
    ].drop_duplicates().sort_values("sex_male")
)

print("\nTumor Grade encoding:")
print(
    cox_df[
        ["Tumor Grade", "tumor_grade_numeric"]
    ].drop_duplicates().sort_values("tumor_grade_numeric")
)


# ============================================================
# REMOVE UNMAPPED VALUES
# ============================================================

cox_df = cox_df.dropna(
    subset=[
        "sex_male",
        "tumor_grade_numeric"
    ]
).copy()


cox_df["sex_male"] = (
    cox_df["sex_male"]
    .astype(int)
)

cox_df["tumor_grade_numeric"] = (
    cox_df["tumor_grade_numeric"]
    .astype(int)
)


# ============================================================
# CREATE FINAL COX DATASET
# ============================================================

model_df = cox_df[
    [
        "Case Submitter ID",
        "survival_time_days",
        "event",
        "age_years",
        "sex_male",
        "tumor_grade_numeric"
    ]
].copy()


# ============================================================
# REMOVE DUPLICATE PATIENTS
# ============================================================

duplicate_count = model_df[
    "Case Submitter ID"
].duplicated().sum()

print("\nDuplicate patients:")
print(duplicate_count)

if duplicate_count > 0:

    model_df = model_df.drop_duplicates(
        subset=["Case Submitter ID"],
        keep="first"
    )


# ============================================================
# SAVE COX DATASET
# ============================================================

cox_dataset_path = os.path.join(
    RESULTS_DIR,
    "cox_model_dataset.csv"
)

model_df.to_csv(
    cox_dataset_path,
    index=False
)

print(
    "\nSaved Cox model dataset:"
)

print(
    cox_dataset_path
)


# ============================================================
# FINAL DATA SUMMARY
# ============================================================

print("\n" + "-" * 75)
print("FINAL COX MODEL DATASET")
print("-" * 75)

print(
    "Patients:",
    len(model_df)
)

print(
    "Events:",
    int(
        model_df["event"].sum()
    )
)

print(
    "Censored:",
    int(
        (model_df["event"] == 0).sum()
    )
)

print(
    "\nPredictors:"
)

print(
    " - Age"
)

print(
    " - Sex (Male vs Female)"
)

print(
    " - Tumor Grade"
)


# ============================================================
# FIT COX MODEL
# ============================================================

cox_model_data = model_df[
    [
        "survival_time_days",
        "event",
        "age_years",
        "sex_male",
        "tumor_grade_numeric"
    ]
].copy()


print("\n" + "-" * 75)
print("FITTING COX PROPORTIONAL HAZARDS MODEL")
print("-" * 75)


cph = CoxPHFitter()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    cph.fit(
        cox_model_data,
        duration_col="survival_time_days",
        event_col="event"
    )


# ============================================================
# COX MODEL SUMMARY
# ============================================================

print("\n" + "-" * 75)
print("COX MODEL SUMMARY")
print("-" * 75)

cph.print_summary()


# ============================================================
# EXTRACT RESULTS
# ============================================================

summary = cph.summary.copy()


# ============================================================
# HAZARD RATIO RESULTS
# ============================================================

results = pd.DataFrame(
    {
        "variable": summary.index,
        "coefficient": summary["coef"].values,
        "hazard_ratio": summary["exp(coef)"].values,
        "lower_95_CI": summary["exp(coef) lower 95%"].values,
        "upper_95_CI": summary["exp(coef) upper 95%"].values,
        "p_value": summary["p"].values,
        "z_score": summary["z"].values
    }
)


# ============================================================
# ADD HUMAN-READABLE VARIABLE NAMES
# ============================================================

variable_names = {
    "age_years": "Age",
    "sex_male": "Sex: Male vs Female",
    "tumor_grade_numeric": "Tumor Grade"
}

results["variable"] = results[
    "variable"
].map(
    lambda x: variable_names.get(
        x,
        x
    )
)


# ============================================================
# SAVE COX RESULTS
# ============================================================

cox_results_path = os.path.join(
    RESULTS_DIR,
    "cox_clinical_results.csv"
)

results.to_csv(
    cox_results_path,
    index=False
)

print(
    "\nSaved Cox results:"
)

print(
    cox_results_path
)


# ============================================================
# CONCORDANCE INDEX
# ============================================================

concordance_index = cph.concordance_index_

print("\n" + "-" * 75)
print("MODEL PERFORMANCE")
print("-" * 75)

print(
    "Concordance Index:",
    concordance_index
)


# ============================================================
# PROPORTIONAL HAZARDS TEST
# ============================================================

print("\n" + "-" * 75)
print("PROPORTIONAL HAZARDS ASSUMPTION TEST")
print("-" * 75)

try:

    ph_test = proportional_hazard_test(
        cph,
        cox_model_data,
        time_transform="rank"
    )

    print(
        ph_test.summary
    )

    ph_test_path = os.path.join(
        RESULTS_DIR,
        "cox_proportional_hazards_test.csv"
    )

    ph_test.summary.to_csv(
        ph_test_path
    )

    print(
        "\nSaved:"
    )

    print(
        ph_test_path
    )

except Exception as error:

    ph_test = None

    print(
        "\nProportional hazards test could not be completed:"
    )

    print(
        error
    )


# ============================================================
# HAZARD RATIO PLOT
# ============================================================

plot_results = results.copy()

plot_results = plot_results.sort_values(
    "hazard_ratio"
)

y_positions = np.arange(
    len(plot_results)
)

hazard_ratios = (
    plot_results["hazard_ratio"]
    .values
)

lower_errors = (
    hazard_ratios
    - plot_results["lower_95_CI"].values
)

upper_errors = (
    plot_results["upper_95_CI"].values
    - hazard_ratios
)

plt.figure(
    figsize=(10, 6)
)

plt.errorbar(
    hazard_ratios,
    y_positions,
    xerr=[
        lower_errors,
        upper_errors
    ],
    fmt="o",
    capsize=5
)

plt.axvline(
    x=1,
    linestyle="--"
)

plt.yticks(
    y_positions,
    plot_results["variable"]
)

plt.xlabel(
    "Hazard Ratio"
)

plt.ylabel(
    "Clinical Variable"
)

plt.title(
    "Cox Proportional Hazards - Hazard Ratios"
)

plt.grid(
    axis="x",
    alpha=0.25
)

plt.tight_layout()

hazard_ratio_plot_path = os.path.join(
    RESULTS_DIR,
    "cox_hazard_ratios.png"
)

plt.savefig(
    hazard_ratio_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nSaved hazard ratio plot:"
)

print(
    hazard_ratio_plot_path
)


# ============================================================
# CREATE HUMAN-READABLE SUMMARY
# ============================================================

summary_path = os.path.join(
    RESULTS_DIR,
    "cox_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PERSON B - COX PROPORTIONAL HAZARDS ANALYSIS\n"
    )

    f.write(
        "=" * 65
        + "\n\n"
    )

    f.write(
        "MODEL\n"
    )

    f.write(
        "-" * 30
        + "\n"
    )

    f.write(
        "Outcome: survival_time_days + event\n"
    )

    f.write(
        "Predictors: Age, Sex, Tumor Grade\n"
    )

    f.write(
        "Tumor Stage: excluded because all values were "
        "'Not Reported'\n\n"
    )

    f.write(
        "COHORT\n"
    )

    f.write(
        "-" * 30
        + "\n"
    )

    f.write(
        f"Patients: {len(model_df)}\n"
    )

    f.write(
        f"Events: {int(model_df['event'].sum())}\n"
    )

    f.write(
        f"Censored: "
        f"{int((model_df['event'] == 0).sum())}\n\n"
    )

    f.write(
        "CONCORDANCE INDEX\n"
    )

    f.write(
        "-" * 30
        + "\n"
    )

    f.write(
        f"{concordance_index:.4f}\n\n"
    )

    f.write(
        "HAZARD RATIOS\n"
    )

    f.write(
        "-" * 30
        + "\n"
    )

    for _, row in results.iterrows():

        f.write(
            f"\n{row['variable']}\n"
        )

        f.write(
            f"  Hazard Ratio: "
            f"{row['hazard_ratio']:.4f}\n"
        )

        f.write(
            f"  95% CI: "
            f"{row['lower_95_CI']:.4f} - "
            f"{row['upper_95_CI']:.4f}\n"
        )

        f.write(
            f"  p-value: "
            f"{row['p_value']:.6f}\n"
        )

    if ph_test is not None:

        f.write(
            "\n\nPROPORTIONAL HAZARDS TEST\n"
        )

        f.write(
            "-" * 30
            + "\n"
        )

        for variable, row in ph_test.summary.iterrows():

            p_value = row["p"]

            f.write(
                f"{variable}: "
                f"p = {p_value:.6f}\n"
            )


print(
    "\nSaved Cox summary:"
)

print(
    summary_path
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 75)
print("COX PROPORTIONAL HAZARDS ANALYSIS COMPLETE")
print("=" * 75)

print(
    "\nOutput directory:"
)

print(
    RESULTS_DIR
)

print(
    "\nGenerated files:"
)

for filename in sorted(
    os.listdir(RESULTS_DIR)
):

    print(
        " -",
        filename
    )

print(
    "\nDone."
)