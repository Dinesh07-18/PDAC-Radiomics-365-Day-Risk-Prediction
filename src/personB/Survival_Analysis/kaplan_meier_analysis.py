import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


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
    "personB"
)

SURVIVAL_RESULTS_DIR = os.path.join(
    RESULTS_DIR,
    "06_Survival_Analysis"
)

os.makedirs(
    SURVIVAL_RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# INPUT FILE
# ============================================================

SURVIVAL_FILE = os.path.join(
    DATA_DIR,
    "clinical_survival.csv"
)

print("=" * 70)
print("PERSON B - KAPLAN-MEIER SURVIVAL ANALYSIS")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nInput file:")
print(SURVIVAL_FILE)


if not os.path.exists(SURVIVAL_FILE):
    raise FileNotFoundError(
        f"Could not find:\n{SURVIVAL_FILE}"
    )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    SURVIVAL_FILE
)

print("\nDataset shape:", df.shape)

print("\nColumns:")
print(list(df.columns))


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Case Submitter ID",
    "survival_time_days",
    "event"
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
# CLEAN SURVIVAL DATA
# ============================================================

survival_df = df[
    required_columns
].copy()

survival_df["survival_time_days"] = pd.to_numeric(
    survival_df["survival_time_days"],
    errors="coerce"
)

survival_df["event"] = pd.to_numeric(
    survival_df["event"],
    errors="coerce"
)

survival_df = survival_df.dropna(
    subset=[
        "survival_time_days",
        "event"
    ]
)

survival_df = survival_df[
    survival_df["survival_time_days"] >= 0
].copy()

survival_df["event"] = survival_df["event"].astype(int)

survival_df = survival_df[
    survival_df["event"].isin([0, 1])
].copy()


# ============================================================
# BASIC SURVIVAL DATA SUMMARY
# ============================================================

print("\n" + "-" * 70)
print("SURVIVAL DATA SUMMARY")
print("-" * 70)

print(
    "Patients:",
    len(survival_df)
)

print(
    "Events:",
    int(
        (survival_df["event"] == 1).sum()
    )
)

print(
    "Censored:",
    int(
        (survival_df["event"] == 0).sum()
    )
)

print(
    "Minimum survival:",
    survival_df["survival_time_days"].min(),
    "days"
)

print(
    "Maximum survival:",
    survival_df["survival_time_days"].max(),
    "days"
)

print(
    "Median observed follow-up:",
    survival_df["survival_time_days"].median(),
    "days"
)


# ============================================================
# SAVE CLEAN SURVIVAL DATA USED FOR KM
# ============================================================

clean_survival_path = os.path.join(
    SURVIVAL_RESULTS_DIR,
    "km_analysis_dataset.csv"
)

survival_df.to_csv(
    clean_survival_path,
    index=False
)

print(
    "\nSaved:",
    clean_survival_path
)


# ============================================================
# OVERALL KAPLAN-MEIER
# ============================================================

kmf = KaplanMeierFitter()

kmf.fit(
    durations=survival_df["survival_time_days"],
    event_observed=survival_df["event"],
    label="All Patients"
)


# ============================================================
# MEDIAN SURVIVAL
# ============================================================

median_survival = kmf.median_survival_time_

print("\n" + "-" * 70)
print("OVERALL KAPLAN-MEIER RESULTS")
print("-" * 70)

print(
    "Median survival:",
    median_survival
)


# ============================================================
# OVERALL KM CURVE
# ============================================================

plt.figure(
    figsize=(10, 7)
)

kmf.plot_survival_function(
    ci_show=True
)

plt.title(
    "Overall Kaplan-Meier Survival Curve"
)

plt.xlabel(
    "Time (days)"
)

plt.ylabel(
    "Survival Probability"
)

plt.ylim(
    0,
    1.05
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()

overall_km_path = os.path.join(
    SURVIVAL_RESULTS_DIR,
    "overall_kaplan_meier_curve.png"
)

plt.savefig(
    overall_km_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved:",
    overall_km_path
)


# ============================================================
# SURVIVAL PROBABILITY TABLE
# ============================================================

survival_function = kmf.survival_function_.copy()

survival_function = survival_function.reset_index()

survival_function.columns = [
    "time_days",
    "survival_probability"
]

survival_probability_path = os.path.join(
    SURVIVAL_RESULTS_DIR,
    "overall_survival_probability.csv"
)

survival_function.to_csv(
    survival_probability_path,
    index=False
)

print(
    "Saved:",
    survival_probability_path
)


# ============================================================
# CREATE 365-DAY RISK CLASSIFICATION
# ============================================================
#
# Class 0:
#   survival >= 365 days
#
# Class 1:
#   death/event before 365 days
#
# Censored before 365 days:
#   excluded
#
# This is the same target definition used in
# the project's classification analysis.
# ============================================================

def create_risk_label(row):

    survival_time = row["survival_time_days"]
    event = row["event"]

    if survival_time >= 365:
        return 0

    elif (
        event == 1
        and survival_time < 365
    ):
        return 1

    else:
        return np.nan


survival_df["risk_label"] = survival_df.apply(
    create_risk_label,
    axis=1
)

classification_df = survival_df.dropna(
    subset=["risk_label"]
).copy()

classification_df["risk_label"] = (
    classification_df["risk_label"]
    .astype(int)
)


# ============================================================
# CLASSIFICATION COHORT SUMMARY
# ============================================================

print("\n" + "-" * 70)
print("365-DAY RISK COHORT")
print("-" * 70)

print(
    "Total eligible:",
    len(classification_df)
)

print(
    "\nRisk distribution:"
)

print(
    classification_df[
        "risk_label"
    ].value_counts().sort_index()
)

print(
    "\nExcluded before 365 days:",
    len(survival_df)
    - len(classification_df)
)


# ============================================================
# LOW-RISK / HIGH-RISK DATA
# ============================================================

low_risk = classification_df[
    classification_df["risk_label"] == 0
].copy()

high_risk = classification_df[
    classification_df["risk_label"] == 1
].copy()


print(
    "\nLow-risk patients:",
    len(low_risk)
)

print(
    "High-risk patients:",
    len(high_risk)
)


# ============================================================
# KAPLAN-MEIER BY 365-DAY RISK GROUP
# ============================================================

km_low = KaplanMeierFitter()

km_high = KaplanMeierFitter()

km_low.fit(
    durations=low_risk["survival_time_days"],
    event_observed=low_risk["event"],
    label="Lower Risk (Class 0)"
)

km_high.fit(
    durations=high_risk["survival_time_days"],
    event_observed=high_risk["event"],
    label="Higher Risk (Class 1)"
)


# ============================================================
# GROUP MEDIAN SURVIVAL
# ============================================================

low_median = km_low.median_survival_time_

high_median = km_high.median_survival_time_

print("\n" + "-" * 70)
print("MEDIAN SURVIVAL BY RISK GROUP")
print("-" * 70)

print(
    "Lower-risk median survival:",
    low_median
)

print(
    "Higher-risk median survival:",
    high_median
)


# ============================================================
# LOG-RANK TEST
# ============================================================

logrank_result = logrank_test(
    low_risk["survival_time_days"],
    high_risk["survival_time_days"],
    event_observed_A=low_risk["event"],
    event_observed_B=high_risk["event"]
)

print("\n" + "-" * 70)
print("LOG-RANK TEST")
print("-" * 70)

print(
    "Test statistic:",
    logrank_result.test_statistic
)

print(
    "p-value:",
    logrank_result.p_value
)


# ============================================================
# SAVE LOG-RANK RESULT
# ============================================================

logrank_summary = pd.DataFrame(
    {
        "test": ["Log-rank"],
        "test_statistic": [
            logrank_result.test_statistic
        ],
        "p_value": [
            logrank_result.p_value
        ],
        "lower_risk_n": [
            len(low_risk)
        ],
        "higher_risk_n": [
            len(high_risk)
        ],
        "lower_risk_median_days": [
            low_median
        ],
        "higher_risk_median_days": [
            high_median
        ]
    }
)

logrank_path = os.path.join(
    SURVIVAL_RESULTS_DIR,
    "risk_group_logrank_test.csv"
)

logrank_summary.to_csv(
    logrank_path,
    index=False
)

print(
    "Saved:",
    logrank_path
)


# ============================================================
# HIGH-RISK VS LOW-RISK KM CURVE
# ============================================================

plt.figure(
    figsize=(10, 7)
)

km_low.plot_survival_function(
    ci_show=True
)

km_high.plot_survival_function(
    ci_show=True
)

plt.title(
    "Kaplan-Meier Survival by 365-Day Risk Group"
)

plt.xlabel(
    "Time (days)"
)

plt.ylabel(
    "Survival Probability"
)

plt.ylim(
    0,
    1.05
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

risk_km_path = os.path.join(
    SURVIVAL_RESULTS_DIR,
    "risk_group_kaplan_meier_curve.png"
)

plt.savefig(
    risk_km_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved:",
    risk_km_path
)


# ============================================================
# SURVIVAL PROBABILITY AT IMPORTANT TIME POINTS
# ============================================================

time_points = [
    90,
    180,
    365,
    730,
    1095
]

survival_probability_rows = []

for time_point in time_points:

    low_probability = float(
        km_low.predict(
            time_point
        )
    )

    high_probability = float(
        km_high.predict(
            time_point
        )
    )

    survival_probability_rows.append(
        {
            "time_days": time_point,
            "lower_risk_survival_probability":
                low_probability,
            "higher_risk_survival_probability":
                high_probability
        }
    )


risk_survival_probability = pd.DataFrame(
    survival_probability_rows
)

risk_probability_path = os.path.join(
    SURVIVAL_RESULTS_DIR,
    "risk_group_survival_probability.csv"
)

risk_survival_probability.to_csv(
    risk_probability_path,
    index=False
)

print(
    "Saved:",
    risk_probability_path
)


# ============================================================
# FINAL SUMMARY TEXT
# ============================================================

summary_path = os.path.join(
    SURVIVAL_RESULTS_DIR,
    "kaplan_meier_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PERSON B - KAPLAN-MEIER SURVIVAL ANALYSIS\n"
    )

    f.write(
        "=" * 60
        + "\n\n"
    )

    f.write(
        "OVERALL COHORT\n"
    )

    f.write(
        "-" * 30
        + "\n"
    )

    f.write(
        f"Patients: {len(survival_df)}\n"
    )

    f.write(
        f"Events: {(survival_df['event'] == 1).sum()}\n"
    )

    f.write(
        f"Censored: {(survival_df['event'] == 0).sum()}\n"
    )

    f.write(
        f"Median KM survival: {median_survival}\n\n"
    )

    f.write(
        "365-DAY RISK COHORT\n"
    )

    f.write(
        "-" * 30
        + "\n"
    )

    f.write(
        f"Eligible patients: {len(classification_df)}\n"
    )

    f.write(
        f"Lower risk (Class 0): {len(low_risk)}\n"
    )

    f.write(
        f"Higher risk (Class 1): {len(high_risk)}\n"
    )

    f.write(
        f"Excluded censored before 365 days: "
        f"{len(survival_df) - len(classification_df)}\n\n"
    )

    f.write(
        "RISK GROUP SURVIVAL\n"
    )

    f.write(
        "-" * 30
        + "\n"
    )

    f.write(
        f"Lower-risk median survival: {low_median}\n"
    )

    f.write(
        f"Higher-risk median survival: {high_median}\n"
    )

    f.write(
        f"Log-rank statistic: "
        f"{logrank_result.test_statistic}\n"
    )

    f.write(
        f"Log-rank p-value: "
        f"{logrank_result.p_value}\n"
    )


print(
    "\nSaved:",
    summary_path
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("KAPLAN-MEIER ANALYSIS COMPLETE")
print("=" * 70)

print("\nOutput directory:")
print(SURVIVAL_RESULTS_DIR)

print("\nGenerated files:")

for filename in sorted(
    os.listdir(SURVIVAL_RESULTS_DIR)
):
    print(
        " -",
        filename
    )

print("\nDone.")