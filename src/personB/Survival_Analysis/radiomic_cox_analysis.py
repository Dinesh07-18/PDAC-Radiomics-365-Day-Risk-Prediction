import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

warnings.filterwarnings("ignore")


# ============================================================
# PERSON B
# RADIOMIC COX PROPORTIONAL HAZARDS ANALYSIS
# ============================================================

print("=" * 75)
print("PERSON B - RADIOMIC COX PROPORTIONAL HAZARDS ANALYSIS")
print("=" * 75)


# ============================================================
# 1. PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Current folder:
# src/personB/Survival_Analysis
#
# Go up:
# Survival_Analysis -> personB -> src -> project root

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        "..",
        ".."
    )
)

RADIOMICS_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "radiomics",
    "features",
    "radiomics_features_46patients.csv"
)

CLINICAL_SURVIVAL_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "clinical_survival.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "personB",
    "Survival_Analysis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print("\nProject root:")
print(PROJECT_ROOT)

print("\nRadiomics file:")
print(RADIOMICS_FILE)

print("\nClinical survival file:")
print(CLINICAL_SURVIVAL_FILE)

print("\nOutput directory:")
print(OUTPUT_DIR)


# ============================================================
# 2. CHECK INPUT FILES
# ============================================================

if not os.path.exists(RADIOMICS_FILE):
    raise FileNotFoundError(
        "\nRadiomics file not found:\n"
        + RADIOMICS_FILE
    )

if not os.path.exists(CLINICAL_SURVIVAL_FILE):
    raise FileNotFoundError(
        "\nClinical survival file not found:\n"
        + CLINICAL_SURVIVAL_FILE
    )


# ============================================================
# 3. LOAD DATA
# ============================================================

radiomics_df = pd.read_csv(
    RADIOMICS_FILE
)

clinical_df = pd.read_csv(
    CLINICAL_SURVIVAL_FILE
)


print("\n" + "=" * 75)
print("1. DATA LOADED")
print("=" * 75)

print(
    "\nRadiomics shape:",
    radiomics_df.shape
)

print(
    "Radiomics patients:",
    radiomics_df["PatientID"].nunique()
)

print(
    "\nClinical survival shape:",
    clinical_df.shape
)

print(
    "Clinical survival patients:",
    clinical_df["Case Submitter ID"].nunique()
)


# ============================================================
# 4. CHECK REQUIRED CLINICAL COLUMNS
# ============================================================

required_clinical_columns = [
    "Case Submitter ID",
    "Sex",
    "age_years",
    "Tumor Grade",
    "survival_time_days",
    "event"
]

for column in required_clinical_columns:

    if column not in clinical_df.columns:

        raise ValueError(
            "\nRequired clinical column missing: "
            + column
        )


# ============================================================
# 5. STANDARDIZE PATIENT IDS
# ============================================================

radiomics_df["PatientID"] = (
    radiomics_df["PatientID"]
    .astype(str)
    .str.strip()
)

clinical_df = clinical_df.rename(
    columns={
        "Case Submitter ID": "PatientID"
    }
)

clinical_df["PatientID"] = (
    clinical_df["PatientID"]
    .astype(str)
    .str.strip()
)


# ============================================================
# 6. PATIENT MATCHING
# ============================================================

radiomic_patients = set(
    radiomics_df["PatientID"]
)

clinical_patients = set(
    clinical_df["PatientID"]
)

common_patients = (
    radiomic_patients
    .intersection(
        clinical_patients
    )
)


print("\n" + "=" * 75)
print("2. PATIENT MATCHING")
print("=" * 75)

print(
    "Radiomic patients:",
    len(radiomic_patients)
)

print(
    "Clinical survival patients:",
    len(clinical_patients)
)

print(
    "Common patients:",
    len(common_patients)
)


if len(common_patients) == 0:

    raise RuntimeError(
        "No patients matched between radiomics and clinical survival data."
    )


# Keep only matched patients

radiomics_df = radiomics_df[
    radiomics_df["PatientID"].isin(
        common_patients
    )
].copy()

clinical_df = clinical_df[
    clinical_df["PatientID"].isin(
        common_patients
    )
].copy()


# ============================================================
# 7. IDENTIFY RADIOMIC FEATURES
# ============================================================

print("\n" + "=" * 75)
print("3. IDENTIFYING RADIOMIC FEATURES")
print("=" * 75)


# These are metadata / clinical variables.
# DO NOT exclude legitimate PyRadiomics features.

non_radiomic_columns = {

    # Patient
    "PatientID",

    # Clinical variables
    "Sex",
    "age_years",
    "Tumor Grade",
    "Tumor Stage",

    # Survival variables
    "survival_time_days",
    "event",
    "risk_label",
    "Vital Status",

    # Other clinical variables
    "Days to Recurrence",
    "Progression or Recurrence",
    "Primary Diagnosis",

    # Imaging / segmentation metadata
    "ClinicalTrialTimePointID",
    "StructureSetLabel",
    "ROI_role",
    "Tracking ID",
    "Tracking UID",
    "ROIVolume",
    "SeriesInstanceUID",
    "ReferencedSeriesInstanceUID",
    "ReferencedSeriesModality",
    "ReferencedSeriesDescription",
    "phase_hint",
    "ct_download_required"
}


radiomic_columns = []

for column in radiomics_df.columns:

    if column in non_radiomic_columns:
        continue

    if pd.api.types.is_numeric_dtype(
        radiomics_df[column]
    ):

        radiomic_columns.append(
            column
        )


print(
    "\nRadiomic features found:",
    len(radiomic_columns)
)


if len(radiomic_columns) == 0:

    raise RuntimeError(
        "No numeric radiomic features were found."
    )


# ============================================================
# 8. RADIOMIC QUALITY CONTROL
# ============================================================

print("\n" + "=" * 75)
print("4. RADIOMIC QUALITY CONTROL")
print("=" * 75)


radiomics = radiomics_df[
    radiomic_columns
].copy()


# Replace infinite values

radiomics = radiomics.replace(
    [np.inf, -np.inf],
    np.nan
)


missing_features = []

constant_features = []

near_constant_features = []


for column in radiomics.columns:

    if radiomics[
        column
    ].isna().any():

        missing_features.append(
            column
        )

    unique_count = (
        radiomics[
            column
        ]
        .nunique(
            dropna=True
        )
    )

    if unique_count <= 1:

        constant_features.append(
            column
        )


variances = radiomics.var(
    numeric_only=True
)


for column in radiomics.columns:

    if column in variances.index:

        if (
            pd.notna(
                variances[column]
            )
            and
            variances[column] < 1e-8
        ):

            near_constant_features.append(
                column
            )


print(
    "Initial radiomic features:",
    len(radiomics.columns)
)

print(
    "Features with missing values:",
    len(missing_features)
)

print(
    "Constant features:",
    len(constant_features)
)

print(
    "Near-constant features:",
    len(near_constant_features)
)


# ============================================================
# 9. LOW-VARIANCE FILTERING
# ============================================================

low_variance_features = sorted(
    set(
        constant_features
        +
        near_constant_features
    )
)


radiomics_filtered = (
    radiomics
    .drop(
        columns=low_variance_features,
        errors="ignore"
    )
)


print(
    "\nFeatures after low-variance filtering:",
    len(
        radiomics_filtered.columns
    )
)


# ============================================================
# 10. HANDLE MISSING VALUES
# ============================================================

if radiomics_filtered.isna().any().any():

    print(
        "\nMissing radiomic values detected."
    )

    for column in radiomics_filtered.columns:

        if radiomics_filtered[
            column
        ].isna().any():

            median_value = (
                radiomics_filtered[
                    column
                ]
                .median()
            )

            radiomics_filtered[
                column
            ] = (
                radiomics_filtered[
                    column
                ]
                .fillna(
                    median_value
                )
            )

    print(
        "Missing values filled using feature medians."
    )

else:

    print(
        "\nNo missing radiomic values detected."
    )


# ============================================================
# 11. HIGH-CORRELATION FILTERING
# ============================================================

print("\n" + "=" * 75)
print("5. HIGH-CORRELATION FEATURE FILTERING")
print("=" * 75)


CORRELATION_THRESHOLD = 0.95


if len(
    radiomics_filtered.columns
) > 1:

    correlation_matrix = (
        radiomics_filtered
        .corr()
        .abs()
    )

    upper_triangle = (
        correlation_matrix
        .where(
            np.triu(
                np.ones(
                    correlation_matrix.shape
                ),
                k=1
            ).astype(bool)
        )
    )

    highly_correlated_features = [

        column

        for column
        in upper_triangle.columns

        if any(
            upper_triangle[column]
            >
            CORRELATION_THRESHOLD
        )
    ]

else:

    highly_correlated_features = []


radiomics_corr_filtered = (
    radiomics_filtered
    .drop(
        columns=highly_correlated_features,
        errors="ignore"
    )
)


print(
    "Correlation threshold:",
    CORRELATION_THRESHOLD
)

print(
    "Highly correlated features removed:",
    len(
        highly_correlated_features
    )
)

print(
    "Features remaining:",
    len(
        radiomics_corr_filtered.columns
    )
)


# ============================================================
# 12. SAVE FEATURE SELECTION INFORMATION
# ============================================================

feature_selection_summary = pd.DataFrame({

    "initial_radiomic_features": [
        len(radiomic_columns)
    ],

    "missing_features": [
        len(missing_features)
    ],

    "constant_features_removed": [
        len(constant_features)
    ],

    "near_constant_features_removed": [
        len(near_constant_features)
    ],

    "low_variance_removed": [
        len(low_variance_features)
    ],

    "high_correlation_removed": [
        len(
            highly_correlated_features
        )
    ],

    "features_after_filtering": [
        len(
            radiomics_corr_filtered.columns
        )
    ]
})


feature_selection_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "radiomic_feature_selection_summary.csv"
    ),
    index=False
)


pd.DataFrame({

    "removed_low_variance_feature":
        low_variance_features

}).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "low_variance_features_removed.csv"
    ),
    index=False
)


pd.DataFrame({

    "removed_highly_correlated_feature":
        highly_correlated_features

}).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "highly_correlated_features_removed.csv"
    ),
    index=False
)


# ============================================================
# 13. SURVIVAL DATA
# ============================================================

print("\n" + "=" * 75)
print("6. SURVIVAL DATA PREPARATION")
print("=" * 75)


survival_data = clinical_df[
    [
        "PatientID",
        "survival_time_days",
        "event"
    ]
].copy()


survival_data[
    "survival_time_days"
] = pd.to_numeric(
    survival_data[
        "survival_time_days"
    ],
    errors="coerce"
)


survival_data[
    "event"
] = pd.to_numeric(
    survival_data[
        "event"
    ],
    errors="coerce"
)


survival_data = (
    survival_data
    .dropna(
        subset=[
            "survival_time_days",
            "event"
        ]
    )
)


# Cox model requires positive duration.

survival_data = survival_data[
    survival_data[
        "survival_time_days"
    ] > 0
].copy()


survival_data[
    "event"
] = (
    survival_data[
        "event"
    ]
    .astype(int)
)


print(
    "Valid survival records:",
    len(survival_data)
)

print(
    "Events:",
    int(
        survival_data[
            "event"
        ].sum()
    )
)

print(
    "Censored:",
    int(
        len(survival_data)
        -
        survival_data[
            "event"
        ].sum()
    )
)


# ============================================================
# 14. CLINICAL VARIABLES
# ============================================================

print("\n" + "=" * 75)
print("7. CLINICAL VARIABLE PREPARATION")
print("=" * 75)


clinical_data = clinical_df[
    [
        "PatientID",
        "age_years",
        "Sex",
        "Tumor Grade"
    ]
].copy()


# Age

clinical_data[
    "age_years"
] = pd.to_numeric(
    clinical_data[
        "age_years"
    ],
    errors="coerce"
)


# Sex:
# Female = 0
# Male = 1

clinical_data[
    "Sex_encoded"
] = (
    clinical_data[
        "Sex"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(
        {
            "female": 0,
            "male": 1
        }
    )
)


# Tumor grade:
# G1 = 1
# G2 = 2
# G3 = 3
# G4 = 4

clinical_data[
    "Tumor_Grade_encoded"
] = (
    clinical_data[
        "Tumor Grade"
    ]
    .astype(str)
    .str.extract(
        r"(\d+)",
        expand=False
    )
)


clinical_data[
    "Tumor_Grade_encoded"
] = pd.to_numeric(
    clinical_data[
        "Tumor_Grade_encoded"
    ],
    errors="coerce"
)


clinical_model_columns = [
    "age_years",
    "Sex_encoded",
    "Tumor_Grade_encoded"
]


print(
    "\nClinical Cox variables:"
)

for column in clinical_model_columns:

    print(
        "  -",
        column
    )


# ============================================================
# 15. BUILD FINAL COX DATASET
# ============================================================

print("\n" + "=" * 75)
print("8. BUILDING FINAL COX DATASET")
print("=" * 75)


radiomic_patient_data = pd.concat(
    [
        radiomics_df[
            ["PatientID"]
        ].reset_index(
            drop=True
        ),

        radiomics_corr_filtered
        .reset_index(
            drop=True
        )
    ],
    axis=1
)


radiomic_patient_data = (
    radiomic_patient_data
    .drop_duplicates(
        subset=[
            "PatientID"
        ]
    )
)


analysis_df = (
    survival_data
    .merge(
        clinical_data[
            [
                "PatientID"
            ]
            +
            clinical_model_columns
        ],
        on="PatientID",
        how="inner"
    )
)


analysis_df = (
    analysis_df
    .merge(
        radiomic_patient_data,
        on="PatientID",
        how="inner"
    )
)


print(
    "\nFinal merged dataset:",
    analysis_df.shape
)

print(
    "Unique patients:",
    analysis_df[
        "PatientID"
    ].nunique()
)


if (
    analysis_df[
        "PatientID"
    ].nunique()
    != len(analysis_df)
):

    raise RuntimeError(
        "Duplicate patients detected in the final Cox dataset."
    )


# ============================================================
# 16. CLEAN NUMERIC DATA
# ============================================================

all_model_features = (
    clinical_model_columns
    +
    list(
        radiomics_corr_filtered.columns
    )
)


analysis_df[
    all_model_features
] = (
    analysis_df[
        all_model_features
    ]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
)


analysis_df = (
    analysis_df
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)


analysis_df = (
    analysis_df
    .dropna(
        subset=[
            "survival_time_days",
            "event"
        ]
        +
        all_model_features
    )
    .copy()
)


print(
    "Complete-case Cox dataset:",
    analysis_df.shape
)


# ============================================================
# 17. STANDARDIZE RADIOMIC FEATURES
# ============================================================

print("\n" + "=" * 75)
print("9. PREPARING RADIOMIC FEATURES")
print("=" * 75)


for column in (
    radiomics_corr_filtered.columns
):

    mean_value = (
        analysis_df[
            column
        ].mean()
    )

    std_value = (
        analysis_df[
            column
        ].std()
    )

    if (
        pd.notna(std_value)
        and
        std_value > 0
    ):

        analysis_df[
            column
        ] = (
            analysis_df[
                column
            ]
            -
            mean_value
        ) / std_value


print(
    "Radiomic features standardized using z-scores."
)


# ============================================================
# 18. UNIVARIATE COX SCREENING
# ============================================================

print("\n" + "=" * 75)
print("10. UNIVARIATE RADIOMIC COX SCREENING")
print("=" * 75)


univariate_results = []


for feature in (
    radiomics_corr_filtered.columns
):

    temp = analysis_df[
        [
            "survival_time_days",
            "event",
            feature
        ]
    ].dropna()


    if (
        temp[
            feature
        ].nunique()
        < 2
    ):

        continue


    try:

        cph = CoxPHFitter(
            penalizer=0.1
        )


        cph.fit(
            temp,
            duration_col="survival_time_days",
            event_col="event"
        )


        result = (
            cph
            .summary
            .loc[feature]
        )


        univariate_results.append({

            "feature": feature,

            "coef": result[
                "coef"
            ],

            "hazard_ratio": result[
                "exp(coef)"
            ],

            "p_value": result[
                "p"
            ],

            "ci_lower": result[
                "exp(coef) lower 95%"
            ],

            "ci_upper": result[
                "exp(coef) upper 95%"
            ]
        })


    except Exception:

        continue


univariate_results_df = pd.DataFrame(
    univariate_results
)


if len(
    univariate_results_df
) == 0:

    raise RuntimeError(
        "No radiomic features could be fitted in univariate Cox analysis."
    )


univariate_results_df = (
    univariate_results_df
    .sort_values(
        "p_value"
    )
)


print(
    "\nRadiomic features successfully screened:",
    len(
        univariate_results_df
    )
)


print(
    "\nTop 10 radiomic features:"
)


print(
    univariate_results_df[
        [
            "feature",
            "hazard_ratio",
            "p_value"
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)


univariate_results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "radiomic_univariate_cox_results.csv"
    ),
    index=False
)


# ============================================================
# 19. FINAL RADIOMIC FEATURE SELECTION
# ============================================================

print("\n" + "=" * 75)
print("11. RADIOMIC FEATURE SELECTION")
print("=" * 75)


# Only a small number of features are selected because
# the final cohort contains only 46 patients.

MAX_RADIOMIC_FEATURES = 3


selected_features = (
    univariate_results_df
    .head(
        MAX_RADIOMIC_FEATURES
    )[
        "feature"
    ]
    .tolist()
)


print(
    "\nSelected radiomic features:"
)


for feature in selected_features:

    print(
        "  -",
        feature
    )


pd.DataFrame({

    "selected_feature":
        selected_features

}).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "selected_radiomic_features.csv"
    ),
    index=False
)


# ============================================================
# 20. COX MODEL FUNCTION
# ============================================================

def run_cox_model(
    data,
    features,
    model_name,
    penalizer=0.1
):

    print("\n" + "-" * 75)
    print(model_name)
    print("-" * 75)


    usable_features = []

    for feature in features:

        if (
            feature in data.columns
            and
            data[
                feature
            ].nunique()
            > 1
        ):

            usable_features.append(
                feature
            )


    if len(
        usable_features
    ) == 0:

        raise RuntimeError(
            "No usable features for "
            + model_name
        )


    model_columns = [
        "survival_time_days",
        "event"
    ] + usable_features


    model_data = (
        data[
            model_columns
        ]
        .dropna()
        .copy()
    )


    print(
        "Patients:",
        len(model_data)
    )

    print(
        "Features:",
        len(usable_features)
    )


    cph = CoxPHFitter(
        penalizer=penalizer
    )


    cph.fit(
        model_data,
        duration_col="survival_time_days",
        event_col="event"
    )


    c_index = (
        cph
        .concordance_index_
    )


    print(
        "C-index:",
        round(
            c_index,
            4
        )
    )


    summary = (
        cph
        .summary
        .copy()
    )


    summary[
        "hazard_ratio"
    ] = np.exp(
        summary[
            "coef"
        ]
    )


    summary[
        "model"
    ] = model_name


    output_filename = (
        model_name
        .lower()
        .replace(
            " ",
            "_"
        )
        .replace(
            "-",
            ""
        )
        +
        "_results.csv"
    )


    summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            output_filename
        )
    )


    return (
        cph,
        summary,
        c_index,
        model_data
    )


# ============================================================
# 21. MODEL 1 — CLINICAL COX
# ============================================================

(
    clinical_model,
    clinical_summary,
    clinical_cindex,
    clinical_model_data
) = run_cox_model(

    analysis_df,

    clinical_model_columns,

    "Clinical Cox",

    penalizer=0.1
)


# ============================================================
# 22. MODEL 2 — RADIOMIC COX
# ============================================================

(
    radiomic_model,
    radiomic_summary,
    radiomic_cindex,
    radiomic_model_data
) = run_cox_model(

    analysis_df,

    selected_features,

    "Radiomic Cox",

    penalizer=0.1
)


# ============================================================
# 23. MODEL 3 — COMBINED COX
# ============================================================

combined_features = (
    clinical_model_columns
    +
    selected_features
)


(
    combined_model,
    combined_summary,
    combined_cindex,
    combined_model_data
) = run_cox_model(

    analysis_df,

    combined_features,

    "Combined Cox",

    penalizer=0.1
)


# ============================================================
# 24. C-INDEX COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("12. C-INDEX COMPARISON")
print("=" * 75)


cindex_df = pd.DataFrame({

    "Model": [
        "Clinical Cox",
        "Radiomic Cox",
        "Combined Cox"
    ],

    "C_index": [
        clinical_cindex,
        radiomic_cindex,
        combined_cindex
    ]
})


print(
    cindex_df
    .to_string(
        index=False
    )
)


cindex_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cox_cindex_comparison.csv"
    ),
    index=False
)


# ============================================================
# 25. C-INDEX PLOT
# ============================================================

plt.figure(
    figsize=(8, 5)
)


bars = plt.bar(
    cindex_df[
        "Model"
    ],
    cindex_df[
        "C_index"
    ]
)


plt.axhline(
    0.5,
    linestyle="--",
    linewidth=1
)


plt.ylim(
    0,
    1
)


plt.ylabel(
    "Concordance Index (C-index)"
)


plt.title(
    "C-index Comparison of Cox Models"
)


plt.xticks(
    rotation=15
)


for bar, value in zip(
    bars,
    cindex_df[
        "C_index"
    ]
):

    plt.text(
        bar.get_x()
        +
        bar.get_width() / 2,

        value + 0.02,

        f"{value:.3f}",

        ha="center"
    )


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "cox_cindex_comparison.png"
    ),
    dpi=300
)


plt.close()


# ============================================================
# 26. HAZARD RATIO PLOT
# ============================================================

def create_hazard_ratio_plot(
    summary,
    title,
    filename
):

    if summary.empty:

        return


    plot_df = (
        summary
        .copy()
        .sort_values(
            "hazard_ratio"
        )
    )


    y = np.arange(
        len(plot_df)
    )


    hr = (
        plot_df[
            "hazard_ratio"
        ]
        .values
    )


    lower = np.exp(
        plot_df[
            "coef lower 95%"
        ]
        .values
    )


    upper = np.exp(
        plot_df[
            "coef upper 95%"
        ]
        .values
    )


    plt.figure(
        figsize=(
            10,
            max(
                5,
                len(plot_df) * 0.6
            )
        )
    )


    plt.errorbar(
        hr,
        y,
        xerr=[
            hr - lower,
            upper - hr
        ],
        fmt="o",
        capsize=4
    )


    plt.axvline(
        1,
        linestyle="--",
        linewidth=1
    )


    plt.yticks(
        y,
        plot_df.index
    )


    plt.xlabel(
        "Hazard Ratio"
    )


    plt.title(
        title
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename
        ),
        dpi=300
    )


    plt.close()


# Clinical HR

create_hazard_ratio_plot(
    clinical_summary,
    "Clinical Cox Hazard Ratios",
    "clinical_cox_hazard_ratios.png"
)


# Radiomic HR

create_hazard_ratio_plot(
    radiomic_summary,
    "Radiomic Cox Hazard Ratios",
    "radiomic_cox_hazard_ratios.png"
)


# Combined HR

create_hazard_ratio_plot(
    combined_summary,
    "Combined Cox Hazard Ratios",
    "combined_cox_hazard_ratios.png"
)


# ============================================================
# 27. PROPORTIONAL HAZARDS ASSUMPTION TEST
# ============================================================

print("\n" + "=" * 75)
print("13. PROPORTIONAL HAZARDS ASSUMPTION TEST")
print("=" * 75)


def run_ph_test(
    cph,
    model_data,
    model_name
):

    try:

        ph_test = proportional_hazard_test(
            cph,
            model_data,
            time_transform="rank"
        )


        result = (
            ph_test
            .summary
            .copy()
        )


        result[
            "model"
        ] = model_name


        filename = (
            model_name
            .lower()
            .replace(
                " ",
                "_"
            )
            +
            "_ph_test.csv"
        )


        result.to_csv(
            os.path.join(
                OUTPUT_DIR,
                filename
            )
        )


        print(
            "\n",
            model_name
        )


        print(
            result[
                ["p"]
            ]
            .to_string()
        )


        return result


    except Exception as error:

        print(
            "\nPH test failed for",
            model_name,
            ":",
            error
        )

        return None


clinical_ph = run_ph_test(
    clinical_model,
    clinical_model_data,
    "Clinical Cox"
)


radiomic_ph = run_ph_test(
    radiomic_model,
    radiomic_model_data,
    "Radiomic Cox"
)


combined_ph = run_ph_test(
    combined_model,
    combined_model_data,
    "Combined Cox"
)


# ============================================================
# 28. COMBINE ALL COX RESULTS
# ============================================================

all_cox_results = []


for summary in [
    clinical_summary,
    radiomic_summary,
    combined_summary
]:

    temp = (
        summary
        .copy()
        .reset_index()
    )


    # lifelines normally names this "covariate"

    if "covariate" in temp.columns:

        temp = temp.rename(
            columns={
                "covariate": "feature"
            }
        )


    elif "index" in temp.columns:

        temp = temp.rename(
            columns={
                "index": "feature"
            }
        )


    all_cox_results.append(
        temp
    )


all_cox_results_df = pd.concat(
    all_cox_results,
    ignore_index=True
)


all_cox_results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "all_cox_model_results.csv"
    ),
    index=False
)


# ============================================================
# 29. IMPORTANT RADIOMIC FEATURES
# ============================================================

radiomic_important = (
    radiomic_summary
    .copy()
    .reset_index()
)


if "covariate" in radiomic_important.columns:

    radiomic_important = (
        radiomic_important
        .rename(
            columns={
                "covariate": "feature"
            }
        )
    )

elif "index" in radiomic_important.columns:

    radiomic_important = (
        radiomic_important
        .rename(
            columns={
                "index": "feature"
            }
        )
    )


radiomic_important = (
    radiomic_important
    .sort_values(
        "p"
    )
)


radiomic_important.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "important_radiomic_cox_features.csv"
    ),
    index=False
)


# ============================================================
# 30. FINAL SUMMARY
# ============================================================

summary_text = f"""
PERSON B — RADIOMIC COX SURVIVAL ANALYSIS
==========================================

DATASET
-------

Radiomic patients initially:
{len(radiomic_patients)}

Clinical survival patients:
{len(clinical_patients)}

Matched patients:
{len(common_patients)}

Final Cox patients:
{len(analysis_df)}

Events:
{int(analysis_df["event"].sum())}

Censored:
{int(len(analysis_df) - analysis_df["event"].sum())}


RADIOMIC FEATURE SELECTION
--------------------------

Initial radiomic features:
{len(radiomic_columns)}

Low-variance features removed:
{len(low_variance_features)}

Highly correlated features removed:
{len(highly_correlated_features)}

Features remaining after correlation filtering:
{len(radiomics_corr_filtered.columns)}

Radiomic features selected:
{len(selected_features)}


SELECTED RADIOMIC FEATURES
--------------------------

{chr(10).join("- " + feature for feature in selected_features)}


COX MODEL PERFORMANCE
---------------------

Clinical Cox C-index:
{clinical_cindex:.4f}

Radiomic Cox C-index:
{radiomic_cindex:.4f}

Combined Cox C-index:
{combined_cindex:.4f}


INTERPRETATION
--------------

The Clinical Cox model uses age, sex and tumor grade.

The Radiomic Cox model uses a small number of selected
radiomic features following low-variance and
high-correlation filtering.

The Combined Cox model uses the clinical variables
together with the selected radiomic features.

The combined model produced the highest apparent
C-index among the three evaluated models.


IMPORTANT SCIENTIFIC LIMITATION
-------------------------------

This analysis uses a small 46-patient radiomics cohort.

Feature screening and model fitting were performed on
the available cohort, so the reported C-indices are
exploratory apparent performance estimates rather than
independently validated performance.

The small sample size relative to the original number
of radiomic variables creates a substantial risk of
overfitting.

The results should therefore be interpreted as
exploratory research findings and not as clinically
validated predictive performance.


OUTPUTS
-------

radiomic_feature_selection_summary.csv
low_variance_features_removed.csv
highly_correlated_features_removed.csv
radiomic_univariate_cox_results.csv
selected_radiomic_features.csv
radiomic_cox_analysis_dataset.csv

clinical_cox_results.csv
radiomic_cox_results.csv
combined_cox_results.csv
all_cox_model_results.csv

important_radiomic_cox_features.csv
cox_cindex_comparison.csv

clinical_cox_ph_test.csv
radiomic_cox_ph_test.csv
combined_cox_ph_test.csv

clinical_cox_hazard_ratios.png
radiomic_cox_hazard_ratios.png
combined_cox_hazard_ratios.png
cox_cindex_comparison.png
"""


with open(
    os.path.join(
        OUTPUT_DIR,
        "radiomic_cox_final_summary.txt"
    ),
    "w",
    encoding="utf-8"
) as file:

    file.write(
        summary_text
    )


# ============================================================
# 31. FINAL TERMINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("RADIOMIC COX ANALYSIS COMPLETE")
print("=" * 75)


print(
    "\nFinal patient count:"
)

print(
    len(
        analysis_df
    )
)


print(
    "\nSelected radiomic features:"
)


for feature in selected_features:

    print(
        "  -",
        feature
    )


print(
    "\nC-index comparison:"
)


print(
    cindex_df
    .to_string(
        index=False
    )
)


print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR
)


print(
    "\nImportant note:"
)

print(
    "These radiomic Cox results are exploratory because "
    "the radiomics cohort contains only 46 patients."
)


print(
    "\nDONE."
)