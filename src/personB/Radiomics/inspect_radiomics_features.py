import os
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT = os.path.abspath(".")

INPUT_FILE = os.path.join(
    PROJECT,
    "data",
    "radiomics",
    "features",
    "radiomics_features_46patients.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT,
    "data",
    "radiomics",
    "features"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "radiomics_feature_quality_report.csv"
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("RADIOMICS FEATURE QUALITY INSPECTION")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        "Radiomics feature file not found:\n"
        + INPUT_FILE
    )


df = pd.read_csv(
    INPUT_FILE
)


print()
print("Dataset shape:")
print(df.shape)


print()
print(
    "Number of patients:",
    df["PatientID"].nunique()
)


# ============================================================
# IDENTIFY CORE COLUMNS
# ============================================================

core_columns = [

    "PatientID",

    "risk_label",

    "age_years",

    "Sex",

    "Tumor Grade",

    "StructureSetLabel",

    "ROI_role",

    "ROIVolume",

    "normalized_phase",

    "ReferencedSeriesDescription"

]


core_columns_present = [

    c
    for c in core_columns
    if c in df.columns

]


print()
print("Core/metadata columns:")
print("-" * 50)

for c in core_columns_present:

    print(c)


# ============================================================
# IDENTIFY RADIOMIC FEATURES
# ============================================================

radiomic_columns = [

    c
    for c in df.columns
    if c not in core_columns_present

]


print()
print(
    "Radiomic feature columns:",
    len(radiomic_columns)
)


# ============================================================
# DUPLICATE PATIENT CHECK
# ============================================================

print()
print("=" * 70)
print("PATIENT CHECK")
print("=" * 70)


duplicate_patients = (
    df["PatientID"]
    .duplicated()
    .sum()
)


print(
    "Duplicate patients:",
    duplicate_patients
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print()
print("Risk distribution:")

print(
    df["risk_label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# DATA TYPES
# ============================================================

print()
print("=" * 70)
print("DATA TYPE CHECK")
print("=" * 70)


numeric_features = []

non_numeric_features = []


for column in radiomic_columns:

    if pd.api.types.is_numeric_dtype(
        df[column]
    ):

        numeric_features.append(
            column
        )

    else:

        non_numeric_features.append(
            column
        )


print(
    "Numeric radiomic features:",
    len(numeric_features)
)


print(
    "Non-numeric columns among candidate features:",
    len(non_numeric_features)
)


if non_numeric_features:

    print()
    print(
        "Non-numeric columns:"
    )

    for c in non_numeric_features:

        print(c)


# ============================================================
# MISSING VALUES
# ============================================================

print()
print("=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)


missing_counts = (
    df[numeric_features]
    .isna()
    .sum()
)


missing_features = (
    missing_counts[
        missing_counts > 0
    ]
    .sort_values(
        ascending=False
    )
)


print(
    "Features with missing values:",
    len(missing_features)
)


if len(missing_features) > 0:

    print()
    print(
        missing_features
    )


# ============================================================
# INFINITE VALUES
# ============================================================

print()
print("=" * 70)
print("INFINITE VALUE CHECK")
print("=" * 70)


inf_counts = {}

for column in numeric_features:

    values = (
        pd.to_numeric(
            df[column],
            errors="coerce"
        )
    )

    count = np.isinf(
        values
    ).sum()

    if count > 0:

        inf_counts[column] = int(
            count
        )


print(
    "Features containing Inf:",
    len(inf_counts)
)


if inf_counts:

    for column, count in inf_counts.items():

        print(
            column,
            "=>",
            count
        )


# ============================================================
# CONSTANT FEATURES
# ============================================================

print()
print("=" * 70)
print("CONSTANT FEATURE CHECK")
print("=" * 70)


constant_features = []


for column in numeric_features:

    unique_count = (
        df[column]
        .nunique(
            dropna=False
        )
    )

    if unique_count <= 1:

        constant_features.append(
            column
        )


print(
    "Constant features:",
    len(constant_features)
)


if constant_features:

    print()

    for column in constant_features:

        print(column)


# ============================================================
# NEAR-CONSTANT FEATURES
# ============================================================

print()
print("=" * 70)
print("NEAR-CONSTANT FEATURE CHECK")
print("=" * 70)


near_constant_features = []


for column in numeric_features:

    counts = (
        df[column]
        .value_counts(
            dropna=False
        )
    )

    if len(counts) == 0:

        continue


    dominant_fraction = (
        counts.iloc[0]
        / len(df)
    )


    if dominant_fraction >= 0.95:

        near_constant_features.append({

            "Feature":
                column,

            "DominantFraction":
                dominant_fraction

        })


print(
    "Near-constant features:",
    len(near_constant_features)
)


if near_constant_features:

    for item in near_constant_features:

        print(
            item["Feature"],
            "=>",
            round(
                item["DominantFraction"],
                3
            )
        )


# ============================================================
# UNIQUE VALUE CHECK
# ============================================================

print()
print("=" * 70)
print("FEATURE VARIABILITY")
print("=" * 70)


variability_records = []


for column in numeric_features:

    series = pd.to_numeric(
        df[column],
        errors="coerce"
    )


    variability_records.append({

        "Feature":
            column,

        "UniqueValues":
            series.nunique(),

        "MissingValues":
            series.isna().sum(),

        "Mean":
            series.mean(),

        "Std":
            series.std(),

        "Min":
            series.min(),

        "Max":
            series.max(),

        "Median":
            series.median()

    })


variability_df = pd.DataFrame(
    variability_records
)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

print()
print("=" * 70)
print("CORRELATION CHECK")
print("=" * 70)


clean_numeric = (
    df[numeric_features]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)


correlation_matrix = (
    clean_numeric
    .corr()
    .abs()
)


# Remove diagonal.

correlation_matrix = (
    correlation_matrix
    .where(
        ~np.eye(
            correlation_matrix.shape[0],
            dtype=bool
        )
    )
)


high_corr_pairs = []


threshold = 0.95


for i, feature_a in enumerate(
    correlation_matrix.columns
):

    for j, feature_b in enumerate(
        correlation_matrix.columns
    ):

        if j <= i:

            continue


        correlation = (
            correlation_matrix.loc[
                feature_a,
                feature_b
            ]
        )


        if (
            pd.notna(correlation)
            and correlation >= threshold
        ):

            high_corr_pairs.append({

                "Feature_A":
                    feature_a,

                "Feature_B":
                    feature_b,

                "Absolute_Correlation":
                    correlation

            })


print(
    "Highly correlated feature pairs (|r| >= 0.95):",
    len(high_corr_pairs)
)


if high_corr_pairs:

    print()
    print(
        "First 20 highly correlated pairs:"
    )

    for pair in high_corr_pairs[:20]:

        print(
            pair["Feature_A"],
            "<-->",
            pair["Feature_B"],
            "r=",
            round(
                pair["Absolute_Correlation"],
                4
            )
        )


# ============================================================
# CREATE FEATURE QUALITY REPORT
# ============================================================

quality_records = []


for column in numeric_features:

    series = pd.to_numeric(
        df[column],
        errors="coerce"
    )


    quality_records.append({

        "Feature":
            column,

        "DataType":
            str(df[column].dtype),

        "UniqueValues":
            series.nunique(),

        "MissingValues":
            int(series.isna().sum()),

        "InfiniteValues":
            int(
                np.isinf(
                    series
                ).sum()
            ),

        "Mean":
            series.mean(),

        "Std":
            series.std(),

        "Min":
            series.min(),

        "Median":
            series.median(),

        "Max":
            series.max(),

        "Constant":
            series.nunique() <= 1,

        "NearConstant_95pct":
            (
                series.value_counts(
                    dropna=False
                ).iloc[0]
                / len(series)
                >= 0.95
            )

    })


quality_df = pd.DataFrame(
    quality_records
)


# ============================================================
# SAVE REPORT
# ============================================================

quality_df.to_csv(
    REPORT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("QUALITY INSPECTION COMPLETE")
print("=" * 70)


print(
    "Total dataset columns:",
    len(df.columns)
)


print(
    "Numeric radiomic features:",
    len(numeric_features)
)


print(
    "Missing-value features:",
    len(missing_features)
)


print(
    "Infinite-value features:",
    len(inf_counts)
)


print(
    "Constant features:",
    len(constant_features)
)


print(
    "Near-constant features:",
    len(near_constant_features)
)


print(
    "High-correlation pairs:",
    len(high_corr_pairs)
)


print()
print(
    "Quality report:"
)

print(
    os.path.abspath(
        REPORT_FILE
    )
)


print()
print("=" * 70)