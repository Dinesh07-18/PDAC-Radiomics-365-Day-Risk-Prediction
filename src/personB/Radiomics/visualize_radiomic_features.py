import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PERSON B
# RADIOMIC FEATURE VISUALIZATION
# ============================================================


# ------------------------------------------------------------
# 1. PROJECT ROOT
# ------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        ".."
    )
)

print("=" * 80)
print("PERSON B - RADIOMIC FEATURE VISUALIZATION")
print("=" * 80)

print()
print("Project root:")
print(PROJECT_ROOT)


# ------------------------------------------------------------
# 2. INPUT FILE
# ------------------------------------------------------------

FEATURE_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "radiomics",
    "features",
    "radiomics_features_46patients.csv"
)

if not os.path.exists(FEATURE_FILE):
    raise FileNotFoundError(
        "\nRadiomics feature file not found:\n"
        f"{FEATURE_FILE}"
    )

print()
print("Radiomics feature file:")
print(FEATURE_FILE)


# ------------------------------------------------------------
# 3. OUTPUT DIRECTORY
# ------------------------------------------------------------

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "personB",
    "final_visualizations"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print()
print("Output directory:")
print(OUTPUT_DIR)


# ============================================================
# 4. LOAD DATA
# ============================================================

print()
print("Loading radiomic feature matrix...")

df = pd.read_csv(
    FEATURE_FILE
)

print()
print("Dataset shape:")
print(df.shape)

print()
print("Number of patients:")
print(len(df))


# ============================================================
# 5. IDENTIFY RADIOMIC FEATURES
# ============================================================

print()
print("Identifying radiomic features...")


metadata_columns = [
    "PatientID",
    "risk_label",
    "age_years",
    "Sex",
    "Tumor Grade",
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
]


radiomic_features = []

for column in df.columns:

    if column in metadata_columns:
        continue

    if pd.api.types.is_numeric_dtype(
        df[column]
    ):
        radiomic_features.append(
            column
        )


print()
print(
    "Radiomic features found:",
    len(radiomic_features)
)


if len(radiomic_features) == 0:
    raise RuntimeError(
        "No numeric radiomic features were found."
    )


# ============================================================
# 6. CREATE RADIOMICS DATAFRAME
# ============================================================

radiomics_df = df[
    radiomic_features
].copy()

print()
print(
    "Radiomic matrix shape:",
    radiomics_df.shape
)


# ============================================================
# 7. DATA QUALITY CHECK
# ============================================================

print()
print("Checking radiomic data quality...")


missing_values = int(
    radiomics_df.isna().sum().sum()
)


numeric_array = radiomics_df.to_numpy(
    dtype=float
)


infinite_values = int(
    np.isinf(
        numeric_array
    ).sum()
)


constant_features = []

for column in radiomic_features:

    if radiomics_df[
        column
    ].nunique() <= 1:

        constant_features.append(
            column
        )


print()
print(
    "Missing values:",
    missing_values
)

print(
    "Infinite values:",
    infinite_values
)

print(
    "Constant features:",
    len(constant_features)
)


# ============================================================
# 8. FEATURE STATISTICS
# ============================================================

print()
print("Calculating feature statistics...")


feature_stats_rows = []


for feature in radiomic_features:

    values = (
        radiomics_df[
            feature
        ]
        .astype(float)
    )

    feature_stats_rows.append(
        {
            "feature": feature,
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "max": values.max(),
            "variance": values.var(),
            "unique_values": values.nunique()
        }
    )


feature_stats = pd.DataFrame(
    feature_stats_rows
)


feature_stats = (
    feature_stats
    .sort_values(
        "variance",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


statistics_file = os.path.join(
    OUTPUT_DIR,
    "radiomic_feature_statistics.csv"
)


feature_stats.to_csv(
    statistics_file,
    index=False
)


print()
print(
    "Feature statistics saved:"
)

print(
    statistics_file
)


# ============================================================
# 9. TOP FEATURES FOR DISTRIBUTION
# ============================================================

TOP_DISTRIBUTION_FEATURES = 12


top_distribution = (
    feature_stats
    .head(
        TOP_DISTRIBUTION_FEATURES
    )[
        "feature"
    ]
    .tolist()
)


print()
print(
    "Top features for distribution plot:"
)


for i, feature in enumerate(
    top_distribution,
    start=1
):

    print(
        f"{i:2d}. {feature}"
    )


# ============================================================
# 10. STANDARDIZED FEATURE DISTRIBUTION
# ============================================================

print()
print(
    "Creating feature distribution visualization..."
)


distribution_data = []


for feature in top_distribution:

    values = (
        radiomics_df[
            feature
        ]
        .astype(float)
    )

    mean_value = values.mean()
    std_value = values.std()

    if std_value == 0:

        standardized = (
            values - mean_value
        )

    else:

        standardized = (
            values - mean_value
        ) / std_value

    distribution_data.append(
        standardized.to_numpy()
    )


fig, ax = plt.subplots(
    figsize=(14, 8)
)


ax.boxplot(
    distribution_data,
    tick_labels=top_distribution,
    showfliers=True
)


ax.set_title(
    "Distribution of Top Radiomic Features",
    fontsize=15
)

ax.set_xlabel(
    "Radiomic Feature",
    fontsize=12
)

ax.set_ylabel(
    "Standardized Feature Value",
    fontsize=12
)


plt.xticks(
    rotation=75,
    ha="right"
)


plt.tight_layout()


distribution_file = os.path.join(
    OUTPUT_DIR,
    "radiomic_feature_distributions.png"
)


plt.savefig(
    distribution_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print()
print(
    "Distribution plot saved:"
)

print(
    distribution_file
)


# ============================================================
# 11. CORRELATION ANALYSIS
# ============================================================

print()
print(
    "Calculating radiomic feature correlations..."
)


TOP_CORRELATION_FEATURES = 20


top_correlation = (
    feature_stats
    .head(
        TOP_CORRELATION_FEATURES
    )[
        "feature"
    ]
    .tolist()
)


correlation_df = radiomics_df[
    top_correlation
].copy()


correlation_matrix = (
    correlation_df.corr()
)


print()
print(
    "Correlation matrix shape:"
)

print(
    correlation_matrix.shape
)


# ============================================================
# 12. CORRELATION HEATMAP
# ============================================================

print()
print(
    "Creating correlation heatmap..."
)


fig, ax = plt.subplots(
    figsize=(14, 12)
)


heatmap = ax.imshow(
    correlation_matrix.to_numpy(),
    aspect="auto",
    vmin=-1,
    vmax=1
)


positions = np.arange(
    len(top_correlation)
)


ax.set_xticks(
    positions
)

ax.set_yticks(
    positions
)


ax.set_xticklabels(
    top_correlation,
    rotation=75,
    ha="right",
    fontsize=8
)


ax.set_yticklabels(
    top_correlation,
    fontsize=8
)


ax.set_title(
    "Correlation Heatmap of Top Radiomic Features",
    fontsize=15
)


# ------------------------------------------------------------
# Correlation values inside cells
# ------------------------------------------------------------

for i in range(
    len(top_correlation)
):

    for j in range(
        len(top_correlation)
    ):

        value = correlation_matrix.iloc[
            i,
            j
        ]

        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=6
        )


# ------------------------------------------------------------
# Colorbar
# ------------------------------------------------------------

colorbar = fig.colorbar(
    heatmap,
    ax=ax
)


colorbar.set_label(
    "Pearson Correlation",
    rotation=270,
    labelpad=18
)


plt.tight_layout()


correlation_file = os.path.join(
    OUTPUT_DIR,
    "radiomic_correlation_heatmap.png"
)


plt.savefig(
    correlation_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print()
print(
    "Correlation heatmap saved:"
)

print(
    correlation_file
)


# ============================================================
# 13. HIGH-CORRELATION FEATURE PAIRS
# ============================================================

print()
print(
    "Finding highly correlated feature pairs..."
)


high_correlation_pairs = []


for i in range(
    len(top_correlation)
):

    for j in range(
        i + 1,
        len(top_correlation)
    ):

        feature_a = top_correlation[
            i
        ]

        feature_b = top_correlation[
            j
        ]

        correlation_value = (
            correlation_matrix
            .loc[
                feature_a,
                feature_b
            ]
        )


        if abs(
            correlation_value
        ) >= 0.90:

            high_correlation_pairs.append(
                {
                    "feature_1": feature_a,
                    "feature_2": feature_b,
                    "correlation": correlation_value
                }
            )


# ------------------------------------------------------------
# Create dataframe
# ------------------------------------------------------------

if len(
    high_correlation_pairs
) > 0:

    high_corr_df = pd.DataFrame(
        high_correlation_pairs
    )

    high_corr_df[
        "absolute_correlation"
    ] = high_corr_df[
        "correlation"
    ].abs()

    high_corr_df = (
        high_corr_df
        .sort_values(
            "absolute_correlation",
            ascending=False
        )
        .drop(
            columns=[
                "absolute_correlation"
            ]
        )
        .reset_index(
            drop=True
        )
    )

else:

    high_corr_df = pd.DataFrame(
        columns=[
            "feature_1",
            "feature_2",
            "correlation"
        ]
    )


high_corr_file = os.path.join(
    OUTPUT_DIR,
    "high_correlation_radiomic_pairs.csv"
)


high_corr_df.to_csv(
    high_corr_file,
    index=False
)


print()
print(
    "Highly correlated pairs saved:"
)

print(
    high_corr_file
)


print()
print(
    "High-correlation pairs found:",
    len(high_corr_df)
)


# ============================================================
# 14. SUMMARY FILE
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "radiomic_visualization_summary.txt"
)


with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PERSON B - RADIOMIC VISUALIZATION SUMMARY\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )

    f.write(
        f"Patients: {len(df)}\n"
    )

    f.write(
        f"Radiomic features: "
        f"{len(radiomic_features)}\n"
    )

    f.write(
        f"Missing values: "
        f"{missing_values}\n"
    )

    f.write(
        f"Infinite values: "
        f"{infinite_values}\n"
    )

    f.write(
        f"Constant features: "
        f"{len(constant_features)}\n"
    )

    f.write(
        f"Distribution features: "
        f"{len(top_distribution)}\n"
    )

    f.write(
        f"Correlation features: "
        f"{len(top_correlation)}\n"
    )

    f.write(
        f"High-correlation pairs "
        f"(absolute r >= 0.90): "
        f"{len(high_corr_df)}\n\n"
    )

    f.write(
        "Top variance features:\n"
    )

    for i, feature in enumerate(
        top_distribution,
        start=1
    ):

        f.write(
            f"{i}. {feature}\n"
        )


print()
print(
    "Summary saved:"
)

print(
    summary_file
)


# ============================================================
# 15. FINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print("RADIOMIC VISUALIZATION COMPLETE")
print("=" * 80)


print()
print("Generated files:")


print()
print(
    "1. Radiomic feature distributions:"
)

print(
    distribution_file
)


print()
print(
    "2. Radiomic correlation heatmap:"
)

print(
    correlation_file
)


print()
print(
    "3. Radiomic feature statistics:"
)

print(
    statistics_file
)


print()
print(
    "4. Highly correlated feature pairs:"
)

print(
    high_corr_file
)


print()
print(
    "5. Visualization summary:"
)

print(
    summary_file
)


print()
print(
    "PERSON B RADIOMIC VISUALIZATION "
    "COMPLETED SUCCESSFULLY."
)


print("=" * 80)