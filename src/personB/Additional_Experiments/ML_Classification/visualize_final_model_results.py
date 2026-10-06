import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATH SETUP
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Script location:
# src\personB\Additional_Experiments\ML_Classification
#
# Go up:
# ML_Classification -> Additional_Experiments -> personB -> src -> project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "..", "..")
)

# Classification experiment root
RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "personB",
    "Additional_Experiments",
    "ML_Classification"
)

# Existing result folders
MODEL_COMPARISON_DIR = os.path.join(
    RESULTS_DIR,
    "Model_Comparison"
)

MODEL_EVALUATION_DIR = os.path.join(
    RESULTS_DIR,
    "Model_Evaluation"
)

FINAL_SUMMARIES_DIR = os.path.join(
    RESULTS_DIR,
    "Final_Summaries"
)

os.makedirs(MODEL_COMPARISON_DIR, exist_ok=True)
os.makedirs(MODEL_EVALUATION_DIR, exist_ok=True)
os.makedirs(FINAL_SUMMARIES_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_column_name(column):
    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def normalize_dataframe_columns(df):
    df = df.copy()
    df.columns = [
        normalize_column_name(c)
        for c in df.columns
    ]
    return df


def find_column(df, candidates):

    normalized = {
        normalize_column_name(col): col
        for col in df.columns
    }

    for candidate in candidates:

        candidate_norm = normalize_column_name(
            candidate
        )

        if candidate_norm in normalized:
            return normalized[candidate_norm]

    return None


def safe_filename(text):

    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def find_file_recursive(root, filename):

    matches = glob.glob(
        os.path.join(
            root,
            "**",
            filename
        ),
        recursive=True
    )

    if matches:
        return matches[0]

    return None


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PERSON B - ADDITIONAL ML CLASSIFICATION VISUALIZATION")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nClassification results directory:")
print(RESULTS_DIR)


# ============================================================
# FIND MODEL COMPARISON SUMMARY
# ============================================================

summary_candidates = [
    "model_comparison_v2_summary.csv",
    "model_comparison_summary.csv"
]

summary_path = None

for filename in summary_candidates:

    summary_path = find_file_recursive(
        RESULTS_DIR,
        filename
    )

    if summary_path is not None:
        break


if summary_path is None:

    raise FileNotFoundError(
        "\nCould not find model comparison summary CSV inside:\n"
        + RESULTS_DIR
    )


print("\nModel comparison summary:")
print(summary_path)


# ============================================================
# LOAD SUMMARY
# ============================================================

results = pd.read_csv(
    summary_path
)

print("\nResults shape:")
print(results.shape)

print("\nColumns:")
print(list(results.columns))


# ============================================================
# NORMALIZE COLUMNS
# ============================================================

results_normalized = normalize_dataframe_columns(
    results
)


required_columns = [
    "classifier",
    "feature_set",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
]


missing = [
    column
    for column in required_columns
    if column not in results_normalized.columns
]


if missing:

    raise ValueError(
        "Missing required columns:\n"
        + str(missing)
    )


# Convert metrics to numeric
for column in [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
]:

    results_normalized[column] = pd.to_numeric(
        results_normalized[column],
        errors="coerce"
    )


# ============================================================
# BEST MODEL BY ROC-AUC
# ============================================================

best_index = results_normalized[
    "roc_auc"
].idxmax()

best_model = results_normalized.loc[
    best_index
]

best_classifier = best_model[
    "classifier"
]

best_feature_set = best_model[
    "feature_set"
]

best_roc_auc = float(
    best_model["roc_auc"]
)


print("\n" + "-" * 70)
print("BEST MODEL BY ROC-AUC")
print("-" * 70)

print("Classifier :", best_classifier)
print("Feature Set:", best_feature_set)
print("ROC-AUC    :", f"{best_roc_auc:.4f}")


# ============================================================
# SAVE FINAL MODEL COMPARISON TABLE
# ============================================================

comparison_table_path = os.path.join(
    MODEL_COMPARISON_DIR,
    "final_model_comparison_table.csv"
)

results.to_csv(
    comparison_table_path,
    index=False
)

print(
    "\nSaved:",
    comparison_table_path
)


# ============================================================
# GENERAL METRIC PLOTS
# ============================================================

metric_information = [

    (
        "accuracy",
        "Accuracy",
        "accuracy_model_comparison.png"
    ),

    (
        "precision",
        "Precision",
        "precision_model_comparison.png"
    ),

    (
        "recall",
        "Recall",
        "recall_model_comparison.png"
    ),

    (
        "f1",
        "F1 Score",
        "f1_model_comparison.png"
    ),

    (
        "roc_auc",
        "ROC-AUC",
        "roc_auc_model_comparison.png"
    )
]


for metric_column, metric_title, filename in metric_information:

    plot_df = results_normalized.copy()

    plot_df["label"] = (
        plot_df["feature_set"].astype(str)
        + " + "
        + plot_df["classifier"].astype(str)
    )

    plot_df = plot_df.sort_values(
        metric_column,
        ascending=True
    )

    plt.figure(
        figsize=(11, 7)
    )

    plt.barh(
        plot_df["label"],
        plot_df[metric_column]
    )

    plt.xlabel(metric_title)
    plt.ylabel("Model")

    plt.title(
        f"Model Comparison - {metric_title}"
    )

    plt.xlim(0, 1)

    for i, value in enumerate(
        plot_df[metric_column]
    ):

        if pd.notna(value):

            plt.text(
                value + 0.01,
                i,
                f"{value:.3f}",
                va="center"
            )

    plt.tight_layout()

    output_path = os.path.join(
        MODEL_COMPARISON_DIR,
        filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output_path
    )


# ============================================================
# CLINICAL VS RADIOMICS VS COMBINED
# ============================================================

available_feature_sets = [
    "clinical_only",
    "radiomics_only",
    "combined"
]


comparison_df = results_normalized[
    results_normalized[
        "feature_set"
    ].isin(
        available_feature_sets
    )
].copy()


if not comparison_df.empty:

    # Prefer Logistic Regression
    lr_df = comparison_df[
        comparison_df[
            "classifier"
        ]
        .astype(str)
        .str.lower()
        .str.contains("logistic")
    ].copy()

    if not lr_df.empty:
        comparison_df = lr_df

    comparison_df = comparison_df.drop_duplicates(
        subset=["feature_set"],
        keep="first"
    )

    ordered_sets = [
        "clinical_only",
        "radiomics_only",
        "combined"
    ]

    x_labels = [
        "Clinical Only",
        "Radiomics Only",
        "Combined"
    ]

    values = []

    for feature_set in ordered_sets:

        row = comparison_df[
            comparison_df[
                "feature_set"
            ] == feature_set
        ]

        if len(row) > 0:

            values.append(
                float(
                    row.iloc[0]["roc_auc"]
                )
            )

        else:

            values.append(
                np.nan
            )

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        x_labels,
        values
    )

    plt.ylabel("ROC-AUC")
    plt.xlabel("Feature Set")

    plt.title(
        "Clinical vs Radiomics vs Combined - ROC-AUC"
    )

    plt.ylim(0, 1)

    for i, value in enumerate(values):

        if pd.notna(value):

            plt.text(
                i,
                value + 0.025,
                f"{value:.3f}",
                ha="center"
            )

    plt.tight_layout()

    output_path = os.path.join(
        MODEL_COMPARISON_DIR,
        "clinical_radiomics_combined_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output_path
    )


# ============================================================
# FIND OOF PREDICTION FILES
# ============================================================

print("\n" + "-" * 70)
print("SEARCHING FOR OOF PREDICTIONS")
print("-" * 70)


oof_files = glob.glob(
    os.path.join(
        RESULTS_DIR,
        "**",
        "*oof_predictions.csv"
    ),
    recursive=True
)


print(
    "OOF files found:",
    len(oof_files)
)

for file in sorted(oof_files):

    print(
        " -",
        file
    )


# ============================================================
# MATCH BEST MODEL OOF FILE
# ============================================================

best_classifier_clean = safe_filename(
    best_classifier
)

best_feature_set_clean = safe_filename(
    best_feature_set
)


preferred_oof_names = [

    (
        f"{best_classifier_clean}_"
        f"{best_feature_set_clean}_"
        f"oof_predictions.csv"
    ),

    (
        f"{best_classifier_clean.lower()}_"
        f"{best_feature_set_clean.lower()}_"
        f"oof_predictions.csv"
    )
]


best_oof_path = None


# Exact filename search
for file in oof_files:

    basename = os.path.basename(
        file
    ).lower()

    for candidate in preferred_oof_names:

        if basename == candidate.lower():

            best_oof_path = file
            break

    if best_oof_path is not None:
        break


# Fallback search
if best_oof_path is None:

    for file in oof_files:

        basename = os.path.basename(
            file
        ).lower()

        if (
            "logisticregression" in basename
            and "clinical_only" in basename
        ):

            best_oof_path = file
            break


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion_matrix_created = False


if best_oof_path is not None:

    print(
        "\nBest OOF file:"
    )

    print(
        best_oof_path
    )

    oof = pd.read_csv(
        best_oof_path
    )

    print(
        "\nOOF shape:",
        oof.shape
    )

    print(
        "OOF columns:",
        list(oof.columns)
    )


    oof_normalized = normalize_dataframe_columns(
        oof
    )


    actual_column = find_column(
        oof_normalized,
        [
            "actual",
            "y_true",
            "true",
            "actual_label",
            "true_label"
        ]
    )


    predicted_column = find_column(
        oof_normalized,
        [
            "predicted",
            "y_pred",
            "prediction",
            "predicted_label",
            "prediction_label"
        ]
    )


    print(
        "\nDetected actual column:",
        actual_column
    )

    print(
        "Detected predicted column:",
        predicted_column
    )


    if (
        actual_column is not None
        and predicted_column is not None
    ):

        actual = pd.to_numeric(
            oof_normalized[
                actual_column
            ],
            errors="coerce"
        )

        predicted = pd.to_numeric(
            oof_normalized[
                predicted_column
            ],
            errors="coerce"
        )


        valid = (
            actual.notna()
            & predicted.notna()
        )


        actual = actual[
            valid
        ].astype(int)

        predicted = predicted[
            valid
        ].astype(int)


        # Manual binary confusion matrix
        tn = int(
            (
                (actual == 0)
                & (predicted == 0)
            ).sum()
        )

        fp = int(
            (
                (actual == 0)
                & (predicted == 1)
            ).sum()
        )

        fn = int(
            (
                (actual == 1)
                & (predicted == 0)
            ).sum()
        )

        tp = int(
            (
                (actual == 1)
                & (predicted == 1)
            ).sum()
        )


        cm = np.array(
            [
                [tn, fp],
                [fn, tp]
            ]
        )


        print("\nConfusion Matrix:")
        print(cm)

        print("\nTN:", tn)
        print("FP:", fp)
        print("FN:", fn)
        print("TP:", tp)


        # Plot
        plt.figure(
            figsize=(8, 7)
        )

        plt.imshow(
            cm,
            interpolation="nearest"
        )

        plt.title(
            "Best Model Confusion Matrix\n"
            "Clinical Only + Logistic Regression"
        )

        plt.colorbar()

        tick_marks = np.arange(2)

        plt.xticks(
            tick_marks,
            [
                "Lower Risk (0)",
                "Higher Risk (1)"
            ]
        )

        plt.yticks(
            tick_marks,
            [
                "Lower Risk (0)",
                "Higher Risk (1)"
            ]
        )

        plt.xlabel(
            "Predicted Class"
        )

        plt.ylabel(
            "Actual Class"
        )


        for i in range(
            cm.shape[0]
        ):

            for j in range(
                cm.shape[1]
            ):

                plt.text(
                    j,
                    i,
                    str(cm[i, j]),
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=18,
                    fontweight="bold"
                )


        plt.tight_layout()


        confusion_output = os.path.join(
            MODEL_EVALUATION_DIR,
            "best_model_confusion_matrix.png"
        )


        plt.savefig(
            confusion_output,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


        print(
            "\nSaved:",
            confusion_output
        )


        confusion_matrix_created = True


# ============================================================
# SAVE BEST MODEL SUMMARY
# ============================================================

best_summary_path = os.path.join(
    FINAL_SUMMARIES_DIR,
    "best_model_summary.txt"
)


with open(
    best_summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PERSON B - ADDITIONAL ML CLASSIFICATION\n"
    )

    f.write(
        "BEST MODEL SUMMARY\n"
    )

    f.write(
        "=" * 55
        + "\n\n"
    )

    f.write(
        f"Feature set: {best_feature_set}\n"
    )

    f.write(
        f"Model: {best_classifier}\n"
    )

    f.write(
        f"ROC-AUC: {best_roc_auc:.4f}\n"
    )

    f.write(
        f"Accuracy: {best_model['accuracy']:.4f}\n"
    )

    f.write(
        f"Precision: {best_model['precision']:.4f}\n"
    )

    f.write(
        f"Recall: {best_model['recall']:.4f}\n"
    )

    f.write(
        f"F1: {best_model['f1']:.4f}\n"
    )


    if confusion_matrix_created:

        f.write(
            "\nConfusion Matrix\n"
        )

        f.write(
            "----------------\n"
        )

        f.write(
            f"TN: {tn}\n"
        )

        f.write(
            f"FP: {fp}\n"
        )

        f.write(
            f"FN: {fn}\n"
        )

        f.write(
            f"TP: {tp}\n"
        )

        f.write(
            "\nMatrix:\n"
        )

        f.write(
            str(cm)
            + "\n"
        )


print(
    "\nSaved:",
    best_summary_path
)


# ============================================================
# FINAL VISUALIZATION SUMMARY
# ============================================================

summary_output_path = os.path.join(
    FINAL_SUMMARIES_DIR,
    "final_visualization_summary.txt"
)


with open(
    summary_output_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PERSON B - ADDITIONAL ML CLASSIFICATION\n"
    )

    f.write(
        "FINAL VISUALIZATION SUMMARY\n"
    )

    f.write(
        "=" * 60
        + "\n\n"
    )

    f.write(
        "Best model selected using ROC-AUC:\n"
    )

    f.write(
        f"Feature Set: {best_feature_set}\n"
    )

    f.write(
        f"Classifier: {best_classifier}\n"
    )

    f.write(
        f"ROC-AUC: {best_roc_auc:.4f}\n\n"
    )

    f.write(
        "This folder contains additional ML classification "
        "experiments and is separate from the main Person B "
        "radiomics and survival-analysis contribution.\n"
    )


print(
    "Saved:",
    summary_output_path
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION VISUALIZATION COMPLETE")
print("=" * 70)

print("\nMain classification results:")
print(
    RESULTS_DIR
)

print("\nModel comparison:")
print(
    MODEL_COMPARISON_DIR
)

print("\nModel evaluation:")
print(
    MODEL_EVALUATION_DIR
)

print("\nFinal summaries:")
print(
    FINAL_SUMMARIES_DIR
)

print("\nDone.")