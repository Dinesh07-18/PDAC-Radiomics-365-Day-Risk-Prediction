import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# PATHS
# ============================================================

PROJECT = os.path.abspath(".")

RESULTS_DIR = os.path.join(
    PROJECT,
    "data",
    "radiomics",
    "results"
)

OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    "final_visualizations"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FILES
# ============================================================

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "model_comparison_v2_summary.csv"
)

FOLD_FILE = os.path.join(
    RESULTS_DIR,
    "model_comparison_v2_fold_results.csv"
)

CLINICAL_PRED = os.path.join(
    RESULTS_DIR,
    "LogisticRegression_clinical_only_oof_predictions.csv"
)

RADIO_PRED = os.path.join(
    RESULTS_DIR,
    "LogisticRegression_radiomics_only_oof_predictions.csv"
)

COMBINED_PRED = os.path.join(
    RESULTS_DIR,
    "LogisticRegression_combined_oof_predictions.csv"
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 80)
print("FINAL PDAC RESULTS AND VISUALIZATION")
print("=" * 80)


# ============================================================
# LOAD RESULTS
# ============================================================

summary = pd.read_csv(
    SUMMARY_FILE
)

fold_results = pd.read_csv(
    FOLD_FILE
)


print()
print("Summary loaded:")
print(summary.shape)

print()
print("Fold results loaded:")
print(fold_results.shape)


# ============================================================
# ROUND VALUES
# ============================================================

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC"
]

summary_display = summary.copy()

for column in metric_columns:

    summary_display[column] = (
        summary_display[column]
        .round(4)
    )


# ============================================================
# SAVE CLEAN SUMMARY
# ============================================================

clean_summary_file = os.path.join(
    OUTPUT_DIR,
    "final_model_comparison_table.csv"
)

summary_display.to_csv(
    clean_summary_file,
    index=False
)


# ============================================================
# 1. MODEL ROC-AUC COMPARISON
# ============================================================

print()
print("Creating ROC-AUC comparison...")


plot_data = summary.copy()

plot_data["Model"] = (
    plot_data["Classifier"]
    + " - "
    + plot_data["Feature_Set"]
)


plot_data = plot_data.sort_values(
    "ROC_AUC",
    ascending=True
)


plt.figure(
    figsize=(11, 7)
)

plt.barh(
    plot_data["Model"],
    plot_data["ROC_AUC"]
)

plt.axvline(
    0.5,
    linestyle="--"
)

plt.xlim(
    0,
    1
)

plt.xlabel(
    "ROC-AUC"
)

plt.ylabel(
    "Model"
)

plt.title(
    "PDAC 365-Day Risk Prediction: ROC-AUC Comparison"
)

plt.tight_layout()

roc_auc_file = os.path.join(
    OUTPUT_DIR,
    "roc_auc_model_comparison.png"
)

plt.savefig(
    roc_auc_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 2. PERFORMANCE COMPARISON
# ============================================================

print(
    "Creating performance comparison..."
)


main_models = summary[
    (
        (summary["Classifier"] == "LogisticRegression")
    )
].copy()


main_models["Model"] = (
    main_models["Feature_Set"]
    .replace(
        {
            "clinical_only":
                "Clinical only",

            "radiomics_only":
                "Radiomics only",

            "combined":
                "Clinical + Radiomics"
        }
    )
)


metrics_for_plot = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC"
]


x = np.arange(
    len(main_models)
)

width = 0.15


plt.figure(
    figsize=(12, 7)
)


for i, metric in enumerate(
    metrics_for_plot
):

    plt.bar(
        x + i * width,
        main_models[metric],
        width,
        label=metric
    )


plt.xticks(
    x + width * 2,
    main_models["Model"]
)

plt.ylim(
    0,
    1
)

plt.ylabel(
    "Score"
)

plt.title(
    "Clinical vs Radiomics vs Combined Performance"
)

plt.legend()

plt.tight_layout()


performance_file = os.path.join(
    OUTPUT_DIR,
    "clinical_radiomics_combined_comparison.png"
)


plt.savefig(
    performance_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 3. BEST MODEL CONFUSION MATRIX
# ============================================================

print(
    "Creating confusion matrix..."
)


best_prediction_file = (
    CLINICAL_PRED
)


best_predictions = pd.read_csv(
    best_prediction_file
)


cm = confusion_matrix(

    best_predictions["Actual"],

    best_predictions["Predicted"]

)


plt.figure(
    figsize=(7, 6)
)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Class 0",
        "Class 1"
    ]
)


disp.plot(
    values_format="d"
)

plt.title(
    "Clinical Logistic Regression - Confusion Matrix"
)

plt.tight_layout()


cm_file = os.path.join(
    OUTPUT_DIR,
    "best_model_confusion_matrix.png"
)


plt.savefig(
    cm_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 4. ROC CURVES - THREE LOGISTIC MODELS
# ============================================================

print(
    "Creating ROC curves..."
)


prediction_files = {

    "Clinical only":
        CLINICAL_PRED,

    "Radiomics only":
        RADIO_PRED,

    "Clinical + Radiomics":
        COMBINED_PRED

}


plt.figure(
    figsize=(9, 7)
)


for model_name, file_path in (
    prediction_files.items()
):


    predictions = pd.read_csv(
        file_path
    )


    y_true = predictions[
        "Actual"
    ]


    y_prob = predictions[
        "Probability_Class1"
    ]


    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )


    roc_auc = auc(
        fpr,
        tpr
    )


    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=(
            f"{model_name} "
            f"(AUC = {roc_auc:.3f})"
        )
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Chance"
)


plt.xlim(
    0,
    1
)

plt.ylim(
    0,
    1.05
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curves - Logistic Regression Models"
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()


roc_file = os.path.join(
    OUTPUT_DIR,
    "logistic_regression_roc_curves.png"
)


plt.savefig(
    roc_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 5. FOLD-TO-FOLD AUC VARIABILITY
# ============================================================

print(
    "Creating cross-validation variability plot..."
)


lr_folds = fold_results[
    fold_results["Classifier"]
    == "LogisticRegression"
].copy()


lr_folds["Model"] = (
    lr_folds["Feature_Set"]
    .replace(
        {
            "clinical_only":
                "Clinical only",

            "radiomics_only":
                "Radiomics only",

            "combined":
                "Clinical + Radiomics"
        }
    )
)


plt.figure(
    figsize=(10, 7)
)


models_order = [
    "Clinical only",
    "Radiomics only",
    "Clinical + Radiomics"
]


box_data = []


for model in models_order:

    box_data.append(
        lr_folds[
            lr_folds["Model"]
            == model
        ]["ROC_AUC"].values
    )


plt.boxplot(
    box_data,
    labels=models_order
)


plt.axhline(
    0.5,
    linestyle="--"
)


plt.ylabel(
    "Fold ROC-AUC"
)

plt.title(
    "Cross-Validation ROC-AUC Variability"
)

plt.xticks(
    rotation=15
)

plt.tight_layout()


cv_file = os.path.join(
    OUTPUT_DIR,
    "cross_validation_auc_variability.png"
)


plt.savefig(
    cv_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 6. FINAL RESEARCH SUMMARY
# ============================================================

clinical_result = summary[
    (
        (summary["Classifier"]
         == "LogisticRegression")
        &
        (summary["Feature_Set"]
         == "clinical_only")
    )
].iloc[0]


radiomics_result = summary[
    (
        (summary["Classifier"]
         == "LogisticRegression")
        &
        (summary["Feature_Set"]
         == "radiomics_only")
    )
].iloc[0]


combined_result = summary[
    (
        (summary["Classifier"]
         == "LogisticRegression")
        &
        (summary["Feature_Set"]
         == "combined")
    )
].iloc[0]


print()
print("=" * 80)
print("KEY RESULTS")
print("=" * 80)


print()
print(
    "Clinical-only ROC-AUC:",
    round(
        clinical_result["ROC_AUC"],
        4
    )
)


print(
    "Radiomics-only ROC-AUC:",
    round(
        radiomics_result["ROC_AUC"],
        4
    )
)


print(
    "Combined ROC-AUC:",
    round(
        combined_result["ROC_AUC"],
        4
    )
)


print()
print(
    "Clinical-only Recall:",
    round(
        clinical_result["Recall"],
        4
    )
)


print(
    "Radiomics-only Recall:",
    round(
        radiomics_result["Recall"],
        4
    )
)


print(
    "Combined Recall:",
    round(
        combined_result["Recall"],
        4
    )
)


# ============================================================
# 7. AUTOMATIC CONCLUSION
# ============================================================

clinical_auc = clinical_result[
    "ROC_AUC"
]

radiomics_auc = radiomics_result[
    "ROC_AUC"
]

combined_auc = combined_result[
    "ROC_AUC"
]


if combined_auc > clinical_auc:

    conclusion = (
        "Radiomics improved ROC-AUC "
        "over the clinical-only model."
    )

else:

    conclusion = (
        "Radiomics did not improve "
        "ROC-AUC over the clinical-only model."
    )


print()
print(
    "Conclusion:"
)

print(
    conclusion
)


# ============================================================
# 8. WRITE TEXT REPORT
# ============================================================

report_file = os.path.join(
    OUTPUT_DIR,
    "final_results_summary.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PDAC 365-DAY RISK CLASSIFICATION\n"
    )

    f.write(
        "FINAL RESULTS SUMMARY\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )


    f.write(
        "Dataset:\n"
    )

    f.write(
        "46 CT-eligible patients\n"
    )

    f.write(
        "Class 0: 33\n"
    )

    f.write(
        "Class 1: 13\n\n"
    )


    f.write(
        "LOGISTIC REGRESSION RESULTS\n"
    )

    f.write(
        "-" * 70
        + "\n"
    )


    for name, result in [

        (
            "Clinical-only",
            clinical_result
        ),

        (
            "Radiomics-only",
            radiomics_result
        ),

        (
            "Clinical + Radiomics",
            combined_result
        )

    ]:

        f.write(
            f"{name}\n"
        )

        f.write(
            f"Accuracy: "
            f"{result['Accuracy']:.4f}\n"
        )

        f.write(
            f"Precision: "
            f"{result['Precision']:.4f}\n"
        )

        f.write(
            f"Recall: "
            f"{result['Recall']:.4f}\n"
        )

        f.write(
            f"F1: "
            f"{result['F1']:.4f}\n"
        )

        f.write(
            f"ROC-AUC: "
            f"{result['ROC_AUC']:.4f}\n\n"
        )


    f.write(
        "INTERPRETATION\n"
    )

    f.write(
        "-" * 70
        + "\n"
    )

    f.write(
        conclusion
        + "\n\n"
    )


    f.write(
        "The results should be interpreted cautiously "
        "because the final radiomics cohort contains "
        "only 46 patients and 13 Class-1 cases, while "
        "107 radiomic features were extracted.\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print("FILES CREATED")
print("=" * 80)

print()
print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)


print()
print(
    "1.",
    roc_auc_file
)

print(
    "2.",
    performance_file
)

print(
    "3.",
    cm_file
)

print(
    "4.",
    roc_file
)

print(
    "5.",
    cv_file
)

print(
    "6.",
    clean_summary_file
)

print(
    "7.",
    report_file
)


print()
print("=" * 80)
print("FINAL VISUALIZATION COMPLETE")
print("=" * 80)