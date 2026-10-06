import os
import warnings
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT = os.path.abspath(".")

RADIOMICS_FILE = os.path.join(
    PROJECT,
    "data",
    "radiomics",
    "features",
    "radiomics_features_46patients.csv"
)

CLINICAL_FILE = os.path.join(
    PROJECT,
    "data",
    "processed",
    "classification_365days.csv"
)

RESULTS_DIR = os.path.join(
    PROJECT,
    "data",
    "radiomics",
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 75)
print("PDAC 365-DAY RISK MODEL COMPARISON")
print("=" * 75)


# ============================================================
# LOAD DATA
# ============================================================

radiomics = pd.read_csv(
    RADIOMICS_FILE
)

clinical = pd.read_csv(
    CLINICAL_FILE
)

print()
print("Radiomics shape:", radiomics.shape)
print("Clinical shape:", clinical.shape)


# ============================================================
# NORMALIZE PATIENT IDS
# ============================================================

radiomics["PatientID"] = (
    radiomics["PatientID"]
    .astype(str)
    .str.strip()
)

clinical["Case Submitter ID"] = (
    clinical["Case Submitter ID"]
    .astype(str)
    .str.strip()
)


# ============================================================
# PREPARE CLINICAL DATA
# ============================================================

print()
print("=" * 75)
print("PREPARING CLINICAL DATA")
print("=" * 75)


clinical_columns = [
    "Case Submitter ID",
    "age_years",
    "Sex",
    "Tumor Grade",
    "risk_label"
]


available = [
    c
    for c in clinical_columns
    if c in clinical.columns
]


print()
print("Clinical columns available:")

for c in available:
    print(" ", c)


required = [
    "Case Submitter ID",
    "age_years",
    "Sex",
    "Tumor Grade",
    "risk_label"
]


missing_required = [
    c
    for c in required
    if c not in clinical.columns
]


if missing_required:

    raise RuntimeError(
        "Required clinical columns missing:\n"
        + str(missing_required)
    )


# Use the clinical file as the authoritative source
# for clinical predictors and target.

clinical_model_data = clinical[
    required
].copy()


clinical_model_data = (
    clinical_model_data
    .rename(
        columns={
            "Case Submitter ID": "PatientID"
        }
    )
)


# ============================================================
# MERGE
# ============================================================

data = radiomics.merge(
    clinical_model_data,
    on="PatientID",
    how="inner",
    suffixes=(
        "_radiomics",
        "_clinical"
    )
)


print()
print("=" * 75)
print("MERGED DATASET")
print("=" * 75)


print(
    "Patients after merge:",
    data["PatientID"].nunique()
)

print(
    "Rows:",
    len(data)
)


# ============================================================
# TARGET
# ============================================================

# The target from the clinical classification file
# is authoritative.

if "risk_label_clinical" in data.columns:

    data["risk_label"] = (
        pd.to_numeric(
            data["risk_label_clinical"],
            errors="coerce"
        )
    )

elif "risk_label" in data.columns:

    data["risk_label"] = (
        pd.to_numeric(
            data["risk_label"],
            errors="coerce"
        )
    )

else:

    raise RuntimeError(
        "Could not find clinical risk_label."
    )


data = data.dropna(
    subset=["risk_label"]
).copy()


data["risk_label"] = (
    data["risk_label"]
    .astype(int)
)


# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicate_count = (
    data["PatientID"]
    .duplicated()
    .sum()
)


print(
    "Duplicate patients:",
    duplicate_count
)


if duplicate_count != 0:

    raise RuntimeError(
        "Duplicate patients detected."
    )


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print()
print("Target distribution:")

print(
    data["risk_label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# CLINICAL FEATURES
# ============================================================

clinical_features = [
    "age_years_clinical",
    "Sex_clinical",
    "Tumor Grade_clinical"
]


# Make sure they exist.

missing_clinical = [
    c
    for c in clinical_features
    if c not in data.columns
]


if missing_clinical:

    raise RuntimeError(
        "Clinical features missing after merge:\n"
        + str(missing_clinical)
    )


print()
print("Clinical features used:")

for c in clinical_features:
    print(" ", c)


# ============================================================
# RADIOMIC FEATURES
# ============================================================

# IMPORTANT:
# Only take the original radiomics columns.
# Clinical metadata must NOT accidentally enter
# the radiomics feature set.

radiomics_excluded = {

    "PatientID",

    "risk_label",

    "risk_label_clinical",

    "age_years",
    "age_years_radiomics",
    "Sex",
    "Sex_radiomics",
    "Tumor Grade",
    "Tumor Grade_radiomics",

    "StructureSetLabel",
    "ROI_role",
    "ROIVolume",
    "normalized_phase",
    "ReferencedSeriesDescription",

    "Case Submitter ID"

}


radiomic_features = [

    column

    for column in radiomics.columns

    if column not in radiomics_excluded

]


# Keep numeric features only.

radiomic_features = [

    column

    for column in radiomic_features

    if pd.api.types.is_numeric_dtype(
        data[column]
    )

]


print()
print(
    "Radiomic features:",
    len(radiomic_features)
)


if len(radiomic_features) == 0:

    raise RuntimeError(
        "No radiomic features found."
    )


# ============================================================
# CORRELATION FILTER
# ============================================================

class CorrelationFilter(
    BaseEstimator,
    TransformerMixin
):

    def __init__(
        self,
        threshold=0.95
    ):

        self.threshold = threshold


    def fit(
        self,
        X,
        y=None
    ):

        X_df = pd.DataFrame(
            X
        )


        correlation = (
            X_df.corr()
            .abs()
        )


        upper = correlation.where(

            np.triu(
                np.ones(
                    correlation.shape,
                    dtype=bool
                ),
                k=1
            )
        )


        self.features_to_drop_ = [

            column

            for column in upper.columns

            if any(
                upper[column]
                > self.threshold
            )

        ]


        self.keep_indices_ = [

            i

            for i, column
            in enumerate(
                X_df.columns
            )

            if column not in
            self.features_to_drop_

        ]


        return self


    def transform(
        self,
        X
    ):

        X_array = np.asarray(
            X
        )

        return X_array[
            :,
            self.keep_indices_
        ]


# ============================================================
# SAFE FEATURE COUNT
# ============================================================

def safe_k(
    n_features
):

    return max(
        1,
        min(
            10,
            n_features
        )
    )


# ============================================================
# CLASSIFIER
# ============================================================

def create_classifier():

    return LogisticRegression(

        max_iter=5000,

        class_weight="balanced",

        solver="liblinear",

        random_state=42

    )


# ============================================================
# CROSS VALIDATION
# ============================================================

N_SPLITS = 5


cv = StratifiedKFold(

    n_splits=N_SPLITS,

    shuffle=True,

    random_state=42

)


# ============================================================
# CLINICAL PREPROCESSOR
# ============================================================

clinical_numeric = [
    "age_years_clinical"
]


clinical_categorical = [
    "Sex_clinical",
    "Tumor Grade_clinical"
]


clinical_preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",

            Pipeline(
                steps=[

                    (
                        "imputer",

                        SimpleImputer(
                            strategy="median"
                        )
                    ),

                    (
                        "scaler",

                        StandardScaler()
                    )

                ]
            ),

            clinical_numeric

        ),

        (
            "categorical",

            Pipeline(
                steps=[

                    (
                        "imputer",

                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),

                    (
                        "onehot",

                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        )
                    )

                ]
            ),

            clinical_categorical

        )

    ],

    remainder="drop"

)


# ============================================================
# RADIOMICS PIPELINE
# ============================================================

radiomics_pipeline = Pipeline(

    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "correlation_filter",

            CorrelationFilter(
                threshold=0.95
            )
        ),

        (
            "scaler",

            StandardScaler()
        ),

        (
            "selector",

            SelectKBest(
                score_func=f_classif,
                k=safe_k(
                    len(radiomic_features)
                )
            )
        )

    ]

)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    model_name
):

    print()
    print("=" * 75)
    print(
        "MODEL:",
        model_name
    )
    print("=" * 75)


    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    patient_ids_all = []


    fold_records = []


    for fold, (
        train_idx,
        test_idx
    ) in enumerate(

        cv.split(
            X,
            y
        ),

        start=1

    ):


        print()
        print(
            f"Fold {fold}/{N_SPLITS}"
        )


        X_train = X.iloc[
            train_idx
        ].copy()


        X_test = X.iloc[
            test_idx
        ].copy()


        y_train = y.iloc[
            train_idx
        ].copy()


        y_test = y.iloc[
            test_idx
        ].copy()


        model.fit(
            X_train,
            y_train
        )


        y_pred = model.predict(
            X_test
        )


        y_prob = (
            model.predict_proba(
                X_test
            )[:, 1]
        )


        y_true_all.extend(
            y_test.tolist()
        )

        y_pred_all.extend(
            y_pred.tolist()
        )

        y_prob_all.extend(
            y_prob.tolist()
        )


        patient_ids_all.extend(
            data.iloc[
                test_idx
            ]["PatientID"]
            .tolist()
        )


        accuracy = accuracy_score(
            y_test,
            y_pred
        )


        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )


        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )


        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )


        auc = roc_auc_score(
            y_test,
            y_prob
        )


        fold_records.append({

            "Model":
                model_name,

            "Fold":
                fold,

            "Accuracy":
                accuracy,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1":
                f1,

            "ROC_AUC":
                auc

        })


        print(
            "Accuracy:",
            round(
                accuracy,
                4
            )
        )

        print(
            "Precision:",
            round(
                precision,
                4
            )
        )

        print(
            "Recall:",
            round(
                recall,
                4
            )
        )

        print(
            "F1:",
            round(
                f1,
                4
            )
        )

        print(
            "ROC-AUC:",
            round(
                auc,
                4
            )
        )


    # ========================================================
    # OUT-OF-FOLD RESULTS
    # ========================================================

    y_true_all = np.asarray(
        y_true_all
    )

    y_pred_all = np.asarray(
        y_pred_all
    )

    y_prob_all = np.asarray(
        y_prob_all
    )


    accuracy = accuracy_score(
        y_true_all,
        y_pred_all
    )


    precision = precision_score(
        y_true_all,
        y_pred_all,
        zero_division=0
    )


    recall = recall_score(
        y_true_all,
        y_pred_all,
        zero_division=0
    )


    f1 = f1_score(
        y_true_all,
        y_pred_all,
        zero_division=0
    )


    auc = roc_auc_score(
        y_true_all,
        y_prob_all
    )


    cm = confusion_matrix(
        y_true_all,
        y_pred_all
    )


    print()
    print("OUT-OF-FOLD RESULTS")

    print(
        "Accuracy:",
        round(
            accuracy,
            4
        )
    )

    print(
        "Precision:",
        round(
            precision,
            4
        )
    )

    print(
        "Recall:",
        round(
            recall,
            4
        )
    )

    print(
        "F1:",
        round(
            f1,
            4
        )
    )

    print(
        "ROC-AUC:",
        round(
            auc,
            4
        )
    )


    print()
    print("Confusion Matrix:")

    print(cm)


    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    predictions = pd.DataFrame({

        "PatientID":
            patient_ids_all,

        "Actual":
            y_true_all,

        "Predicted":
            y_pred_all,

        "Probability_Class1":
            y_prob_all

    })


    prediction_file = os.path.join(

        RESULTS_DIR,

        model_name
        + "_oof_predictions.csv"

    )


    predictions.to_csv(
        prediction_file,
        index=False
    )


    summary = {

        "Model":
            model_name,

        "Patients":
            len(y_true_all),

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1":
            f1,

        "ROC_AUC":
            auc,

        "TN":
            int(cm[0, 0]),

        "FP":
            int(cm[0, 1]),

        "FN":
            int(cm[1, 0]),

        "TP":
            int(cm[1, 1])

    }


    return (
        summary,
        fold_records
    )


# ============================================================
# PREPARE TARGET
# ============================================================

y = data[
    "risk_label"
]


# ============================================================
# CLINICAL-ONLY DATA
# ============================================================

X_clinical = data[
    clinical_features
].copy()


# ============================================================
# RADIOMICS-ONLY DATA
# ============================================================

X_radiomics = data[
    radiomic_features
].copy()


# ============================================================
# MODEL 1: CLINICAL ONLY
# ============================================================

clinical_model = Pipeline(

    steps=[

        (
            "preprocessor",

            clinical_preprocessor
        ),

        (
            "classifier",

            create_classifier()
        )

    ]

)


clinical_summary, clinical_folds = (
    evaluate_model(

        clinical_model,

        X_clinical,

        y,

        "clinical_only"

    )
)


# ============================================================
# MODEL 2: RADIOMICS ONLY
# ============================================================

radiomics_model = Pipeline(

    steps=[

        (
            "radiomics_preprocessing",

            radiomics_pipeline
        ),

        (
            "classifier",

            create_classifier()
        )

    ]

)


radiomics_summary, radiomics_folds = (
    evaluate_model(

        radiomics_model,

        X_radiomics,

        y,

        "radiomics_only"

    )
)


# ============================================================
# MODEL 3: COMBINED
# ============================================================

def evaluate_combined():

    print()
    print("=" * 75)
    print(
        "MODEL: combined_clinical_radiomics"
    )
    print("=" * 75)


    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    patient_ids_all = []


    fold_records = []


    for fold, (
        train_idx,
        test_idx
    ) in enumerate(

        cv.split(
            data,
            y
        ),

        start=1

    ):


        print()
        print(
            f"Fold {fold}/{N_SPLITS}"
        )


        train = data.iloc[
            train_idx
        ].copy()


        test = data.iloc[
            test_idx
        ].copy()


        y_train = train[
            "risk_label"
        ]


        y_test = test[
            "risk_label"
        ]


        # ----------------------------------------------------
        # CLINICAL
        # ----------------------------------------------------

        clinical_train = train[
            clinical_features
        ]


        clinical_test = test[
            clinical_features
        ]


        clinical_processor = (
            clinical_preprocessor
        )


        clinical_train_array = (
            clinical_processor.fit_transform(
                clinical_train
            )
        )


        clinical_test_array = (
            clinical_processor.transform(
                clinical_test
            )
        )


        # ----------------------------------------------------
        # RADIOMICS
        # ----------------------------------------------------

        radio_train = train[
            radiomic_features
        ]


        radio_test = test[
            radiomic_features
        ]


        # Imputation

        imputer = SimpleImputer(
            strategy="median"
        )


        radio_train_imp = (
            imputer.fit_transform(
                radio_train
            )
        )


        radio_test_imp = (
            imputer.transform(
                radio_test
            )
        )


        # Correlation filtering

        correlation_filter = (
            CorrelationFilter(
                threshold=0.95
            )
        )


        radio_train_filtered = (
            correlation_filter.fit_transform(
                radio_train_imp
            )
        )


        radio_test_filtered = (
            correlation_filter.transform(
                radio_test_imp
            )
        )


        # Scaling

        scaler = StandardScaler()


        radio_train_scaled = (
            scaler.fit_transform(
                radio_train_filtered
            )
        )


        radio_test_scaled = (
            scaler.transform(
                radio_test_filtered
            )
        )


        # Feature selection

        k = safe_k(
            radio_train_scaled.shape[1]
        )


        selector = SelectKBest(

            score_func=f_classif,

            k=k

        )


        radio_train_selected = (
            selector.fit_transform(
                radio_train_scaled,
                y_train
            )
        )


        radio_test_selected = (
            selector.transform(
                radio_test_scaled
            )
        )


        print(
            "Radiomics after correlation filter:",
            radio_train_filtered.shape[1]
        )


        print(
            "Radiomics selected:",
            radio_train_selected.shape[1]
        )


        # ----------------------------------------------------
        # COMBINE
        # ----------------------------------------------------

        X_train_combined = np.hstack(

            [

                clinical_train_array,

                radio_train_selected

            ]

        )


        X_test_combined = np.hstack(

            [

                clinical_test_array,

                radio_test_selected

            ]

        )


        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        classifier = (
            create_classifier()
        )


        classifier.fit(
            X_train_combined,
            y_train
        )


        y_pred = classifier.predict(
            X_test_combined
        )


        y_prob = (
            classifier.predict_proba(
                X_test_combined
            )[:, 1]
        )


        # ----------------------------------------------------
        # SAVE OOF
        # ----------------------------------------------------

        y_true_all.extend(
            y_test.tolist()
        )

        y_pred_all.extend(
            y_pred.tolist()
        )

        y_prob_all.extend(
            y_prob.tolist()
        )

        patient_ids_all.extend(
            test[
                "PatientID"
            ].tolist()
        )


        # ----------------------------------------------------
        # FOLD METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )


        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )


        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )


        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )


        auc = roc_auc_score(
            y_test,
            y_prob
        )


        fold_records.append({

            "Model":
                "combined",

            "Fold":
                fold,

            "Accuracy":
                accuracy,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1":
                f1,

            "ROC_AUC":
                auc

        })


        print(
            "Accuracy:",
            round(
                accuracy,
                4
            )
        )

        print(
            "Precision:",
            round(
                precision,
                4
            )
        )

        print(
            "Recall:",
            round(
                recall,
                4
            )
        )

        print(
            "F1:",
            round(
                f1,
                4
            )
        )

        print(
            "ROC-AUC:",
            round(
                auc,
                4
            )
        )


    # ========================================================
    # OVERALL
    # ========================================================

    y_true_all = np.asarray(
        y_true_all
    )

    y_pred_all = np.asarray(
        y_pred_all
    )

    y_prob_all = np.asarray(
        y_prob_all
    )


    accuracy = accuracy_score(
        y_true_all,
        y_pred_all
    )


    precision = precision_score(
        y_true_all,
        y_pred_all,
        zero_division=0
    )


    recall = recall_score(
        y_true_all,
        y_pred_all,
        zero_division=0
    )


    f1 = f1_score(
        y_true_all,
        y_pred_all,
        zero_division=0
    )


    auc = roc_auc_score(
        y_true_all,
        y_prob_all
    )


    cm = confusion_matrix(
        y_true_all,
        y_pred_all
    )


    print()
    print("OUT-OF-FOLD RESULTS")


    print(
        "Accuracy:",
        round(
            accuracy,
            4
        )
    )


    print(
        "Precision:",
        round(
            precision,
            4
        )
    )


    print(
        "Recall:",
        round(
            recall,
            4
        )
    )


    print(
        "F1:",
        round(
            f1,
            4
        )
    )


    print(
        "ROC-AUC:",
        round(
            auc,
            4
        )
    )


    print()
    print("Confusion Matrix:")

    print(cm)


    predictions = pd.DataFrame({

        "PatientID":
            patient_ids_all,

        "Actual":
            y_true_all,

        "Predicted":
            y_pred_all,

        "Probability_Class1":
            y_prob_all

    })


    predictions.to_csv(

        os.path.join(

            RESULTS_DIR,

            "combined_oof_predictions.csv"

        ),

        index=False

    )


    summary = {

        "Model":
            "combined",

        "Patients":
            len(y_true_all),

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1":
            f1,

        "ROC_AUC":
            auc,

        "TN":
            int(cm[0, 0]),

        "FP":
            int(cm[0, 1]),

        "FN":
            int(cm[1, 0]),

        "TP":
            int(cm[1, 1])

    }


    return (
        summary,
        fold_records
    )


combined_summary, combined_folds = (
    evaluate_combined()
)


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

summary_df = pd.DataFrame([

    clinical_summary,

    radiomics_summary,

    combined_summary

])


summary_file = os.path.join(

    RESULTS_DIR,

    "model_comparison_summary.csv"

)


summary_df.to_csv(

    summary_file,

    index=False

)


fold_df = pd.DataFrame(

    clinical_folds
    + radiomics_folds
    + combined_folds

)


fold_file = os.path.join(

    RESULTS_DIR,

    "model_comparison_fold_results.csv"

)


fold_df.to_csv(

    fold_file,

    index=False

)


# ============================================================
# FINAL TABLE
# ============================================================

print()
print("=" * 75)
print("FINAL MODEL COMPARISON")
print("=" * 75)

print()


print(

    summary_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC"
        ]
    ].to_string(
        index=False
    )

)


print()
print(
    "Summary:"
)

print(
    os.path.abspath(
        summary_file
    )
)


print()
print(
    "Fold results:"
)

print(
    os.path.abspath(
        fold_file
    )
)


print()
print("=" * 75)
print("MODEL COMPARISON COMPLETE")
print("=" * 75)