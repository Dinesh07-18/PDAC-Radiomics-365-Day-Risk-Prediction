import os
import warnings
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

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
print("=" * 80)
print("PDAC 365-DAY RISK MODEL COMPARISON - EXPERIMENT 2")
print("=" * 80)


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
# NORMALIZE IDS
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

clinical_required = [
    "Case Submitter ID",
    "age_years",
    "Sex",
    "Tumor Grade",
    "risk_label"
]


missing = [
    c
    for c in clinical_required
    if c not in clinical.columns
]


if missing:

    raise RuntimeError(
        "Missing clinical columns: "
        + str(missing)
    )


clinical_data = clinical[
    clinical_required
].copy()


clinical_data = clinical_data.rename(
    columns={
        "Case Submitter ID": "PatientID",
        "age_years": "age_years_clinical",
        "Sex": "Sex_clinical",
        "Tumor Grade": "Tumor Grade_clinical",
        "risk_label": "risk_label_clinical"
    }
)


# ============================================================
# MERGE
# ============================================================

data = radiomics.merge(
    clinical_data,
    on="PatientID",
    how="inner"
)


print()
print("=" * 80)
print("MERGED DATASET")
print("=" * 80)

print(
    "Patients:",
    data["PatientID"].nunique()
)

print(
    "Rows:",
    len(data)
)


# ============================================================
# TARGET
# ============================================================

data["risk_label"] = (
    pd.to_numeric(
        data["risk_label_clinical"],
        errors="coerce"
    )
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

duplicates = (
    data["PatientID"]
    .duplicated()
    .sum()
)


print(
    "Duplicate patients:",
    duplicates
)


if duplicates != 0:

    raise RuntimeError(
        "Duplicate patient IDs detected."
    )


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print()
print("Risk distribution:")

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


print()
print("Clinical features:")

for c in clinical_features:
    print(" ", c)


# ============================================================
# RADIOMIC FEATURES
# ============================================================

metadata_columns = {

    "PatientID",

    "risk_label",

    "risk_label_clinical",

    "age_years",

    "Sex",

    "Tumor Grade",

    "StructureSetLabel",

    "ROI_role",

    "ROIVolume",

    "normalized_phase",

    "ReferencedSeriesDescription"

}


radiomic_features = [

    c

    for c in radiomics.columns

    if c not in metadata_columns

]


radiomic_features = [

    c

    for c in radiomic_features

    if pd.api.types.is_numeric_dtype(
        data[c]
    )

]


print()
print(
    "Radiomic features:",
    len(radiomic_features)
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

        X_df = pd.DataFrame(X)

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

        X_array = np.asarray(X)

        return X_array[
            :,
            self.keep_indices_
        ]


# ============================================================
# FEATURE COUNT
# ============================================================

def get_k(
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
# CLASSIFIERS
# ============================================================

def make_models():

    return {

        "LogisticRegression":

            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="liblinear",
                random_state=42
            ),

        "SVM":

            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=42
            ),

        "RandomForest":

            RandomForestClassifier(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )

    }


# ============================================================
# CLINICAL PREPROCESSOR
# ============================================================

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

            [
                "age_years_clinical"
            ]

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

            [
                "Sex_clinical",
                "Tumor Grade_clinical"
            ]

        )

    ],

    remainder="drop"

)


# ============================================================
# RADIOMICS PREPROCESSOR
# ============================================================

def make_radiomics_pipeline():

    return Pipeline(

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
                    k=get_k(
                        len(
                            radiomic_features
                        )
                    )
                )

            )

        ]

    )


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)


# ============================================================
# GENERAL MODEL EVALUATION
# ============================================================

def evaluate_standard_model(

    X,
    y,
    model,
    model_name,
    feature_set

):

    print()
    print("-" * 80)

    print(
        "MODEL:",
        model_name,
        "| FEATURE SET:",
        feature_set
    )

    print("-" * 80)


    y_true_all = []

    y_pred_all = []

    y_prob_all = []

    patient_ids_all = []

    fold_results = []


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


        fold_results.append({

            "Classifier":
                model_name,

            "Feature_Set":
                feature_set,

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


        print()
        print(
            "Fold",
            fold,
            "| AUC:",
            round(
                auc,
                4
            )
        )


    # ========================================================
    # OUT-OF-FOLD METRICS
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
    print(
        "OUT-OF-FOLD:"
    )

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
    print(
        "Confusion Matrix:"
    )

    print(cm)


    # ========================================================
    # SAVE OOF PREDICTIONS
    # ========================================================

    safe_name = (
        model_name
        + "_"
        + feature_set
    )


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

            safe_name
            + "_oof_predictions.csv"

        ),

        index=False

    )


    summary = {

        "Classifier":
            model_name,

        "Feature_Set":
            feature_set,

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
        fold_results
    )


# ============================================================
# DATA
# ============================================================

y = data[
    "risk_label"
]


X_clinical = data[
    clinical_features
].copy()


X_radiomics = data[
    radiomic_features
].copy()


# ============================================================
# RUN CLINICAL + RADIOMICS MODELS
# ============================================================

all_summaries = []

all_folds = []


models = make_models()


# ------------------------------------------------------------
# CLINICAL ONLY
# ------------------------------------------------------------

for model_name, classifier in models.items():

    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",

                clinical_preprocessor
            ),

            (
                "classifier",

                classifier
            )

        ]

    )


    summary, folds = evaluate_standard_model(

        X_clinical,

        y,

        pipeline,

        model_name,

        "clinical_only"

    )


    all_summaries.append(
        summary
    )

    all_folds.extend(
        folds
    )


# ------------------------------------------------------------
# RADIOMICS ONLY
# ------------------------------------------------------------

for model_name, classifier in models.items():

    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",

                make_radiomics_pipeline()
            ),

            (
                "classifier",

                classifier
            )

        ]

    )


    summary, folds = evaluate_standard_model(

        X_radiomics,

        y,

        pipeline,

        model_name,

        "radiomics_only"

    )


    all_summaries.append(
        summary
    )

    all_folds.extend(
        folds
    )


# ============================================================
# COMBINED MODEL
# ============================================================

def evaluate_combined(

    model_name,
    classifier

):

    print()
    print("=" * 80)

    print(
        "MODEL:",
        model_name,
        "| FEATURE SET: combined"
    )

    print("=" * 80)


    y_true_all = []

    y_pred_all = []

    y_prob_all = []

    patient_ids_all = []

    fold_results = []


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

        clinical_processor = (
            clinical_preprocessor
        )


        clinical_train = (
            clinical_processor
            .fit_transform(
                train[
                    clinical_features
                ]
            )
        )


        clinical_test = (
            clinical_processor
            .transform(
                test[
                    clinical_features
                ]
            )
        )


        # ----------------------------------------------------
        # RADIOMICS
        # ----------------------------------------------------

        imputer = SimpleImputer(
            strategy="median"
        )


        radio_train = (
            imputer.fit_transform(
                train[
                    radiomic_features
                ]
            )
        )


        radio_test = (
            imputer.transform(
                test[
                    radiomic_features
                ]
            )
        )


        # ----------------------------------------------------
        # CORRELATION FILTER
        # ----------------------------------------------------

        corr_filter = (
            CorrelationFilter(
                threshold=0.95
            )
        )


        radio_train = (
            corr_filter.fit_transform(
                radio_train
            )
        )


        radio_test = (
            corr_filter.transform(
                radio_test
            )
        )


        # ----------------------------------------------------
        # SCALING
        # ----------------------------------------------------

        scaler = StandardScaler()


        radio_train = (
            scaler.fit_transform(
                radio_train
            )
        )


        radio_test = (
            scaler.transform(
                radio_test
            )
        )


        # ----------------------------------------------------
        # FEATURE SELECTION
        # ----------------------------------------------------

        k = get_k(
            radio_train.shape[1]
        )


        selector = SelectKBest(

            score_func=f_classif,

            k=k

        )


        radio_train = (
            selector.fit_transform(
                radio_train,
                y_train
            )
        )


        radio_test = (
            selector.transform(
                radio_test
            )
        )


        # ----------------------------------------------------
        # COMBINE
        # ----------------------------------------------------

        X_train = np.hstack(

            [
                clinical_train,
                radio_train
            ]

        )


        X_test = np.hstack(

            [
                clinical_test,
                radio_test
            ]

        )


        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        classifier.fit(
            X_train,
            y_train
        )


        y_pred = classifier.predict(
            X_test
        )


        y_prob = (
            classifier.predict_proba(
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
            test[
                "PatientID"
            ].tolist()
        )


        # ----------------------------------------------------
        # FOLD METRICS
        # ----------------------------------------------------

        fold_accuracy = accuracy_score(
            y_test,
            y_pred
        )

        fold_precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        fold_recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        fold_f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        fold_auc = roc_auc_score(
            y_test,
            y_prob
        )


        fold_results.append({

            "Classifier":
                model_name,

            "Feature_Set":
                "combined",

            "Fold":
                fold,

            "Accuracy":
                fold_accuracy,

            "Precision":
                fold_precision,

            "Recall":
                fold_recall,

            "F1":
                fold_f1,

            "ROC_AUC":
                fold_auc

        })


        print(
            "Fold",
            fold,
            "| AUC:",
            round(
                fold_auc,
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
    print(
        "OUT-OF-FOLD:"
    )

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
    print(
        "Confusion Matrix:"
    )

    print(cm)


    safe_name = (
        model_name
        + "_combined"
    )


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

            safe_name
            + "_oof_predictions.csv"

        ),

        index=False

    )


    summary = {

        "Classifier":
            model_name,

        "Feature_Set":
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
        fold_results
    )


# ============================================================
# RUN COMBINED
# ============================================================

for model_name, classifier in models.items():

    summary, folds = evaluate_combined(

        model_name,

        classifier

    )


    all_summaries.append(
        summary
    )

    all_folds.extend(
        folds
    )


# ============================================================
# SAVE RESULTS
# ============================================================

summary_df = pd.DataFrame(
    all_summaries
)


fold_df = pd.DataFrame(
    all_folds
)


summary_file = os.path.join(

    RESULTS_DIR,

    "model_comparison_v2_summary.csv"

)


fold_file = os.path.join(

    RESULTS_DIR,

    "model_comparison_v2_fold_results.csv"

)


summary_df.to_csv(
    summary_file,
    index=False
)


fold_df.to_csv(
    fold_file,
    index=False
)


# ============================================================
# FINAL TABLE
# ============================================================

print()
print("=" * 80)
print("FINAL EXPERIMENT 2 COMPARISON")
print("=" * 80)

print()


display_columns = [

    "Classifier",

    "Feature_Set",

    "Accuracy",

    "Precision",

    "Recall",

    "F1",

    "ROC_AUC"

]


print(

    summary_df[
        display_columns
    ].sort_values(
        "ROC_AUC",
        ascending=False
    ).to_string(
        index=False
    )

)


print()
print(
    "Summary file:"
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
print("=" * 80)
print("EXPERIMENT 2 COMPLETE")
print("=" * 80)