import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from imblearn.over_sampling import SMOTENC


# -----------------------------------------
# Load classification dataset
# -----------------------------------------

df = pd.read_csv("data/processed/classification_365days.csv")

X = df[["age_years", "Sex", "Tumor Grade"]].copy()
y = df["risk_label"].copy()


# -----------------------------------------
# 5-fold stratified cross-validation
# -----------------------------------------

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


nb_accuracy = []
nb_f1 = []
nb_auc = []

lr_accuracy = []
lr_f1 = []
lr_auc = []


# -----------------------------------------
# Run each fold
# -----------------------------------------

for fold, (train_index, test_index) in enumerate(
    skf.split(X, y), start=1
):

    X_train = X.iloc[train_index].copy()
    X_test = X.iloc[test_index].copy()

    y_train = y.iloc[train_index].copy()
    y_test = y.iloc[test_index].copy()


    # -------------------------------------
    # Encode categorical features
    # -------------------------------------

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    X_train[["Sex", "Tumor Grade"]] = encoder.fit_transform(
        X_train[["Sex", "Tumor Grade"]]
    )

    X_test[["Sex", "Tumor Grade"]] = encoder.transform(
        X_test[["Sex", "Tumor Grade"]]
    )


    # -------------------------------------
    # SMOTENC ONLY on training fold
    # -------------------------------------

    smote = SMOTENC(
        categorical_features=[1, 2],
        random_state=42
    )

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train
    )


    # -------------------------------------
    # Naive Bayes
    # -------------------------------------

    nb = GaussianNB()

    nb.fit(
        X_train_smote,
        y_train_smote
    )

    nb_pred = nb.predict(X_test)
    nb_prob = nb.predict_proba(X_test)[:, 1]

    nb_accuracy.append(
        accuracy_score(y_test, nb_pred)
    )

    nb_f1.append(
        f1_score(y_test, nb_pred)
    )

    nb_auc.append(
        roc_auc_score(y_test, nb_prob)
    )


    # -------------------------------------
    # Logistic Regression
    # -------------------------------------

    lr = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    lr.fit(
        X_train_smote,
        y_train_smote
    )

    lr_pred = lr.predict(X_test)
    lr_prob = lr.predict_proba(X_test)[:, 1]

    lr_accuracy.append(
        accuracy_score(y_test, lr_pred)
    )

    lr_f1.append(
        f1_score(y_test, lr_pred)
    )

    lr_auc.append(
        roc_auc_score(y_test, lr_prob)
    )


    print(f"\nFold {fold}")
    print("-" * 30)
    print("Naive Bayes:")
    print("  Accuracy:", round(nb_accuracy[-1], 4))
    print("  F1:", round(nb_f1[-1], 4))
    print("  ROC-AUC:", round(nb_auc[-1], 4))

    print("Logistic Regression:")
    print("  Accuracy:", round(lr_accuracy[-1], 4))
    print("  F1:", round(lr_f1[-1], 4))
    print("  ROC-AUC:", round(lr_auc[-1], 4))


# -----------------------------------------
# Calculate mean and standard deviation
# -----------------------------------------

print("\n")
print("=" * 50)
print("5-FOLD CROSS-VALIDATION RESULTS")
print("=" * 50)

print("\nNaive Bayes + SMOTENC")
print(
    "Accuracy:",
    f"{np.mean(nb_accuracy):.4f} +/- {np.std(nb_accuracy):.4f}"
)
print(
    "F1:",
    f"{np.mean(nb_f1):.4f} +/- {np.std(nb_f1):.4f}"
)
print(
    "ROC-AUC:",
    f"{np.mean(nb_auc):.4f} +/- {np.std(nb_auc):.4f}"
)


print("\nLogistic Regression + SMOTENC")
print(
    "Accuracy:",
    f"{np.mean(lr_accuracy):.4f} +/- {np.std(lr_accuracy):.4f}"
)
print(
    "F1:",
    f"{np.mean(lr_f1):.4f} +/- {np.std(lr_f1):.4f}"
)
print(
    "ROC-AUC:",
    f"{np.mean(lr_auc):.4f} +/- {np.std(lr_auc):.4f}"
)