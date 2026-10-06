import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTENC


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv(
    "data/processed/classification_365days.csv"
)


# --------------------------------------------------
# 2. Select features and target
# --------------------------------------------------

X = df[
    [
        "age_years",
        "Sex",
        "Tumor Grade"
    ]
].copy()

y = df["risk_label"]


# --------------------------------------------------
# 3. Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 4. Encode categorical features
# --------------------------------------------------

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


# --------------------------------------------------
# 5. Apply SMOTENC to training data ONLY
# --------------------------------------------------

smote = SMOTENC(
    categorical_features=[1, 2],
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


# --------------------------------------------------
# 6. Train Gaussian Naive Bayes
# --------------------------------------------------

model = GaussianNB()

model.fit(
    X_train_smote,
    y_train_smote
)


# --------------------------------------------------
# 7. Predict on untouched test set
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 8. Evaluate
# --------------------------------------------------

print("Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))