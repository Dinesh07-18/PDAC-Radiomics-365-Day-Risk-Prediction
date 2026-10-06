import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTENC


# --------------------------------------------------
# 1. Load classification dataset
# --------------------------------------------------

df = pd.read_csv("data/processed/classification_365days.csv")


# --------------------------------------------------
# 2. Separate features and target
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
# 5. Apply SMOTENC ONLY to training data
# --------------------------------------------------

categorical_features = [1, 2]

smote = SMOTENC(
    categorical_features=categorical_features,
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

print("Before SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

print("\nTraining shape before SMOTE:")
print(X_train.shape)

print("\nTraining shape after SMOTE:")
print(X_train_smote.shape)

print("\nTesting shape:")
print(X_test.shape)

print("\nFirst 5 training rows after SMOTE:")
print(X_train_smote.head())