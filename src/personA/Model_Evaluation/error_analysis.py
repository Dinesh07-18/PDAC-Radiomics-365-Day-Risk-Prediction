import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTENC

# Load dataset
df = pd.read_csv("data/processed/classification_365days.csv")

# Features and target
X = df[["age_years", "Sex", "Tumor Grade"]].copy()
y = df["risk_label"]

# Keep patient IDs for error analysis
patient_ids = df["Case Submitter ID"]

# Same split used in our original model
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X,
    y,
    patient_ids,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Encode categorical features
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

# Apply SMOTENC only to training data
smote = SMOTENC(
    categorical_features=[1, 2],
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

# Train Naive Bayes
model = GaussianNB()
model.fit(X_train_smote, y_train_smote)

# Predict test set
y_pred = model.predict(X_test)

# Create error-analysis table
results = X_test.copy()
results["Patient_ID"] = id_test.values
results["Actual"] = y_test.values
results["Predicted"] = y_pred

# Identify incorrect predictions
errors = results[results["Actual"] != results["Predicted"]].copy()

# Add error type
errors["Error_Type"] = errors.apply(
    lambda row:
        "False Positive"
        if row["Actual"] == 0 and row["Predicted"] == 1
        else "False Negative",
    axis=1
)

# Arrange columns
errors = errors[
    [
        "Patient_ID",
        "age_years",
        "Sex",
        "Tumor Grade",
        "Actual",
        "Predicted",
        "Error_Type"
    ]
]

print("Naive Bayes + SMOTENC Error Analysis")
print("=====================================")

print("\nTotal test patients:", len(results))
print("Correct predictions:", (results["Actual"] == results["Predicted"]).sum())
print("Incorrect predictions:", len(errors))

print("\nError types:")
print(errors["Error_Type"].value_counts())

print("\nIncorrect predictions:")
print(errors.to_string(index=False))

# Save results
errors.to_csv(
    "results/personA_error_analysis.csv",
    index=False
)

print("\nSaved to:")
print("results/personA_error_analysis.csv")