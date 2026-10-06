import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTENC


# Load classification dataset
df = pd.read_csv("data/processed/classification_365days.csv")

X = df[["age_years", "Sex", "Tumor Grade"]].copy()
y = df["risk_label"]


# Same train/test split used throughout Person A's experiments
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
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


# Calculate permutation importance on untouched test data
result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="accuracy",
    n_repeats=20,
    random_state=42
)


# Create results table
importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance_Mean": result.importances_mean,
    "Importance_STD": result.importances_std
})

importance_df = importance_df.sort_values(
    "Importance_Mean",
    ascending=False
)

print("Permutation Feature Importance:")
print(importance_df)


# Save results
importance_df.to_csv(
    "results/personA_feature_importance.csv",
    index=False
)


# Plot
plt.figure(figsize=(7, 5))

plt.bar(
    importance_df["Feature"],
    importance_df["Importance_Mean"]
)

plt.xlabel("Feature")
plt.ylabel("Mean Accuracy Decrease")
plt.title("Naive Bayes Feature Importance")

plt.tight_layout()

plt.savefig(
    "results/personA_feature_importance.png",
    dpi=300
)

plt.show()