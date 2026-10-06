import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from imblearn.over_sampling import SMOTENC


# Load dataset
df = pd.read_csv("data/processed/classification_365days.csv")

X = df[["age_years", "Sex", "Tumor Grade"]].copy()
y = df["risk_label"]


# Same split as before
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# One-hot encode categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical",
         OneHotEncoder(handle_unknown="ignore"),
         ["Sex", "Tumor Grade"]),

        ("numeric",
         StandardScaler(),
         ["age_years"])
    ]
)


# Logistic Regression pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        random_state=42,
        max_iter=1000
    ))
])


# Train
model.fit(X_train, y_train)


# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# Results
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)


print("Logistic Regression with One-Hot Encoding")
print("=" * 50)

print("\nAccuracy:", accuracy)
print("ROC-AUC:", auc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
