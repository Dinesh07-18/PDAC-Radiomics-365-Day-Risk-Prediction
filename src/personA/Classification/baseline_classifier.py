import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load classification dataset
df = pd.read_csv("data/processed/classification_365days.csv")

# Features and target
X = df[["age_years", "Sex", "Tumor Grade"]]
y = df["risk_label"]

# Same train/test split used in our other models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Majority-class baseline
baseline = DummyClassifier(
    strategy="most_frequent"
)

baseline.fit(X_train, y_train)

# Predictions
y_pred = baseline.predict(X_test)

# Results
accuracy = accuracy_score(y_test, y_pred)

print("Majority-Class Baseline")
print("=======================")

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

print("\nAccuracy:", accuracy)
print("Accuracy (%):", round(accuracy * 100, 2))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))