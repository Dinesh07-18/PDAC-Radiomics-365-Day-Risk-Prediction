import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTENC


# Load data
df = pd.read_csv("data/processed/classification_365days.csv")

X = df[["age_years", "Sex", "Tumor Grade"]].copy()
y = df["risk_label"]


# Same train/test split used throughout
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


# SMOTENC on training data only
smote = SMOTENC(
    categorical_features=[1, 2],
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


# Naive Bayes
nb = GaussianNB()
nb.fit(X_train_smote, y_train_smote)

nb_prob = nb.predict_proba(X_test)[:, 1]


# Logistic Regression
lr = LogisticRegression(
    random_state=42,
    max_iter=1000
)

lr.fit(X_train_smote, y_train_smote)

lr_prob = lr.predict_proba(X_test)[:, 1]


# ROC curves
nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_prob)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)

nb_auc = roc_auc_score(y_test, nb_prob)
lr_auc = roc_auc_score(y_test, lr_prob)


# Plot
plt.figure(figsize=(7, 6))

plt.plot(
    nb_fpr,
    nb_tpr,
    label=f"Naive Bayes (AUC = {nb_auc:.3f})"
)

plt.plot(
    lr_fpr,
    lr_tpr,
    label=f"Logistic Regression (AUC = {lr_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Person A Classification Models")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/personA_roc_curves.png",
    dpi=300
)

plt.show()

print("Naive Bayes ROC-AUC:", nb_auc)
print("Logistic Regression ROC-AUC:", lr_auc)
print("ROC curve saved to results/personA_roc_curves.png")