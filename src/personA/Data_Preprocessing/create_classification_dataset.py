import pandas as pd

# Load cleaned clinical dataset
df = pd.read_csv("data/processed/clinical_survival.csv")

# Remove the patient with zero recorded follow-up
df = df[df["survival_time_days"] > 0].copy()

# 365-day classification threshold
threshold = 365

# Keep only patients whose outcome is known at the 365-day point:
# 1. Death before 365 days -> high risk
# 2. Follow-up/survival at least 365 days -> low risk
df = df[
    (df["survival_time_days"] >= threshold)
    | ((df["event"] == 1) & (df["survival_time_days"] < threshold))
].copy()

# Create target
# 0 = survived at least 365 days
# 1 = died before 365 days
df["risk_label"] = (
    (df["event"] == 1)
    & (df["survival_time_days"] < threshold)
).astype(int)

# Keep only features needed for classification
classification_df = df[
    [
        "Case Submitter ID",
        "age_years",
        "Sex",
        "Tumor Grade",
        "risk_label"
    ]
].copy()

# Save
output_file = "data/processed/classification_365days.csv"
classification_df.to_csv(output_file, index=False)

print("Classification dataset created!")
print("Shape:", classification_df.shape)

print("\nRisk labels:")
print(classification_df["risk_label"].value_counts())

print("\nColumns:")
print(classification_df.columns.tolist())

print("\nFirst 5 rows:")
print(classification_df.head())