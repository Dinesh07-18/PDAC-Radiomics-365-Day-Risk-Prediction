import pandas as pd

# --------------------------------------------------
# 1. Load the original PDC clinical data
# --------------------------------------------------

input_file = "data/raw/PDC_study_clinical_10042026_224233.csv"

df = pd.read_csv(input_file, dtype=str)

# Convert blank strings to missing values
df = df.replace(r"^\s*$", pd.NA, regex=True)


# --------------------------------------------------
# 2. Keep only pancreatic ductal adenocarcinoma
# --------------------------------------------------

df = df[
    df["Disease Type"] == "Pancreatic Ductal Adenocarcinoma"
].copy()


# --------------------------------------------------
# 3. Convert important numeric columns
# --------------------------------------------------

df["Age at Diagnosis"] = pd.to_numeric(
    df["Age at Diagnosis"],
    errors="coerce"
)

df["Days to Death"] = pd.to_numeric(
    df["Days to Death"],
    errors="coerce"
)

df["Days to Last Follow Up"] = pd.to_numeric(
    df["Days to Last Follow Up"],
    errors="coerce"
)


# --------------------------------------------------
# 4. Create survival time
#
# For a patient who died:
#       survival time = Days to Death
#
# For a patient who is alive:
#       survival time = Days to Last Follow Up
# --------------------------------------------------

df["survival_time_days"] = (
    df["Days to Death"]
    .fillna(df["Days to Last Follow Up"])
)


# --------------------------------------------------
# 5. Create event variable
#
# 1 = death occurred
# 0 = censored/alive
# --------------------------------------------------

df["event"] = df["Vital Status"].map({
    "Dead": 1,
    "Alive": 0
})


# --------------------------------------------------
# 6. Keep only patients with known survival information
# --------------------------------------------------

df = df[
    df["event"].notna()
    & df["survival_time_days"].notna()
].copy()


# --------------------------------------------------
# 7. Convert age from days to years
# --------------------------------------------------

df["age_years"] = df["Age at Diagnosis"] / 365.25


# --------------------------------------------------
# 8. Select the columns we need
# --------------------------------------------------

clean_df = df[
    [
        "Case Submitter ID",
        "Sex",
        "age_years",
        "Vital Status",
        "survival_time_days",
        "event",
        "Days to Recurrence",
        "Progression or Recurrence",
        "Primary Diagnosis",
        "Tumor Grade",
        "Tumor Stage"
    ]
].copy()


# --------------------------------------------------
# 9. Save the cleaned dataset
# --------------------------------------------------

output_file = "data/processed/clinical_survival.csv"

clean_df.to_csv(output_file, index=False)

print("Clean dataset created!")
print("Number of patients:", len(clean_df))
print("\nVital status:")
print(clean_df["Vital Status"].value_counts())

print("\nFirst 5 rows:")
print(clean_df.head())