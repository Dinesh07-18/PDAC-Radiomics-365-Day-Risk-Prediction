import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("data/processed/clinical_survival.csv")

# --------------------------------------------------
# 1. Basic information
# --------------------------------------------------

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

# --------------------------------------------------
# 2. Vital status
# --------------------------------------------------

print("\nVital Status:")
print(df["Vital Status"].value_counts())

# --------------------------------------------------
# 3. Sex
# --------------------------------------------------

print("\nSex:")
print(df["Sex"].value_counts())

# --------------------------------------------------
# 4. Age statistics
# --------------------------------------------------

print("\nAge statistics:")
print(df["age_years"].describe())

# --------------------------------------------------
# 5. Survival-time statistics
# --------------------------------------------------

print("\nSurvival time statistics (days):")
print(df["survival_time_days"].describe())

# --------------------------------------------------
# 6. Event statistics
# --------------------------------------------------

print("\nEvent:")
print(df["event"].value_counts())

# --------------------------------------------------
# 7. Plot 1: Vital Status
# --------------------------------------------------

df["Vital Status"].value_counts().plot(kind="bar")

plt.title("Vital Status of Patients")
plt.xlabel("Vital Status")
plt.ylabel("Number of Patients")
plt.tight_layout()

plt.savefig("results_vital_status.png")
plt.show()

# --------------------------------------------------
# 8. Plot 2: Age distribution
# --------------------------------------------------

df["age_years"].plot(kind="hist", bins=15)

plt.title("Age Distribution")
plt.xlabel("Age (years)")
plt.ylabel("Number of Patients")
plt.tight_layout()

plt.savefig("results_age_distribution.png")
plt.show()

# --------------------------------------------------
# 9. Plot 3: Survival-time distribution
# --------------------------------------------------

df["survival_time_days"].plot(kind="hist", bins=20)

plt.title("Survival Time Distribution")
plt.xlabel("Survival Time (days)")
plt.ylabel("Number of Patients")
plt.tight_layout()

plt.savefig("results_survival_distribution.png")
plt.show()