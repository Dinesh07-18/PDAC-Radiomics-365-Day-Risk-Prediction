import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\DELL\Downloads\ml_mini_project\ml_mini_project")

metadata_file = (
    ROOT / "data" / "radiomics" / "metadata"
    / "Metadata_Report_CPTAC_PDA_2025_10_20-1.csv"
)

cohort_file = (
    ROOT / "data" / "radiomics" / "metadata"
    / "personB_radiomics_candidate_cohort.csv"
)

output_file = (
    ROOT / "data" / "radiomics" / "metadata"
    / "personB_rtstruct_download_manifest.csv"
)

metadata = pd.read_csv(metadata_file)
cohort = pd.read_csv(cohort_file)

# Patients for whom we have CT candidates
ct_patients = set(
    cohort.loc[cohort["CT_candidate"] == True, "PatientID"]
)

# Select the relevant pre-dose tumor segmentations
rtstruct = metadata[
    metadata["PatientID"].isin(ct_patients)
    & (metadata["Annotation Type"].astype(str).str.lower() == "segmentation")
    & (metadata["ClinicalTrialTimePointID"].astype(str).str.lower() == "pre-dose")
    & (metadata["StructureSetLabel"].isin(["PANCREAS1", "PANCREAS2"]))
    & (metadata["ReferencedSeriesModality"].astype(str).str.upper() == "CT")
].copy()

# One row per RTSTRUCT series
rtstruct = rtstruct.drop_duplicates(subset=["SeriesInstanceUID"])

columns = [
    "PatientID",
    "ClinicalTrialTimePointID",
    "SeriesInstanceUID",
    "StructureSetLabel",
    "ROIVolume",
    "Annotation Type",
    "ReferencedSeriesInstanceUID",
    "ReferencedSeriesDescription",
    "ReferencedSeriesModality"
]

rtstruct[columns].to_csv(output_file, index=False)

print()
print("=== RTSTRUCT MANIFEST CREATED ===")
print("Patients:", rtstruct["PatientID"].nunique())
print("RTSTRUCT series:", len(rtstruct))
print("Output:", output_file)
print()

print("Patients:")
for patient in sorted(rtstruct["PatientID"].unique()):
    print(patient)