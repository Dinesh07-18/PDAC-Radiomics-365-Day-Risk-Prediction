import os
import re
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT = os.path.abspath(".")

CT_MANIFEST = os.path.join(
    PROJECT,
    "data",
    "radiomics",
    "metadata",
    "personB_ct_download_manifest.csv"
)

MASK_AUDIT = os.path.join(
    PROJECT,
    "data",
    "radiomics",
    "metadata",
    "mask_generation_audit.csv"
)

OUTPUT = os.path.join(
    PROJECT,
    "data",
    "radiomics",
    "metadata",
    "final_patient_radiomics_manifest.csv"
)


# ============================================================
# CHECK FILES
# ============================================================

print()
print("=" * 70)
print("PATIENT-LEVEL RADIOMICS COHORT SELECTION")
print("=" * 70)

if not os.path.exists(CT_MANIFEST):

    raise FileNotFoundError(
        "CT manifest not found:\n"
        + CT_MANIFEST
    )

if not os.path.exists(MASK_AUDIT):

    raise FileNotFoundError(
        "Mask audit file not found:\n"
        + MASK_AUDIT
    )


# ============================================================
# LOAD DATA
# ============================================================

print()
print("Loading CT manifest...")

ct_df = pd.read_csv(
    CT_MANIFEST
)

print(
    "CT manifest rows:",
    len(ct_df)
)


print()
print("Loading mask audit...")

mask_df = pd.read_csv(
    MASK_AUDIT
)

print(
    "Mask audit rows:",
    len(mask_df)
)


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

ct_df.columns = [
    str(c).strip()
    for c in ct_df.columns
]

mask_df.columns = [
    str(c).strip()
    for c in mask_df.columns
]


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_ct_columns = [

    "PatientID",

    "risk_label",

    "age_years",

    "Sex",

    "Tumor Grade",

    "ClinicalTrialTimePointID",

    "StructureSetLabel",

    "ROI_role",

    "ROIVolume",

    "ReferencedSeriesInstanceUID",

    "ReferencedSeriesModality",

    "ReferencedSeriesDescription",

    "phase_hint"

]


required_mask_columns = [

    "PatientID",

    "RTSTRUCT_SeriesInstanceUID",

    "Referenced_CT_SeriesInstanceUID",

    "ROI_Name",

    "Mask_File",

    "Status"

]


missing_ct = [
    c
    for c in required_ct_columns
    if c not in ct_df.columns
]


missing_mask = [
    c
    for c in required_mask_columns
    if c not in mask_df.columns
]


if missing_ct:

    raise RuntimeError(
        "Missing CT manifest columns:\n"
        + "\n".join(missing_ct)
    )


if missing_mask:

    raise RuntimeError(
        "Missing mask audit columns:\n"
        + "\n".join(missing_mask)
    )


# ============================================================
# KEEP ONLY SUCCESSFULLY GENERATED MASKS
# ============================================================

mask_df = mask_df[
    mask_df["Status"]
    .astype(str)
    .str.startswith("SUCCESS")
].copy()


print()
print(
    "Successful mask records:",
    len(mask_df)
)


# ============================================================
# NORMALIZE UIDs
# ============================================================

ct_df[
    "ReferencedSeriesInstanceUID"
] = (
    ct_df[
        "ReferencedSeriesInstanceUID"
    ]
    .astype(str)
    .str.strip()
)


mask_df[
    "Referenced_CT_SeriesInstanceUID"
] = (
    mask_df[
        "Referenced_CT_SeriesInstanceUID"
    ]
    .astype(str)
    .str.strip()
)


ct_df[
    "SeriesInstanceUID"
] = (
    ct_df[
        "SeriesInstanceUID"
    ]
    .astype(str)
    .str.strip()
)


mask_df[
    "RTSTRUCT_SeriesInstanceUID"
] = (
    mask_df[
        "RTSTRUCT_SeriesInstanceUID"
    ]
    .astype(str)
    .str.strip()
)


# ============================================================
# CONVERT ROI VOLUME TO NUMERIC
# ============================================================

ct_df["ROIVolume"] = pd.to_numeric(
    ct_df["ROIVolume"],
    errors="coerce"
)


# ============================================================
# NORMALIZE PHASE
# ============================================================

def normalize_phase(value):

    if pd.isna(value):

        return "unspecified"

    value = str(value).strip().lower()

    if value == "":

        return "unspecified"

    if "venous" in value:

        return "venous"

    if "arterial" in value:

        return "arterial"

    if (
        "contrast" in value
        or "+k" in value
        or "+c" in value
    ):

        return "contrast_unspecified"

    return "unspecified"


ct_df["normalized_phase"] = (
    ct_df["phase_hint"]
    .apply(normalize_phase)
)


# ============================================================
# PHASE PRIORITY
# ============================================================

phase_priority = {

    "venous": 1,

    "arterial": 2,

    "contrast_unspecified": 3,

    "unspecified": 4

}


ct_df["phase_priority"] = (
    ct_df["normalized_phase"]
    .map(phase_priority)
    .fillna(4)
    .astype(int)
)


# ============================================================
# REMOVE NON-CT RECORDS
# ============================================================

ct_df = ct_df[
    ct_df[
        "ReferencedSeriesModality"
    ]
    .astype(str)
    .str.upper()
    .eq("CT")
].copy()


# ============================================================
# KEEP ONLY PRE-DOSE
# ============================================================

ct_df = ct_df[
    ct_df[
        "ClinicalTrialTimePointID"
    ]
    .astype(str)
    .str.lower()
    .eq("pre-dose")
].copy()


# ============================================================
# MERGE MASK INFORMATION
# ============================================================

print()
print("Matching CT records with generated masks...")


merged = ct_df.merge(

    mask_df[

        [
            "PatientID",

            "RTSTRUCT_SeriesInstanceUID",

            "Referenced_CT_SeriesInstanceUID",

            "ROI_Name",

            "Mask_File",

            "Status"

        ]

    ],

    left_on=[
        "PatientID",
        "SeriesInstanceUID",
        "ReferencedSeriesInstanceUID"
    ],

    right_on=[
        "PatientID",
        "RTSTRUCT_SeriesInstanceUID",
        "Referenced_CT_SeriesInstanceUID"
    ],

    how="inner"

)


print(
    "Matched CT/mask rows:",
    len(merged)
)


# ============================================================
# CHECK PATIENT COUNT
# ============================================================

unique_patients = (
    merged["PatientID"]
    .nunique()
)


print(
    "Unique patients with CT + mask:",
    unique_patients
)


# ============================================================
# SELECT ONE CT/MASK PER PATIENT
# ============================================================

selected_rows = []

selection_log = []


for patient_id, group in merged.groupby(
    "PatientID",
    sort=True
):

    group = group.copy()


    # --------------------------------------------------------
    # STEP 1:
    # Best phase
    # --------------------------------------------------------

    best_priority = (
        group["phase_priority"]
        .min()
    )


    candidates = group[
        group["phase_priority"]
        == best_priority
    ].copy()


    # --------------------------------------------------------
    # STEP 2:
    # Largest annotated ROI
    # --------------------------------------------------------

    candidates = candidates.sort_values(

        by=[
            "ROIVolume",
            "ReferencedSeriesDescription",
            "ReferencedSeriesInstanceUID"
        ],

        ascending=[
            False,
            True,
            True
        ],

        na_position="last"

    )


    selected = candidates.iloc[0]


    selected_rows.append(
        selected
    )


    # --------------------------------------------------------
    # SELECTION LOG
    # --------------------------------------------------------

    selection_log.append({

        "PatientID":
            patient_id,

        "Candidates":
            len(group),

        "Selected_Phase":
            selected["normalized_phase"],

        "Selected_Phase_Priority":
            selected["phase_priority"],

        "Selected_ROI_Volume":
            selected["ROIVolume"],

        "Selected_CT_UID":
            selected[
                "ReferencedSeriesInstanceUID"
            ],

        "Selected_RTSTRUCT_UID":
            selected[
                "SeriesInstanceUID"
            ],

        "Selected_CT_Description":
            selected[
                "ReferencedSeriesDescription"
            ],

        "Selected_Mask":
            selected[
                "Mask_File"
            ]

    })


# ============================================================
# CREATE FINAL DATAFRAME
# ============================================================

final_df = pd.DataFrame(
    selected_rows
)


# ============================================================
# SORT BY PATIENT
# ============================================================

final_df = final_df.sort_values(
    by="PatientID"
).reset_index(
    drop=True
)


# ============================================================
# ADD SAMPLE ID
# ============================================================

final_df.insert(
    0,
    "sample_id",
    range(
        1,
        len(final_df) + 1
    )
)


# ============================================================
# CREATE CLEAN FINAL COLUMNS
# ============================================================

final_columns = [

    "sample_id",

    "PatientID",

    "risk_label",

    "age_years",

    "Sex",

    "Tumor Grade",

    "ClinicalTrialTimePointID",

    "StructureSetLabel",

    "ROI_role",

    "ROI_Name",

    "ROIVolume",

    "normalized_phase",

    "phase_priority",

    "ReferencedSeriesDescription",

    "ReferencedSeriesInstanceUID",

    "SeriesInstanceUID",

    "Mask_File"

]


final_df = final_df[
    final_columns
]


# ============================================================
# SAVE FINAL MANIFEST
# ============================================================

final_df.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# SAVE SELECTION LOG
# ============================================================

selection_log_df = pd.DataFrame(
    selection_log
)


selection_log_file = os.path.join(

    PROJECT,

    "data",

    "radiomics",

    "metadata",

    "patient_ct_selection_log.csv"

)


selection_log_df.to_csv(
    selection_log_file,
    index=False
)


# ============================================================
# FINAL STATISTICS
# ============================================================

print()
print("=" * 70)
print("FINAL PATIENT-LEVEL COHORT")
print("=" * 70)

print(
    "Patients selected:",
    len(final_df)
)


print()
print("Phase distribution:")

print(
    final_df[
        "normalized_phase"
    ].value_counts()
)


print()
print("Risk distribution:")

print(
    final_df[
        "risk_label"
    ].value_counts()
    .sort_index()
)


# ============================================================
# CHECK DUPLICATE PATIENTS
# ============================================================

duplicate_patients = (
    final_df[
        "PatientID"
    ]
    .duplicated()
    .sum()
)


print()
print(
    "Duplicate patients:",
    duplicate_patients
)


if duplicate_patients != 0:

    raise RuntimeError(
        "ERROR: Duplicate patients exist "
        "in the final cohort."
    )


# ============================================================
# CHECK DUPLICATE CT SERIES
# ============================================================

duplicate_ct = (
    final_df[
        "ReferencedSeriesInstanceUID"
    ]
    .duplicated()
    .sum()
)


print(
    "Duplicate CT series:",
    duplicate_ct
)


# ============================================================
# CHECK MASK FILES
# ============================================================

missing_masks = []


for mask_path in final_df["Mask_File"]:

    if not os.path.exists(mask_path):

        missing_masks.append(
            mask_path
        )


print(
    "Missing mask files:",
    len(missing_masks)
)


if missing_masks:

    print()
    print(
        "Missing mask files:"
    )

    for path in missing_masks:

        print(
            path
        )

    raise RuntimeError(
        "Some selected masks do not exist."
    )


# ============================================================
# CLASS COUNTS
# ============================================================

class_counts = (
    final_df[
        "risk_label"
    ]
    .value_counts()
    .sort_index()
)


print()
print("Risk class counts:")

for label, count in class_counts.items():

    print(
        f"Class {label}: {count}"
    )


# ============================================================
# OUTPUT FILES
# ============================================================

print()
print("=" * 70)

print(
    "FINAL MANIFEST:"
)

print(
    os.path.abspath(
        OUTPUT
    )
)


print()
print(
    "SELECTION LOG:"
)

print(
    os.path.abspath(
        selection_log_file
    )
)


print()
print("=" * 70)

print(
    "PATIENT-LEVEL SELECTION COMPLETE."
)

print("=" * 70)