import os
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt


# ============================================================
# PERSON B
# CT + SEGMENTATION OVERLAY VISUALIZATION
# ============================================================


# ------------------------------------------------------------
# 1. PROJECT ROOT
# ------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        ".."
    )
)


print("=" * 80)
print("PERSON B - CT + SEGMENTATION OVERLAY")
print("=" * 80)

print()
print("Project root:")
print(PROJECT_ROOT)


# ------------------------------------------------------------
# 2. DIRECTORIES
# ------------------------------------------------------------

CT_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "radiomics",
    "ct",
    "cptac_pda"
)

MASK_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "radiomics",
    "masks"
)

METADATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "radiomics",
    "metadata"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "personB",
    "final_visualizations"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print()
print("Output directory:")
print(OUTPUT_DIR)


# ------------------------------------------------------------
# 3. PATIENT
# ------------------------------------------------------------

PATIENT_ID = "C3L-00401"


print()
print("Patient:")
print(PATIENT_ID)


# ============================================================
# 4. FIND MASK FILES
# ============================================================

patient_mask_dir = os.path.join(
    MASK_ROOT,
    PATIENT_ID
)


if not os.path.exists(
    patient_mask_dir
):

    raise FileNotFoundError(
        f"\nMask directory not found:\n"
        f"{patient_mask_dir}"
    )


mask_files = glob.glob(
    os.path.join(
        patient_mask_dir,
        "*.nii.gz"
    )
)


if len(mask_files) == 0:

    raise FileNotFoundError(
        f"\nNo NIfTI mask files found for "
        f"{PATIENT_ID}\n"
        f"Directory:\n{patient_mask_dir}"
    )


print()
print(
    "Mask files found:",
    len(mask_files)
)


for i, path in enumerate(
    mask_files,
    start=1
):

    print(
        f"  {i}. {os.path.basename(path)}"
    )


# ------------------------------------------------------------
# Select first mask
# ------------------------------------------------------------

mask_path = mask_files[0]


print()
print("Selected mask:")
print(mask_path)


# ============================================================
# 5. LOAD MASK
# ============================================================

print()
print("Loading mask...")


mask_image = sitk.ReadImage(
    mask_path
)


mask_array = sitk.GetArrayFromImage(
    mask_image
)


print(
    "Mask size:",
    mask_image.GetSize()
)


print(
    "Mask array shape:",
    mask_array.shape
)


mask_voxels = int(
    np.sum(
        mask_array > 0
    )
)


print(
    "Mask foreground voxels:",
    mask_voxels
)


if mask_voxels == 0:

    raise RuntimeError(
        "Selected mask contains "
        "no foreground voxels."
    )


# ============================================================
# 6. READ RTSTRUCT REFERENCE TABLE
# ============================================================

reference_file = os.path.join(
    METADATA_ROOT,
    "rtstruct_reference_check.csv"
)


if not os.path.exists(
    reference_file
):

    raise FileNotFoundError(
        "\nRTSTRUCT reference file not found:\n"
        f"{reference_file}"
    )


print()
print(
    "Reading RTSTRUCT reference mapping..."
)


reference_df = pd.read_csv(
    reference_file
)


print(
    "Reference rows:",
    len(reference_df)
)


print(
    "Reference columns:",
    list(reference_df.columns)
)


# ============================================================
# 7. VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "PatientID",
    "RTSTRUCT_SeriesInstanceUID",
    "Referenced_CT_SeriesInstanceUID"
]


for column in required_columns:

    if column not in reference_df.columns:

        raise RuntimeError(
            f"\nRequired column missing:\n"
            f"{column}\n\n"
            f"Available columns:\n"
            f"{list(reference_df.columns)}"
        )


# ============================================================
# 8. FILTER TO CURRENT PATIENT
# ============================================================

patient_reference_df = reference_df[
    reference_df[
        "PatientID"
    ].astype(str)
    == PATIENT_ID
].copy()


if len(patient_reference_df) == 0:

    raise RuntimeError(
        f"\nNo reference rows found "
        f"for patient {PATIENT_ID}."
    )


print()
print(
    "Reference rows for patient:",
    len(patient_reference_df)
)


# ============================================================
# 9. IDENTIFY CT UID FROM MASK FILENAME
# ============================================================

mask_filename = os.path.basename(
    mask_path
)


print()
print("Mask filename:")
print(mask_filename)


# ------------------------------------------------------------
# IMPORTANT:
#
# The mask filename contains the REFERENCED CT UID,
# with dots replaced by underscores.
#
# Example:
#
# CT UID:
# 1.3.6.1.4.1.14519.5.2.1...
#
# Mask filename:
# 1_3_6_1_4_1_14519_5_2_1...
#
# Therefore we compare the normalized CT UID against
# the mask filename.
# ------------------------------------------------------------

ct_series_uid = None


for candidate_uid in (
    patient_reference_df[
        "Referenced_CT_SeriesInstanceUID"
    ]
    .dropna()
    .astype(str)
    .unique()
):

    normalized_uid = candidate_uid.replace(
        ".",
        "_"
    )


    if normalized_uid in mask_filename:

        ct_series_uid = candidate_uid

        break


if ct_series_uid is None:

    raise RuntimeError(
        "\nCould not identify the referenced CT "
        "SeriesInstanceUID from the mask filename.\n\n"
        f"Patient: {PATIENT_ID}\n"
        f"Mask filename: {mask_filename}\n\n"
        "Referenced CT UIDs checked:\n"
        + "\n".join(
            str(x)
            for x in patient_reference_df[
                "Referenced_CT_SeriesInstanceUID"
            ]
            .dropna()
            .unique()
        )
    )


print()
print(
    "Referenced CT SeriesInstanceUID identified:"
)


print(
    ct_series_uid
)


# ============================================================
# 10. FIND ASSOCIATED RTSTRUCT UID
# ============================================================

matching_reference_rows = patient_reference_df[
    patient_reference_df[
        "Referenced_CT_SeriesInstanceUID"
    ].astype(str)
    == str(ct_series_uid)
]


if len(matching_reference_rows) == 0:

    raise RuntimeError(
        "\nCould not find RTSTRUCT mapping "
        "for selected CT series."
    )


rtstruct_uids = (
    matching_reference_rows[
        "RTSTRUCT_SeriesInstanceUID"
    ]
    .dropna()
    .astype(str)
    .unique()
)


print()
print(
    "Associated RTSTRUCT UID(s):"
)


for uid in rtstruct_uids:

    print(
        " ",
        uid
    )


# ============================================================
# 11. FIND PATIENT CT DIRECTORY
# ============================================================

patient_ct_root = os.path.join(
    CT_ROOT,
    PATIENT_ID
)


if not os.path.exists(
    patient_ct_root
):

    raise FileNotFoundError(
        "\nPatient CT directory not found:\n"
        f"{patient_ct_root}"
    )


print()
print(
    "Patient CT directory:"
)


print(
    patient_ct_root
)


# ============================================================
# 12. FIND ALL DICOM FILES
# ============================================================

print()
print(
    "Searching for DICOM files..."
)


all_dicom_files = glob.glob(
    os.path.join(
        patient_ct_root,
        "**",
        "*.dcm"
    ),
    recursive=True
)


print(
    "DICOM files found:",
    len(all_dicom_files)
)


if len(all_dicom_files) == 0:

    raise RuntimeError(
        "No DICOM files found for patient."
    )


# ============================================================
# 13. DISCOVER CT SERIES
# ============================================================

print()
print(
    "Discovering CT series..."
)


series_file_map = {}


for file_path in all_dicom_files:

    try:

        reader = sitk.ImageFileReader()

        reader.SetFileName(
            file_path
        )

        reader.ReadImageInformation()


        # ----------------------------------------------------
        # DICOM Modality
        # ----------------------------------------------------

        modality = ""


        if reader.HasMetaDataKey(
            "0008|0060"
        ):

            modality = (
                reader.GetMetaData(
                    "0008|0060"
                )
                .strip()
                .upper()
            )


        if modality != "CT":

            continue


        # ----------------------------------------------------
        # DICOM Series Instance UID
        # ----------------------------------------------------

        if not reader.HasMetaDataKey(
            "0020|000e"
        ):

            continue


        series_uid = (
            reader.GetMetaData(
                "0020|000e"
            )
            .strip()
        )


        if series_uid == "":

            continue


        if series_uid not in series_file_map:

            series_file_map[
                series_uid
            ] = []


        series_file_map[
            series_uid
        ].append(
            file_path
        )


    except Exception:

        continue


print()
print(
    "CT series discovered:",
    len(series_file_map)
)


if len(series_file_map) == 0:

    raise RuntimeError(
        "No CT series could be discovered."
    )


# ============================================================
# 14. CHECK SELECTED CT SERIES
# ============================================================

print()
print(
    "Checking referenced CT series..."
)


if ct_series_uid not in series_file_map:

    print()
    print(
        "ERROR: Referenced CT series "
        "was not found in downloaded files."
    )


    print()
    print(
        "Available CT series:"
    )


    for uid, files in series_file_map.items():

        print(
            f"  {uid} -> {len(files)} files"
        )


    raise RuntimeError(
        "\nThe referenced CT series is not "
        "present in the downloaded patient data."
    )


ct_files = series_file_map[
    ct_series_uid
]


print()
print(
    "Referenced CT series FOUND."
)


print(
    "CT files in series:",
    len(ct_files)
)


# ============================================================
# 15. LOAD CT SERIES
# ============================================================

print()
print(
    "Loading referenced CT series..."
)


reader = sitk.ImageSeriesReader()


# ------------------------------------------------------------
# Try SimpleITK's normal series discovery first
# ------------------------------------------------------------

series_file_names = (
    sitk.ImageSeriesReader
    .GetGDCMSeriesFileNames(
        patient_ct_root,
        ct_series_uid
    )
)


# ------------------------------------------------------------
# Fallback:
# use our manually discovered files
# ------------------------------------------------------------

if len(series_file_names) == 0:

    print()
    print(
        "GDCM automatic lookup returned "
        "zero files."
    )


    print(
        "Using manually discovered CT files."
    )


    series_file_names = sorted(
        ct_files
    )


print()
print(
    "CT slices selected:",
    len(series_file_names)
)


if len(series_file_names) == 0:

    raise RuntimeError(
        "No CT slices available."
    )


reader.SetFileNames(
    series_file_names
)


ct_image = reader.Execute()


ct_array = sitk.GetArrayFromImage(
    ct_image
)


print()
print(
    "CT size:",
    ct_image.GetSize()
)


print(
    "CT array shape:",
    ct_array.shape
)


# ============================================================
# 16. CHECK CT AND MASK DIMENSIONS
# ============================================================

print()
print(
    "Checking CT and mask dimensions..."
)


print(
    "CT shape:",
    ct_array.shape
)


print(
    "Mask shape:",
    mask_array.shape
)


if ct_array.shape != mask_array.shape:

    raise RuntimeError(
        "\nCT and mask dimensions do not match.\n\n"
        f"CT shape: {ct_array.shape}\n"
        f"Mask shape: {mask_array.shape}\n\n"
        "The selected CT series and mask "
        "do not have the same voxel grid."
    )


print()
print(
    "CT and mask dimensions MATCH."
)


# ============================================================
# 17. CHECK IMAGE GEOMETRY
# ============================================================

print()
print(
    "Checking image geometry..."
)


print(
    "CT spacing:",
    ct_image.GetSpacing()
)


print(
    "Mask spacing:",
    mask_image.GetSpacing()
)


print(
    "CT origin:",
    ct_image.GetOrigin()
)


print(
    "Mask origin:",
    mask_image.GetOrigin()
)


# ============================================================
# 18. FIND BEST MASK SLICE
# ============================================================

print()
print(
    "Finding slice with largest "
    "segmentation area..."
)


mask_area_per_slice = np.sum(
    mask_array > 0,
    axis=(1, 2)
)


best_slice = int(
    np.argmax(
        mask_area_per_slice
    )
)


best_area = int(
    mask_area_per_slice[
        best_slice
    ]
)


print(
    "Best slice:",
    best_slice
)


print(
    "Segmentation pixels:",
    best_area
)


# ============================================================
# 19. EXTRACT CT AND MASK SLICE
# ============================================================

ct_slice = ct_array[
    best_slice
]


mask_slice = mask_array[
    best_slice
]


# ============================================================
# 20. SOFT-TISSUE WINDOW
# ============================================================

WINDOW_CENTER = 50
WINDOW_WIDTH = 400


window_min = (
    WINDOW_CENTER
    - WINDOW_WIDTH / 2
)


window_max = (
    WINDOW_CENTER
    + WINDOW_WIDTH / 2
)


ct_windowed = np.clip(
    ct_slice,
    window_min,
    window_max
)


# ============================================================
# 21. CREATE VISUALIZATION
# ============================================================

print()
print(
    "Creating visualization..."
)


fig, ax = plt.subplots(
    figsize=(9, 8)
)


# ------------------------------------------------------------
# CT image
# ------------------------------------------------------------

ax.imshow(
    ct_windowed,
    cmap="gray",
    vmin=window_min,
    vmax=window_max
)


# ------------------------------------------------------------
# Segmentation overlay
# ------------------------------------------------------------

overlay = np.ma.masked_where(
    mask_slice == 0,
    mask_slice
)


ax.imshow(
    overlay,
    cmap="autumn",
    alpha=0.45
)


# ------------------------------------------------------------
# Segmentation contour
# ------------------------------------------------------------

ax.contour(
    mask_slice > 0,
    levels=[0.5],
    linewidths=1.5
)


# ------------------------------------------------------------
# Title
# ------------------------------------------------------------

ax.set_title(
    "CT Image with Pancreas Segmentation\n"
    f"Patient: {PATIENT_ID} | "
    f"Slice: {best_slice}",
    fontsize=14
)


ax.axis(
    "off"
)


# ============================================================
# 22. SAVE FIGURE
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    f"{PATIENT_ID}_ct_segmentation_overlay.png"
)


plt.tight_layout()


plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 23. FINAL REPORT
# ============================================================

print()
print("=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)


print()
print(
    "Patient:"
)

print(
    PATIENT_ID
)


print()
print(
    "Referenced CT UID:"
)

print(
    ct_series_uid
)


print()
print(
    "Associated RTSTRUCT UID(s):"
)

for uid in rtstruct_uids:

    print(
        " ",
        uid
    )


print()
print(
    "CT slices:"
)

print(
    len(series_file_names)
)


print()
print(
    "Mask foreground voxels:"
)

print(
    mask_voxels
)


print()
print(
    "Best segmentation slice:"
)

print(
    best_slice
)


print()
print(
    "Saved visualization:"
)

print(
    output_file
)


print()
print(
    "PERSON B CT + SEGMENTATION "
    "VISUALIZATION COMPLETED SUCCESSFULLY."
)


print("=" * 80)