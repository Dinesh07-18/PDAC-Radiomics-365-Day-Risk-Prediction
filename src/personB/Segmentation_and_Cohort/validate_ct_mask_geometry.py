import os
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom


# ============================================================
# PERSON B
# CT + MASK GEOMETRY VALIDATION
#
# IMPORTANT:
# This script ONLY checks geometry.
# It does NOT modify CT files or masks.
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
print("PERSON B - CT + MASK GEOMETRY VALIDATION")
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


# ------------------------------------------------------------
# 3. PATIENT
# ------------------------------------------------------------

PATIENT_ID = "C3L-00401"


print()
print("Patient:")
print(PATIENT_ID)


# ============================================================
# 4. LOAD RTSTRUCT REFERENCE TABLE
# ============================================================

reference_file = os.path.join(
    METADATA_ROOT,
    "rtstruct_reference_check.csv"
)


if not os.path.exists(reference_file):

    raise FileNotFoundError(
        f"\nReference file not found:\n"
        f"{reference_file}"
    )


reference_df = pd.read_csv(
    reference_file
)


required_columns = [
    "PatientID",
    "RTSTRUCT_SeriesInstanceUID",
    "Referenced_CT_SeriesInstanceUID"
]


for column in required_columns:

    if column not in reference_df.columns:

        raise RuntimeError(
            f"\nMissing required column: {column}\n"
            f"Available columns:\n"
            f"{list(reference_df.columns)}"
        )


patient_reference = reference_df[
    reference_df[
        "PatientID"
    ].astype(str)
    == PATIENT_ID
].copy()


if len(patient_reference) == 0:

    raise RuntimeError(
        f"\nNo reference rows found "
        f"for {PATIENT_ID}"
    )


print()
print(
    "RTSTRUCT reference rows:",
    len(patient_reference)
)


# ============================================================
# 5. FIND MASK
# ============================================================

patient_mask_dir = os.path.join(
    MASK_ROOT,
    PATIENT_ID
)


mask_files = glob.glob(
    os.path.join(
        patient_mask_dir,
        "*.nii.gz"
    )
)


if len(mask_files) == 0:

    raise FileNotFoundError(
        f"\nNo masks found:\n"
        f"{patient_mask_dir}"
    )


print()
print(
    "Masks found:",
    len(mask_files)
)


for i, mask_file in enumerate(
    mask_files,
    start=1
):

    print(
        f"  {i}. {os.path.basename(mask_file)}"
    )


# ------------------------------------------------------------
# Select first mask
# ------------------------------------------------------------

mask_path = mask_files[0]


print()
print("Selected mask:")
print(mask_path)


# ============================================================
# 6. IDENTIFY REFERENCED CT UID
# ============================================================

mask_filename = os.path.basename(
    mask_path
)


ct_series_uid = None


for candidate_uid in (
    patient_reference[
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
        "\nCould not identify CT UID "
        "from mask filename."
    )


print()
print(
    "Referenced CT SeriesInstanceUID:"
)

print(
    ct_series_uid
)


# ============================================================
# 7. LOAD MASK
# ============================================================

print()
print("Loading mask...")


mask_image = sitk.ReadImage(
    mask_path
)


mask_array = sitk.GetArrayFromImage(
    mask_image
)


print()
print(
    "Mask size:",
    mask_image.GetSize()
)


print(
    "Mask array shape:",
    mask_array.shape
)


print(
    "Mask spacing:",
    mask_image.GetSpacing()
)


print(
    "Mask origin:",
    mask_image.GetOrigin()
)


print(
    "Mask direction:",
    mask_image.GetDirection()
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


# ============================================================
# 8. FIND PATIENT CT DIRECTORY
# ============================================================

patient_ct_root = os.path.join(
    CT_ROOT,
    PATIENT_ID
)


if not os.path.exists(
    patient_ct_root
):

    raise FileNotFoundError(
        f"\nCT directory not found:\n"
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
# 9. FIND DICOM FILES BELONGING TO REFERENCED CT
# ============================================================

all_dicom_files = glob.glob(
    os.path.join(
        patient_ct_root,
        "**",
        "*.dcm"
    ),
    recursive=True
)


print()
print(
    "Total DICOM files:",
    len(all_dicom_files)
)


ct_files = []


for file_path in all_dicom_files:

    try:

        ds = pydicom.dcmread(
            file_path,
            stop_before_pixels=True
        )


        modality = str(
            getattr(
                ds,
                "Modality",
                ""
            )
        ).strip().upper()


        series_uid = str(
            getattr(
                ds,
                "SeriesInstanceUID",
                ""
            )
        ).strip()


        if (
            modality == "CT"
            and series_uid == ct_series_uid
        ):

            ct_files.append(
                file_path
            )


    except Exception:

        continue


print()
print(
    "Referenced CT DICOM files:",
    len(ct_files)
)


if len(ct_files) == 0:

    raise RuntimeError(
        "\nNo DICOM files found for "
        "the referenced CT series."
    )


# ============================================================
# 10. READ CT SLICE GEOMETRY
# ============================================================

print()
print(
    "Reading CT slice geometry..."
)


ct_slice_records = []


for file_path in ct_files:

    try:

        ds = pydicom.dcmread(
            file_path,
            stop_before_pixels=True
        )


        instance_number = getattr(
            ds,
            "InstanceNumber",
            None
        )


        image_position = getattr(
            ds,
            "ImagePositionPatient",
            None
        )


        image_orientation = getattr(
            ds,
            "ImageOrientationPatient",
            None
        )


        pixel_spacing = getattr(
            ds,
            "PixelSpacing",
            None
        )


        slice_thickness = getattr(
            ds,
            "SliceThickness",
            None
        )


        if image_position is None:

            continue


        x = float(
            image_position[0]
        )

        y = float(
            image_position[1]
        )

        z = float(
            image_position[2]
        )


        ct_slice_records.append(
            {
                "file": file_path,
                "instance_number": instance_number,
                "x": x,
                "y": y,
                "z": z,
                "orientation": str(
                    image_orientation
                ),
                "pixel_spacing": str(
                    pixel_spacing
                ),
                "slice_thickness": slice_thickness
            }
        )


    except Exception:

        continue


if len(ct_slice_records) == 0:

    raise RuntimeError(
        "Could not read CT slice positions."
    )


ct_geometry_df = pd.DataFrame(
    ct_slice_records
)


# ------------------------------------------------------------
# Sort by physical Z position
# ------------------------------------------------------------

ct_geometry_df = (
    ct_geometry_df
    .sort_values(
        "z"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 11. CT Z-POSITION ANALYSIS
# ============================================================

ct_z = (
    ct_geometry_df[
        "z"
    ]
    .to_numpy(
        dtype=float
    )
)


ct_z_differences = np.diff(
    ct_z
)


print()
print(
    "CT slices with valid position:",
    len(ct_z)
)


print(
    "CT minimum Z:",
    float(
        np.min(ct_z)
    )
)


print(
    "CT maximum Z:",
    float(
        np.max(ct_z)
    )
)


if len(ct_z_differences) > 0:

    print()
    print(
        "CT Z-spacing statistics:"
    )

    print(
        "  Minimum:",
        float(
            np.min(
                np.abs(
                    ct_z_differences
                )
            )
        )
    )

    print(
        "  Maximum:",
        float(
            np.max(
                np.abs(
                    ct_z_differences
                )
            )
        )
    )

    print(
        "  Median:",
        float(
            np.median(
                np.abs(
                    ct_z_differences
                )
            )
        )
    )

    print(
        "  Mean:",
        float(
            np.mean(
                np.abs(
                    ct_z_differences
                )
            )
        )
    )


# ============================================================
# 12. DETECT NON-UNIFORM CT SPACING
# ============================================================

nonuniform_indices = []


if len(ct_z_differences) > 1:

    median_spacing = np.median(
        np.abs(
            ct_z_differences
        )
    )


    tolerance = max(
        0.05,
        median_spacing * 0.10
    )


    for i, difference in enumerate(
        np.abs(
            ct_z_differences
        )
    ):

        if abs(
            difference
            - median_spacing
        ) > tolerance:

            nonuniform_indices.append(
                i
            )


print()
print(
    "Non-uniform Z intervals:",
    len(nonuniform_indices)
)


if len(nonuniform_indices) > 0:

    print()
    print(
        "Examples of unusual Z intervals:"
    )


    for index in nonuniform_indices[:10]:

        print(
            f"  Between slice {index} "
            f"and {index + 1}: "
            f"{abs(ct_z_differences[index]):.6f} mm"
        )


# ============================================================
# 13. LOAD CT WITH SIMPLEITK
# ============================================================

print()
print(
    "Loading CT volume..."
)


# Sort files according to physical Z position.

sorted_ct_files = (
    ct_geometry_df[
        "file"
    ]
    .tolist()
)


reader = sitk.ImageSeriesReader()


reader.SetFileNames(
    sorted_ct_files
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


print(
    "CT spacing:",
    ct_image.GetSpacing()
)


print(
    "CT origin:",
    ct_image.GetOrigin()
)


print(
    "CT direction:",
    ct_image.GetDirection()
)


# ============================================================
# 14. DIMENSION CHECK
# ============================================================

print()
print(
    "Checking array dimensions..."
)


dimension_match = (
    ct_array.shape
    ==
    mask_array.shape
)


print(
    "CT shape:",
    ct_array.shape
)


print(
    "Mask shape:",
    mask_array.shape
)


print(
    "Dimensions match:",
    dimension_match
)


# ============================================================
# 15. PHYSICAL GEOMETRY COMPARISON
# ============================================================

print()
print(
    "Comparing physical geometry..."
)


ct_spacing = np.array(
    ct_image.GetSpacing(),
    dtype=float
)


mask_spacing = np.array(
    mask_image.GetSpacing(),
    dtype=float
)


ct_origin = np.array(
    ct_image.GetOrigin(),
    dtype=float
)


mask_origin = np.array(
    mask_image.GetOrigin(),
    dtype=float
)


spacing_difference = (
    mask_spacing
    -
    ct_spacing
)


origin_difference = (
    mask_origin
    -
    ct_origin
)


print()
print(
    "CT spacing:"
)

print(
    ct_spacing
)


print()
print(
    "Mask spacing:"
)

print(
    mask_spacing
)


print()
print(
    "Spacing difference:"
)

print(
    spacing_difference
)


print()
print(
    "CT origin:"
)

print(
    ct_origin
)


print()
print(
    "Mask origin:"
)

print(
    mask_origin
)


print()
print(
    "Origin difference:"
)

print(
    origin_difference
)


# ============================================================
# 16. GEOMETRY MATCH TEST
# ============================================================

spacing_matches = np.allclose(
    ct_spacing,
    mask_spacing,
    atol=0.01
)


origin_matches = np.allclose(
    ct_origin,
    mask_origin,
    atol=1.0
)


direction_matches = np.allclose(
    np.array(
        ct_image.GetDirection()
    ),
    np.array(
        mask_image.GetDirection()
    ),
    atol=1e-3
)


print()
print(
    "Geometry checks:"
)


print(
    "  Dimensions:",
    dimension_match
)


print(
    "  Spacing:",
    spacing_matches
)


print(
    "  Origin:",
    origin_matches
)


print(
    "  Direction:",
    direction_matches
)


# ============================================================
# 17. CALCULATE PHYSICAL EXTENTS
# ============================================================

print()
print(
    "Calculating physical extents..."
)


ct_size = np.array(
    ct_image.GetSize(),
    dtype=float
)


mask_size = np.array(
    mask_image.GetSize(),
    dtype=float
)


# Approximate physical extent:
# (size - 1) * spacing

ct_extent = (
    ct_size - 1
) * ct_spacing


mask_extent = (
    mask_size - 1
) * mask_spacing


print()
print(
    "CT physical extent:"
)

print(
    ct_extent
)


print()
print(
    "Mask physical extent:"
)

print(
    mask_extent
)


print()
print(
    "Physical extent difference:"
)

print(
    mask_extent
    -
    ct_extent
)


# ============================================================
# 18. MASK PHYSICAL BOUNDING BOX
# ============================================================

print()
print(
    "Calculating mask foreground bounding box..."
)


foreground_indices = np.argwhere(
    mask_array > 0
)


if len(foreground_indices) == 0:

    raise RuntimeError(
        "Mask contains no foreground."
    )


mask_z_indices = (
    foreground_indices[:, 0]
)


mask_y_indices = (
    foreground_indices[:, 1]
)


mask_x_indices = (
    foreground_indices[:, 2]
)


print()
print(
    "Mask foreground slice range:"
)

print(
    int(
        np.min(
            mask_z_indices
        )
    ),
    "to",
    int(
        np.max(
            mask_z_indices
        )
    )
)


print(
    "Mask foreground Y range:"
)

print(
    int(
        np.min(
            mask_y_indices
        )
    ),
    "to",
    int(
        np.max(
            mask_y_indices
        )
    )
)


print(
    "Mask foreground X range:"
)

print(
    int(
        np.min(
            mask_x_indices
        )
    ),
    "to",
    int(
        np.max(
            mask_x_indices
        )
    )
)


# ============================================================
# 19. SAVE VALIDATION REPORT
# ============================================================

report_file = os.path.join(
    OUTPUT_DIR,
    f"{PATIENT_ID}_ct_mask_geometry_report.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PERSON B - CT + MASK GEOMETRY VALIDATION\n"
    )

    f.write(
        "=" * 70
        + "\n\n"
    )

    f.write(
        f"Patient: {PATIENT_ID}\n"
    )

    f.write(
        f"Mask: {mask_path}\n"
    )

    f.write(
        f"Referenced CT UID: {ct_series_uid}\n\n"
    )


    f.write(
        "CT\n"
    )

    f.write(
        f"Shape: {ct_array.shape}\n"
    )

    f.write(
        f"Spacing: {ct_spacing}\n"
    )

    f.write(
        f"Origin: {ct_origin}\n"
    )

    f.write(
        f"Direction: {ct_image.GetDirection()}\n"
    )

    f.write(
        f"Z minimum: {np.min(ct_z)}\n"
    )

    f.write(
        f"Z maximum: {np.max(ct_z)}\n"
    )

    f.write(
        f"Z intervals: {len(ct_z_differences)}\n"
    )

    f.write(
        f"Non-uniform intervals: "
        f"{len(nonuniform_indices)}\n\n"
    )


    f.write(
        "MASK\n"
    )

    f.write(
        f"Shape: {mask_array.shape}\n"
    )

    f.write(
        f"Spacing: {mask_spacing}\n"
    )

    f.write(
        f"Origin: {mask_origin}\n"
    )

    f.write(
        f"Direction: {mask_image.GetDirection()}\n"
    )

    f.write(
        f"Foreground voxels: {mask_voxels}\n\n"
    )


    f.write(
        "GEOMETRY CHECKS\n"
    )

    f.write(
        f"Dimensions match: "
        f"{dimension_match}\n"
    )

    f.write(
        f"Spacing matches: "
        f"{spacing_matches}\n"
    )

    f.write(
        f"Origin matches: "
        f"{origin_matches}\n"
    )

    f.write(
        f"Direction matches: "
        f"{direction_matches}\n\n"
    )


    f.write(
        "CT physical extent:\n"
    )

    f.write(
        f"{ct_extent}\n\n"
    )


    f.write(
        "Mask physical extent:\n"
    )

    f.write(
        f"{mask_extent}\n\n"
    )


    f.write(
        "Physical extent difference:\n"
    )

    f.write(
        f"{mask_extent - ct_extent}\n\n"
    )


    f.write(
        "Mask foreground slice range:\n"
    )

    f.write(
        f"{np.min(mask_z_indices)} "
        f"to "
        f"{np.max(mask_z_indices)}\n"
    )


# ============================================================
# 20. FINAL STATUS
# ============================================================

print()
print("=" * 80)
print("GEOMETRY VALIDATION COMPLETE")
print("=" * 80)


print()
print(
    "Dimensions match:"
)

print(
    dimension_match
)


print()
print(
    "Spacing matches:"
)

print(
    spacing_matches
)


print()
print(
    "Origin matches:"
)

print(
    origin_matches
)


print()
print(
    "Direction matches:"
)

print(
    direction_matches
)


print()
print(
    "Non-uniform CT intervals:"
)

print(
    len(nonuniform_indices)
)


print()
print(
    "Validation report saved:"
)

print(
    report_file
)


print()
print(
    "NO CT OR MASK FILES WERE MODIFIED."
)


print("=" * 80)