import os
import pandas as pd
import pydicom
import SimpleITK as sitk

from radiomics import featureextractor


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT = os.path.abspath(".")

MANIFEST = os.path.join(
    PROJECT,
    "data",
    "radiomics",
    "metadata",
    "final_patient_radiomics_manifest.csv"
)

CT_ROOT = os.path.join(
    PROJECT,
    "data",
    "radiomics",
    "ct"
)


# ============================================================
# LOAD FINAL PATIENT MANIFEST
# ============================================================

print()
print("=" * 70)
print("PYRADIOMICS ONE-PATIENT VALIDATION")
print("=" * 70)

if not os.path.exists(MANIFEST):

    raise FileNotFoundError(
        "Final patient manifest not found:\n"
        + MANIFEST
    )


df = pd.read_csv(MANIFEST)


print(
    "Patients in manifest:",
    len(df)
)


# ============================================================
# SELECT FIRST PATIENT
# ============================================================

row = df.iloc[0]

patient_id = str(
    row["PatientID"]
).strip()

ct_uid = str(
    row["ReferencedSeriesInstanceUID"]
).strip()

mask_file = str(
    row["Mask_File"]
).strip()


print()
print("TEST PATIENT")
print("-" * 70)

print(
    "Patient ID:",
    patient_id
)

print(
    "Risk label:",
    row["risk_label"]
)

print(
    "CT Series UID:",
    ct_uid
)

print(
    "Mask:",
    mask_file
)


# ============================================================
# CHECK MASK
# ============================================================

if not os.path.exists(mask_file):

    raise FileNotFoundError(
        "Mask file not found:\n"
        + mask_file
    )


# ============================================================
# FIND REFERENCED CT SERIES
# ============================================================

print()
print("Searching for referenced CT series...")

ct_series_dir = None

files_scanned = 0

for root, dirs, files in os.walk(CT_ROOT):

    for filename in files:

        if not filename.lower().endswith(".dcm"):
            continue

        path = os.path.join(
            root,
            filename
        )

        files_scanned += 1

        try:

            ds = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                specific_tags=[
                    "Modality",
                    "SeriesInstanceUID"
                ]
            )

            modality = str(
                getattr(
                    ds,
                    "Modality",
                    ""
                )
            )

            series_uid = str(
                getattr(
                    ds,
                    "SeriesInstanceUID",
                    ""
                )
            )

            if (
                modality == "CT"
                and series_uid == ct_uid
            ):

                ct_series_dir = root

                break

        except Exception:

            continue

    if ct_series_dir is not None:
        break


print(
    "DICOM files scanned:",
    files_scanned
)


if ct_series_dir is None:

    raise RuntimeError(
        "Could not find the referenced CT series:\n"
        + ct_uid
    )


print(
    "CT directory found:"
)

print(
    ct_series_dir
)


# ============================================================
# LOAD CT SERIES
# ============================================================

print()
print("Loading CT series...")

reader = sitk.ImageSeriesReader()

series_ids = (
    reader.GetGDCMSeriesIDs(
        ct_series_dir
    )
)


if not series_ids:

    raise RuntimeError(
        "SimpleITK could not find "
        "a DICOM series."
    )


selected_series = None


for sid in series_ids:

    dicom_files = (
        reader.GetGDCMSeriesFileNames(
            ct_series_dir,
            sid
        )
    )

    if not dicom_files:
        continue

    try:

        ds = pydicom.dcmread(
            dicom_files[0],
            stop_before_pixels=True,
            specific_tags=[
                "SeriesInstanceUID"
            ]
        )

        uid = str(
            getattr(
                ds,
                "SeriesInstanceUID",
                ""
            )
        )

        if uid == ct_uid:

            selected_series = sid

            break

    except Exception:

        continue


if selected_series is None:

    raise RuntimeError(
        "Could not identify the exact CT series."
    )


ct_files = (
    reader.GetGDCMSeriesFileNames(
        ct_series_dir,
        selected_series
    )
)


reader.SetFileNames(
    ct_files
)


ct_image = reader.Execute()


print(
    "CT loaded successfully."
)

print(
    "CT size:",
    ct_image.GetSize()
)

print(
    "CT spacing:",
    ct_image.GetSpacing()
)

print(
    "Number of CT slices:",
    len(ct_files)
)


# ============================================================
# LOAD MASK
# ============================================================

print()
print("Loading mask...")

mask_image = sitk.ReadImage(
    mask_file
)


print(
    "Mask loaded successfully."
)

print(
    "Mask size:",
    mask_image.GetSize()
)

print(
    "Mask spacing:",
    mask_image.GetSpacing()
)


# ============================================================
# CHECK GEOMETRY
# ============================================================

print()
print("=" * 70)
print("GEOMETRY CHECK")
print("=" * 70)


print(
    "CT size:",
    ct_image.GetSize()
)

print(
    "Mask size:",
    mask_image.GetSize()
)


if (
    ct_image.GetSize()
    != mask_image.GetSize()
):

    raise RuntimeError(
        "CT and mask sizes do not match."
    )


print(
    "Size check: PASS"
)


print()
print(
    "CT spacing:",
    ct_image.GetSpacing()
)

print(
    "Mask spacing:",
    mask_image.GetSpacing()
)


# ============================================================
# CHECK PHYSICAL GEOMETRY
# ============================================================

def close_tuple(
    a,
    b,
    tolerance=1e-4
):

    return all(
        abs(x - y) <= tolerance
        for x, y in zip(a, b)
    )


if not close_tuple(
    ct_image.GetSpacing(),
    mask_image.GetSpacing()
):

    raise RuntimeError(
        "CT and mask spacing do not match."
    )


if not close_tuple(
    ct_image.GetOrigin(),
    mask_image.GetOrigin()
):

    raise RuntimeError(
        "CT and mask origin do not match."
    )


if not close_tuple(
    ct_image.GetDirection(),
    mask_image.GetDirection()
):

    raise RuntimeError(
        "CT and mask direction do not match."
    )


print(
    "Spacing check: PASS"
)

print(
    "Origin check: PASS"
)

print(
    "Direction check: PASS"
)


# ============================================================
# CHECK MASK VOXELS
# ============================================================

print()
print("Checking mask voxels...")

mask_array = sitk.GetArrayFromImage(
    mask_image
)


unique_values = set(
    mask_array.flatten().tolist()
)


print(
    "Unique mask values:",
    sorted(unique_values)
)


voxel_count = int(
    (mask_array > 0).sum()
)


print(
    "Mask voxels:",
    voxel_count
)


if voxel_count == 0:

    raise RuntimeError(
        "Mask contains zero voxels."
    )


print(
    "Mask voxel check: PASS"
)


# ============================================================
# CHECK MASK PHYSICAL VOLUME
# ============================================================

spacing = mask_image.GetSpacing()

voxel_volume_mm3 = (
    spacing[0]
    * spacing[1]
    * spacing[2]
)


mask_volume_mm3 = (
    voxel_count
    * voxel_volume_mm3
)


mask_volume_cm3 = (
    mask_volume_mm3 / 1000.0
)


print()
print(
    "Voxel volume:",
    voxel_volume_mm3,
    "mm³"
)

print(
    "Mask volume:",
    round(
        mask_volume_cm3,
        3
    ),
    "cm³"
)


# ============================================================
# CHECK MASK OVERLAP WITH CT
# ============================================================

print()
print("Checking CT/mask spatial overlap...")

ct_size = ct_image.GetSize()

mask_size = mask_image.GetSize()


if ct_size != mask_size:

    raise RuntimeError(
        "CT and mask do not have identical dimensions."
    )


# ============================================================
# CREATE PYRADIOMICS EXTRACTOR
# ============================================================

print()
print("=" * 70)
print("INITIALIZING PYRADIOMICS")
print("=" * 70)


settings = {

    # Use original CT resolution initially.
    # We will decide on resampling later.

    "binWidth": 25,

    "normalize": False,

    "minimumROIDimensions": 3,

    "geometryTolerance": 1e-4,

    "correctMask": False,

    "label": 1

}


extractor = featureextractor.RadiomicsFeatureExtractor(
    **settings
)


# ============================================================
# ENABLE FEATURE CLASSES
# ============================================================

extractor.disableAllFeatures()

extractor.enableFeatureClassByName(
    "firstorder"
)

extractor.enableFeatureClassByName(
    "shape"
)

extractor.enableFeatureClassByName(
    "glcm"
)

extractor.enableFeatureClassByName(
    "glrlm"
)

extractor.enableFeatureClassByName(
    "glszm"
)

extractor.enableFeatureClassByName(
    "gldm"
)

extractor.enableFeatureClassByName(
    "ngtdm"
)


print(
    "Feature classes enabled:"
)

print(
    "First Order"
)

print(
    "Shape"
)

print(
    "GLCM"
)

print(
    "GLRLM"
)

print(
    "GLSZM"
)

print(
    "GLDM"
)

print(
    "NGTDM"
)


# ============================================================
# EXTRACT FEATURES
# ============================================================

print()
print("=" * 70)
print("EXTRACTING FEATURES")
print("=" * 70)

print(
    "This may take some time..."
)


result = extractor.execute(
    ct_image,
    mask_image
)


# ============================================================
# SEPARATE DIAGNOSTICS AND FEATURES
# ============================================================

feature_dict = {}

diagnostic_count = 0
feature_count = 0


for key, value in result.items():

    if key.startswith(
        "diagnostics_"
    ):

        diagnostic_count += 1

    else:

        feature_dict[key] = value

        feature_count += 1


# ============================================================
# DISPLAY DIAGNOSTICS
# ============================================================

print()
print("=" * 70)
print("PYRADIOMICS DIAGNOSTICS")
print("=" * 70)


for key, value in result.items():

    if key.startswith(
        "diagnostics_"
    ):

        print(
            f"{key}: {value}"
        )


# ============================================================
# DISPLAY FEATURE RESULTS
# ============================================================

print()
print("=" * 70)
print("FEATURE EXTRACTION RESULT")
print("=" * 70)


print(
    "Diagnostic values:",
    diagnostic_count
)

print(
    "Radiomic features:",
    feature_count
)


# ============================================================
# CHECK NUMERIC FEATURES
# ============================================================

numeric_features = {}

non_numeric_features = []


for key, value in feature_dict.items():

    try:

        numeric_value = float(
            value
        )

        numeric_features[key] = (
            numeric_value
        )

    except Exception:

        non_numeric_features.append(
            key
        )


print()
print(
    "Numeric radiomic features:",
    len(numeric_features)
)

print(
    "Non-numeric radiomic features:",
    len(non_numeric_features)
)


# ============================================================
# CHECK NaN / INF
# ============================================================

import math


nan_features = []
inf_features = []


for key, value in numeric_features.items():

    if math.isnan(value):

        nan_features.append(
            key
        )

    elif math.isinf(value):

        inf_features.append(
            key
        )


print(
    "NaN features:",
    len(nan_features)
)

print(
    "Infinite features:",
    len(inf_features)
)


# ============================================================
# SHOW SAMPLE FEATURES
# ============================================================

print()
print("=" * 70)
print("SAMPLE FEATURES")
print("=" * 70)


shown = 0


for key, value in numeric_features.items():

    print(
        f"{key}: {value}"
    )

    shown += 1

    if shown >= 20:
        break


# ============================================================
# SAVE TEST OUTPUT
# ============================================================

output_dir = os.path.join(

    PROJECT,

    "data",

    "radiomics",

    "features"

)


os.makedirs(
    output_dir,
    exist_ok=True
)


output_file = os.path.join(

    output_dir,

    "test_one_patient_radiomics.csv"

)


test_row = {

    "PatientID":
        patient_id,

    "risk_label":
        row["risk_label"]

}


test_row.update(
    numeric_features
)


test_df = pd.DataFrame(
    [test_row]
)


test_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 70)
print("ONE-PATIENT PYRADIOMICS TEST COMPLETE")
print("=" * 70)

print(
    "Patient:",
    patient_id
)

print(
    "Radiomic features:",
    len(numeric_features)
)

print(
    "NaN:",
    len(nan_features)
)

print(
    "Inf:",
    len(inf_features)
)

print()
print(
    "Output:"
)

print(
    os.path.abspath(
        output_file
    )
)

print()
print("=" * 70)