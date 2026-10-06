import os
import math
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

OUTPUT_DIR = os.path.join(
    PROJECT,
    "data",
    "radiomics",
    "features"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "radiomics_features_46patients.csv"
)

LOG_FILE = os.path.join(
    OUTPUT_DIR,
    "radiomics_extraction_log.csv"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("FULL 46-PATIENT PYRADIOMICS EXTRACTION")
print("=" * 70)


# ============================================================
# CHECK MANIFEST
# ============================================================

if not os.path.exists(MANIFEST):

    raise FileNotFoundError(
        "Final patient manifest not found:\n"
        + MANIFEST
    )


# ============================================================
# LOAD MANIFEST
# ============================================================

df = pd.read_csv(
    MANIFEST
)


print()
print(
    "Patients in manifest:",
    len(df)
)


if len(df) != 46:

    print()
    print(
        "WARNING: Expected 46 patients, "
        "but found:",
        len(df)
    )


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [

    "PatientID",

    "risk_label",

    "ReferencedSeriesInstanceUID",

    "Mask_File"

]


missing_columns = [

    column
    for column in required_columns
    if column not in df.columns

]


if missing_columns:

    raise RuntimeError(
        "Missing required manifest columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# NORMALIZE CT UID
# ============================================================

df[
    "ReferencedSeriesInstanceUID"
] = (

    df[
        "ReferencedSeriesInstanceUID"
    ]
    .astype(str)
    .str.strip()

)


# ============================================================
# BUILD CT SERIES INDEX
# ============================================================

print()
print("=" * 70)
print("BUILDING CT SERIES INDEX")
print("=" * 70)

print(
    "Scanning CT DICOM files..."
)

ct_series = {}

files_scanned = 0
dicom_errors = 0


for root, dirs, files in os.walk(
    CT_ROOT
):

    for filename in files:

        if not filename.lower().endswith(
            ".dcm"
        ):
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


            if modality != "CT":

                continue


            uid = str(
                getattr(
                    ds,
                    "SeriesInstanceUID",
                    ""
                )
            ).strip()


            if not uid:

                continue


            if uid not in ct_series:

                ct_series[uid] = root


        except Exception:

            dicom_errors += 1


print(
    "DICOM files scanned:",
    files_scanned
)

print(
    "CT series indexed:",
    len(ct_series)
)

print(
    "Unreadable files:",
    dicom_errors
)


# ============================================================
# VERIFY ALL REQUIRED CT SERIES EXIST
# ============================================================

missing_ct_series = []


for uid in df[
    "ReferencedSeriesInstanceUID"
].unique():

    if uid not in ct_series:

        missing_ct_series.append(
            uid
        )


if missing_ct_series:

    print()
    print(
        "Missing CT series:",
        len(missing_ct_series)
    )

    for uid in missing_ct_series:

        print(
            uid
        )

    raise RuntimeError(
        "One or more required CT series "
        "could not be found."
    )


# ============================================================
# CREATE PYRADIOMICS EXTRACTOR
# ============================================================

print()
print("=" * 70)
print("INITIALIZING PYRADIOMICS")
print("=" * 70)


settings = {

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
# ENABLE SAME FEATURES AS ONE-PATIENT TEST
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


print()
print(
    "PyRadiomics configuration:"
)

print(
    "binWidth = 25"
)

print(
    "normalize = False"
)

print(
    "minimumROIDimensions = 3"
)

print(
    "correctMask = False"
)

print(
    "Enabled feature classes:"
)

print(
    "  - First Order"
)

print(
    "  - Shape"
)

print(
    "  - GLCM"
)

print(
    "  - GLRLM"
)

print(
    "  - GLSZM"
)

print(
    "  - GLDM"
)

print(
    "  - NGTDM"
)


# ============================================================
# EXTRACTION LOG
# ============================================================

log_records = []


# ============================================================
# RESULTS
# ============================================================

results = []


successful = 0
failed = 0


# ============================================================
# PROCESS EACH PATIENT
# ============================================================

for index, row in df.iterrows():

    patient_number = index + 1


    patient_id = str(
        row["PatientID"]
    ).strip()


    risk_label = int(
        row["risk_label"]
    )


    ct_uid = str(
        row[
            "ReferencedSeriesInstanceUID"
        ]
    ).strip()


    mask_file = str(
        row["Mask_File"]
    ).strip()


    print()
    print("=" * 70)

    print(
        f"[{patient_number}/{len(df)}] "
        f"Patient: {patient_id}"
    )

    print(
        "Risk label:",
        risk_label
    )

    print(
        "CT UID:",
        ct_uid
    )


    # ========================================================
    # GET CT DIRECTORY
    # ========================================================

    ct_series_dir = ct_series[
        ct_uid
    ]


    print(
        "CT directory:"
    )

    print(
        ct_series_dir
    )


    # ========================================================
    # CHECK MASK
    # ========================================================

    if not os.path.exists(mask_file):

        print(
            "ERROR: Mask file not found."
        )

        log_records.append({

            "PatientID":
                patient_id,

            "risk_label":
                risk_label,

            "CT_UID":
                ct_uid,

            "Mask_File":
                mask_file,

            "Status":
                "FAILED_MASK_NOT_FOUND",

            "Feature_Count":
                0

        })

        failed += 1

        continue


    # ========================================================
    # LOAD CT SERIES
    # ========================================================

    try:

        reader = (
            sitk.ImageSeriesReader()
        )


        series_ids = (
            reader.GetGDCMSeriesIDs(
                ct_series_dir
            )
        )


        if not series_ids:

            raise RuntimeError(
                "No DICOM series found."
            )


        selected_series = None


        # ----------------------------------------------------
        # FIND EXACT CT SERIES
        # ----------------------------------------------------

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
                ).strip()


                if uid == ct_uid:

                    selected_series = sid

                    break


            except Exception:

                continue


        if selected_series is None:

            raise RuntimeError(
                "Exact CT SeriesInstanceUID "
                "could not be matched."
            )


        # ----------------------------------------------------
        # GET CT FILES
        # ----------------------------------------------------

        ct_files = (
            reader.GetGDCMSeriesFileNames(
                ct_series_dir,
                selected_series
            )
        )


        if not ct_files:

            raise RuntimeError(
                "No DICOM files found "
                "for selected CT series."
            )


        reader.SetFileNames(
            ct_files
        )


        ct_image = reader.Execute()


        print(
            "CT loaded:"
        )

        print(
            "  Size:",
            ct_image.GetSize()
        )

        print(
            "  Spacing:",
            ct_image.GetSpacing()
        )

        print(
            "  Slices:",
            len(ct_files)
        )


        # ====================================================
        # LOAD MASK
        # ====================================================

        mask_image = sitk.ReadImage(
            mask_file
        )


        print(
            "Mask loaded:"
        )

        print(
            "  Size:",
            mask_image.GetSize()
        )

        print(
            "  Spacing:",
            mask_image.GetSpacing()
        )


        # ====================================================
        # GEOMETRY CHECK
        # ====================================================

        if (
            ct_image.GetSize()
            != mask_image.GetSize()
        ):

            raise RuntimeError(

                "CT and mask dimensions "
                "do not match.\n"

                f"CT = "
                f"{ct_image.GetSize()}\n"

                f"Mask = "
                f"{mask_image.GetSize()}"

            )


        if not all(

            abs(a - b) <= 1e-4

            for a, b in zip(

                ct_image.GetSpacing(),

                mask_image.GetSpacing()

            )

        ):

            raise RuntimeError(
                "CT and mask spacing "
                "do not match."
            )


        if not all(

            abs(a - b) <= 1e-4

            for a, b in zip(

                ct_image.GetOrigin(),

                mask_image.GetOrigin()

            )

        ):

            raise RuntimeError(
                "CT and mask origin "
                "do not match."
            )


        if not all(

            abs(a - b) <= 1e-4

            for a, b in zip(

                ct_image.GetDirection(),

                mask_image.GetDirection()

            )

        ):

            raise RuntimeError(
                "CT and mask direction "
                "do not match."
            )


        print(
            "Geometry: PASS"
        )


        # ====================================================
        # CHECK MASK CONTENT
        # ====================================================

        mask_array = (
            sitk.GetArrayFromImage(
                mask_image
            )
        )


        voxel_count = int(
            (
                mask_array > 0
            ).sum()
        )


        print(
            "Mask voxels:",
            voxel_count
        )


        if voxel_count == 0:

            raise RuntimeError(
                "Mask contains zero voxels."
            )


        # ====================================================
        # EXTRACT FEATURES
        # ====================================================

        print(
            "Extracting radiomic features..."
        )


        result = extractor.execute(

            ct_image,

            mask_image

        )


        # ====================================================
        # KEEP ONLY NUMERIC FEATURES
        # ====================================================

        numeric_features = {}

        nan_count = 0
        inf_count = 0
        non_numeric_count = 0


        for key, value in result.items():

            # Skip PyRadiomics diagnostics.

            if key.startswith(
                "diagnostics_"
            ):

                continue


            try:

                numeric_value = float(
                    value
                )


                if math.isnan(
                    numeric_value
                ):

                    nan_count += 1

                    continue


                if math.isinf(
                    numeric_value
                ):

                    inf_count += 1

                    continue


                numeric_features[key] = (
                    numeric_value
                )


            except Exception:

                non_numeric_count += 1


        feature_count = len(
            numeric_features
        )


        print(
            "Features extracted:",
            feature_count
        )

        print(
            "NaN:",
            nan_count
        )

        print(
            "Inf:",
            inf_count
        )


        # ====================================================
        # CREATE PATIENT RESULT
        # ====================================================

        patient_result = {

            "PatientID":
                patient_id,

            "risk_label":
                risk_label

        }


        # Add clinical metadata if available.

        for column in [

            "age_years",

            "Sex",

            "Tumor Grade",

            "StructureSetLabel",

            "ROI_role",

            "ROIVolume",

            "normalized_phase",

            "ReferencedSeriesDescription"

        ]:

            if column in row.index:

                patient_result[column] = (
                    row[column]
                )


        # Add radiomic features.

        patient_result.update(
            numeric_features
        )


        results.append(
            patient_result
        )


        # ====================================================
        # SAVE AFTER EVERY SUCCESS
        # ====================================================

        current_results = pd.DataFrame(
            results
        )


        current_results.to_csv(

            OUTPUT_FILE,

            index=False

        )


        # ====================================================
        # LOG
        # ====================================================

        log_records.append({

            "PatientID":
                patient_id,

            "risk_label":
                risk_label,

            "CT_UID":
                ct_uid,

            "Mask_File":
                mask_file,

            "Status":
                "SUCCESS",

            "Feature_Count":
                feature_count,

            "NaN_Count":
                nan_count,

            "Inf_Count":
                inf_count,

            "NonNumeric_Count":
                non_numeric_count,

            "Mask_Voxel_Count":
                voxel_count

        })


        # Save log continuously.

        pd.DataFrame(
            log_records
        ).to_csv(

            LOG_FILE,

            index=False

        )


        successful += 1


        print(
            "Status: SUCCESS"
        )


    # ========================================================
    # HANDLE FAILURE
    # ========================================================

    except Exception as e:

        print()
        print(
            "ERROR:"
        )

        print(
            repr(e)
        )


        log_records.append({

            "PatientID":
                patient_id,

            "risk_label":
                risk_label,

            "CT_UID":
                ct_uid,

            "Mask_File":
                mask_file,

            "Status":
                "FAILED: "
                + str(e),

            "Feature_Count":
                0

        })


        pd.DataFrame(
            log_records
        ).to_csv(

            LOG_FILE,

            index=False

        )


        failed += 1


# ============================================================
# FINAL SAVE
# ============================================================

if results:

    final_features = pd.DataFrame(
        results
    )


    final_features.to_csv(

        OUTPUT_FILE,

        index=False

    )

else:

    final_features = pd.DataFrame()


pd.DataFrame(
    log_records
).to_csv(

    LOG_FILE,

    index=False

)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("=== FULL RADIOMICS EXTRACTION COMPLETE ===")
print("=" * 70)

print(
    "Patients in manifest:",
    len(df)
)

print(
    "Successful:",
    successful
)

print(
    "Failed:",
    failed
)


if not final_features.empty:

    print()
    print(
        "Final feature matrix shape:"
    )

    print(
        final_features.shape
    )


    print()
    print(
        "Patients in feature matrix:",
        final_features[
            "PatientID"
        ].nunique()
    )


    print()
    print(
        "Risk distribution:"
    )

    print(
        final_features[
            "risk_label"
        ].value_counts()
        .sort_index()
    )


print()
print(
    "Feature output:"
)

print(
    os.path.abspath(
        OUTPUT_FILE
    )
)


print()
print(
    "Extraction log:"
)

print(
    os.path.abspath(
        LOG_FILE
    )
)


print()
print("=" * 70)

if failed == 0:

    print(
        "ALL PATIENTS EXTRACTED SUCCESSFULLY."
    )

else:

    print(
        "WARNING:",
        failed,
        "PATIENT(S) FAILED."
    )

print("=" * 70)