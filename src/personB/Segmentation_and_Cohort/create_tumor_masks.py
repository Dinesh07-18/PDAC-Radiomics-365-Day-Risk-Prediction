import os
import csv
import pydicom
import SimpleITK as sitk
from rt_utils import RTStructBuilder


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT = os.path.abspath(".")

CT_ROOT = os.path.join(
    PROJECT,
    "data",
    "radiomics",
    "ct"
)

RTSTRUCT_ROOT = os.path.join(
    PROJECT,
    "data",
    "radiomics",
    "annotations"
)

METADATA = os.path.join(
    PROJECT,
    "data",
    "radiomics",
    "metadata",
    "rtstruct_reference_check.csv"
)

MASK_ROOT = os.path.join(
    PROJECT,
    "data",
    "radiomics",
    "masks"
)

AUDIT_FILE = os.path.join(
    PROJECT,
    "data",
    "radiomics",
    "metadata",
    "mask_generation_audit.csv"
)

os.makedirs(
    MASK_ROOT,
    exist_ok=True
)


# ============================================================
# FIND ALL RTSTRUCT FILES
# ============================================================

print()
print("=" * 70)
print("SCANNING RTSTRUCT FILES")
print("=" * 70)

rtstruct_by_uid = {}

for root, dirs, files in os.walk(RTSTRUCT_ROOT):

    for file in files:

        if not file.lower().endswith(".dcm"):
            continue

        path = os.path.join(root, file)

        try:

            ds = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                specific_tags=[
                    "Modality",
                    "SeriesInstanceUID",
                    "PatientID"
                ]
            )

            modality = str(
                getattr(
                    ds,
                    "Modality",
                    ""
                )
            )

            if modality != "RTSTRUCT":
                continue

            uid = getattr(
                ds,
                "SeriesInstanceUID",
                None
            )

            if uid:

                rtstruct_by_uid[str(uid)] = path

        except Exception as e:

            print(
                "Could not read RTSTRUCT:",
                path
            )

            print(
                "Error:",
                e
            )


print(
    "RTSTRUCT files found:",
    len(rtstruct_by_uid)
)


# ============================================================
# FIND ALL CT SERIES
# ============================================================

print()
print("=" * 70)
print("SCANNING CT DICOM FILES")
print("=" * 70)

print(
    "This may take a little while..."
)

ct_series = {}

for root, dirs, files in os.walk(CT_ROOT):

    for file in files:

        if not file.lower().endswith(".dcm"):
            continue

        path = os.path.join(root, file)

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

            uid = getattr(
                ds,
                "SeriesInstanceUID",
                None
            )

            if uid:

                uid = str(uid)

                if uid not in ct_series:

                    ct_series[uid] = root

        except Exception:
            pass


print(
    "CT series found:",
    len(ct_series)
)


# ============================================================
# CHECK METADATA FILE
# ============================================================

if not os.path.exists(METADATA):

    raise FileNotFoundError(
        "Metadata file not found:\n"
        + METADATA
    )


# ============================================================
# READ RTSTRUCT REFERENCE METADATA
# ============================================================

print()
print("=" * 70)
print("READING RTSTRUCT REFERENCE METADATA")
print("=" * 70)

with open(
    METADATA,
    "r",
    encoding="utf-8-sig"
) as f:

    reader = csv.DictReader(f)

    records = list(reader)


print(
    "Metadata records:",
    len(records)
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [

    "PatientID",

    "RTSTRUCT_SeriesInstanceUID",

    "Referenced_CT_SeriesInstanceUID",

    "ROI_Number",

    "ROI_Name",

    "RTSTRUCT_File"

]


if records:

    available_columns = records[0].keys()

    missing_columns = [

        col
        for col in required_columns
        if col not in available_columns

    ]

    if missing_columns:

        raise RuntimeError(
            "The metadata CSV is missing these "
            "required columns:\n"
            + "\n".join(missing_columns)
        )


# ============================================================
# PROCESS RTSTRUCTS
# ============================================================

audit = []

success = 0
failed = 0


for i, record in enumerate(
    records,
    start=1
):

    print()
    print("=" * 70)

    # --------------------------------------------------------
    # BASIC METADATA
    # --------------------------------------------------------

    patient_id = str(
        record.get(
            "PatientID",
            ""
        )
    ).strip()

    rt_uid = str(
        record.get(
            "RTSTRUCT_SeriesInstanceUID",
            ""
        )
    ).strip()

    ct_uid = str(
        record.get(
            "Referenced_CT_SeriesInstanceUID",
            ""
        )
    ).strip()

    roi_number = str(
        record.get(
            "ROI_Number",
            ""
        )
    ).strip()

    roi_name = str(
        record.get(
            "ROI_Name",
            ""
        )
    ).strip()

    rt_file = str(
        record.get(
            "RTSTRUCT_File",
            ""
        )
    ).strip()


    print(
        f"[{i}/{len(records)}] Patient: "
        f"{patient_id}"
    )

    print(
        "ROI:",
        roi_name
    )

    print(
        "RTSTRUCT UID:",
        rt_uid
    )

    print(
        "CT UID:",
        ct_uid
    )


    # ========================================================
    # VALIDATE RTSTRUCT
    # ========================================================

    if rt_uid not in rtstruct_by_uid:

        print(
            "ERROR: RTSTRUCT file not found."
        )

        audit.append({

            "PatientID":
                patient_id,

            "RTSTRUCT_SeriesInstanceUID":
                rt_uid,

            "Referenced_CT_SeriesInstanceUID":
                ct_uid,

            "ROI_Number":
                roi_number,

            "ROI_Name":
                roi_name,

            "RTSTRUCT_File":
                rt_file,

            "CT_Directory":
                "",

            "Mask_File":
                "",

            "Status":
                "FAILED_RTSTRUCT_NOT_FOUND"

        })

        failed += 1

        continue


    # Always use the actual discovered RTSTRUCT file.
    actual_rt_file = rtstruct_by_uid[rt_uid]


    # ========================================================
    # VALIDATE CT
    # ========================================================

    if ct_uid not in ct_series:

        print(
            "ERROR: Referenced CT series not found."
        )

        audit.append({

            "PatientID":
                patient_id,

            "RTSTRUCT_SeriesInstanceUID":
                rt_uid,

            "Referenced_CT_SeriesInstanceUID":
                ct_uid,

            "ROI_Number":
                roi_number,

            "ROI_Name":
                roi_name,

            "RTSTRUCT_File":
                actual_rt_file,

            "CT_Directory":
                "",

            "Mask_File":
                "",

            "Status":
                "FAILED_CT_NOT_FOUND"

        })

        failed += 1

        continue


    ct_dir = ct_series[ct_uid]


    # ========================================================
    # LOAD RTSTRUCT
    # ========================================================

    try:

        rtstruct = RTStructBuilder.create_from(

            dicom_series_path=ct_dir,

            rt_struct_path=actual_rt_file

        )


        roi_names = rtstruct.get_roi_names()


        print(
            "ROIs available:",
            roi_names
        )


        # ====================================================
        # FIND CORRECT ROI
        # ====================================================

        selected_roi = None


        # -----------------------------------------------
        # EXACT MATCH
        # -----------------------------------------------

        if roi_name in roi_names:

            selected_roi = roi_name


        # -----------------------------------------------
        # NORMALIZED MATCH
        # -----------------------------------------------

        if selected_roi is None:

            target = " ".join(
                roi_name.lower().split()
            )

            for candidate in roi_names:

                normalized = " ".join(
                    candidate.lower().split()
                )

                if normalized == target:

                    selected_roi = candidate

                    break


        # -----------------------------------------------
        # SINGLE ROI FALLBACK
        # -----------------------------------------------

        if (
            selected_roi is None
            and len(roi_names) == 1
        ):

            selected_roi = roi_names[0]


        # -----------------------------------------------
        # ROI NOT FOUND
        # -----------------------------------------------

        if selected_roi is None:

            print(
                "ERROR: Could not identify ROI."
            )

            audit.append({

                "PatientID":
                    patient_id,

                "RTSTRUCT_SeriesInstanceUID":
                    rt_uid,

                "Referenced_CT_SeriesInstanceUID":
                    ct_uid,

                "ROI_Number":
                    roi_number,

                "ROI_Name":
                    roi_name,

                "RTSTRUCT_File":
                    actual_rt_file,

                "CT_Directory":
                    ct_dir,

                "Mask_File":
                    "",

                "Status":
                    "FAILED_ROI_NOT_FOUND"

            })

            failed += 1

            continue


        print(
            "Selected ROI:",
            selected_roi
        )


        # ====================================================
        # GENERATE 3D MASK
        # ====================================================

        print(
            "Generating 3D mask..."
        )


        # rt-utils 1.2.7 expects ROI name
        # as a positional argument.

        mask = rtstruct.get_roi_mask_by_name(
            selected_roi
        )


        # ====================================================
        # LOAD EXACT CT SERIES
        # ====================================================

        reader = sitk.ImageSeriesReader()


        series_ids = (
            reader.GetGDCMSeriesIDs(
                ct_dir
            )
        )


        if not series_ids:

            raise RuntimeError(
                "SimpleITK could not find "
                "a DICOM series in:\n"
                + ct_dir
            )


        selected_series = None


        # ====================================================
        # MATCH EXACT CT SERIES INSTANCE UID
        # ====================================================

        for sid in series_ids:

            files = (
                reader.GetGDCMSeriesFileNames(
                    ct_dir,
                    sid
                )
            )


            if not files:
                continue


            try:

                first_ds = pydicom.dcmread(
                    files[0],
                    stop_before_pixels=True,
                    specific_tags=[
                        "SeriesInstanceUID"
                    ]
                )


                found_uid = str(
                    getattr(
                        first_ds,
                        "SeriesInstanceUID",
                        ""
                    )
                )


                if found_uid == ct_uid:

                    selected_series = sid

                    break


            except Exception:

                pass


        if selected_series is None:

            raise RuntimeError(
                "Could not match the referenced "
                "CT SeriesInstanceUID inside "
                "the CT directory."
            )


        # ====================================================
        # READ CT
        # ====================================================

        ct_files = (
            reader.GetGDCMSeriesFileNames(
                ct_dir,
                selected_series
            )
        )


        if not ct_files:

            raise RuntimeError(
                "No DICOM files found for "
                "the selected CT series."
            )


        reader.SetFileNames(
            ct_files
        )


        ct_image = reader.Execute()


        print(
            "CT size:",
            ct_image.GetSize()
        )


        print(
            "RT-Utils mask shape:",
            mask.shape
        )


        # ====================================================
        # CHECK MASK DIMENSIONS
        # ====================================================

        ct_size = ct_image.GetSize()


        # SimpleITK:
        #
        # GetSize() =
        # (X, Y, Z)
        #
        # RT-Utils:
        #
        # mask =
        # (Rows, Columns, Slices)
        #
        # Therefore expected RT-Utils shape is:
        #
        # (Y, X, Z)

        expected_rtutils_shape = (

            ct_size[1],

            ct_size[0],

            ct_size[2]

        )


        if mask.shape != expected_rtutils_shape:

            raise RuntimeError(

                "RT-Utils mask dimensions do not "
                "match the CT dimensions.\n"

                f"Mask shape = {mask.shape}\n"

                f"Expected RT-Utils shape = "
                f"{expected_rtutils_shape}"

            )


        # ====================================================
        # CONVERT AXES
        # ====================================================

        # RT-Utils:
        #
        # (Rows, Columns, Slices)
        #
        # becomes:
        #
        # (Slices, Rows, Columns)
        #
        # for SimpleITK.

        mask_for_sitk = mask.transpose(
            2,
            0,
            1
        )


        print(
            "Mask shape after transpose:",
            mask_for_sitk.shape
        )


        # ====================================================
        # CREATE SIMPLEITK MASK
        # ====================================================

        mask_image = sitk.GetImageFromArray(
            mask_for_sitk.astype("uint8")
        )


        print(
            "SimpleITK mask size:",
            mask_image.GetSize()
        )


        # ====================================================
        # FINAL SIZE CHECK
        # ====================================================

        if (
            mask_image.GetSize()
            != ct_image.GetSize()
        ):

            raise RuntimeError(

                "Final mask size does not "
                "match CT size.\n"

                f"Mask = "
                f"{mask_image.GetSize()}\n"

                f"CT = "
                f"{ct_image.GetSize()}"

            )


        # ====================================================
        # COPY CT GEOMETRY
        # ====================================================

        mask_image.SetSpacing(
            ct_image.GetSpacing()
        )

        mask_image.SetOrigin(
            ct_image.GetOrigin()
        )

        mask_image.SetDirection(
            ct_image.GetDirection()
        )


        # ====================================================
        # CREATE PATIENT DIRECTORY
        # ====================================================

        patient_dir = os.path.join(
            MASK_ROOT,
            patient_id
        )


        os.makedirs(
            patient_dir,
            exist_ok=True
        )


        # ====================================================
        # SAFE ROI NAME
        # ====================================================

        safe_roi = (

            selected_roi

            .replace(
                " ",
                "_"
            )

            .replace(
                "/",
                "_"
            )

            .replace(
                "\\",
                "_"
            )

            .replace(
                ":",
                "_"
            )

        )


        # ====================================================
        # SAFE CT UID
        # ====================================================

        # The CT UID is included so that:
        #
        # arterial mask != venous mask
        #
        # and no file can accidentally overwrite
        # another CT/mask pair.

        safe_ct_uid = (

            ct_uid

            .replace(
                ".",
                "_"
            )

        )


        # ====================================================
        # UNIQUE MASK FILE
        # ====================================================

        mask_file = os.path.join(

            patient_dir,

            f"{patient_id}_"
            f"{safe_roi}_"
            f"{safe_ct_uid}.nii.gz"

        )


        # ====================================================
        # SAVE MASK
        # ====================================================

        sitk.WriteImage(

            mask_image,

            mask_file

        )


        print()
        print(
            "MASK SAVED:"
        )

        print(
            mask_file
        )


        # ====================================================
        # VERIFY FILE EXISTS
        # ====================================================

        if not os.path.exists(mask_file):

            raise RuntimeError(
                "Mask file was not created."
            )


        # ====================================================
        # MASK VOXEL COUNT
        # ====================================================

        voxel_count = int(
            sitk.GetArrayFromImage(
                mask_image
            ).sum()
        )


        print(
            "Mask voxel count:",
            voxel_count
        )


        if voxel_count == 0:

            print(
                "WARNING: Mask contains zero voxels."
            )


        # ====================================================
        # AUDIT SUCCESS
        # ====================================================

        audit.append({

            "PatientID":
                patient_id,

            "RTSTRUCT_SeriesInstanceUID":
                rt_uid,

            "Referenced_CT_SeriesInstanceUID":
                ct_uid,

            "ROI_Number":
                roi_number,

            "ROI_Name":
                selected_roi,

            "RTSTRUCT_File":
                actual_rt_file,

            "CT_Directory":
                ct_dir,

            "Mask_File":
                mask_file,

            "Status":
                "SUCCESS"

        })


        success += 1


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        print()
        print(
            "ERROR while generating mask:"
        )

        print(
            repr(e)
        )


        audit.append({

            "PatientID":
                patient_id,

            "RTSTRUCT_SeriesInstanceUID":
                rt_uid,

            "Referenced_CT_SeriesInstanceUID":
                ct_uid,

            "ROI_Number":
                roi_number,

            "ROI_Name":
                roi_name,

            "RTSTRUCT_File":
                actual_rt_file,

            "CT_Directory":
                ct_dir,

            "Mask_File":
                "",

            "Status":
                "FAILED_EXCEPTION: "
                + str(e)

        })


        failed += 1


# ============================================================
# SAVE AUDIT FILE
# ============================================================

fields = [

    "PatientID",

    "RTSTRUCT_SeriesInstanceUID",

    "Referenced_CT_SeriesInstanceUID",

    "ROI_Number",

    "ROI_Name",

    "RTSTRUCT_File",

    "CT_Directory",

    "Mask_File",

    "Status"

]


with open(
    AUDIT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()

    writer.writerows(audit)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("=== MASK GENERATION COMPLETE ===")
print("=" * 70)

print(
    "Total RTSTRUCTs:",
    len(records)
)

print(
    "Successful:",
    success
)

print(
    "Failed:",
    failed
)

print()

print(
    "Masks directory:"
)

print(
    os.path.abspath(
        MASK_ROOT
    )
)

print()

print(
    "Audit file:"
)

print(
    os.path.abspath(
        AUDIT_FILE
    )
)

print()
print("=" * 70)

if failed == 0:

    print(
        "ALL MASKS GENERATED SUCCESSFULLY."
    )

else:

    print(
        "WARNING:",
        failed,
        "MASK(S) FAILED."
    )

print("=" * 70)