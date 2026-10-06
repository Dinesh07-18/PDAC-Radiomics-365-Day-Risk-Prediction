import os
import csv
import pydicom

ROOT = r"data\radiomics\annotations"
OUTPUT = r"data\radiomics\metadata\rtstruct_reference_check.csv"

rows = []

for root, dirs, files in os.walk(ROOT):
    for file in files:
        if not file.lower().endswith(".dcm"):
            continue

        path = os.path.join(root, file)

        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)

            if getattr(ds, "Modality", "") != "RTSTRUCT":
                continue

            patient_id = str(getattr(ds, "PatientID", ""))
            rt_uid = str(getattr(ds, "SeriesInstanceUID", ""))

            referenced_ct_uids = set()

            for frame_ref in getattr(
                ds, "ReferencedFrameOfReferenceSequence", []
            ):
                for study_ref in getattr(
                    frame_ref, "RTReferencedStudySequence", []
                ):
                    for series_ref in getattr(
                        study_ref, "RTReferencedSeriesSequence", []
                    ):
                        uid = getattr(series_ref, "SeriesInstanceUID", None)
                        if uid:
                            referenced_ct_uids.add(str(uid))

            roi_names = {}

            for roi in getattr(ds, "StructureSetROISequence", []):
                roi_names[int(roi.ROINumber)] = str(
                    getattr(roi, "ROIName", "")
                )

            contour_numbers = set()

            for contour in getattr(ds, "ROIContourSequence", []):
                contour_numbers.add(
                    int(contour.ReferencedROINumber)
                )

            for roi_number, roi_name in roi_names.items():
                rows.append({
                    "PatientID": patient_id,
                    "RTSTRUCT_SeriesInstanceUID": rt_uid,
                    "Referenced_CT_SeriesInstanceUID": ";".join(
                        sorted(referenced_ct_uids)
                    ),
                    "ROI_Number": roi_number,
                    "ROI_Name": roi_name,
                    "Has_Contour": roi_number in contour_numbers,
                    "RTSTRUCT_File": path,
                })

        except Exception as e:
            print("ERROR:", path)
            print(repr(e))

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

fields = [
    "PatientID",
    "RTSTRUCT_SeriesInstanceUID",
    "Referenced_CT_SeriesInstanceUID",
    "ROI_Number",
    "ROI_Name",
    "Has_Contour",
    "RTSTRUCT_File",
]

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print()
print("=== RTSTRUCT REFERENCE CHECK COMPLETE ===")
print("Rows:", len(rows))
print("Output:", os.path.abspath(OUTPUT))

unique_refs = set()

for r in rows:
    if r["Referenced_CT_SeriesInstanceUID"]:
        for uid in r["Referenced_CT_SeriesInstanceUID"].split(";"):
            unique_refs.add(uid)

print("Unique referenced CT series:", len(unique_refs))
print()

empty = sum(
    1 for r in rows
    if not r["Referenced_CT_SeriesInstanceUID"]
)

print("Rows with missing CT reference:", empty)