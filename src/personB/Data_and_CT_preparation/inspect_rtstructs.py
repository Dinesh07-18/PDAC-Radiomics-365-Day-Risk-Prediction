import os
import csv
import pydicom

ROOT = r"data\radiomics\annotations"
OUTPUT = r"data\radiomics\metadata\rtstruct_roi_inventory.csv"

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
            rtstruct_uid = str(getattr(ds, "SeriesInstanceUID", ""))

            # Referenced CT series UID
            referenced_ct_uids = set()

            for study_seq in getattr(ds, "ReferencedStudySequence", []):
                for series_seq in getattr(
                    study_seq, "RTReferencedSeriesSequence", []
                ):
                    uid = getattr(series_seq, "SeriesInstanceUID", None)
                    if uid:
                        referenced_ct_uids.add(str(uid))

            # ROI names and types
            roi_names = {}
            for roi in getattr(ds, "StructureSetROISequence", []):
                roi_number = int(roi.ROINumber)
                roi_names[roi_number] = str(
                    getattr(roi, "ROIName", "")
                )

            roi_types = {}
            for obs in getattr(ds, "RTROIObservationsSequence", []):
                roi_number = int(obs.ReferencedROINumber)
                roi_types[roi_number] = str(
                    getattr(obs, "RTROIInterpretedType", "")
                )

            # ROI contour numbers
            contour_roi_numbers = set()
            for contour in getattr(ds, "ROIContourSequence", []):
                contour_roi_numbers.add(int(contour.ReferencedROINumber))

            if not roi_names:
                rows.append({
                    "PatientID": patient_id,
                    "RTSTRUCT_SeriesInstanceUID": rtstruct_uid,
                    "Referenced_CT_SeriesInstanceUID": ";".join(
                        sorted(referenced_ct_uids)
                    ),
                    "ROI_Number": "",
                    "ROI_Name": "",
                    "ROI_Type": "",
                    "Has_Contour": "",
                    "File": path,
                })
                continue

            for roi_number, roi_name in roi_names.items():
                rows.append({
                    "PatientID": patient_id,
                    "RTSTRUCT_SeriesInstanceUID": rtstruct_uid,
                    "Referenced_CT_SeriesInstanceUID": ";".join(
                        sorted(referenced_ct_uids)
                    ),
                    "ROI_Number": roi_number,
                    "ROI_Name": roi_name,
                    "ROI_Type": roi_types.get(roi_number, ""),
                    "Has_Contour": roi_number in contour_roi_numbers,
                    "File": path,
                })

        except Exception as e:
            print("ERROR:", path)
            print(" ", repr(e))

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

fieldnames = [
    "PatientID",
    "RTSTRUCT_SeriesInstanceUID",
    "Referenced_CT_SeriesInstanceUID",
    "ROI_Number",
    "ROI_Name",
    "ROI_Type",
    "Has_Contour",
    "File",
]

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print()
print("=== RTSTRUCT INSPECTION COMPLETE ===")
print("ROI rows:", len(rows))
print("Output:", os.path.abspath(OUTPUT))
print("Unique patients:", len(set(r["PatientID"] for r in rows)))
print(
    "Unique RTSTRUCT series:",
    len(set(r["RTSTRUCT_SeriesInstanceUID"] for r in rows))
)
print()