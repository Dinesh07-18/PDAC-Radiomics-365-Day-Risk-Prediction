import os
import pydicom

ROOT = r"data\radiomics\ct"

uids = set()
n = 0
bad = 0

for root, dirs, files in os.walk(ROOT):
    for file in files:
        if not file.lower().endswith(".dcm"):
            continue

        path = os.path.join(root, file)

        try:
            ds = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                specific_tags=["SeriesInstanceUID"]
            )

            uid = getattr(ds, "SeriesInstanceUID", None)

            if uid:
                uids.add(str(uid))
                n += 1

        except Exception:
            bad += 1

print()
print("=== CT SERIES CHECK ===")
print("DICOM files scanned:", n)
print("Unique CT series found:", len(uids))
print("Unreadable:", bad)