import csv
import io
import time
import zipfile
from pathlib import Path

import requests
import urllib3

# Disable the SSL warning caused by the TCIA certificate chain
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ============================================================
# CONFIGURATION
# ============================================================

MANIFEST = Path(
    "data/radiomics/metadata/personB_ct_download_manifest.csv"
)

OUT_ROOT = Path(
    "data/radiomics/ct"
)

LOG_FILE = Path(
    "data/radiomics/metadata/ct_download_log.csv"
)

BASE_URL = (
    "https://services.cancerimagingarchive.net/"
    "services/v2/TCIA/query/getImage"
)

# Maximum time allowed for one download attempt
DOWNLOAD_TIMEOUT = 900

# Number of attempts for each CT series
MAX_ATTEMPTS = 3

# Seconds to wait between failed attempts
RETRY_DELAY = 10


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_name(value):
    """
    Convert a string into a Windows-safe folder/file name.
    """
    return "".join(
        c if c.isalnum() or c in "-_." else "_"
        for c in str(value)
    )


# ============================================================
# MAIN DOWNLOAD FUNCTION
# ============================================================

def main():

    # --------------------------------------------------------
    # Check manifest
    # --------------------------------------------------------

    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST}"
        )

    # Create required directories
    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Read manifest
    # --------------------------------------------------------

    with MANIFEST.open(
        newline="",
        encoding="utf-8-sig"
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    # --------------------------------------------------------
    # Remove duplicate CT SeriesInstanceUIDs
    # --------------------------------------------------------

    unique_series = {}

    for row in rows:

        uid = row[
            "ReferencedSeriesInstanceUID"
        ]

        if uid:
            unique_series.setdefault(
                uid,
                row
            )

    print()
    print("=" * 60)
    print("TCIA CPTAC-PDA CT DOWNLOAD")
    print("=" * 60)
    print()

    print(
        f"Manifest rows: {len(rows)}"
    )

    print(
        f"Unique CT series to download: "
        f"{len(unique_series)}"
    )

    print()

    # --------------------------------------------------------
    # Download log
    # --------------------------------------------------------

    log_rows = []

    total = len(unique_series)

    # --------------------------------------------------------
    # Process every CT series
    # --------------------------------------------------------

    for index, (uid, row) in enumerate(
        unique_series.items(),
        start=1
    ):

        patient = safe_name(
            row["PatientID"]
        )

        description = safe_name(
            row["ReferencedSeriesDescription"]
        )[:80]

        # Folder for this CT series
        series_dir = (
            OUT_ROOT
            / patient
            / uid
        )

        series_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Check whether this series was already downloaded
        # ----------------------------------------------------

        existing_dicom = list(
            series_dir.rglob("*.dcm")
        )

        if existing_dicom:

            print(
                f"[{index}/{total}] "
                f"SKIP {patient} | "
                f"already has "
                f"{len(existing_dicom)} DICOM files"
            )

            log_rows.append({
                "PatientID": patient,
                "SeriesInstanceUID": uid,
                "status": "already_present",
                "files": len(existing_dicom),
                "description": row[
                    "ReferencedSeriesDescription"
                ],
            })

            continue

        # ----------------------------------------------------
        # Start download
        # ----------------------------------------------------

        print(
            f"[{index}/{total}] "
            f"Downloading {patient} | "
            f"{description}"
        )

        downloaded = False
        last_error = ""

        # ----------------------------------------------------
        # Retry loop
        # ----------------------------------------------------

        for attempt in range(
            1,
            MAX_ATTEMPTS + 1
        ):

            try:

                print(
                    f"    Attempt "
                    f"{attempt}/{MAX_ATTEMPTS}"
                )

                response = requests.get(
                    BASE_URL,
                    params={
                        "SeriesInstanceUID": uid
                    },
                    timeout=DOWNLOAD_TIMEOUT,
                    verify=False
                )

                response.raise_for_status()

                # ------------------------------------------------
                # Make sure TCIA returned a ZIP file
                # ------------------------------------------------

                if not response.content.startswith(
                    b"PK"
                ):

                    raise RuntimeError(
                        "TCIA did not return a ZIP archive."
                    )

                # ------------------------------------------------
                # Extract ZIP
                # ------------------------------------------------

                with zipfile.ZipFile(
                    io.BytesIO(
                        response.content
                    )
                ) as archive:

                    archive.extractall(
                        series_dir
                    )

                # ------------------------------------------------
                # Count extracted files
                # ------------------------------------------------

                files = [
                    p
                    for p in series_dir.rglob("*")
                    if p.is_file()
                ]

                print(
                    f"    SUCCESS: "
                    f"extracted "
                    f"{len(files)} files"
                )

                log_rows.append({
                    "PatientID": patient,
                    "SeriesInstanceUID": uid,
                    "status": "downloaded",
                    "files": len(files),
                    "description": row[
                        "ReferencedSeriesDescription"
                    ],
                })

                downloaded = True

                break

            # ----------------------------------------------------
            # Network/request error
            # ----------------------------------------------------

            except requests.exceptions.RequestException as e:

                last_error = str(e)

                print(
                    f"    Attempt "
                    f"{attempt} failed:"
                )

                print(
                    f"    {e}"
                )

                if attempt < MAX_ATTEMPTS:

                    print(
                        f"    Retrying in "
                        f"{RETRY_DELAY} seconds..."
                    )

                    time.sleep(
                        RETRY_DELAY
                    )

            # ----------------------------------------------------
            # Any other error
            # ----------------------------------------------------

            except Exception as e:

                last_error = str(e)

                print(
                    f"    Attempt "
                    f"{attempt} failed:"
                )

                print(
                    f"    {e}"
                )

                if attempt < MAX_ATTEMPTS:

                    print(
                        f"    Retrying in "
                        f"{RETRY_DELAY} seconds..."
                    )

                    time.sleep(
                        RETRY_DELAY
                    )

        # --------------------------------------------------------
        # If all attempts failed
        # --------------------------------------------------------

        if not downloaded:

            print(
                f"    FAILED after "
                f"{MAX_ATTEMPTS} attempts"
            )

            log_rows.append({
                "PatientID": patient,
                "SeriesInstanceUID": uid,
                "status": "ERROR",
                "files": 0,
                "description": row[
                    "ReferencedSeriesDescription"
                ],
            })

        # Small delay before next series
        time.sleep(1)

    # ============================================================
    # SAVE DOWNLOAD LOG
    # ============================================================

    with LOG_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "PatientID",
                "SeriesInstanceUID",
                "status",
                "files",
                "description",
            ]
        )

        writer.writeheader()

        writer.writerows(
            log_rows
        )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    downloaded_count = sum(
        1
        for row in log_rows
        if row["status"] == "downloaded"
    )

    existing_count = sum(
        1
        for row in log_rows
        if row["status"] == "already_present"
    )

    failed_count = sum(
        1
        for row in log_rows
        if row["status"] == "ERROR"
    )

    print()
    print("=" * 60)
    print("DOWNLOAD FINISHED")
    print("=" * 60)

    print(
        f"Successfully downloaded: "
        f"{downloaded_count}"
    )

    print(
        f"Already present: "
        f"{existing_count}"
    )

    print(
        f"Failed: "
        f"{failed_count}"
    )

    print()
    print(
        f"Log file:"
    )

    print(
        LOG_FILE
    )

    print()
    print(
        "CT files are stored in:"
    )

    print(
        OUT_ROOT
    )

    print(
        "=" * 60
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()