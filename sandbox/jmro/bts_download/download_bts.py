"""
Download BTS On-Time Performance data for all months in 2009-2018.
Selects ALL available fields and downloads one CSV (zip) per year/month.
"""

import os
import time
import zipfile
import requests
from bs4 import BeautifulSoup

BASE_URL = (
    "https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr="
)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS = range(2009, 2019)  # 2009-2018
MONTHS = range(1, 13)  # 1-12

# All 109 data field checkboxes
ALL_FIELDS = [
    "YEAR",
    "QUARTER",
    "MONTH",
    "DAY_OF_MONTH",
    "DAY_OF_WEEK",
    "FL_DATE",
    "OP_UNIQUE_CARRIER",
    "OP_CARRIER_AIRLINE_ID",
    "OP_CARRIER",
    "TAIL_NUM",
    "OP_CARRIER_FL_NUM",
    "ORIGIN_AIRPORT_ID",
    "ORIGIN_AIRPORT_SEQ_ID",
    "ORIGIN_CITY_MARKET_ID",
    "ORIGIN",
    "ORIGIN_CITY_NAME",
    "ORIGIN_STATE_ABR",
    "ORIGIN_STATE_FIPS",
    "ORIGIN_STATE_NM",
    "ORIGIN_WAC",
    "DEST_AIRPORT_ID",
    "DEST_AIRPORT_SEQ_ID",
    "DEST_CITY_MARKET_ID",
    "DEST",
    "DEST_CITY_NAME",
    "DEST_STATE_ABR",
    "DEST_STATE_FIPS",
    "DEST_STATE_NM",
    "DEST_WAC",
    "CRS_DEP_TIME",
    "DEP_TIME",
    "DEP_DELAY",
    "DEP_DELAY_NEW",
    "DEP_DEL15",
    "DEP_DELAY_GROUP",
    "DEP_TIME_BLK",
    "TAXI_OUT",
    "WHEELS_OFF",
    "WHEELS_ON",
    "TAXI_IN",
    "CRS_ARR_TIME",
    "ARR_TIME",
    "ARR_DELAY",
    "ARR_DELAY_NEW",
    "ARR_DEL15",
    "ARR_DELAY_GROUP",
    "ARR_TIME_BLK",
    "CANCELLED",
    "CANCELLATION_CODE",
    "DIVERTED",
    "CRS_ELAPSED_TIME",
    "ACTUAL_ELAPSED_TIME",
    "AIR_TIME",
    "FLIGHTS",
    "DISTANCE",
    "DISTANCE_GROUP",
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "FIRST_DEP_TIME",
    "TOTAL_ADD_GTIME",
    "LONGEST_ADD_GTIME",
    "DIV_AIRPORT_LANDINGS",
    "DIV_REACHED_DEST",
    "DIV_ACTUAL_ELAPSED_TIME",
    "DIV_ARR_DELAY",
    "DIV_DISTANCE",
    "DIV1_AIRPORT",
    "DIV1_AIRPORT_ID",
    "DIV1_AIRPORT_SEQ_ID",
    "DIV1_WHEELS_ON",
    "DIV1_TOTAL_GTIME",
    "DIV1_LONGEST_GTIME",
    "DIV1_WHEELS_OFF",
    "DIV1_TAIL_NUM",
    "DIV2_AIRPORT",
    "DIV2_AIRPORT_ID",
    "DIV2_AIRPORT_SEQ_ID",
    "DIV2_WHEELS_ON",
    "DIV2_TOTAL_GTIME",
    "DIV2_LONGEST_GTIME",
    "DIV2_WHEELS_OFF",
    "DIV2_TAIL_NUM",
    "DIV3_AIRPORT",
    "DIV3_AIRPORT_ID",
    "DIV3_AIRPORT_SEQ_ID",
    "DIV3_WHEELS_ON",
    "DIV3_TOTAL_GTIME",
    "DIV3_LONGEST_GTIME",
    "DIV3_WHEELS_OFF",
    "DIV3_TAIL_NUM",
    "DIV4_AIRPORT",
    "DIV4_AIRPORT_ID",
    "DIV4_AIRPORT_SEQ_ID",
    "DIV4_WHEELS_ON",
    "DIV4_TOTAL_GTIME",
    "DIV4_LONGEST_GTIME",
    "DIV4_WHEELS_OFF",
    "DIV4_TAIL_NUM",
    "DIV5_AIRPORT",
    "DIV5_AIRPORT_ID",
    "DIV5_AIRPORT_SEQ_ID",
    "DIV5_WHEELS_ON",
    "DIV5_TOTAL_GTIME",
    "DIV5_LONGEST_GTIME",
    "DIV5_WHEELS_OFF",
    "DIV5_TAIL_NUM",
]


def get_form_tokens(session):
    """GET the page and extract ASP.NET hidden form tokens."""
    resp = session.get(BASE_URL, timeout=60)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    form = soup.find("form", {"id": "form1"})
    tokens = {}
    for inp in form.find_all("input", {"type": "hidden"}):
        name = inp.get("name", "")
        val = inp.get("value", "")
        if name:
            tokens[name] = val
    return tokens


def download_month(session, tokens, year, month, retry_count=3):
    """Download data for a single year/month combination."""
    output_zip = os.path.join(OUTPUT_DIR, f"bts_{year}_{month:02d}.zip")
    output_csv = os.path.join(OUTPUT_DIR, f"bts_{year}_{month:02d}.csv")

    # Skip if already downloaded
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 1000:
        print(
            f"  [SKIP] {year}-{month:02d} already exists ({os.path.getsize(output_csv):,} bytes)"
        )
        return True

    for attempt in range(1, retry_count + 1):
        try:
            # Re-fetch tokens for each attempt to get fresh viewstate
            if attempt > 1:
                tokens = get_form_tokens(session)

            # Build POST data
            data = dict(tokens)
            data["cboGeography"] = "All"
            data["cboYear"] = str(year)
            data["cboPeriod"] = str(month)
            for field in ALL_FIELDS:
                data[field] = "on"
            data["btnDownload"] = "Download"

            resp = session.post(BASE_URL, data=data, timeout=300, stream=True)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            content_disp = resp.headers.get("Content-Disposition", "")

            if (
                "application" in content_type
                or "zip" in content_type
                or "attachment" in content_disp
            ):
                # Save the zip file
                with open(output_zip, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
                file_size = os.path.getsize(output_zip)
                print(f"  [OK] Downloaded {year}-{month:02d} ({file_size:,} bytes)")
                return extract_zip(output_zip, year, month)
            else:
                if attempt < retry_count:
                    print(
                        f"  [WARN] {year}-{month:02d} attempt {attempt}: got HTML, retrying..."
                    )
                    tokens = get_form_tokens(session)
                    time.sleep(5)
                else:
                    print(
                        f"  [FAIL] {year}-{month:02d}: server returned HTML instead of file"
                    )
                    return False

        except requests.exceptions.RequestException as e:
            if attempt < retry_count:
                print(
                    f"  [WARN] {year}-{month:02d} attempt {attempt}: {e}, retrying in 10s..."
                )
                time.sleep(10)
            else:
                print(f"  [FAIL] {year}-{month:02d}: {e}")
                return False
    return False


def extract_zip(zip_path, year, month):
    """Extract CSV from the downloaded zip file."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_files = [f for f in zf.namelist() if f.endswith(".csv")]
            if csv_files:
                extracted = zf.extract(csv_files[0], OUTPUT_DIR)
                target = os.path.join(OUTPUT_DIR, f"bts_{year}_{month:02d}.csv")
                if os.path.abspath(extracted) != os.path.abspath(target):
                    if os.path.exists(target):
                        os.remove(target)
                    os.rename(extracted, target)
                size = os.path.getsize(target)
                print(f"  [EXTRACTED] {year}-{month:02d}: {size:,} bytes")
                # Clean up zip
                os.remove(zip_path)
                return True
            else:
                print(
                    f"  [WARN] {year}-{month:02d}: zip contains no CSV: {zf.namelist()}"
                )
                return False
    except zipfile.BadZipFile:
        print(f"  [FAIL] {year}-{month:02d}: corrupt zip file")
        os.remove(zip_path)
        return False


def main():
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": BASE_URL,
        }
    )

    total = len(list(YEARS)) * 12
    success = 0
    failed = []

    print(f"Downloading BTS On-Time Performance data")
    print(f"  Years: 2009-2018 | Months: 1-12 | Total: {total} files")
    print(f"  Fields: {len(ALL_FIELDS)} (all available)")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Get initial tokens
    print("Fetching initial ASP.NET form tokens...")
    tokens = get_form_tokens(session)
    print(f"  Got {len(tokens)} tokens")

    for year in YEARS:
        for month in MONTHS:
            idx = success + len(failed) + 1
            print(f"\n[{idx}/{total}] {year}-{month:02d}")
            ok = download_month(session, tokens, year, month)
            if ok:
                success += 1
            else:
                failed.append(f"{year}-{month:02d}")

            # Re-fetch tokens periodically to avoid stale viewstate
            if idx % 6 == 0:
                try:
                    tokens = get_form_tokens(session)
                except Exception:
                    pass

            # Rate limiting - be respectful
            time.sleep(1)

    print("\n" + "=" * 60)
    print(f"Complete: {success}/{total} files downloaded successfully.")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")
    else:
        print("All downloads completed successfully!")

    # Summary of files
    csv_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv")])
    total_size = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in csv_files)
    print(f"\n{len(csv_files)} CSV files, total size: {total_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    main()
