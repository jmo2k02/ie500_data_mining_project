"""Test download of a single month to validate the approach."""

import os
import sys
import requests
from bs4 import BeautifulSoup

BASE_URL = (
    "https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr="
)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

session = requests.Session()
session.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": BASE_URL,
    }
)

# Step 1: GET page to get tokens
print("Step 1: Fetching page for ASP.NET tokens...")
resp = session.get(BASE_URL, timeout=60)
print(f"  Status: {resp.status_code}")

soup = BeautifulSoup(resp.text, "html.parser")
tokens = {}
for name in [
    "__VIEWSTATE",
    "__VIEWSTATEGENERATOR",
    "__EVENTVALIDATION",
    "__EVENTTARGET",
    "__EVENTARGUMENT",
    "__LASTFOCUS",
    "affiliate",
]:
    tag = soup.find("input", {"name": name})
    if tag:
        tokens[name] = tag.get("value", "")
        print(f"  Token {name}: {len(tokens[name])} chars")

# Step 2: POST for year=2009, month=1
print("\nStep 2: POSTing for 2009-01...")
data = dict(tokens)
data["cboGeography"] = "All"
data["cboYear"] = "2009"
data["cboPeriod"] = "1"
for field in ALL_FIELDS:
    data[field] = "on"
data["chkAllVars"] = "on"
data["chkAllGroups"] = "on"
data["chkDownloadZip"] = "on"
data["btnDownload"] = "Download"

resp = session.post(BASE_URL, data=data, timeout=300, stream=True)
print(f"  Status: {resp.status_code}")
print(f"  Content-Type: {resp.headers.get('Content-Type', 'N/A')}")
print(f"  Content-Disposition: {resp.headers.get('Content-Disposition', 'N/A')}")
print(f"  Content-Length: {resp.headers.get('Content-Length', 'N/A')}")

content_type = resp.headers.get("Content-Type", "")
content_disp = resp.headers.get("Content-Disposition", "")

if (
    "application" in content_type
    or "zip" in content_type
    or "octet" in content_type
    or "attachment" in content_disp
):
    output_path = os.path.join(OUTPUT_DIR, "test_2009_01.zip")
    total_bytes = 0
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            total_bytes += len(chunk)
    print(f"\n  SUCCESS! Downloaded {total_bytes:,} bytes to {output_path}")

    # Try to list zip contents
    import zipfile

    try:
        with zipfile.ZipFile(output_path) as zf:
            print(f"  Zip contents: {zf.namelist()}")
            for name in zf.namelist():
                info = zf.getinfo(name)
                print(f"    {name}: {info.file_size:,} bytes")
    except Exception as e:
        print(f"  Not a valid zip: {e}")
        # Maybe it's a raw CSV?
        with open(output_path, "rb") as f:
            first_bytes = f.read(500)
        print(f"  First bytes: {first_bytes[:200]}")
else:
    # Got HTML back
    print(f"\n  Got HTML response (not a file download)")
    # Check if it's an error or what
    text = resp.text
    print(f"  Response length: {len(text)} chars")
    # Save for debugging
    debug_path = os.path.join(OUTPUT_DIR, "debug_response.html")
    with open(debug_path, "w") as f:
        f.write(text[:50000])
    print(f"  Saved first 50k chars to {debug_path}")

    # Check for error messages
    soup2 = BeautifulSoup(text, "html.parser")
    errors = soup2.find_all(class_=lambda x: x and "error" in x.lower()) if True else []
    if errors:
        for e in errors:
            print(f"  Error: {e.text.strip()}")
