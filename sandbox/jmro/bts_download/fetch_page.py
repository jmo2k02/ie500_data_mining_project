"""Fetch the BTS page and analyze its form structure."""

import requests
from bs4 import BeautifulSoup

url = "https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr="

session = requests.Session()
session.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
)

resp = session.get(url)
print(f"Status: {resp.status_code}")

soup = BeautifulSoup(resp.text, "html.parser")

# Find year dropdown
year_select = soup.find(
    "select", {"id": lambda x: x and "year" in x.lower()}
) or soup.find("select", {"name": lambda x: x and "year" in x.lower()})
if year_select:
    print(
        f"\n=== Year dropdown: id={year_select.get('id')}, name={year_select.get('name')} ==="
    )
    options = year_select.find_all("option")
    for opt in options:
        print(f"  value='{opt.get('value')}' text='{opt.text.strip()}'")

# Find period/month dropdown
period_select = soup.find(
    "select", {"id": lambda x: x and ("period" in x.lower() or "month" in x.lower())}
) or soup.find(
    "select", {"name": lambda x: x and ("period" in x.lower() or "month" in x.lower())}
)
if period_select:
    print(
        f"\n=== Period dropdown: id={period_select.get('id')}, name={period_select.get('name')} ==="
    )
    options = period_select.find_all("option")
    for opt in options:
        print(f"  value='{opt.get('value')}' text='{opt.text.strip()}'")

# Find all select elements if above didn't work
print("\n=== All <select> elements ===")
for sel in soup.find_all("select"):
    print(
        f"  id={sel.get('id')}, name={sel.get('name')}, options_count={len(sel.find_all('option'))}"
    )
    for opt in sel.find_all("option")[:5]:
        print(f"    value='{opt.get('value')}' text='{opt.text.strip()}'")
    if len(sel.find_all("option")) > 5:
        print(f"    ... and {len(sel.find_all('option')) - 5} more")

# Find all checkboxes (fields)
checkboxes = soup.find_all("input", {"type": "checkbox"})
print(f"\n=== Checkboxes (fields): {len(checkboxes)} ===")
for cb in checkboxes:
    print(f"  name={cb.get('name')}, id={cb.get('id')}, checked={cb.get('checked')}")

# Find hidden fields (__VIEWSTATE, __EVENTVALIDATION, etc.)
print("\n=== Hidden fields ===")
for hidden in soup.find_all("input", {"type": "hidden"}):
    name = hidden.get("name", "")
    val = hidden.get("value", "")
    print(f"  name={name}, value_len={len(val)}")

# Find submit/download button
print("\n=== Buttons/Submits ===")
for btn in soup.find_all("input", {"type": ["submit", "button"]}):
    print(f"  name={btn.get('name')}, id={btn.get('id')}, value={btn.get('value')}")
for btn in soup.find_all("button"):
    print(f"  name={btn.get('name')}, id={btn.get('id')}, text={btn.text.strip()}")
