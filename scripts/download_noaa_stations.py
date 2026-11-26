"""
download_noaa_stations.py   (version 2)
Downloads hourly observations for ALL Bangladesh weather stations in NOAA's
global station archive (Global Hourly / ISD) - 38 BMD stations incl. Bhola,
Khepupara, Hatia, Mongla, Chandpur, Serajgonj. Free, no account.

Output (folder noaa_stations):
  noaa_bd_stations.csv        - station id, name, lat, lon, period
  <STATIONID>_<YEAR>.csv      - one file per station per year (2006-2025)
                                wind is in the WND column: "dir,q,type,speed*10,q"

Version 1 used an old NOAA address that now returns an error page; this
version deletes those bad .gz files and uses the current address.

Run:   double-click Get-NOAA.bat
"""
import os, csv, io, time, glob, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "noaa_stations")
os.makedirs(OUT, exist_ok=True)
YEARS = range(2006, 2026)
SOURCES = [
    "https://www.ncei.noaa.gov/data/global-hourly/access/{y}/{sid}.csv",
    "https://noaa-global-hourly-pds.s3.amazonaws.com/{y}/{sid}.csv",
]

# remove the broken files from version 1
for f in glob.glob(os.path.join(OUT, "*.gz")):
    os.remove(f)

def get(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            time.sleep(5 * (attempt + 1))
        except Exception:
            time.sleep(5 * (attempt + 1))
    return None

def looks_ok(data):
    return data is not None and data[:20].lstrip().startswith(b'"STATION"')

# station list (already downloaded by version 1; re-read it)
lst = os.path.join(OUT, "noaa_bd_stations.csv")
with open(lst) as fh:
    stations = list(csv.DictReader(fh))
print(f"{len(stations)} Bangladesh stations")

for s in stations:
    sid = f"{s['usaf']}{s['wban']}"
    got = 0
    for y in YEARS:
        target = os.path.join(OUT, f"{sid}_{y}.csv")
        if os.path.exists(target):
            got += 1; continue
        for src in SOURCES:
            data = get(src.format(y=y, sid=sid))
            if looks_ok(data):
                with open(target, "wb") as fh: fh.write(data)
                got += 1
                break
    print(f"{sid} {s['name']:<26} years downloaded: {got}")
print("Done. Files are in", OUT)
