"""
download_station_data.py
Downloads observed hourly wind from Bangladesh airport weather stations
(METAR reports) from the Iowa Environmental Mesonet archive - free, no account.

Output (in the folder station_data):
  bd_stations.csv            - station code, name, latitude, longitude
  metar_<CODE>.csv           - every report: time (UTC), wind direction (deg),
                               wind speed (knots), gust (knots)

Run:   python download_station_data.py
Then tell Claude - it will clean the data and compare it with ERA5.
"""
import os, time, json, urllib.request, urllib.parse, csv

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "station_data")
os.makedirs(OUT, exist_ok=True)

START, END = "2006-01-01", "2026-01-01"
BASE = "https://mesonet.agron.iastate.edu"

def get(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            print("   retrying after error:", e)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("could not download " + url)

# 1) list of Bangladesh stations in the archive
gj = json.loads(get(BASE + "/geojson/network/BD__ASOS.geojson"))
stations = []
for f in gj["features"]:
    p = f["properties"]; lon, lat = f["geometry"]["coordinates"][:2]
    stations.append({"code": p["sid"], "name": p.get("sname", ""), "lat": lat, "lon": lon})
with open(os.path.join(OUT, "bd_stations.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["code", "name", "lat", "lon"])
    w.writeheader(); w.writerows(stations)
print(f"{len(stations)} stations found:", ", ".join(s["code"] for s in stations))

# 2) wind reports, one station at a time (the server limits big requests)
for s in stations:
    target = os.path.join(OUT, f"metar_{s['code']}.csv")
    if os.path.exists(target) and os.path.getsize(target) > 1000:
        print("skip", s["code"], "(already downloaded)"); continue
    q = urllib.parse.urlencode([
        ("station", s["code"]), ("data", "drct"), ("data", "sknt"), ("data", "gust"),
        ("sts", START + "T00:00:00Z"), ("ets", END + "T00:00:00Z"),
        ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "yes"),
        ("missing", "empty"), ("trace", "empty"), ("report_type", "3"), ("report_type", "4"),
    ])
    print("downloading", s["code"], s["name"], "...")
    text = get(BASE + "/cgi-bin/request/asos.py?" + q)
    with open(target, "w", encoding="utf-8") as fh:
        fh.write(text)
    print("   rows:", text.count("\n") - 1)
    time.sleep(5)
print("Done. Files are in", OUT)
