"""
download_era5_hourly.py   (version 5 - safe to start many times)
Downloads hourly ERA5 wind (10 m and 100 m, u and v) for Bangladesh,
2016-2025, into the folder era5_hourly, 6 months per request.

What is new in version 5
  1. It asks Copernicus first. Before sending anything it reads your request
     list on the Copernicus website:
       - a request still waiting/running  -> it waits for that one, sends nothing new
       - two or more waiting for the SAME months -> keeps the oldest, cancels the rest
       - a finished request not yet saved -> downloads and saves it
     It keeps up to 3 requests waiting at Copernicus at the same time (their limit),
     each for different months, and sends a new one whenever one finishes.
  2. Only one copy can run. If you double-click Get-Data.bat while it is already
     running, the second window says so and closes.
  3. Internet drops are fine: it keeps retrying every 2 minutes.
Files are named by what is actually inside them: era5_wind_YYYY_MMtoMM.nc

Run:  double-click Get-Data.bat
"""
import os, re, sys, json, time, glob
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "era5_hourly")
os.makedirs(OUT, exist_ok=True)
LEDGER = os.path.join(OUT, "_jobs.json")
YEARS = range(2016, 2026)          # 10 years: 2016-2025
BLOCK = 6                          # months per request (6 is accepted by Copernicus)
MAX_ACTIVE = 3                     # Copernicus lets you have 3 requests waiting at once
DATASET = "reanalysis-era5-single-levels"
AREA = [26.7, 88.0, 20.5, 92.7]
VARS = ["10m_u_component_of_wind", "10m_v_component_of_wind",
        "100m_u_component_of_wind", "100m_v_component_of_wind"]
WAIT = 300                         # seconds between status checks

def say(*a): print(time.strftime("%d-%b %H:%M"), *a, flush=True)

# ---------- only one copy at a time ----------
_lockfile = open(os.path.join(OUT, "_running.lock"), "a+")
try:
    import msvcrt
    msvcrt.locking(_lockfile.fileno(), msvcrt.LK_NBLCK, 1)
except ImportError:
    pass
except OSError:
    print("\nThe ERA5 download is ALREADY RUNNING in another window. Nothing to do here.")
    time.sleep(8); sys.exit(0)

# ---------- Copernicus connection ----------
cfg = {}
with open(os.path.join(HERE, ".cdsapirc")) as f:
    for line in f:
        if ":" in line:
            k, v = line.split(":", 1); cfg[k.strip()] = v.strip()
API = cfg["url"].rstrip("/") + "/retrieve/v1"
S = requests.Session(); S.headers["PRIVATE-TOKEN"] = cfg["key"]

def call(method, path, **kw):
    """HTTP call that survives internet drops (retries every 2 min)."""
    url = path if path.startswith("http") else API + path
    while True:
        try:
            r = S.request(method, url, timeout=120, **kw)
            if r.status_code >= 500:
                raise requests.ConnectionError(f"server error {r.status_code}")
            return r
        except (requests.ConnectionError, requests.Timeout) as e:
            say(f"No connection to Copernicus ({str(e)[:80]}) - retrying in 2 min")
            time.sleep(120)

# ---------- local bookkeeping ----------
def load_ledger():
    try:
        return json.load(open(LEDGER))
    except Exception:
        return {"saved": [], "skipped": []}
def save_ledger(L): json.dump(L, open(LEDGER, "w"), indent=1)

def done_months():
    done = set()
    for f in glob.glob(os.path.join(OUT, "era5_wind_*.nc")):
        m = re.match(r"era5_wind_(\d{4})_(\d{2})(?:to(\d{2}))?\.nc$", os.path.basename(f))
        if m:
            y, a, b = int(m[1]), int(m[2]), int(m[3] or m[2])
            done.update((y, k) for k in range(a, b + 1))
    return done

def next_block(busy=()):
    done = done_months() | set(busy)
    for y in YEARS:
        todo = [m for m in range(1, 13) if (y, m) not in done]
        if todo:
            run = [todo[0]]
            for m in todo[1:]:
                if m == run[-1] + 1 and len(run) < BLOCK: run.append(m)
                else: break
            return y, run
    return None

def job_period(job_id):
    """(year, (months...)) of a job, read from Copernicus; None if unknown."""
    r = call("GET", f"/jobs/{job_id}", params={"request": "true"})
    try:
        meta = r.json().get("metadata", {}).get("request", {})
        ids = meta.get("ids", meta)
        year = ids["year"]; months = ids["month"]
        year = int(year[0] if isinstance(year, list) else year)
        months = tuple(sorted(int(m) for m in (months if isinstance(months, list) else [months])))
        return year, months
    except Exception:
        return None

def list_jobs():
    r = call("GET", "/jobs", params={"limit": 100})
    jobs = r.json().get("jobs", []) if r.ok else []
    return [j for j in jobs if j.get("processID", DATASET) == DATASET]

def save_result(job_id, L):
    """Download a finished job; name the file by the months it really contains."""
    r = call("GET", f"/jobs/{job_id}/results")
    if not r.ok:
        say(f"Result of {job_id[:8]} no longer available - skipping"); L["skipped"].append(job_id); return
    href = r.json()["asset"]["value"]["href"]
    tmp = os.path.join(OUT, f"_download_{job_id[:8]}.part")
    say(f"Downloading finished request {job_id[:8]} ...")
    while True:
        try:
            with requests.get(href, stream=True, timeout=300) as d:
                d.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in d.iter_content(1 << 20): fh.write(chunk)
            break
        except Exception as e:
            say(f"Download interrupted ({str(e)[:80]}) - retrying in 2 min"); time.sleep(120)
    import xarray as xr
    with xr.open_dataset(tmp) as ds:
        tname = "valid_time" if "valid_time" in ds.coords else "time"
        t = ds[tname].to_index()
    years = sorted(set(t.year))
    if len(years) != 1 or years[0] not in YEARS:
        os.remove(tmp); say(f"  {job_id[:8]} is for {years} - not needed, discarded")
        L["skipped"].append(job_id); return
    y, a, b = years[0], t.month.min(), t.month.max()
    name = f"era5_wind_{y}_{a:02d}.nc" if a == b else f"era5_wind_{y}_{a:02d}to{b:02d}.nc"
    if all((y, m) in done_months() for m in range(a, b + 1)):
        os.remove(tmp); say(f"  {name} already saved earlier - duplicate discarded")
    else:
        os.replace(tmp, os.path.join(OUT, name)); say(f"  SAVED {name}")
    L["saved"].append(job_id)

# ---------- main loop ----------
say(f"ERA5 hourly download {YEARS[0]}-{YEARS[-1]}, {BLOCK} months per request. Leave this window open.")
L = load_ledger()
while True:
    jobs = list_jobs()
    # 1) save anything that finished
    for j in jobs:
        jid = j["jobID"]
        if j.get("status") == "successful" and jid not in L["saved"] and jid not in L["skipped"]:
            p = job_period(jid)
            if p is not None and (p[0] not in YEARS or all((p[0], m) in done_months() for m in p[1])):
                L["skipped"].append(jid); save_ledger(L); continue      # already have it / not needed
            save_result(jid, L); save_ledger(L)
    # 2) look at waiting/running requests; cancel duplicates and ones no longer needed
    active = [j for j in jobs if j.get("status") in ("accepted", "running")]
    active.sort(key=lambda j: j.get("created", ""))
    keep, seen = [], set()
    for j in active:
        p = job_period(j["jobID"])
        unneeded = p is not None and (p[0] not in YEARS or all((p[0], m) in done_months() for m in p[1]))
        if p is not None and (p in seen or unneeded):
            if j.get("status") == "accepted":
                call("DELETE", f"/jobs/{j['jobID']}")
                say(f"Cancelled duplicate/unneeded request {j['jobID'][:8]} ({p[0]} months {p[1][0]}-{p[1][-1]})")
                continue
        if p is not None: seen.add(p)
        keep.append((j, p))
    busy = {(p[0], m) for j, p in keep if p for m in p[1]}
    if keep:
        desc = ", ".join(f"{p[0]} months {p[1][0]}-{p[1][-1]} ({j['status']})" if p else f"{j['jobID'][:8]} ({j['status']})"
                         for j, p in keep)
        say(f"At Copernicus now ({len(keep)}/{MAX_ACTIVE}): {desc}")
    # 3) top up to MAX_ACTIVE requests, each for months nobody has asked for yet
    sent = 0
    while len(keep) + sent < MAX_ACTIVE:
        nb = next_block(busy)
        if nb is None:
            break
        y, run = nb
        req = {"product_type": ["reanalysis"], "variable": VARS, "year": [str(y)],
               "month": [f"{m:02d}" for m in run], "day": [f"{d:02d}" for d in range(1, 32)],
               "time": [f"{h:02d}:00" for h in range(24)], "area": AREA,
               "data_format": "netcdf", "download_format": "unarchived"}
        r = call("POST", f"/processes/{DATASET}/execution", json={"inputs": req})
        if r.ok:
            say(f"Sent request for {y} months {run[0]}-{run[-1]} (id {r.json().get('jobID','?')[:8]})")
            busy.update((y, m) for m in run); sent += 1
            time.sleep(5)
            continue
        msg = r.text[:300]
        if "cost" in msg or "too large" in msg:
            BLOCK = max(1, BLOCK // 2); say(f"Too big for Copernicus - now {BLOCK} month(s) per request")
            continue
        say(f"Copernicus refused a new request ({r.status_code}): {msg[:120]} - will try again later")
        break
    if not keep and sent == 0 and next_block() is None:
        say("ALL DONE - every month 2016-2025 is saved in", OUT); break
    say(f"Next check in {WAIT // 60} min.")
    time.sleep(WAIT)
input("Press Enter to close")
