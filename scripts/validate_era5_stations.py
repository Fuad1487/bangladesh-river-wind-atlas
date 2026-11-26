"""
validate_era5_stations.py
Compares ERA5 10 m wind (your bangladesh_wind_2020..2025.nc files) with
observed wind at Bangladesh weather stations (NOAA Global Hourly / BMD synoptic
reports, plus Dhaka and Chattogram airport METARs), 2016-2025.

Method
- ERA5 10 m speed = sqrt(u10^2 + v10^2), bilinearly interpolated to each station.
- Matched only at the exact times both exist (every hour that both have).
- Station observations kept only if NOAA quality code is 0/1/4/5/9 and speed
  is not missing. Calms (0 m/s) are kept - they are real reports.
- Stations used only if they have ≥ 1000 matched pairs.
Outputs: station_validation.csv, seasonal_validation.csv, monthly_cycle.csv,
         fig_*.png
Run from D:\\Wind Prediction:  python validation_2016_2025\\validate_era5_stations.py
"""
import glob, os, sys
import numpy as np, pandas as pd, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "validation_2016_2025")
os.makedirs(OUT, exist_ok=True)

SEASON = {12: "Winter (DJF)", 1: "Winter (DJF)", 2: "Winter (DJF)",
          3: "Pre-monsoon (MAM)", 4: "Pre-monsoon (MAM)", 5: "Pre-monsoon (MAM)",
          6: "Monsoon (JJAS)", 7: "Monsoon (JJAS)", 8: "Monsoon (JJAS)", 9: "Monsoon (JJAS)",
          10: "Post-monsoon (ON)", 11: "Post-monsoon (ON)"}

# ---------- ERA5 ----------
ds = xr.concat([xr.open_dataset(f)[["u10", "v10"]].load()
                for f in sorted(glob.glob(os.path.join(ROOT, "era5_hourly", "era5_wind_20*.nc")))], dim="valid_time")
ws10 = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2).load()
era_times = pd.DatetimeIndex(ws10["valid_time"].values)

# ---------- stations ----------
meta = pd.read_csv(os.path.join(ROOT, "noaa_stations", "noaa_bd_stations.csv"), dtype=str)
meta["sid"] = meta["usaf"] + meta["wban"]
meta["lat"] = meta["lat"].astype(float); meta["lon"] = meta["lon"].astype(float)

def read_noaa(sid):
    files = sorted(glob.glob(os.path.join(ROOT, "noaa_stations", f"{sid}_20[12][0-9].csv")))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f, usecols=["DATE", "WND", "REPORT_TYPE"], dtype=str) for f in files])
    w = df["WND"].str.split(",", expand=True)
    spd = pd.to_numeric(w[3], errors="coerce")
    ok = (spd != 9999) & w[4].isin(["0", "1", "4", "5", "9"])
    out = pd.DataFrame({"time": pd.to_datetime(df["DATE"]), "obs": spd / 10.0,
                        "type": df["REPORT_TYPE"]})[ok]
    # prefer synoptic (FM-12) reports when a station has both
    if (out["type"] == "FM-12").sum() > 500:
        out = out[out["type"] == "FM-12"]
    return out.drop_duplicates("time").set_index("time")["obs"]

def read_metar(code):
    f = os.path.join(ROOT, "station_data", f"metar_{code}.csv")
    df = pd.read_csv(f, usecols=["valid", "sknt"])
    df["time"] = pd.to_datetime(df["valid"])
    df = df.dropna(subset=["sknt"])
    df = df[(df["time"] >= "2016-01-01") & (df["time"] < "2026-01-01")]
    # METARs are at :00/:30 or :50 - keep reports within 10 minutes of the hour
    df["hr"] = df["time"].dt.round("h")
    df = df[(df["time"] - df["hr"]).abs() <= pd.Timedelta("10min")]
    return df.drop_duplicates("hr").set_index("hr")["sknt"] * 0.514444

series = []
for _, m in meta.iterrows():
    s = read_noaa(m["sid"])
    if s is not None and len(s):
        series.append((m["sid"], m["name"].strip().title(), m["lat"], m["lon"], "BMD synoptic (NOAA)", s))
mm = {"VGHS": ("Dhaka airport (METAR)", 23.8433, 90.3978), "VGEG": ("Chattogram airport (METAR)", 22.2496, 91.8133)}
for code, (nm, la, lo) in mm.items():
    if os.path.exists(os.path.join(ROOT, "station_data", f"metar_{code}.csv")):
        series.append((code, nm, la, lo, "Airport METAR (IEM)", read_metar(code)))

rows, pairs_all = [], []
for sid, name, lat, lon, src, s in series:
    s = s[s.index.isin(era_times)]
    if len(s) < 1000:
        continue
    e = ws10.interp(latitude=lat, longitude=lon).sel(valid_time=s.index).values
    p = pd.DataFrame({"obs": s.values, "era5": e}, index=s.index).dropna()
    if len(p) < 1000:
        continue
    p["sid"], p["name"], p["month"] = sid, name, p.index.month
    pairs_all.append(p)
    mon = p.groupby("month")[["obs", "era5"]].mean()
    rows.append({
        "station_id": sid, "station": name, "source": src, "lat": lat, "lon": lon,
        "n_pairs": len(p), "pct_calm_obs": round(100 * (p["obs"] == 0).mean(), 1),
        "obs_mean_ms": round(p["obs"].mean(), 2), "era5_mean_ms": round(p["era5"].mean(), 2),
        "bias_ms": round((p["era5"] - p["obs"]).mean(), 2),
        "rmse_ms": round(np.sqrt(((p["era5"] - p["obs"]) ** 2).mean()), 2),
        "r_3hourly": round(p[["obs", "era5"]].corr().iloc[0, 1], 2),
        "r_monthly_cycle": round(mon.corr().iloc[0, 1], 2),
        "ratio_obs_to_era5": round(p["obs"].mean() / p["era5"].mean(), 2),
    })

st = pd.DataFrame(rows).sort_values("lat", ascending=False)
# quality flag: very sheltered / mostly-calm stations
st["quality"] = np.where((st["pct_calm_obs"] > 50) | (st["obs_mean_ms"] < 1.0), "poor exposure", "usable")
st.to_csv(os.path.join(OUT, "station_validation.csv"), index=False)

pairs = pd.concat(pairs_all)
pairs["season"] = pairs["month"].map(SEASON)
good = set(st.loc[st["quality"] == "usable", "station_id"])
srows = []
for (label, sub) in [("All stations", pairs), ("Usable stations", pairs[pairs["sid"].isin(good)])]:
    for season, g in sub.groupby("season"):
        srows.append({"set": label, "season": season, "n_pairs": len(g),
                      "obs_mean_ms": round(g["obs"].mean(), 2), "era5_mean_ms": round(g["era5"].mean(), 2),
                      "bias_ms": round((g["era5"] - g["obs"]).mean(), 2),
                      "rmse_ms": round(np.sqrt(((g["era5"] - g["obs"]) ** 2).mean()), 2),
                      "r": round(g[["obs", "era5"]].corr().iloc[0, 1], 2)})
pd.DataFrame(srows).to_csv(os.path.join(OUT, "seasonal_validation.csv"), index=False)

mc = pairs[pairs["sid"].isin(good)].groupby("month")[["obs", "era5"]].mean().round(2)
mc.to_csv(os.path.join(OUT, "monthly_cycle.csv"))

# ---------- figures ----------
fig, ax = plt.subplots(figsize=(6, 6))
for q, c in [("usable", "tab:blue"), ("poor exposure", "tab:gray")]:
    d = st[st["quality"] == q]
    ax.scatter(d["obs_mean_ms"], d["era5_mean_ms"], c=c, label=q, s=40)
for _, r in st.iterrows():
    ax.annotate(r["station"][:12], (r["obs_mean_ms"], r["era5_mean_ms"]), fontsize=6, xytext=(3, 2), textcoords="offset points")
lim = [0, max(st["era5_mean_ms"].max(), st["obs_mean_ms"].max()) + 0.5]
ax.plot(lim, lim, "k--", lw=0.8, label="1:1")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Observed mean 10 m wind (m/s)"); ax.set_ylabel("ERA5 mean 10 m wind (m/s)")
ax.set_title("Station mean wind vs ERA5, 2016-2025"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_station_means_vs_era5.png"), dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(mc.index, mc["obs"], "o-", label="Stations (usable, mean)")
ax.plot(mc.index, mc["era5"], "s-", label="ERA5 at the same stations")
ax.set_xticks(range(1, 13)); ax.set_xticklabels(list("JFMAMJJASOND"))
ax.set_ylabel("10 m wind speed (m/s)"); ax.set_title("Seasonal cycle: stations vs ERA5, 2016-2025")
ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_monthly_cycle.png"), dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 7))
sc = ax.scatter(st["lon"], st["lat"], c=st["r_3hourly"], cmap="viridis", s=60, vmin=0, vmax=1, edgecolor="k")
for _, r in st.iterrows():
    ax.annotate(r["station"][:12], (r["lon"], r["lat"]), fontsize=6, xytext=(3, 2), textcoords="offset points")
plt.colorbar(sc, label="Correlation r (3-hourly)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_title("ERA5 vs station correlation")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_correlation_map.png"), dpi=200); plt.close(fig)

print(st.to_string(index=False))
print(pd.DataFrame(srows).to_string(index=False))
print(mc.T.to_string())
