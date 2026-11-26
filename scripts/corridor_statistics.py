"""
corridor_statistics.py
Tests whether the river-corridor wind effect holds once latitude, longitude and
distance to the coast are controlled for, using the ERA5 grid nodes
(0.25 deg, ~27 km) inside Bangladesh, 2016-2025, 100 m mean wind speed.

- Speed at each node = mean of the 2-hourly speeds (not of the vector).
- Distance to coast: from a 1 km global land/sea mask (global-land-mask package).
- Corridor flag: node within B km of a river centreline (Padma, Jamuna,
  Brahmaputra, Meghna waypoints from river_wind_analysis.py), B = 10/15/25/30 km.
- OLS: speed ~ latitude + longitude + log(distance to coast) + corridor,
  with standard errors clustered by 1 x 1 degree block (handles the fact that
  neighbouring nodes are not independent).
- Also a test restricted to inland nodes > 150 km from the coast (Jamuna /
  Brahmaputra / upper Padma region), where the sea breeze cannot explain it.
Outputs: corridor_regression.csv, era5_nodes_with_covariates.csv, fig_*.png
"""
import glob, json, math, os, sys
import numpy as np, pandas as pd, xarray as xr
import statsmodels.formula.api as smf
from shapely.geometry import shape, Point
from global_land_mask import globe
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "validation_2016_2025")

RIVERS = {
    "Padma": [(24.60, 88.28), (24.37, 88.60), (24.15, 89.03), (23.93, 89.30), (23.87, 89.75),
              (23.65, 89.83), (23.49, 90.15), (23.43, 90.27), (23.23, 90.63)],
    "Jamuna": [(25.60, 89.68), (25.18, 89.67), (24.85, 89.62), (24.45, 89.70), (24.15, 89.72), (23.87, 89.75)],
    "Brahmaputra": [(25.18, 89.67), (24.93, 89.95), (24.75, 90.40), (24.30, 90.75), (24.05, 90.98)],
    "Meghna": [(24.05, 90.98), (23.85, 90.85), (23.55, 90.70), (23.23, 90.63), (22.75, 90.75),
               (22.20, 90.75), (21.90, 90.90)],
}

def seg_km(lat, lon, a, b):
    ref = (a[0] + b[0]) / 2; kx = 111.32 * math.cos(math.radians(ref)); ky = 110.574
    px, py, ax, ay, bx, by = lon*kx, lat*ky, a[1]*kx, a[0]*ky, b[1]*kx, b[0]*ky
    dx, dy = bx-ax, by-ay
    t = max(0, min(1, ((px-ax)*dx + (py-ay)*dy) / (dx*dx + dy*dy)))
    return math.hypot(px-(ax+t*dx), py-(ay+t*dy))

RIVER_FILE = sys.argv[3] if len(sys.argv) > 3 else None
if RIVER_FILE:   # HydroRIVERS (or any) centreline GeoJSON with a "corridor" property
    from shapely.geometry import shape as _shape
    RIVERS = {}
    for f in json.load(open(RIVER_FILE))["features"]:
        g = _shape(f["geometry"]).simplify(0.002)
        for ln in (list(g.geoms) if g.geom_type == "MultiLineString" else [g]):
            RIVERS.setdefault(f["properties"]["corridor"], []).append([(c[1], c[0]) for c in ln.coords])
else:
    RIVERS = {k: [v] for k, v in RIVERS.items()}

def dist_river(lat, lon, names=None):
    names = names or list(RIVERS)
    return min(seg_km(lat, lon, a, b) for n in names for pl in RIVERS[n] for a, b in zip(pl[:-1], pl[1:]))

# ---- ERA5 100 m mean speed per node ----
files = sorted(glob.glob(os.path.join(ROOT, "era5_hourly", "era5_wind_20*.nc")))
acc, n = None, 0
for f in files:
    d = xr.open_dataset(f)
    s = np.sqrt(d["u100"]**2 + d["v100"]**2)
    acc = s.sum("valid_time") if acc is None else acc + s.sum("valid_time"); n += d.sizes["valid_time"]
mean100 = (acc / n).load()

# ---- Bangladesh land outline ----
gj = json.load(open(os.path.join(ROOT, "data", "bangladesh_outline.geojson")))
geoms = [shape(f["geometry"]) for f in gj["features"]] if "features" in gj else [shape(gj)]
bd = geoms[0]
for g in geoms[1:]: bd = bd.union(g)

# ---- distance to coast from 1 km land/sea mask ----
glat = np.arange(19.5, 27.0, 0.02); glon = np.arange(87.5, 93.5, 0.02)
LA, LO = np.meshgrid(glat, glon, indexing="ij")
ocean = globe.is_ocean(LA, LO)
olat, olon = LA[ocean], LO[ocean]
def dist_coast(lat, lon):
    dy = (olat - lat) * 110.574; dx = (olon - lon) * 111.32 * math.cos(math.radians(lat))
    return float(np.sqrt(dx*dx + dy*dy).min())

rows = []
for la in mean100["latitude"].values:
    for lo in mean100["longitude"].values:
        if not bd.contains(Point(lo, la)):
            continue
        rows.append({"lat": la, "lon": lo, "ws100": float(mean100.sel(latitude=la, longitude=lo)),
                     "dist_coast_km": dist_coast(la, lo), "dist_river_km": dist_river(la, lo),
                     "dist_jamuna_brahm_km": dist_river(la, lo, ["Jamuna", "Brahmaputra"])})
df = pd.DataFrame(rows)
df["log_dcoast"] = np.log(df["dist_coast_km"].clip(lower=1))
df["block"] = (np.floor(df["lat"]).astype(int).astype(str) + "_" + np.floor(df["lon"]).astype(int).astype(str))
df.to_csv(os.path.join(OUT, "era5_nodes_with_covariates.csv"), index=False)

res = []
def fit(sub, label, buf, distcol="dist_river_km"):
    sub = sub.copy(); sub["corridor"] = (sub[distcol] <= buf).astype(int)
    if sub["corridor"].sum() < 3:
        res.append({"test": label, "buffer_km": buf, "n_nodes": len(sub), "n_corridor": int(sub["corridor"].sum())}); return
    raw = sub.groupby("corridor")["ws100"].mean()
    m = smf.ols("ws100 ~ lat + lon + log_dcoast + corridor", data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(sub["block"])[0]})
    ci = m.conf_int().loc["corridor"]
    res.append({"test": label, "buffer_km": buf, "n_nodes": len(sub), "n_corridor": int(sub["corridor"].sum()),
                "raw_corridor_mean": round(raw.get(1, np.nan), 3), "raw_other_mean": round(raw.get(0, np.nan), 3),
                "raw_difference": round(raw.get(1, np.nan) - raw.get(0, np.nan), 3),
                "adjusted_effect_ms": round(m.params["corridor"], 3),
                "ci95_low": round(ci[0], 3), "ci95_high": round(ci[1], 3),
                "p_value": round(m.pvalues["corridor"], 4), "r2_model": round(m.rsquared, 3),
                "coef_log_dcoast": round(m.params["log_dcoast"], 3), "coef_lat": round(m.params["lat"], 3)})

for b in [10, 15, 25, 30]:
    fit(df, "All four rivers, all Bangladesh nodes", b)
inland = df[df["dist_coast_km"] > 150]
for b in [10, 15, 25, 30]:
    fit(inland, "All four rivers, inland nodes only (>150 km from coast)", b)
    fit(inland, "Jamuna + Brahmaputra only, inland nodes (>150 km from coast)", b, "dist_jamuna_brahm_km")

r = pd.DataFrame(res); r.to_csv(os.path.join(OUT, "corridor_regression.csv"), index=False)
print(f"{len(df)} ERA5 nodes inside Bangladesh; {len(inland)} inland >150 km")
print(r.to_string(index=False))

# figure: speed vs distance to coast, corridor nodes highlighted (15 km)
fig, ax = plt.subplots(figsize=(7, 4.5))
c = df["dist_river_km"] <= 15
ax.scatter(df.loc[~c, "dist_coast_km"], df.loc[~c, "ws100"], s=14, c="lightgray", label="Other nodes")
ax.scatter(df.loc[c, "dist_coast_km"], df.loc[c, "ws100"], s=22, c="tab:blue", label="Within 15 km of a river")
ax.set_xlabel("Distance to coast (km)"); ax.set_ylabel("Mean 100 m wind speed (m/s)")
ax.set_title("ERA5 nodes, 2016-2025: wind falls with distance from the sea"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_speed_vs_coast_distance.png"), dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
for lab, mk in [("All four rivers, all Bangladesh nodes", "o-"), ("All four rivers, inland nodes only (>150 km from coast)", "s-"),
                ("Jamuna + Brahmaputra only, inland nodes (>150 km from coast)", "^-")]:
    d = r[r["test"] == lab].dropna(subset=["adjusted_effect_ms"])
    ax.errorbar(d["buffer_km"], d["adjusted_effect_ms"],
                yerr=[d["adjusted_effect_ms"] - d["ci95_low"], d["ci95_high"] - d["adjusted_effect_ms"]],
                fmt=mk, capsize=3, label=lab.split(",")[0] + (" (inland)" if "inland" in lab else " (all)"))
ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("Corridor buffer width (km)")
ax.set_ylabel("Corridor effect after controls (m/s)"); ax.set_title("River-corridor effect with 95% confidence intervals")
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_corridor_effect_by_buffer.png"), dpi=200); plt.close(fig)
