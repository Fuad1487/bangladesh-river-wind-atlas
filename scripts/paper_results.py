"""
paper_results.py
Recomputes every block-level number used in the journal paper from ERA5,
for any period and any river-centreline source.

Inputs (all inside D:\\Wind Prediction unless given):
  ERA5 files        era5_hourly\\era5_wind_*.nc  (or bangladesh_wind_*.nc for the old sample)
  Block list        "Wind Blocks 10km Ranked (2020-2025).xlsx" - block names, centres,
                    district, zone (coastal / river / other) and distance to coast
  Land fraction     data\\wind_blocks_all.geojson
  Rivers            waypoints below, or a GeoJSON of river lines (HydroRIVERS)

Method (identical to the published 2020-2025 version):
  speed = sqrt(u100^2 + v100^2) at every time step; node statistics are averaged
  over time, then bilinearly interpolated to each 10 km block centre.
  WPD = mean(0.5 * 1.225 * V^3); Weibull k = (sd/mean)^-1.086, c = mean/Gamma(1+1/k);
  CF = generic curve, cubic from 3 to 11 m/s, 1 to 25 m/s;
  corridor block = nearest river within 15 km;
  AHP = 0.419*C1 + 0.263*C2 + 0.097*C3 + 0.160*C4 + 0.062*C5 with min-max
  scaling, C3 = 1 - scaled inter-annual CV, C4 = 1 - min(d_river, 60)/60;
  classes at 0.30 / 0.45 / 0.60 / 0.75.

Run:  python paper_results.py <era5 glob> <out folder> [rivers.geojson]
"""
import sys, os, glob, json, math
import numpy as np, pandas as pd, xarray as xr
from scipy.special import gamma

ERA5_GLOB = sys.argv[1]
OUT = sys.argv[2]
RIVER_FILE = sys.argv[3] if len(sys.argv) > 3 else None
BLOCKS_XLSX = os.environ.get("BLOCKS_XLSX", "Wind Blocks 10km Ranked (2020-2025).xlsx")
LANDFRAC = os.environ.get("LANDFRAC", os.path.join("data", "wind_blocks_all.geojson"))
os.makedirs(OUT, exist_ok=True)

WAYPOINTS = {
    "Padma": [(24.60, 88.28), (24.37, 88.60), (24.15, 89.03), (23.93, 89.30), (23.87, 89.75),
              (23.65, 89.83), (23.49, 90.15), (23.43, 90.27), (23.23, 90.63)],
    "Jamuna": [(25.60, 89.68), (25.18, 89.67), (24.85, 89.62), (24.45, 89.70), (24.15, 89.72), (23.87, 89.75)],
    "Brahmaputra": [(25.18, 89.67), (24.93, 89.95), (24.75, 90.40), (24.30, 90.75), (24.05, 90.98)],
    "Meghna": [(24.05, 90.98), (23.85, 90.85), (23.55, 90.70), (23.23, 90.63), (22.75, 90.75),
               (22.20, 90.75), (21.90, 90.90)],
}
W = {"C1": 0.419, "C2": 0.263, "C3": 0.097, "C4": 0.160, "C5": 0.062}
CORRIDOR_KM, C4_CAP_KM = 15.0, 60.0


def seg_km(lat, lon, a, b):
    ref = (a[0] + b[0]) / 2; kx = 111.32 * math.cos(math.radians(ref)); ky = 110.574
    px, py, ax, ay, bx, by = lon * kx, lat * ky, a[1] * kx, a[0] * ky, b[1] * kx, b[0] * ky
    dx, dy = bx - ax, by - ay
    t = 0 if dx == dy == 0 else max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def load_rivers():
    """{name: [polyline, ...]} each polyline a list of (lat, lon)."""
    if not RIVER_FILE:
        return {k: [v] for k, v in WAYPOINTS.items()}
    gj = json.load(open(RIVER_FILE))
    out = {}
    for f in gj["features"]:
        name = f["properties"]["corridor"]
        geom = f["geometry"]
        from shapely.geometry import shape as _shape
        g = _shape(geom).simplify(0.002)   # ~200 m tolerance keeps distances exact to well under 1 km
        lines = list(g.geoms) if g.geom_type == "MultiLineString" else [g]
        for ln in lines:
            out.setdefault(name, []).append([(c[1], c[0]) for c in ln.coords])
    return out


def dist_to(lat, lon, polylines):
    return min(seg_km(lat, lon, a, b) for pl in polylines for a, b in zip(pl[:-1], pl[1:]))


# ---------------- ERA5 node statistics ----------------
files = sorted(glob.glob(ERA5_GLOB))
ds = xr.concat([xr.open_dataset(f)[["u10", "v10", "u100", "v100"]].load() for f in files], "valid_time").sortby("valid_time")
ds = ds.isel(valid_time=~pd.Index(ds.valid_time.values).duplicated())
s = np.sqrt(ds.u100 ** 2 + ds.v100 ** 2); s10 = np.sqrt(ds.u10 ** 2 + ds.v10 ** 2)
mon, yr = s.valid_time.dt.month, s.valid_time.dt.year
years = sorted(set(yr.values.tolist()))
N = {"mean": s.mean("valid_time"), "mean10": s10.mean("valid_time"),
     "wpd": (0.5 * 1.225 * s ** 3).mean("valid_time")}
sd = s.std("valid_time"); k = (sd / N["mean"]) ** -1.086
N["k"] = k; N["c"] = N["mean"] / xr.apply_ufunc(lambda kk: gamma(1 + 1 / kk), k)
N["cf"] = xr.where(s < 3, 0, xr.where(s < 11, (s ** 3 - 27) / (1331 - 27), xr.where(s < 25, 1, 0))).mean("valid_time")
N["jjas"] = s.where(mon.isin([6, 7, 8, 9])).mean("valid_time")
N["djf"] = s.where(mon.isin([12, 1, 2])).mean("valid_time")
N["on"] = s.where(mon.isin([10, 11])).mean("valid_time")
N["p999"] = s.quantile(0.999, "valid_time")
for y in years: N[f"y{y}"] = s.where(yr == y).mean("valid_time")
for m in range(1, 13): N[f"m{m:02d}"] = s.where(mon == m).mean("valid_time")
n_steps = s.sizes["valid_time"]

# ---------------- blocks ----------------
B = pd.read_excel(BLOCKS_XLSX, sheet_name="All blocks (ranked)")
B = B[["Block", "District", "Division", "Zone", "Distance to coast (km)", "Centre lat", "Centre lon"]].copy()
lf = {f["properties"]["block_id"]: f["properties"]["land_fraction"] for f in json.load(open(LANDFRAC))["features"]}
B["land_fraction"] = B.Block.map(lf)
import geopandas as gpd
_g = gpd.read_file(LANDFRAC)[["block_id", "geometry"]].to_crs(32646)   # UTM 46N for areas
B["area_km2"] = B.Block.map(dict(zip(_g.block_id, _g.area / 1e6)))
_la, _lo = np.radians(B["Centre lat"]), np.radians(B["Centre lon"])
B["dist_dhaka_km"] = 6371 * 2 * np.arcsin(np.sqrt(np.sin((_la - math.radians(23.8103)) / 2) ** 2 +
                     np.cos(_la) * math.cos(math.radians(23.8103)) * np.sin((_lo - math.radians(90.4125)) / 2) ** 2))
lat_da = xr.DataArray(B["Centre lat"].clip(20.5, 26.5).values, dims="b")
lon_da = xr.DataArray(B["Centre lon"].clip(88.0, 92.5).values, dims="b")
for key, field in N.items():
    B[key] = field.interp(latitude=lat_da, longitude=lon_da).values
B["alpha"] = np.log(B["mean"] / B["mean10"]) / np.log(10)
ycols = [f"y{y}" for y in years]
B["cv"] = B[ycols].std(axis=1, ddof=0) / B[ycols].mean(axis=1)

rivers = load_rivers()
for name, pls in rivers.items():
    B["d_" + name] = [dist_to(la, lo, pls) for la, lo in zip(B["Centre lat"], B["Centre lon"])]
dcols = ["d_" + n for n in rivers]
B["dist_river"] = B[dcols].min(axis=1)
B["nearest_river"] = B[dcols].idxmin(axis=1).str[2:]
B["corridor"] = np.where(B.dist_river <= CORRIDOR_KM, B.nearest_river, "")
# zone: coastal belt from the coastline (unchanged); river corridor recomputed for these rivers
B["zone"] = np.where(B.Zone == "Coastal belt", "Coastal belt",
                     np.where(B.corridor != "", "River corridors", "Other inland"))

mm = lambda v: (v - v.min()) / (v.max() - v.min())
def ahp(b, w=W, use_c4=True):
    c = {"C1": mm(b["mean"]), "C2": mm(b["wpd"]), "C3": 1 - mm(b["cv"]),
         "C4": 1 - b.dist_river.clip(upper=C4_CAP_KM) / C4_CAP_KM, "C5": mm(b.land_fraction)}
    ww = dict(w)
    if not use_c4:
        ww.pop("C4"); tot = sum(ww.values()); ww = {k: v / tot for k, v in ww.items()}
    return sum(ww[k] * c[k] for k in ww)
def ahp_class(sc):
    return pd.cut(sc, [-1, 0.30, 0.45, 0.60, 0.75, 2], right=False,
                  labels=["Very low", "Low", "Moderate", "High", "Very high"])
B["ahp"] = ahp(B); B["ahp_class"] = ahp_class(B.ahp)
B["ahp_rank"] = B.ahp.rank(ascending=False, method="first").astype(int)
B = B.sort_values("mean", ascending=False).reset_index(drop=True); B["rank"] = B.index + 1

# ---------------- paper numbers ----------------
R = {"period": f"{years[0]}-{years[-1]}", "time_steps": n_steps,
     "rivers": os.path.basename(RIVER_FILE) if RIVER_FILE else "waypoints"}
R["national_mean"] = B["mean"].mean(); R["national_median"] = B["mean"].median()
R["min_block"] = (B["mean"].min(), B.loc[B["mean"].idxmin(), "District"])
R["max_block"] = (B["mean"].max(), B.loc[B["mean"].idxmax(), "District"], B.loc[B["mean"].idxmax(), "Block"])
R["wpd_mean"] = B.wpd.mean(); R["wpd_max"] = B.wpd.max()
R["jjas"] = B.jjas.mean(); R["djf"] = B.djf.mean(); R["on"] = B.on.mean()
R["k_min"], R["k_max"], R["k_mean"] = B.k.min(), B.k.max(), B.k.mean()
over5 = B[B["mean"] > 5]; R["n_over5"] = len(over5)
R["over5_districts"] = over5.District.value_counts().to_dict()
R["over5_zones"] = over5.zone.value_counts().to_dict()

cor = B[B.corridor != ""]
R["n_corridor_blocks"] = len(cor)
T4 = cor.groupby("corridor").agg(blocks=("Block", "size"), mean=("mean", "mean"), max=("mean", "max"),
                                 jjas=("jjas", "mean"), djf=("djf", "mean"), wpd=("wpd", "mean"),
                                 cf=("cf", "mean")).reindex(["Padma", "Jamuna", "Brahmaputra", "Meghna"])
T4.to_csv(os.path.join(OUT, "table4_corridors.csv"))
meg = cor[cor.corridor == "Meghna"]
R["meghna_south_22.8"] = meg[meg["Centre lat"] < 22.8]["mean"].mean()
R["meghna_rest"] = meg[meg["Centre lat"] >= 22.8]["mean"].mean()
R["corridor_mean10"] = cor.mean10.mean(); R["corridor_alpha"] = float(np.log(cor["mean"].mean() / cor.mean10.mean()) / np.log(10))
mon_cols = [f"m{m:02d}" for m in range(1, 13)]
cor.groupby("corridor")[mon_cols].mean().round(2).to_csv(os.path.join(OUT, "fig3_corridor_monthly.csv"))
R["peak_meghna"] = cor[cor.corridor == "Meghna"][mon_cols].mean().max()
R["peak_padma"] = cor[cor.corridor == "Padma"][mon_cols].mean().max()

# latitude bands (interior 22.8-25.8 N, 0.6 deg)
bands = []
for lo in np.arange(22.8, 25.79, 0.6):
    sub = B[(B["Centre lat"] >= lo) & (B["Centre lat"] < lo + 0.6) & (B.zone != "Coastal belt")]
    a, b_ = sub[sub.corridor != ""]["mean"], sub[sub.corridor == ""]["mean"]
    bands.append({"band": f"{lo:.1f}-{lo + 0.6:.1f}", "corridor_mean": a.mean(), "other_mean": b_.mean(),
                  "diff": a.mean() - b_.mean(), "n_corridor": len(a), "n_other": len(b_)})
pd.DataFrame(bands).round(3).to_csv(os.path.join(OUT, "fig4_latitude_bands.csv"), index=False)
R["lat_band_diffs"] = [round(b["diff"], 2) for b in bands]

# coast vs river (Table V)
area_per_block = {"Coastal belt": 9194 / 148, "River corridors": 24570 / 260}
T5 = {}
for z in ["Coastal belt", "River corridors", "Other inland"]:
    sub = B[B.zone == z]
    T5[z] = {"blocks": len(sub), "area_km2": float(sub.area_km2.sum()), "dist_dhaka_km": float(sub.dist_dhaka_km.mean()),
             "mean": sub["mean"].mean(), "wpd": sub.wpd.mean(), "cf": sub.cf.mean(),
             "jjas": sub.jjas.mean(), "djf": sub.djf.mean(), "p999": sub.p999.mean(),
             "cv_interannual": float(sub[ycols].mean().std(ddof=0) / sub[ycols].mean().mean()), "n_ge4": int((sub["mean"] >= 4).sum()),
             "resource_index": float((sub.wpd * sub.area_km2).sum())}
    for m in range(1, 13): T5[z][f"m{m:02d}"] = sub[f"m{m:02d}"].mean()
T5 = pd.DataFrame(T5); T5.to_csv(os.path.join(OUT, "table5_coast_vs_river.csv"))
riv_ge4 = B[(B.zone == "River corridors") & (B["mean"] >= 4)]
R["river_ge4_by_corridor"] = riv_ge4.corridor.value_counts().to_dict()

# AHP
R["ahp_classes"] = B.ahp_class.value_counts().reindex(["Very low", "Low", "Moderate", "High", "Very high"]).to_dict()
vh = B[B.ahp_class == "Very high"]
R["corridor_share_all"] = (B.corridor != "").mean()
R["corridor_share_veryhigh"] = (vh.corridor != "").mean() if len(vh) else float("nan")
topA = set(B.nsmallest(24, "ahp_rank").Block); topW = set(B.nsmallest(24, "rank").Block)
R["top24_overlap"] = len(topA & topW)
R["ahp_top5"] = B.nsmallest(5, "ahp_rank")[["Block", "District", "corridor", "mean", "dist_river", "ahp"]].round(3).values.tolist()
T6 = (cor.sort_values("ahp", ascending=False).groupby("corridor").head(3)
      .sort_values(["corridor", "ahp"], ascending=[True, False])
      [["corridor", "Block", "District", "mean", "wpd", "ahp", "ahp_rank"]])
T6.round(3).to_csv(os.path.join(OUT, "table6_best_ahp_per_corridor.csv"), index=False)

# AHP sensitivity: without C4, and each weight +/-20% (renormalised)
base_top = set(B.nsmallest(24, "ahp_rank").Block)
sens = []
noC4 = ahp(B, use_c4=False); top_noC4 = set(B.loc[noC4.nlargest(24).index, "Block"])
sens.append({"test": "without C4 (river proximity)", "top24_kept": len(base_top & top_noC4),
             "corridor_share_top24": float((B.loc[noC4.nlargest(24).index, "corridor"] != "").mean())})
for kk in W:
    for f in (0.8, 1.2):
        w2 = dict(W); w2[kk] *= f; t = sum(w2.values()); w2 = {a: v / t for a, v in w2.items()}
        sc = ahp(B, w2); top = set(B.loc[sc.nlargest(24).index, "Block"])
        sens.append({"test": f"{kk} x{f}", "top24_kept": len(base_top & top),
                     "corridor_share_top24": float((B.loc[sc.nlargest(24).index, "corridor"] != "").mean())})
pd.DataFrame(sens).to_csv(os.path.join(OUT, "ahp_sensitivity.csv"), index=False)
R["base_corridor_share_top24"] = float((B.nsmallest(24, "ahp_rank").corridor != "").mean())

# block table
cols = ["rank", "Block", "District", "Division", "zone", "Distance to coast (km)", "Centre lat", "Centre lon"] + ycols + \
       ["mean", "wpd", "cf", "k", "c", "jjas", "djf", "nearest_river", "dist_river", "corridor", "ahp", "ahp_class", "ahp_rank"]
B[cols].round(4).to_csv(os.path.join(OUT, "blocks_all.csv"), index=False)

def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return round(float(o), 4)
    if isinstance(o, np.integer): return int(o)
    return o
json.dump(clean(R), open(os.path.join(OUT, "paper_numbers.json"), "w"), indent=1)
print(json.dumps(clean(R), indent=1))
print(T4.round(2)); print(T5.round(3).iloc[:12])
