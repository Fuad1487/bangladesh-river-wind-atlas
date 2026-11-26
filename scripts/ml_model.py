"""
ML wind model v2 for the river-corridor paper (ERA5 hourly 2016-2025).

What it does, in plain words
  1. Reads the 20 ERA5 files and works out the 100 m wind speed for every hour.
  2. Averages it into "typical" values: one number per grid point, month and hour of day
     (0-23 UTC). Training years 2016-2023, test years 2024-2025.
  3. Gives each grid point extra map information: distance to the coast, distance to the
     nearest major river, and whether it is land or sea.
  4. Trains two models (Random Forest and XGBoost) and runs two honest tests:
       Test A (new years):  train on 2016-2023, predict 2024-2025.
                            Compared with simply re-using the 2016-2023 average.
       Test B (new places): hide whole 1-degree squares of the map, train on the rest,
                            predict the hidden points. Compared with inverse-distance
                            interpolation from the neighbouring points.
  5. Fits the final model on all ten years, explains it with SHAP, and writes a new
     prediction grid with all 24 hours (fixes the old 3/9/15/21-hour problem).

Usage:  python ml_model.py <wind prediction folder> <rivers geojson> <out folder>
"""
import sys, os, glob, json, math, time
import numpy as np, pandas as pd, xarray as xr
from scipy.spatial import cKDTree
from shapely.geometry import shape, Point
from global_land_mask import globe
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib

ROOT, RIVERS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

# ------------------------------------------------------------------ 1. aggregate ERA5
files = sorted(glob.glob(os.path.join(ROOT, "era5_hourly", "era5_wind_20*.nc")))
S = None  # sums [year, month, hour, lat, lon]
for f in files:
    d = xr.open_dataset(f)
    lats, lons = d["latitude"].values, d["longitude"].values
    if S is None:
        S = np.zeros((10, 12, 24, len(lats), len(lons))); N = np.zeros((10, 12, 24))
    ws = np.sqrt(d["u100"].values**2 + d["v100"].values**2)
    t = pd.DatetimeIndex(d["valid_time"].values)
    yi, mi, hi = t.year - 2016, t.month - 1, t.hour
    np.add.at(S, (yi, mi, hi), ws)
    np.add.at(N, (yi, mi, hi), 1)
    d.close()
log("aggregated", len(files), "files, hours =", int(N.sum()))
MEAN = S / N[..., None, None]                       # year x month x hour x lat x lon
clim_train = (S[:8].sum(0) / N[:8].sum(0)[..., None, None])   # 2016-2023
clim_test  = (S[8:].sum(0) / N[8:].sum(0)[..., None, None])   # 2024-2025
clim_all   = (S.sum(0) / N.sum(0)[..., None, None])           # 2016-2025

# ------------------------------------------------------------------ 2. node covariates
gj = json.load(open(os.path.join(ROOT, "data", "bangladesh_outline.geojson")))
geoms = [shape(f["geometry"]) for f in gj["features"]] if "features" in gj else [shape(gj)]
bd = geoms[0]
for g in geoms[1:]: bd = bd.union(g)

glat = np.arange(19.0, 28.0, 0.02); glon = np.arange(86.5, 94.5, 0.02)
LA, LO = np.meshgrid(glat, glon, indexing="ij")
oc = globe.is_ocean(LA, LO)
def to_xy(lat, lon):  # simple local projection in km around 23.5N
    return np.c_[(np.asarray(lon) - 90.0) * 111.32 * math.cos(math.radians(23.5)),
                 (np.asarray(lat) - 23.5) * 110.574]
ocean_tree = cKDTree(to_xy(LA[oc], LO[oc]))

rv = json.load(open(RIVERS))
pts = []
for ft in rv["features"]:
    g = shape(ft["geometry"])
    for line in (g.geoms if hasattr(g, "geoms") else [g]):
        L = line.length
        for s in np.linspace(0, L, max(2, int(L / 0.005))):   # about every 0.5 km
            p = line.interpolate(s); pts.append((p.y, p.x))
pts = np.array(pts)
river_tree = cKDTree(to_xy(pts[:, 0], pts[:, 1]))

nodes = []
for i, la in enumerate(lats):
    for j, lo in enumerate(lons):
        xy = to_xy([la], [lo])
        nodes.append({"i": i, "j": j, "latitude": float(la), "longitude": float(lo),
                      "dist_coast_km": float(ocean_tree.query(xy)[0][0]) if globe.is_land(la, lo) else 0.0,
                      "dist_river_km": float(river_tree.query(xy)[0][0]),
                      "land": int(globe.is_land(la, lo)),
                      "in_bangladesh": int(bd.contains(Point(lo, la)))})
nodes = pd.DataFrame(nodes)
nodes["block"] = (np.floor(nodes.latitude).astype(int).astype(str) + "_" +
                  np.floor(nodes.longitude).astype(int).astype(str))
nodes.to_csv(os.path.join(OUT, "node_covariates.csv"), index=False)
log("nodes:", len(nodes), "in Bangladesh:", int(nodes.in_bangladesh.sum()))

def table(clim):
    """node x month x hour rows with the target column ws100."""
    m, h = np.meshgrid(np.arange(12), np.arange(24), indexing="ij")
    out = []
    for r in nodes.itertuples():
        out.append(pd.DataFrame({"node": r.Index, "month": m.ravel() + 1, "hour": h.ravel(),
                                 "ws100": clim[:, :, r.i, r.j].ravel()}))
    t = pd.concat(out, ignore_index=True)
    return t.merge(nodes.drop(columns=["i", "j"]), left_on="node", right_index=True)

TR, TE, ALL = table(clim_train), table(clim_test), table(clim_all)
FEATS = ["latitude", "longitude", "month", "hour", "dist_coast_km", "dist_river_km", "land"]
FEATS_BASIC = ["latitude", "longitude", "month", "hour"]   # the old v1 feature set

RF_PARAMS = dict(n_estimators=300, min_samples_leaf=2, max_features=0.6, n_jobs=-1, random_state=42)
XGB_PARAMS = dict(n_estimators=1500, learning_rate=0.03, max_depth=8, min_child_weight=5,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
                  n_jobs=2, random_state=42)
def rf(): return RandomForestRegressor(**RF_PARAMS)
def xg(): return xgb.XGBRegressor(**XGB_PARAMS)

def metrics(y, p, label, test, subset):
    return {"test": test, "subset": subset, "method": label, "n": len(y),
            "rmse_ms": round(float(np.sqrt(mean_squared_error(y, p))), 3),
            "mae_ms": round(float(mean_absolute_error(y, p)), 3),
            "bias_ms": round(float(np.mean(p - y)), 3),
            "r2": round(float(r2_score(y, p)), 4)}

res = []
# ------------------------------------------------------------------ 3. Test A: new years
log("Test A: train 2016-2023, test 2024-2025")
pred = {"Climatology 2016-2023 (no model)": TR["ws100"].values}
for name, mk, F in [("Random Forest", rf, FEATS), ("XGBoost", xg, FEATS),
                    ("XGBoost, lat/lon/month/hour only", xg, FEATS_BASIC)]:
    m = mk().fit(TR[F], TR["ws100"]); pred[name] = m.predict(TE[F])
try:
    old = joblib.load(os.path.join(ROOT, "wind_model_v1.pkl"))
    pred["Old model v1 (2020-2025 sample)"] = old.predict(TE[FEATS_BASIC])
except Exception as e:
    log("old model could not be loaded:", e)
bd_mask = TE["in_bangladesh"].values == 1
for k, p in pred.items():
    res.append(metrics(TE["ws100"].values, p, k, "A: new years (2024-2025)", "all 475 nodes"))
    res.append(metrics(TE["ws100"].values[bd_mask], p[bd_mask], k, "A: new years (2024-2025)", "199 Bangladesh nodes"))
# how different are the test years themselves from the training years
log("  year-to-year: ", res[-1])

# ------------------------------------------------------------------ 4. Test B: new places
log("Test B: spatial cross-validation by 1-degree blocks")
gkf = GroupKFold(n_splits=5)
oof = {k: np.full(len(TR), np.nan) for k in ["Inverse-distance interpolation", "Random Forest",
                                              "XGBoost", "XGBoost, lat/lon/month/hour only"]}
node_xy = to_xy(nodes.latitude.values, nodes.longitude.values)
fold_of_node = np.zeros(len(nodes), int)
for f, (tr_n, te_n) in enumerate(gkf.split(nodes, groups=nodes["block"])):
    fold_of_node[te_n] = f
    trm, tem = TR["node"].isin(tr_n).values, TR["node"].isin(te_n).values
    for name, mk, F in [("Random Forest", rf, FEATS), ("XGBoost", xg, FEATS),
                        ("XGBoost, lat/lon/month/hour only", xg, FEATS_BASIC)]:
        m = mk().fit(TR.loc[trm, F], TR.loc[trm, "ws100"])
        oof[name][tem] = m.predict(TR.loc[tem, F])
    # IDW (power 2, 8 nearest training nodes) at the same month and hour
    tree = cKDTree(node_xy[tr_n]); dist, idx = tree.query(node_xy[te_n], k=8)
    w = 1.0 / np.maximum(dist, 1e-6)**2; w /= w.sum(1, keepdims=True)
    for a, n in enumerate(te_n):
        nb = tr_n[idx[a]]
        vals = clim_train[:, :, nodes.i.values[nb], nodes.j.values[nb]]    # 12 x 24 x 8
        sel = (TR["node"] == n).values
        est = (vals * w[a]).sum(-1)                                        # 12 x 24
        oof["Inverse-distance interpolation"][sel] = est.ravel()
    log(f"  fold {f+1}/5 done, {len(te_n)} nodes held out")
bdm = TR["in_bangladesh"].values == 1
for k, p in oof.items():
    res.append(metrics(TR["ws100"].values, p, k, "B: new places (1-degree blocks held out)", "all 475 nodes"))
    res.append(metrics(TR["ws100"].values[bdm], p[bdm], k, "B: new places (1-degree blocks held out)", "199 Bangladesh nodes"))
TR.assign(**{f"oof_{k}": v for k, v in oof.items()}).to_csv(os.path.join(OUT, "spatial_cv_predictions.csv"), index=False)

R = pd.DataFrame(res); R.to_csv(os.path.join(OUT, "ml_metrics.csv"), index=False)
print(R.to_string(index=False))

# ------------------------------------------------------------------ 5. final model + SHAP
log("final model on 2016-2025")
final = xg().fit(ALL[FEATS], ALL["ws100"])
final.save_model(os.path.join(OUT, "wind_model_v2_xgboost.json"))
joblib.dump(final, os.path.join(OUT, "wind_model_v2_xgboost.pkl"))
fit_p = final.predict(ALL[FEATS])
res_fit = metrics(ALL["ws100"].values, fit_p, "XGBoost final (training fit, not a test)", "fit", "all 475 nodes")

import shap, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
samp = ALL[FEATS].sample(6000, random_state=1)
sv = shap.TreeExplainer(final).shap_values(samp)
imp = pd.DataFrame({"feature": FEATS, "mean_abs_shap_ms": np.abs(sv).mean(0)}).sort_values("mean_abs_shap_ms", ascending=False)
imp.to_csv(os.path.join(OUT, "shap_importance.csv"), index=False)
NICE = {"latitude": "Latitude", "longitude": "Longitude", "month": "Month", "hour": "Hour (UTC)",
        "dist_coast_km": "Distance to coast (km)", "dist_river_km": "Distance to river (km)", "land": "Land (1) / sea (0)"}
plt.figure()
shap.summary_plot(sv, samp.rename(columns=NICE), show=False, plot_size=(7, 4.2))
plt.xlabel("SHAP value: effect on predicted 100 m wind speed (m/s)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "figures", "shap_summary.png"), dpi=300); plt.close()

# spatial CV scatter (Bangladesh nodes)
fig, ax = plt.subplots(1, 2, figsize=(8, 3.8), sharex=True, sharey=True)
for a, k in zip(ax, ["Inverse-distance interpolation", "XGBoost"]):
    y, p = TR["ws100"].values[bdm], oof[k][bdm]
    a.hexbin(y, p, gridsize=60, bins="log", cmap="Blues", mincnt=1)
    lo, hi = 0, max(y.max(), p.max()) + 0.3
    a.plot([lo, hi], [lo, hi], color="#c0392b", lw=1)
    r = R[(R.method == k) & (R.subset == "199 Bangladesh nodes") & R.test.str.startswith("B")].iloc[0]
    a.set_title(f"{k}\nRMSE {r.rmse_ms:.2f} m/s, R² {r.r2:.3f}", fontsize=9)
    a.set_xlabel("ERA5 100 m wind (m/s)")
ax[0].set_ylabel("Predicted at hidden point (m/s)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "figures", "spatial_cv_scatter.png"), dpi=300); plt.close()

# ------------------------------------------------------------------ 6. new prediction grid (24 h)
grid = ALL[["latitude", "longitude", "month", "hour"]].copy()
grid["predicted_wind_speed_ms"] = fit_p
grid["era5_mean_wind_speed_ms"] = ALL["ws100"].values
grid["wind_power_density_wm2"] = 0.5 * 1.225 * grid["predicted_wind_speed_ms"]**3
grid.sort_values(["latitude", "longitude", "month", "hour"], ascending=[False, True, True, True]) \
    .round(4).to_csv(os.path.join(OUT, "predictions_grid_v2.csv"), index=False)

json.dump({"period": "2016-2025 hourly ERA5", "target": "mean 100 m wind speed per node, month, hour (UTC)",
           "features": FEATS, "rf_params": RF_PARAMS, "xgb_params": XGB_PARAMS,
           "final_model": "XGBoost on 2016-2025", "final_fit": res_fit,
           "rows_train_climatology": len(TR), "nodes": len(nodes),
           "nodes_in_bangladesh": int(nodes.in_bangladesh.sum()),
           "spatial_cv": "GroupKFold(5) on 1-degree blocks; IDW power 2, k=8",
           "shap_importance": imp.to_dict("records")},
          open(os.path.join(OUT, "ml_summary.json"), "w"), indent=1, default=float)
log("done")
