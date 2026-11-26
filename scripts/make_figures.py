"""
make_figures.py - regenerates the paper figures from paper_results.py output.
Run: python make_figures.py <results folder> <validation folder> <figures out>
"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from shapely.geometry import shape

RES, VAL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
BLOCKS_GJ = os.environ.get("BLOCKS_GJ", "data/wind_blocks_all.geojson")
DIST_GJ = os.environ.get("DIST_GJ", "data/bangladesh_districts.geojson")
PERIOD = os.environ.get("PERIOD", "2016–2025")
os.makedirs(OUT, exist_ok=True)

INK, MUTED, GRID = "#1b2433", "#4a4f57", "#e6e8ec"
BLUE, ORANGE, GREEN, GOLD = "#2a78d6", "#eb6834", "#1baa7a", "#eba000"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#9aa3ae",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlecolor": INK,
                     "axes.spines.top": False, "axes.spines.right": False})
WAYPOINTS = {
    "Padma": [(24.60, 88.28), (24.37, 88.60), (24.15, 89.03), (23.93, 89.30), (23.87, 89.75),
              (23.65, 89.83), (23.49, 90.15), (23.43, 90.27), (23.23, 90.63)],
    "Jamuna": [(25.60, 89.68), (25.18, 89.67), (24.85, 89.62), (24.45, 89.70), (24.15, 89.72), (23.87, 89.75)],
    "Brahmaputra": [(25.18, 89.67), (24.93, 89.95), (24.75, 90.40), (24.30, 90.75), (24.05, 90.98)],
    "Meghna": [(24.05, 90.98), (23.85, 90.85), (23.55, 90.70), (23.23, 90.63), (22.75, 90.75),
               (22.20, 90.75), (21.90, 90.90)],
}
LABEL_AT = {"Brahmaputra": (90.5, 24.82), "Jamuna": (89.78, 24.55), "Padma": (88.85, 23.9), "Meghna": (90.75, 23.3)}

RIVERS_GJ = os.environ.get("RIVERS_GJ")
if RIVERS_GJ:
    RLINES = {}
    for f in json.load(open(RIVERS_GJ))["features"]:
        g = shape(f["geometry"])
        RLINES.setdefault(f["properties"]["corridor"], []).extend(
            [list(zip(*ln.xy)) for ln in (g.geoms if g.geom_type == "MultiLineString" else [g])])
else:
    RLINES = {k: [[(p[1], p[0]) for p in v]] for k, v in WAYPOINTS.items()}
B = pd.read_csv(os.path.join(RES, "blocks_all.csv"))
geo = {f["properties"]["block_id"]: shape(f["geometry"]) for f in json.load(open(BLOCKS_GJ))["features"]}
dists = [shape(f["geometry"]) for f in json.load(open(DIST_GJ))["features"]]

def polys(g):
    return [g] if g.geom_type == "Polygon" else list(g.geoms)

def block_map(values, cmap, norm, title, cbar_label, fname, rivers=True, legend=None, colors=None):
    fig, ax = plt.subplots(figsize=(6.2, 7.0))
    for bid, val in values.items():
        col = colors[bid] if colors is not None else cmap(norm(val))
        for p in polys(geo[bid]):
            x, y = p.exterior.xy
            ax.fill(x, y, color=col, lw=0.25, ec="white")
    for d in dists:
        for p in polys(d):
            x, y = p.exterior.xy; ax.plot(x, y, color="white", lw=0.4, alpha=0.8)
    if rivers:
        for name, lines in RLINES.items():
            for ln in lines:
                ax.plot([p[0] for p in ln], [p[1] for p in ln], color=INK, lw=1.4)
            if colors is None:
                ax.text(*LABEL_AT[name], name, fontsize=9, weight="bold", color=INK,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.set_aspect(1 / np.cos(np.radians(23.5)))
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)"); ax.set_title(title, loc="left")
    if colors is None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02); cb.set_label(cbar_label); cb.outline.set_visible(False)
    if legend:
        ax.legend(handles=legend, loc="upper right", frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=200); plt.close(fig)

# Fig 2 - mean speed map
bounds = [1.5, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
cm = ListedColormap(["#cfe2f8", "#9cc3ef", "#5f9fe6", "#2f7ed8", "#1f63b8", "#174a8f", "#0f3466"])
block_map(dict(zip(B.Block, B["mean"])), cm, BoundaryNorm(bounds, cm.N),
          "Ten-year mean wind speed, 1,596 blocks",
          f"Mean wind speed at 100 m, {PERIOD} (m/s)", "fig2_mean_speed_map.png")

# Fig 7 - AHP map
abounds = [0, 0.30, 0.45, 0.60, 0.75, 0.90]
acm = ListedColormap(["#fde3d6", "#f4a582", "#ec7444", "#cf5020", "#8a3010"])
block_map(dict(zip(B.Block, B.ahp)), acm, BoundaryNorm(abounds, acm.N),
          "GIS-AHP wind suitability by block", "AHP suitability score (0–1)", "fig7_ahp_map.png")

# Fig 6 - windy blocks
cols, n_c, n_r = {}, 0, 0
for _, r in B.iterrows():
    if r["mean"] >= 4 and r.zone == "Coastal belt": cols[r.Block] = ORANGE; n_c += 1
    elif r["mean"] >= 4 and r.zone == "River corridors": cols[r.Block] = BLUE; n_r += 1
    else: cols[r.Block] = "#d9dde3"
block_map(dict(zip(B.Block, B["mean"])), None, None, "Location of the windy blocks (≥ 4 m/s)", "",
          "fig6_windy_blocks.png", colors=cols,
          legend=[Patch(color=ORANGE, label=f"Coastal belt, ≥ 4 m/s ({n_c})"),
                  Patch(color=BLUE, label=f"River corridors, ≥ 4 m/s ({n_r})"),
                  Patch(color="#d9dde3", label="Other blocks")])

# Fig 3 - corridor monthly cycle
mc = pd.read_csv(os.path.join(RES, "fig3_corridor_monthly.csv"), index_col=0)
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
fig, ax = plt.subplots(figsize=(7, 4.1))
for name, c in [("Padma", BLUE), ("Jamuna", ORANGE), ("Brahmaputra", GREEN), ("Meghna", GOLD)]:
    v = mc.loc[name].values; ax.plot(range(12), v, "-o", color=c, lw=2.2, ms=4.5, label=name)
ax.set_xticks(range(12)); ax.set_xticklabels(months); ax.grid(color=GRID); ax.set_axisbelow(True)
ax.set_ylabel("Mean wind speed at 100 m (m/s)")
ax.set_title(f"Monthly wind cycle along the four river corridors ({PERIOD})", loc="left")
ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1))
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_corridor_monthly.png"), dpi=200); plt.close(fig)

# Fig 4 - latitude bands
lb = pd.read_csv(os.path.join(RES, "fig4_latitude_bands.csv"))
fig, ax = plt.subplots(figsize=(7, 3.8)); x = np.arange(len(lb)); w = 0.38
b1 = ax.bar(x - w / 2, lb.corridor_mean, w, color=BLUE, label="River corridor (≤15 km)")
b2 = ax.bar(x + w / 2, lb.other_mean, w, color="#a3c6f2", label="Other blocks, same latitude band")
for bars in (b1, b2):
    for r in bars: ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.02, f"{r.get_height():.2f}", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(x); ax.set_xticklabels([b.replace("-", "–") + "°N" for b in lb.band]); ax.set_ylim(2.5, 4.4)
ax.set_ylabel("Mean wind speed (m/s)"); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
ax.set_title("Corridor vs non-corridor blocks at the same latitude", loc="left"); ax.legend(frameon=False, ncol=2, fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_latitude_bands.png"), dpi=200); plt.close(fig)

# Fig 5 - coast vs river months
t5 = pd.read_csv(os.path.join(RES, "table5_coast_vs_river.csv"), index_col=0)
mcol = [f"m{m:02d}" for m in range(1, 13)]
c, r = t5.loc[mcol, "Coastal belt"].values, t5.loc[mcol, "River corridors"].values
fig, ax = plt.subplots(figsize=(6.5, 3.7))
ax.fill_between(range(12), r, c, color=ORANGE, alpha=0.12)
ax.plot(range(12), c, "-o", color=ORANGE, lw=2.4, ms=4.5, label=f"Coastal belt ({int(t5.loc['blocks','Coastal belt'])} blocks)")
ax.plot(range(12), r, "-o", color=BLUE, lw=2.4, ms=4.5, label=f"River corridors ({int(t5.loc['blocks','River corridors'])} blocks)")
ax.set_xticks(range(12)); ax.set_xticklabels(list("JFMAMJJASOND")); ax.grid(color=GRID); ax.set_axisbelow(True)
ax.set_ylabel("Wind speed at 100 m (m/s)"); ax.set_title("Coastal belt vs. river corridors through the year", loc="left")
ax.legend(frameon=False, loc="upper left"); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_coast_vs_river.png"), dpi=200); plt.close(fig)

# Fig 8 - ERA5 vs stations (two panels)
st = pd.read_csv(os.path.join(VAL, "station_validation.csv"))
mcy = pd.read_csv(os.path.join(VAL, "monthly_cycle.csv"), index_col=0)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
good = st.quality == "usable"
a1.scatter(st.loc[~good, "obs_mean_ms"], st.loc[~good, "era5_mean_ms"], s=18, color="#b8bec7", label="Sheltered BMD stations")
a1.scatter(st.loc[good, "obs_mean_ms"], st.loc[good, "era5_mean_ms"], s=26, color=BLUE, label="Well-exposed sites")
for _, rr in st[good].iterrows():
    nm = rr.station.replace(" (METAR)", "").replace(" (NOAA)", "")
    if "METAR" in rr.station: continue
    a1.annotate(nm.replace("Hazrat Shahjalal Intl", "Dhaka").replace("Shah Amanat Intl", "Chattogram").replace("Coxs", "Cox's")[:12],
                (rr.obs_mean_ms, rr.era5_mean_ms), fontsize=7, xytext=(3, -8), textcoords="offset points", color=INK)
a1.plot([0, 4.5], [0, 4.5], "--", color=MUTED, lw=0.8); a1.set_xlim(0, 4.5); a1.set_ylim(0, 4.5)
a1.set_xlabel("Observed mean 10 m wind (m/s)"); a1.set_ylabel("ERA5 mean 10 m wind (m/s)")
a1.set_title("(a) Station means", loc="left", fontsize=10); a1.legend(frameon=False, fontsize=7, loc="lower right")
a2.plot(range(12), mcy["obs"].values, "-o", color=INK, ms=4, lw=2, label="Observed (well-exposed sites)")
a2.plot(range(12), mcy["era5"].values, "-o", color=BLUE, ms=4, lw=2, label="ERA5 at the same sites")
a2.set_xticks(range(12)); a2.set_xticklabels(list("JFMAMJJASOND")); a2.grid(color=GRID); a2.set_axisbelow(True)
a2.set_ylabel("10 m wind speed (m/s)"); a2.set_title("(b) Seasonal cycle", loc="left", fontsize=10); a2.legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig8_era5_validation.png"), dpi=200); plt.close(fig)

# Fig 9 - corridor effect by buffer
cr = pd.read_csv(os.path.join(VAL, "corridor_regression.csv"))
fig, ax = plt.subplots(figsize=(6.5, 3.6))
for lab, c, off, mk, short in [("All four rivers, all Bangladesh nodes", BLUE, -0.8, "o", "All four rivers, national"),
                               ("All four rivers, inland nodes only (>150 km from coast)", "#9aa3ae", 0, "s", "All four rivers, inland"),
                               ("Jamuna + Brahmaputra only, inland nodes (>150 km from coast)", ORANGE, 0.8, "^", "Jamuna–Brahmaputra, inland")]:
    d = cr[cr.test == lab]
    ax.errorbar(d.buffer_km + off, d.adjusted_effect_ms, yerr=[d.adjusted_effect_ms - d.ci95_low, d.ci95_high - d.adjusted_effect_ms],
                fmt=mk, color=c, capsize=3, ms=6, lw=1.4, label=short)
ax.axhline(0, color=INK, lw=0.8); ax.set_xticks([10, 15, 25, 30])
ax.set_xlabel("Corridor buffer width (km)"); ax.set_ylabel("Corridor effect after controls (m/s)")
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
ax.set_title("River-corridor effect with 95% confidence intervals", loc="left"); ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig9_corridor_effect.png"), dpi=200); plt.close(fig)
print("figures written to", OUT)
