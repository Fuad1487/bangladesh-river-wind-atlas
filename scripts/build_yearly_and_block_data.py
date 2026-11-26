"""
build_yearly_and_block_data.py  (v3)

Builds the 10km x 10km block grid for the website.

WHAT CHANGED IN v3
--------------------------------------------------------------------
1. BLOCKS ARE CLIPPED TO BANGLADESH. Each block is the real intersection of
   a 10km cell with the national territory, so edge blocks follow the border
   and coastline instead of sticking out over India or the Bay of Bengal.
   Cells with less than MIN_LAND_FRACTION of their area inside Bangladesh
   are dropped.
2. EVERY BLOCK HAS A STABLE NAME: A0001, A0002, ... counted from the
   north-west corner, row by row. The same block has the same name in every
   year file.
3. DISTRICT = the district that covers the largest part of the block
   (not just the district under the block's centre point).
4. WIND SPEED IS NOW THE MEAN OF SPEEDS, NOT THE SPEED OF A MEAN VECTOR.
   The earlier version averaged the u/v wind components over a whole year
   and then took the length of that vector. Bangladesh's wind reverses with
   the monsoon, so the two seasons partly cancelled and the result
   understated real wind speed (and, because power goes with speed cubed,
   understated power far more). v3 averages |wind| itself over every
   2-hourly sample, and averages power density the same way (mean of
   0.5*rho*v^3, not 0.5*rho*(mean v)^3). Your ranking numbers will be
   higher than the old version's - that is the correction, not a bug.
5. Also writes bangladesh_outline.geojson, the national outline the website
   uses to crop the colour overlay (built from the same district file, so
   the crop and the blocks line up exactly).

REQUIREMENTS
--------------------------------------------------------------------
  pip install shapely
  bangladesh_districts.geojson in this same folder
  bangladesh_wind_2020.nc ... bangladesh_wind_2025.nc in this same folder

HONESTY NOTE ABOUT RESOLUTION
--------------------------------------------------------------------
ERA5's real resolution is ~27km. Block values are interpolated between
those points. They give a smooth, easy-to-read ranking surface; they are not
independently measured 10km data and cannot show true local effects such as
channelling at a particular river bend.

OUTPUTS (put in your website's data/ folder)
--------------------------------------------------------------------
  bangladesh_outline.geojson
  wind_blocks_2020.geojson ... wind_blocks_2025.geojson
  wind_blocks_all.geojson          (all years combined)

Run:   python build_yearly_and_block_data.py
"""

import glob
import json
import math

import numpy as np
import pandas as pd
import xarray as xr

try:
    from shapely.geometry import shape, box, mapping, Polygon, MultiPolygon
    from shapely.ops import unary_union
    from shapely.strtree import STRtree
except ImportError:
    print("Missing library. Run: pip install shapely")
    raise SystemExit(1)

DATA_PATTERN = "bangladesh_wind_*.nc"
DISTRICTS_FILE = "bangladesh_districts.geojson"
AIR_DENSITY = 1.225

BLOCK_SIZE_KM = 10.0
MEAN_LAT_FOR_PROJECTION = 23.5      # sizes the longitude step in degrees
GRID_LA1 = 26.68                    # centre latitude of the top row
GRID_LO1 = 88.00                    # centre longitude of the left column
MIN_LAND_FRACTION = 0.05            # drop cells < 5% inside Bangladesh

RIVERSIDE_THRESHOLD_KM = 15.0
COASTAL_THRESHOLD_KM = 15.0

RIVER_CORRIDORS = {
    "Padma": [(24.60, 88.28), (24.37, 88.60), (24.15, 89.03), (23.93, 89.30),
              (23.87, 89.75), (23.65, 89.83), (23.49, 90.15), (23.43, 90.27),
              (23.23, 90.63)],
    "Jamuna": [(25.60, 89.68), (25.18, 89.67), (24.85, 89.62), (24.45, 89.70),
               (24.15, 89.72), (23.87, 89.75)],
    "Brahmaputra": [(25.18, 89.67), (24.93, 89.95), (24.75, 90.40),
                    (24.30, 90.75), (24.05, 90.98)],
    "Meghna": [(24.05, 90.98), (23.85, 90.85), (23.55, 90.70), (23.23, 90.63),
               (22.75, 90.75), (22.20, 90.75), (21.90, 90.90)],
}
COASTAL_CORRIDORS = {
    "Sundarban-Bhola Coast": [(21.60, 89.20), (21.75, 89.50), (21.90, 89.75),
                              (22.00, 90.10), (22.10, 90.35), (22.00, 90.55),
                              (21.85, 90.70)],
    "Chattogram-Coxs Bazar Coast": [(22.23, 91.80), (22.05, 91.88),
                                    (21.85, 91.93), (21.65, 91.97),
                                    (21.43, 91.98)],
}


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
def point_to_segment_km(lat, lon, lat1, lon1, lat2, lon2):
    ref_lat = (lat1 + lat2) / 2
    kx = 111.320 * math.cos(math.radians(ref_lat))
    ky = 110.574
    px, py = lon * kx, lat * ky
    ax, ay = lon1 * kx, lat1 * ky
    bx, by = lon2 * kx, lat2 * ky
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def min_distance_to_corridor(lat, lon, waypoints):
    return min(
        point_to_segment_km(lat, lon, a[0], a[1], b[0], b[1])
        for a, b in zip(waypoints[:-1], waypoints[1:])
    )


def classify_block(lat, lon):
    riverside = any(min_distance_to_corridor(lat, lon, wp) <= RIVERSIDE_THRESHOLD_KM
                    for wp in RIVER_CORRIDORS.values())
    coastal = any(min_distance_to_corridor(lat, lon, wp) <= COASTAL_THRESHOLD_KM
                  for wp in COASTAL_CORRIDORS.values())
    return coastal, riverside


def round_coords(obj, nd=4):
    if isinstance(obj, (list, tuple)):
        if obj and isinstance(obj[0], (int, float)):
            return [round(obj[0], nd), round(obj[1], nd)]
        return [round_coords(o, nd) for o in obj]
    return obj


def polygons_only(geom):
    """Return a list of Polygon parts from any geometry."""
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        out = []
        for g in geom.geoms:
            out.extend(polygons_only(g))
        return out
    return []


def bilinear(field, lat, lon, la1, lo1, dy, dx):
    """Bilinear interpolation on the native ERA5 grid (rows run north->south).
    Points beyond the grid edge take the edge value."""
    ny, nx = field.shape
    gy = min(max((la1 - lat) / dy, 0), ny - 1)
    gx = min(max((lon - lo1) / dx, 0), nx - 1)
    iy0, ix0 = int(math.floor(gy)), int(math.floor(gx))
    iy1, ix1 = min(iy0 + 1, ny - 1), min(ix0 + 1, nx - 1)
    fy, fx = gy - iy0, gx - ix0
    top = field[iy0, ix0] + (field[iy0, ix1] - field[iy0, ix0]) * fx
    bot = field[iy1, ix0] + (field[iy1, ix1] - field[iy1, ix0]) * fx
    return float(top + (bot - top) * fy)


# ----------------------------------------------------------------------------
def main():
    files = sorted(glob.glob(DATA_PATTERN))
    if not files:
        print(f"No files matching {DATA_PATTERN} found in this folder.")
        return
    try:
        districts_geojson = json.load(open(DISTRICTS_FILE))
    except FileNotFoundError:
        print(f"Missing {DISTRICTS_FILE} - put it in this same folder.")
        return

    # ---- national outline + district shapes -------------------------------
    print("Building the national outline from the district file...")
    dist_names, dist_divs, dist_geoms = [], [], []
    for feat in districts_geojson["features"]:
        dist_names.append(feat["properties"]["district"])
        dist_divs.append(feat["properties"].get("division", ""))
        dist_geoms.append(shape(feat["geometry"]).buffer(0))

    union = unary_union(dist_geoms)
    # close the hairline gaps between adjacent (simplified) districts
    closed = union.buffer(0.004, join_style=2).buffer(-0.004, join_style=2)
    parts = [Polygon(p.exterior) for p in polygons_only(closed) if p.area > 1e-6]
    country = unary_union(parts)
    country_light = country.simplify(0.0015, preserve_topology=True)
    gm = mapping(country_light)
    outline = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {"name": "Bangladesh"},
                      "geometry": {"type": gm["type"],
                                   "coordinates": round_coords(gm["coordinates"], 4)}}],
    }
    with open("bangladesh_outline.geojson", "w") as f:
        json.dump(outline, f)
    print(f"  Saved bangladesh_outline.geojson "
          f"(bounds lon {country.bounds[0]:.2f}-{country.bounds[2]:.2f}, "
          f"lat {country.bounds[1]:.2f}-{country.bounds[3]:.2f})")

    dist_tree = STRtree(dist_geoms)

    # ---- ERA5 statistics ----------------------------------------------------
    print(f"\nLoading {len(files)} file(s): {files}")
    ds = xr.open_mfdataset(DATA_PATTERN, combine="by_coords")
    u = ds["u100"].to_dataframe(name="u100").reset_index()
    v = ds["v100"].to_dataframe(name="v100").reset_index()
    df = u.merge(v, on=["valid_time", "latitude", "longitude"])
    df["speed"] = np.sqrt(df["u100"] ** 2 + df["v100"] ** 2)
    df["speed3"] = df["speed"] ** 3
    df["year"] = pd.to_datetime(df["valid_time"]).dt.year

    grid_lats = sorted(df["latitude"].unique(), reverse=True)
    grid_lons = sorted(df["longitude"].unique())
    nx, ny = len(grid_lons), len(grid_lats)
    la1, lo1 = grid_lats[0], grid_lons[0]
    dx = round(grid_lons[1] - grid_lons[0], 4)
    dy = round(grid_lats[0] - grid_lats[1], 4)
    print(f"Native ERA5 grid: {nx} x {ny}, spacing {dx} deg")

    def to_array(frame, col):
        arr = np.zeros((ny, nx))
        for _, r in frame.iterrows():
            i = int(round((la1 - r["latitude"]) / dy))
            j = int(round((r["longitude"] - lo1) / dx))
            arr[i, j] = r[col]
        return arr

    fields = {}   # key -> (mean speed grid, mean speed^3 grid)
    for year in sorted(df["year"].unique()):
        g = df[df["year"] == year].groupby(["latitude", "longitude"])[["speed", "speed3"]].mean().reset_index()
        fields[str(year)] = (to_array(g, "speed"), to_array(g, "speed3"))
    g = df.groupby(["latitude", "longitude"])[["speed", "speed3"]].mean().reset_index()
    fields["all"] = (to_array(g, "speed"), to_array(g, "speed3"))

    # ---- 10km grid, clipped to Bangladesh ---------------------------------
    lat_step = BLOCK_SIZE_KM / 111.0
    lon_step = BLOCK_SIZE_KM / (111.0 * math.cos(math.radians(MEAN_LAT_FOR_PROJECTION)))
    half_lat, half_lon = lat_step / 2, lon_step / 2
    minx, miny, maxx, maxy = country.bounds

    n_rows = int(math.ceil((GRID_LA1 + half_lat - miny) / lat_step)) + 1
    n_cols = int(math.ceil((maxx + half_lon - GRID_LO1) / lon_step)) + 1
    print(f"\nScanning {n_rows} x {n_cols} candidate cells...")

    blocks = []
    for r in range(n_rows):
        lat = GRID_LA1 - r * lat_step
        if lat + half_lat < miny:
            break
        for c in range(n_cols):
            lon = GRID_LO1 + c * lon_step
            cell = box(lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat)
            if not cell.intersects(country):
                continue
            clipped = cell.intersection(country)
            frac = clipped.area / cell.area
            if frac < MIN_LAND_FRACTION:
                continue

            # district with the largest overlap
            best_i, best_a = None, 0.0
            for i in dist_tree.query(cell):
                a = cell.intersection(dist_geoms[int(i)]).area
                if a > best_a:
                    best_i, best_a = int(i), a
            if best_i is None:
                continue

            coastal, riverside = classify_block(lat, lon)
            blocks.append({
                "row": r, "col": c, "lat": lat, "lon": lon,
                "geom": clipped, "frac": frac,
                "district": dist_names[best_i], "division": dist_divs[best_i],
                "coastal": coastal, "riverside": riverside,
            })

    width = max(4, len(str(len(blocks))))
    for n, b in enumerate(blocks, start=1):
        b["block_id"] = f"A{n:0{width}d}"
    print(f"Kept {len(blocks)} blocks inside Bangladesh "
          f"(A{1:0{width}d} .. A{len(blocks):0{width}d}).")

    grid_meta = {
        "la1": GRID_LA1, "lo1": GRID_LO1,
        "lat_step": lat_step, "lon_step": lon_step,
        "block_size_km": BLOCK_SIZE_KM,
    }

    for key, (spd_grid, cube_grid) in fields.items():
        features = []
        for b in blocks:
            speed = bilinear(spd_grid, b["lat"], b["lon"], la1, lo1, dy, dx)
            power = 0.5 * AIR_DENSITY * bilinear(cube_grid, b["lat"], b["lon"], la1, lo1, dy, dx)
            parts = polygons_only(b["geom"])
            if len(parts) == 1:
                geometry = {"type": "Polygon",
                            "coordinates": round_coords(mapping(parts[0])["coordinates"], 4)}
            else:
                geometry = {"type": "MultiPolygon",
                            "coordinates": [round_coords(mapping(p)["coordinates"], 4) for p in parts]}
            features.append({
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "block_id": b["block_id"],
                    "row": b["row"], "col": b["col"],
                    "lat": round(b["lat"], 4), "lon": round(b["lon"], 4),
                    "district": b["district"], "division": b["division"],
                    "in_bangladesh": True,
                    "is_coastal": b["coastal"], "is_riverside": b["riverside"],
                    "land_fraction": round(b["frac"], 2),
                    "wind_speed_ms": round(speed, 2),
                    "power_density_wm2": round(power, 1),
                },
            })
        out = {"type": "FeatureCollection", "grid": grid_meta,
               "period": key, "features": features}
        name = f"wind_blocks_{key}.geojson"
        with open(name, "w") as f:
            json.dump(out, f)
        print(f"  Saved {name} ({len(features)} blocks)")

    print("\nDone. Copy bangladesh_outline.geojson and every wind_blocks_*.geojson")
    print("into your website's data/ folder.")


if __name__ == "__main__":
    main()
