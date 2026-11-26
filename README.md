# River-corridor wind atlas of Bangladesh

Code and data for the paper **"A GIS-Based Assessment of Wind Energy Potential Along Major River Corridors of Bangladesh"**
(A. H. Fuad and K. A. Al Mamun, United International University).

The scripts turn ten years (2016–2025) of hourly ERA5 reanalysis into a 100 m wind atlas on 1,596 named
10 km × 10 km blocks, check ERA5 against Bangladesh station records, test whether the Padma, Jamuna,
Old Brahmaputra and Meghna corridors are windier than surrounding land, train an explainable ML
climatology model, and rank sites with a five-criterion GIS-AHP model.

## Contents

| Folder | What is in it |
|---|---|
| `scripts/` | All Python code, in the order below |
| `data/` | Block grid, national outline, districts, river centerlines (GeoJSON), coastal-zone table |
| `results/` | Ranked list of all 1,596 blocks, paper tables, corridor regression, station validation, ML metrics and the 24-hour prediction grid |
| `figures/` | Paper figures |

ERA5 and station files are not included (about 400 MB); the scripts download them.

## How to reproduce

```bash
pip install -r requirements.txt
```

1. **ERA5.** Create a free Copernicus CDS account, put your key in `~/.cdsapirc`, then run
   `python scripts/download_era5_hourly.py` (u10, v10, u100, v100; 2016–2025; 20.5–26.7° N, 88.0–92.7° E).
2. **Stations.** `python scripts/download_noaa_stations.py` (NOAA Global Hourly, 36 BMD stations) and
   `python scripts/download_station_data.py` (Iowa Environmental Mesonet METAR, Dhaka and Chattogram).
3. **Block grid.** `python scripts/build_yearly_and_block_data.py` (already built: `data/wind_blocks_all.geojson`).
4. **Validation.** `python scripts/validate_era5_stations.py <project folder> <out folder>`
5. **Atlas, corridors, AHP.**
   `BLOCKS_XLSX=data/wind_blocks_zones.xlsx python scripts/paper_results.py "era5_hourly/era5_wind_20*.nc" results data/bangladesh_rivers_hydrorivers.geojson`
6. **Corridor regression.** `python scripts/corridor_statistics.py <project folder> <out folder> data/bangladesh_rivers_hydrorivers.geojson`
7. **ML model.** `python scripts/ml_model.py <project folder> data/bangladesh_rivers_hydrorivers.geojson results/ml`
8. **Figures.** `python scripts/make_figures.py results results/validation figures`

## Data sources

- ERA5 hourly single levels: Hersbach et al. (2020), Copernicus Climate Data Store.
- Station observations: NOAA Integrated Surface Database (Global Hourly); Iowa Environmental Mesonet ASOS/METAR archive.
- Rivers: HydroRIVERS v1.0 (Lehner & Grill, 2013) for the Padma, Jamuna and upper Meghna; the Old Brahmaputra and
  the Meghna estuary below Chandpur were digitised from waypoints (the `source` field in the GeoJSON says which).
- District boundaries: public administrative boundaries of Bangladesh.

## Licence

Code: MIT. Data and results: CC BY 4.0. See `LICENSE`.

## Contact

Azmal Hossain Fuad, MSc (CSE), United International University, Dhaka.
