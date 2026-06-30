"""
AQI India Observatory — Flask backend
======================================
Serves hourly AQI / pollutant NetCDF grids as PNG map overlays, pixel-level
hourly time series, and summary statistics, for an interactive Leaflet +
Chart.js frontend (templates/index.html).

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000

Place your .nc files (e.g. aqi_2024-02-10.nc) inside ./data/
"""

import io
import os
import re
import glob
import numpy as np
import xarray as xr
from flask import Flask, jsonify, request, send_file, render_template
from PIL import Image
from functools import lru_cache

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FILE_RE = re.compile(r"aqi_(\d{4}-\d{2}-\d{2})\.nc$")

# ----------------------------------------------------------------------------
# Manual overlay calibration
# ----------------------------------------------------------------------------
# If, after restarting the server, the colored overlay still sits a little
# off from the real coastline/border in a consistent direction, dial these in
# (degrees) rather than guessing blind — open the map, compare a sharp known
# point (e.g. where the coast at Mumbai or the Pakistan border should be) to
# where the overlay color edge actually falls, and nudge accordingly.
#   overlay appears too far NORTH ("above" where it should be) -> lower this:
GRID_LAT_OFFSET_DEG = -0.6
#   overlay appears too far EAST -> lower this (negative shifts it west):
GRID_LON_OFFSET_DEG = 0.0

# Upscales the raw grid before colorizing so the rendered PNG blends smoothly
# between neighboring cells (bilinear) instead of showing each grid cell as a
# single flat-colored square. This is what makes the *map itself* look like
# a continuous gradient, matching the legend — not just the color math.
OVERLAY_SMOOTH_FACTOR = 3  # 1 = off (raw pixel grid, blocky)

VARIABLES = ["AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]
 
VAR_UNITS = {
    "AQI": "index",
    "PM2.5": "µg/m³",
    "PM10": "µg/m³",
    "NO2": "ppb",
    "SO2": "ppb",
    "CO": "ppm",
    "Ozone": "ppb",
}

# US EPA style AQI breakpoint colors (value -> RGB)
# NOTE: previously each category had two stops at identical color (e.g. 50/51
# both yellow boundaries), which produced *hard* banded edges between
# categories instead of a smooth gradient. This version uses a single stop per
# breakpoint so colors interpolate continuously across the whole 0-500 range
# (still anchored at the familiar AQI breakpoints), giving a true gradient
# instead of flat color "classes".
AQI_STOPS = [
    (0,   (0, 228, 0)),     # Good - green
    (50,  (0, 228, 0)),
    (100, (255, 255, 0)),   # Moderate - yellow
    (150, (255, 126, 0)),   # USG - orange
    (200, (255, 0, 0)),     # Unhealthy - red
    (300, (143, 63, 151)),  # Very Unhealthy - purple
    (500, (126, 0, 35)),    # Hazardous - maroon
]

# Generic cool->warm gradient used for non-AQI pollutants (normalized 0..1)
GENERIC_STOPS = [
    (0.00, (12, 12, 60)),
    (0.15, (40, 30, 130)),
    (0.35, (110, 30, 160)),
    (0.55, (220, 60, 120)),
    (0.75, (255, 140, 40)),
    (1.00, (255, 245, 120)),
]


# ----------------------------------------------------------------------------
# File discovery / caching
# ----------------------------------------------------------------------------

def list_data_files():
    files = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "aqi_*.nc"))):
        m = FILE_RE.search(os.path.basename(path))
        if m:
            files.append({"id": os.path.basename(path), "date": m.group(1)})
    return files


@lru_cache(maxsize=8)
def open_dataset(file_id):
    path = os.path.join(DATA_DIR, file_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(file_id)
    # Keep file handle open & cached; small enough (~130MB) to load lazily.
    return xr.open_dataset(path)


def safe_var(ds, var):
    if var not in ds.data_vars:
        raise KeyError(var)
    return ds[var]


# ----------------------------------------------------------------------------
# Color mapping helpers
# ----------------------------------------------------------------------------

def smooth_upscale(arr, factor):
    """Bilinear-upscale a 2D array that may contain NaNs, without letting NaN
    (invalid/no-data) cells bleed color into valid neighbors. Uses normalized
    convolution: each output sample is the weighted average of only the valid
    input cells that cover it, so the resampled field stays smooth *within*
    the data region while the data/no-data boundary stays clean.
    """
    if factor <= 1:
        return arr
    h, w = arr.shape
    valid = (~np.isnan(arr)).astype(np.float32)
    filled = np.where(valid > 0, arr, 0.0).astype(np.float32)

    img_vals = Image.fromarray(filled, mode="F")
    img_mask = Image.fromarray(valid, mode="F")

    new_size = (w * factor, h * factor)
    big_vals = np.asarray(img_vals.resize(new_size, Image.Resampling.BILINEAR), dtype=np.float64)
    big_mask = np.asarray(img_mask.resize(new_size, Image.Resampling.BILINEAR), dtype=np.float64)

    out = np.full((new_size[1], new_size[0]), np.nan, dtype=np.float64)
    ok = big_mask > 0.35  # require a reasonable fraction of valid coverage
    out[ok] = big_vals[ok] / big_mask[ok]
    return out


def colorize_aqi(arr):
    """arr: 2D float array of AQI values (NaN allowed) -> RGBA uint8 array."""
    h, w = arr.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    valid = ~np.isnan(arr)
    vals = np.clip(arr, 0, 500)

    xs = np.array([s[0] for s in AQI_STOPS], dtype=np.float64)
    rs = np.array([s[1][0] for s in AQI_STOPS], dtype=np.float64)
    gs = np.array([s[1][1] for s in AQI_STOPS], dtype=np.float64)
    bs = np.array([s[1][2] for s in AQI_STOPS], dtype=np.float64)

    r = np.interp(vals, xs, rs)
    g = np.interp(vals, xs, gs)
    b = np.interp(vals, xs, bs)

    rgba[..., 0] = r.astype(np.uint8)
    rgba[..., 1] = g.astype(np.uint8)
    rgba[..., 2] = b.astype(np.uint8)
    rgba[..., 3] = np.where(valid, 215, 0).astype(np.uint8)
    return rgba


def colorize_generic(arr, vmin, vmax):
    h, w = arr.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    valid = ~np.isnan(arr)
    rng = max(vmax - vmin, 1e-9)
    norm = np.clip((arr - vmin) / rng, 0, 1)

    xs = np.array([s[0] for s in GENERIC_STOPS])
    rs = np.array([s[1][0] for s in GENERIC_STOPS], dtype=np.float64)
    gs = np.array([s[1][1] for s in GENERIC_STOPS], dtype=np.float64)
    bs = np.array([s[1][2] for s in GENERIC_STOPS], dtype=np.float64)

    r = np.interp(norm, xs, rs)
    g = np.interp(norm, xs, gs)
    b = np.interp(norm, xs, bs)

    rgba[..., 0] = r.astype(np.uint8)
    rgba[..., 1] = g.astype(np.uint8)
    rgba[..., 2] = b.astype(np.uint8)
    rgba[..., 3] = np.where(valid, 215, 0).astype(np.uint8)
    return rgba


# ----------------------------------------------------------------------------
# Routes — pages
# ----------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ----------------------------------------------------------------------------
# Routes — API
# ----------------------------------------------------------------------------

@app.route("/api/files")
def api_files():
    return jsonify({"files": list_data_files(), "variables": VARIABLES, "units": VAR_UNITS})


@app.route("/api/meta/<file_id>")
def api_meta(file_id):
    try:
        ds = open_dataset(file_id)
    except FileNotFoundError:
        return jsonify({"error": "file not found"}), 404

    lat = ds["lat"].values
    lon = ds["lon"].values

    lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
    lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))

    # Manual calibration nudge (see constants near the top of the file) — use
    # this if the overlay is consistently offset from the real coastline by a
    # small, fixed amount in one direction.
    lat_min += GRID_LAT_OFFSET_DEG
    lat_max += GRID_LAT_OFFSET_DEG
    lon_min += GRID_LON_OFFSET_DEG
    lon_max += GRID_LON_OFFSET_DEG

    return jsonify({
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "n_lat": int(lat.size),
        "n_lon": int(lon.size),
        "n_hours": int(ds["time"].size),
        "variables": [v for v in VARIABLES if v in ds.data_vars],
    })


@app.route("/api/stats/<file_id>/<var>/<int:hour>")
def api_stats(file_id, var, hour):
    try:
        ds = open_dataset(file_id)
        da = safe_var(ds, var).isel(time=hour)
    except (FileNotFoundError, KeyError, IndexError):
        return jsonify({"error": "invalid request"}), 404

    arr = da.values.astype(np.float64)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return jsonify({"min": None, "max": None, "mean": None, "count": 0})

    return jsonify({
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "count": int(valid.size),
    })


@app.route("/api/overlay/<file_id>/<var>/<int:hour>.png")
def api_overlay(file_id, var, hour):
    try:
        ds = open_dataset(file_id)
        da = safe_var(ds, var).isel(time=hour)
    except (FileNotFoundError, KeyError, IndexError):
        return jsonify({"error": "invalid request"}), 404

    arr = da.values.astype(np.float64)
    arr = np.where(arr < -1e3, np.nan, arr)  # guard against bad sentinel fill values

    if OVERLAY_SMOOTH_FACTOR > 1:
        arr = smooth_upscale(arr, OVERLAY_SMOOTH_FACTOR)

    if var == "AQI":
        rgba = colorize_aqi(arr)
    else:
        vmin = float(request.args.get("vmin", np.nanmin(arr) if np.any(~np.isnan(arr)) else 0))
        vmax = float(request.args.get("vmax", np.nanmax(arr) if np.any(~np.isnan(arr)) else 1))
        rgba = colorize_generic(arr, vmin, vmax)

    # lat is ascending (south->north); image rows must go north->south (top to bottom)
    rgba = np.flipud(rgba)

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    resp = send_file(buf, mimetype="image/png")
    resp.headers["Cache-Control"] = "public, max-age=3600"
    return resp


@app.route("/api/pixel")
def api_pixel():
    file_id = request.args.get("file")
    lat = float(request.args.get("lat"))
    lon = float(request.args.get("lon"))

    try:
        ds = open_dataset(file_id)
    except FileNotFoundError:
        return jsonify({"error": "file not found"}), 404

    lat_arr = ds["lat"].values
    lon_arr = ds["lon"].values
    if lat < lat_arr.min() or lat > lat_arr.max() or lon < lon_arr.min() or lon > lon_arr.max():
        return jsonify({"error": "out of bounds"}), 400

    iy = int(np.argmin(np.abs(lat_arr - lat)))
    ix = int(np.argmin(np.abs(lon_arr - lon)))

    series = {}
    for v in VARIABLES:
        if v not in ds.data_vars:
            continue
        vals = ds[v].isel(lat=iy, lon=ix).values.astype(np.float64)
        vals = np.where(vals < -1e3, np.nan, vals)
        series[v] = [None if np.isnan(x) else round(float(x), 3) for x in vals]

    return jsonify({
        "lat": float(lat_arr[iy]),
        "lon": float(lon_arr[ix]),
        "hours": list(range(len(ds["time"]))),
        "series": series,
    })


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "files": len(list_data_files())})


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Data directory: {DATA_DIR}")
    print(f"Found {len(list_data_files())} NetCDF file(s).")
    app.run(debug=True, host="0.0.0.0", port=8090)
