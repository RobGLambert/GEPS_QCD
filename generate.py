#!/usr/bin/env python3
"""Generate CMC GEPS quartile coefficient of dispersion (QCD) maps.

QCD = (Q3 - Q1) / (Q3 + Q1), computed from the GEPS probability-product GRIB2
files on the ECCC datamart. Writes PNG frames plus a manifest.json that the
static viewer in web/ reads.

Usage:
    python generate.py                       # newest complete run, snowfall
    python generate.py --run 2026102900      # specific run
    python generate.py --product SPRATE RPRATE
    python generate.py --inspect FILE.grib2  # list messages (to check Q1/Q3 indices)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import matplotlib.patheffects as pe # For Median Contour

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
URL = (
    "https://dd.weather.gc.ca/{day}/WXO-DD/ensemble/geps/grib2/products/{hh}/{fff:03d}/"
    "CMC_geps-prob_{var}-Accum-12h_SFC_0_latlon0p5x0p5_{run}_P{fff:03d}_all-products.grib2"
)
HOURS = range(12, 385, 12)

PRODUCTS = {  # datamart variable code -> display label
    "SPRATE": "Snowfall",
    "RPRATE": "Rainfall",
    "IPRATE": "Ice Pellet",
    "FPRATE": "Freezing Rain",
}

# 1-based GRIB message numbers (25th, 75th percentile) inside each *_all-products.grib2.
# The median sits between them. Verify with: python generate.py --inspect FILE.grib2
MSGS = {
    "SPRATE": (13, 15),
    "RPRATE": (15, 17),
    "IPRATE": (12, 14),
    "FPRATE": (12, 14),
}

DOMAIN = dict(lat1=40, lat2=56, lon1=-72, lon2=-48)
MIN_Q3 = 0.1  # mask points where Q3 <= this (raise it to hide noisy near-zero amounts)

# The "Accum-12h" fields are ALREADY 12-hour accumulations (kg/m^2 = mm water equivalent), even though
# the README lists the rate unit kg/(m^2 s). Evidence: raw medians are ~0.6-6.5, not ~1e-5 as a rate would be.
# So no multiplication by seconds; only convert water equivalent to the display unit.
SNOW_TO_LIQUID_RATIO = 10.0       # 10:1 => 1 mm liquid equivalent = 1 cm snow
ICE_PELLET_DEPTH_FACTOR = 1.1    # cm ice per 10 mm liquid equivalent (solid-ice approximation)

DISPLAY_UNITS = {
    "SPRATE": "cm",
    "RPRATE": "mm",
    "IPRATE": "cm",
    "FPRATE": "mm",
}

CITIES = {
    "Halifax": (-63.5728, 44.6476),
    "Fredericton": (-66.6431, 45.9636),
    "St. John's": (-52.7453, 47.5556),
}

_COLORS = [
    (220, 245, 253), (166, 225, 204), (112, 205, 156), (58, 185, 108), (83, 190, 84),
    (142, 206, 70), (202, 222, 56), (237, 221, 51), (237, 194, 59), (236, 167, 67),
    (236, 141, 75),
]
CMAP = LinearSegmentedColormap.from_list(
    "qcd", [(i / 10, tuple(c / 255 for c in rgb)) for i, rgb in enumerate(_COLORS)]
)
OCEAN = (237 / 255, 251 / 255, 255 / 255)
LAND = (251 / 255, 245 / 255, 234 / 255)

UA = {"User-Agent": "geps-qcd-site (github pages build)"}

# Median contour levels in DISPLAY units (cm for snow / ice pellet, mm for rain / freezing rain).
# Starting points only - pick levels that mean something to you (warning criteria etc.),
# and use the diagnostic cell at the bottom to see what the data actually reaches.
# Levels below MIN_Q3 are ignored automatically.
MED_LEVELS = {
    "SPRATE": [0.5, 2, 5, 10, 15, 25, 35, 50, 75, 100],
    "RPRATE": [1, 2, 5, 10, 25, 50, 75, 100, 150, 200, 250],
    "IPRATE": [0.5, 1, 2, 5, 10, 15, 20, 25, 30, 40],
    "FPRATE": [0.5, 1, 2, 5, 10, 15, 20, 25, 30, 40],
}
MED_SMOOTH_PASSES = 2   # 0 = raw field, 1 = light 3x3 smoothing, 2 = heavier. Removes single-cell spikes.


# ----------------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------------
def url_for(run: str, var: str, fff: int) -> str:
    run_date = dt.datetime.strptime(run, "%Y%m%d%H")
    day = run_date.strftime("%Y%m%d")
    return URL.format(day=day, hh=run[8:], fff=fff, var=var, run=run)


def exists(url: str) -> bool:
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=30):
            return True
    except (urllib.error.URLError, TimeoutError):
        return False


def latest_complete_run(var: str) -> str:
    """Newest run whose final forecast hour is already posted."""
    now = dt.datetime.now(dt.timezone.utc)
    for back in (0, 1):
        d = (now - dt.timedelta(days=back)).strftime("%Y%m%d")
        for hh in ("12", "00"):
            run = f"{d}{hh}"
            if dt.datetime.strptime(run, "%Y%m%d%H").replace(tzinfo=dt.timezone.utc) > now:
                continue
            if exists(url_for(run, var, max(HOURS))):
                return run
    sys.exit("No complete GEPS run found on the datamart.")


def download(url: str, dest: Path, tries: int = 3) -> None:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
            return
        except (urllib.error.URLError, TimeoutError):
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


# ----------------------------------------------------------------------------
# Computation + plotting
# ----------------------------------------------------------------------------
def compute_qcd(q1: np.ndarray, q3: np.ndarray) -> np.ma.MaskedArray:
    with np.errstate(divide="ignore", invalid="ignore"):
        qcd = np.abs((q3 - q1) / (q3 + q1))
    qcd = np.ma.masked_invalid(qcd)  # 0/0 -> dry everywhere -> transparent
    return np.ma.masked_where(np.ma.getmaskarray(qcd) | (qcd > 1) | (np.asarray(q3) <= MIN_Q3), qcd)


def read_quartiles(path: Path, var: str):
    import pygrib

    m1, m3 = MSGS[var]
    m2 = m1 + 1                      # median sits between Q1 and Q3
    grbs = pygrib.open(str(path))
    try:
        q1, lats, lons = grbs.message(m1).data(**DOMAIN)
        med, _, _ = grbs.message(m2).data(**DOMAIN)
        q3, _, _ = grbs.message(m3).data(**DOMAIN)
    finally:
        grbs.close()
    return q1, med, q3, lats, lons

def convert_to_display_units(values: np.ndarray, var: str) -> np.ndarray:
    """Convert the 12-hour accumulation (kg m-2 = mm water equivalent) to the displayed unit."""
    water_equiv_mm = np.asarray(values)   # already an accumulation; 1 kg m-2 of water = 1 mm

    if var in ("RPRATE", "FPRATE"):
        return water_equiv_mm  # mm liquid-equivalent accumulation
    if var == "SPRATE":
        # mm water equivalent * snow-to-liquid ratio / 10 mm per cm
        return water_equiv_mm * SNOW_TO_LIQUID_RATIO / 10.0
    if var == "IPRATE":
        # Approximate solid-ice depth: ~1.1 mm ice per mm water equivalent, then mm -> cm.
        return water_equiv_mm * ICE_PELLET_DEPTH_FACTOR / 10.0
    raise ValueError(f"Unknown precipitation product: {var}")


def smooth_field(a, passes=1):
    """Light [1 2 1]/4 smoothing in each direction, numpy only. Masked/missing cells count as 0."""
    a = np.ma.filled(np.ma.asarray(a, dtype=float), 0.0)
    k = (0.25, 0.5, 0.25)
    for _ in range(passes):
        for axis in (0, 1):
            n = a.shape[axis]
            p = np.pad(a, [(1, 1) if ax == axis else (0, 0) for ax in (0, 1)], mode="edge")
            a = (k[0] * np.take(p, range(0, n), axis) +
                 k[1] * np.take(p, range(1, n + 1), axis) +
                 k[2] * np.take(p, range(2, n + 2), axis))
    return a

def plot_frame(q1, med, q3, lats, lons, label: str, run: str, fff: int, out: Path, var: str) -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cf
    from PIL import Image

    init = dt.datetime.strptime(run, "%Y%m%d%H")
    valid = init + dt.timedelta(hours=fff)

    # Convert all percentile fields consistently. QCD remains dimensionless,
    # while MIN_Q3 is now interpreted in the displayed accumulation units.
    q1 = convert_to_display_units(q1, var)
    med = convert_to_display_units(med, var)
    q3 = convert_to_display_units(q3, var)
    qcd = compute_qcd(q1, q3)

    # after: figure shape follows the domain, so the map fills the axes box
    AX_W, AX_H = 0.88, 0.82        # keep in sync with add_axes below
    aspect = (lons.max() - lons.min()) / (lats.max() - lats.min())
    fig = plt.figure(figsize=(18 * aspect * AX_H / AX_W, 18), dpi=60)
    ax = fig.add_axes([0.02, 0.03, AX_W, AX_H], projection=ccrs.PlateCarree())
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()])

    def ne(cat, name, **kw):
        return cf.NaturalEarthFeature(cat, name, "10m", **kw)

    ax.add_feature(ne("physical", "lakes", edgecolor="face", facecolor="none"), edgecolor="lightblue", zorder=6)
    ax.add_feature(ne("physical", "rivers_lake_centerlines", edgecolor="face", facecolor="none"), edgecolor="lightblue", zorder=6)
    ax.add_feature(ne("physical", "ocean", edgecolor="face", facecolor=OCEAN), edgecolor="lightblue", zorder=6)
    ax.add_feature(ne("cultural", "admin_0_countries", facecolor=LAND), edgecolor="black", zorder=6)
    ax.add_feature(ne("physical", "coastline", edgecolor="face", facecolor="none"), edgecolor="black", zorder=11)
    ax.add_feature(ne("cultural", "admin_0_countries", facecolor="none"), edgecolor="black", zorder=11)
    ax.add_feature(ne("cultural", "admin_1_states_provinces_lines", facecolor="none"), edgecolor="darkgray", zorder=11)

    # mesh = ax.contourf(lons, lats, qcd, levels=np.linspace(0, 1, 101), cmap=CMAP, zorder=10)
    mesh = ax.pcolormesh(lons, lats, qcd, cmap=CMAP, vmin=0, vmax=1, shading="nearest", transform=ccrs.PlateCarree(), zorder=10)

    # Median contours. The median is always <= Q3, so any level >= MIN_Q3 already sits inside the
    # area where QCD is drawn; masking the field to the QCD area (as before) was not needed and
    # created stacked lines along its edge. Light smoothing removes single-cell spikes.
    med_s = smooth_field(med, MED_SMOOTH_PASSES)
    levels = [lv for lv in MED_LEVELS[var] if MIN_Q3 <= lv < med_s.max()]
    if levels:
        cs = ax.contour(lons, lats, med_s, levels=levels, colors="k", linewidths=1.2,
                        transform=ccrs.PlateCarree(), zorder=13)
        ax.clabel(cs, fmt="%g", fontsize=16, inline=True, inline_spacing=10)
        for t in cs.labelTexts:                      # white halo so labels read over the colours
            t.set_path_effects([pe.withStroke(linewidth=4, foreground="white")])

    cax = fig.add_axes([0.915, 0.10, 0.014, 0.65])
    cbar = fig.colorbar(mesh, cax=cax, ticks=np.linspace(0, 1, 11))
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label("Quartile Coefficient of Dispersion (QCD)", fontsize=20, labelpad=30)

    # --- 2. ADD AGREEMENT & DISAGREEMENT LABELS ---
    cbar.ax.text(0.5, -0.08, "High Agreement\n(Low Spread)", transform=cbar.ax.transAxes,
                 fontsize=18, va="center", ha="left", fontweight="bold")
    cbar.ax.text(0.5, 1.08, "High Disagreement\n(High Spread)", transform=cbar.ax.transAxes,
                 fontsize=18, va="center", ha="left", fontweight="bold")

    for name, (lon, lat) in CITIES.items():
        ax.text(lon, lat, name, color="k", fontsize=22, ha="center", va="center",
                transform=ccrs.PlateCarree(), zorder=12,
                bbox=dict(boxstyle="round", ec="#121212", fc="#fadede", pad=0.2))

    ax.set_title(f"CMC GEPS\n12h {label} Accumulation QCD", loc="left", fontsize=30, fontweight="bold")
    ax.set_title(
        f"Init {init:%H}Z - {init:%d %b %Y}\nHour T+{fff}\nValid {valid:%H}Z - {valid:%d %b %Y}",
        loc="right", fontsize=20, fontweight="bold",
    )
    fig.text(0.02, 0.005, f"Black contours: ensemble median 12h accumulation ({DISPLAY_UNITS[var]})", fontsize=22)
    fig.text(0.70, 0.005, "Plot Made by: Rob Lambert", fontsize=22, ha="center")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.png")
    fig.savefig(tmp, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)  # the notebook never freed figures; matters across 32 frames

    # Palette-quantize: ~3-4x smaller PNGs, no visible change for flat map colours
    Image.open(tmp).convert("RGB").quantize(256).save(out, optimize=True)
    tmp.unlink()


def worker(job: dict) -> int:
    fff = job["fff"]
    with tempfile.TemporaryDirectory() as td:
        grib = Path(td) / f"{fff:03d}.grib2"
        download(job["url"], grib)
        q1, med, q3, lats, lons = read_quartiles(grib, job["var"])
    plot_frame(q1, med, q3, lats, lons, job["label"], job["run"], fff, Path(job["out"]), job["var"])
    return fff


def prewarm_cartopy() -> None:
    """Fetch Natural Earth files once so worker processes don't race on download."""
    from cartopy.io import shapereader as sr

    for cat, name in [("physical", "lakes"), ("physical", "rivers_lake_centerlines"),
                      ("physical", "ocean"), ("physical", "coastline"),
                      ("cultural", "admin_0_countries"),
                      ("cultural", "admin_1_states_provinces_lines")]:
        sr.natural_earth(resolution="10m", category=cat, name=name)


# ----------------------------------------------------------------------------
def inspect(path: str) -> None:
    import pygrib

    for g in pygrib.open(path):
        print(g)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="dist", help="output directory (default: dist)")
    ap.add_argument("--run", help="run as YYYYMMDDHH (default: newest complete run)")
    ap.add_argument("--product", nargs="+", default=list(PRODUCTS), choices=list(PRODUCTS),
                    help="default: all products")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--inspect", metavar="GRIB2", help="print messages in a file and exit")
    args = ap.parse_args()

    if args.inspect:
        return inspect(args.inspect)

    run = args.run or latest_complete_run(args.product[0])
    print(f"Using run {run}")
    out = Path(args.out)
    prewarm_cartopy()

    manifest = {"run": run, "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "products": {}}
    init = dt.datetime.strptime(run, "%Y%m%d%H")

    for var in args.product:
        label = PRODUCTS[var]
        jobs = [dict(fff=h, run=run, var=var, label=label, url=url_for(run, var, h),
                     out=str(out / "img" / var / f"f{h:03d}.png")) for h in HOURS]
        done = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(worker, j): j["fff"] for j in jobs}
            for f in as_completed(futs):
                try:
                    done.append(f.result())
                    print(f"  {var} T+{futs[f]:03d} done")
                except Exception as e:  # keep going; a missing hour just leaves a gap
                    print(f"  {var} T+{futs[f]:03d} FAILED: {e}", file=sys.stderr)
        if not done:  # skip this product rather than sinking the whole deploy
            print(f"WARNING: no frames produced for {var}; skipping it.", file=sys.stderr)
            continue
        manifest["products"][var] = {
            "label": label,
            "frames": [
                {"hour": h, "valid": (init + dt.timedelta(hours=h)).strftime("%Y-%m-%dT%H:%MZ"),
                 "file": f"img/{var}/f{h:03d}.png"}
                for h in sorted(done)
            ],
        }

    if not manifest["products"]:
        sys.exit("No products produced any frames.")
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "data" / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("Wrote manifest.")


if __name__ == "__main__":
    main()
