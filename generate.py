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

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
URL = (
    "https://dd.weather.gc.ca/{day}/ensemble/geps/grib2/products/{hh}/{fff:03d}/"
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


# ----------------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------------
def url_for(run: str, var: str, fff: int) -> str:
    run_date = dt.datetime.strptime(run, "%Y%m%d%H")
    today = dt.datetime.now(dt.timezone.utc).date()
    delta = (today - run_date.date()).days
    if delta not in (0, 1):
        sys.exit(f"Run {run} is not on the datamart's today/yesterday directories.")
    day = "today" if delta == 0 else "yesterday"
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
    grbs = pygrib.open(str(path))
    try:
        q1, lats, lons = grbs.message(m1).data(**DOMAIN)
        q3, _, _ = grbs.message(m3).data(**DOMAIN)
    finally:
        grbs.close()
    return q1, q3, lats, lons


def plot_frame(q1, q3, lats, lons, label: str, run: str, fff: int, out: Path) -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cf
    from PIL import Image

    init = dt.datetime.strptime(run, "%Y%m%d%H")
    valid = init + dt.timedelta(hours=fff)
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

    mesh = ax.contourf(lons, lats, qcd, levels=np.linspace(0, 1, 101), cmap=CMAP, zorder=10)

    cax = fig.add_axes([0.915, 0.10, 0.014, 0.65])
    cbar = fig.colorbar(mesh, cax=cax, ticks=np.linspace(0, 1, 11))
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(f"QCD 12h {label} Accumulation", fontsize=20, labelpad=30)

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
        q1, q3, lats, lons = read_quartiles(grib, job["var"])
    plot_frame(q1, q3, lats, lons, job["label"], job["run"], fff, Path(job["out"]))
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
