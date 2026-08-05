

import os
import sys
import zipfile
import tempfile
import shutil
import argparse
import logging
from pathlib import Path

import numpy as np

try:
    import xarray as xr
except Exception as exc:                                   
    sys.exit(f"xarray requis : pip install xarray netCDF4 h5netcdf  ({exc})")

try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = os.getcwd()
sys.path.insert(0, _SCRIPT_DIR)
from route_config_v2 import ROUTES, ERA5_CONFIG, EXPECTED_SHORT

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

DATA_DIR = Path("ERA5_data_v2")

VAR_ALIASES = {
    "u10": ["u10", "10m_u_component_of_wind"],
    "v10": ["v10", "10m_v_component_of_wind"],
    "t2m": ["t2m", "2m_temperature"],
    "msl": ["msl", "mean_sea_level_pressure"],
    "swh": ["swh", "significant_height_of_combined_wind_waves_and_swell"],
    "mwp": ["mwp", "mean_wave_period"],
    "mwd": ["mwd", "mean_wave_direction"],
}


# ── Détection de format (magic bytes) ─────────────────────────────────────────

def is_zip(fp: Path) -> bool:
    try:
        with open(fp, "rb") as f:
            return f.read(4) == b"PK\x03\x04"
    except Exception:
        return False


def is_netcdf(fp: Path) -> bool:
    try:
        with open(fp, "rb") as f:
            m = f.read(8)
        return m[:4] in (b"CDF\x01", b"CDF\x02") or m == b"\x89HDF\r\n\x1a\n"
    except Exception:
        return False


def open_nc(fp):
    for eng in ["netcdf4", "h5netcdf", "scipy"]:
        try:
            return xr.open_dataset(str(fp), engine=eng)
        except Exception:
            continue
    return None


def normalise(ds):
    """valid_time→time ; latitude/longitude→lat/lon ; lon→[-180,180] ; tri."""
    if "valid_time" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    for d in ["expver", "number"]:
        if d in ds.dims:
            ds = ds.isel({d: 0}).drop_vars(d, errors="ignore")
        elif d in ds.coords:
            ds = ds.drop_vars(d, errors="ignore")
    ren = {}
    if "latitude" in ds.dims:  ren["latitude"] = "lat"
    if "longitude" in ds.dims: ren["longitude"] = "lon"
    if ren:
        ds = ds.rename(ren)
    if "lon" in ds.coords and float(ds.lon.max()) > 180:
        ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180)
    for c in ["lon", "lat", "time"]:
        if c in ds.coords:
            ds = ds.sortby(c)
    return ds


# ── Lecture d'un fichier CDS (nc simple ou ZIP multi-nc) → dataset fusionné ────

def read_cds_file(fp: Path):
    """
    Renvoie un xr.Dataset normalisé en fusionnant TOUS les NetCDF contenus.
    Gère : NetCDF simple, ZIP à 1 NetCDF, ZIP à plusieurs NetCDF.
    """
    if is_netcdf(fp):
        ds = open_nc(fp)
        return normalise(ds) if ds is not None else None

    if not is_zip(fp):
        log.error(f"  format inconnu (ni NetCDF ni ZIP) : {fp.name}")
        return None

    tmp = Path(tempfile.mkdtemp(prefix="era5v2_"))
    try:
        with zipfile.ZipFile(str(fp), "r") as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".nc")]
            if not names:
                log.error(f"  ZIP sans .nc : {fp.name} (contenu={zf.namelist()})")
                return None
            zf.extractall(str(tmp))
        dss = []
        for n in names:
            ds = open_nc(tmp / n)
            if ds is not None:
                dss.append(normalise(ds).load())
        if not dss:
            return None
        # Fusion de tous les flux contenus dans le ZIP
        merged = dss[0] if len(dss) == 1 else xr.merge(
            dss, compat="override", join="outer")
        return merged
    except zipfile.BadZipFile:
        log.error(f"  ZIP corrompu : {fp.name}")
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Fusion atmosphère ⊕ vagues sur la grille atmosphérique ────────────────────

def merge_atmos_wave(atmos, wave):
    """
    Interpole les variables de vagues (grille 0.5°) sur la grille
    atmosphérique (0.25°, mêmes pas de temps) puis les ajoute au dataset atmos.
    """
    out = atmos
    if wave is not None and {"lat", "lon", "time"}.issubset(wave.dims):
        # Interpolation spatiale (le temps est identique → exact)
        wave_i = wave.interp(lat=atmos["lat"], lon=atmos["lon"],
                             method="linear")
        if not atmos["time"].equals(wave_i["time"]):
            wave_i = wave_i.interp(time=atmos["time"], method="nearest")
        for v in wave_i.data_vars:
            out = out.assign({v: wave_i[v]})
    return out


def resolve_present(ds):
    """Liste des clés courtes présentes (résolution par alias)."""
    present = []
    for short, aliases in VAR_ALIASES.items():
        if any(a in ds.data_vars for a in aliases):
            present.append(short)
    return present


def process_month(route_id, box_suffix, year, month, data_dir, keep_raw):
    folder = data_dir / route_id / box_suffix
    base = f"ERA5_{route_id}_{box_suffix}_{year}_{month}"
    canonical = folder / f"{base}.nc"
    f_atmos = folder / f"{base}_atmos.nc"
    f_wave  = folder / f"{base}_wave.nc"

    # Déjà fusionné et valide ?
    if is_netcdf(canonical):
        ds = open_nc(canonical)
        if ds is not None:
            present = resolve_present(ds); ds.close()
            if "swh" in present and "u10" in present:
                return "skip"

    if not f_atmos.exists():
        return "missing_atmos"
    atmos = read_cds_file(f_atmos)
    if atmos is None:
        return "error_atmos"
    wave = read_cds_file(f_wave) if f_wave.exists() else None
    if wave is None:
        log.warning(f"  {base}: fichier WAVE absent/illisible → swh manquera.")

    merged = merge_atmos_wave(atmos, wave)
    present = resolve_present(merged)

    enc = {v: {"zlib": True, "complevel": 4} for v in merged.data_vars}
    tmp_out = folder / f"{base}.nc.tmp"
    merged.to_netcdf(tmp_out, encoding=enc)
    merged.close()
    atmos.close()
    if wave is not None:
        wave.close()
    os.replace(tmp_out, canonical)

    if not keep_raw:
        f_atmos.unlink(missing_ok=True)
        f_wave.unlink(missing_ok=True)

    size = canonical.stat().st_size / 1024**2
    log.info(f"  ✓ {canonical.name} ({size:.1f} MB) vars={present}")
    return "ok" if "swh" in present else "ok_no_wave"


def main():
    import sys as _sys
    if hasattr(_sys, "ps1") or "ipykernel" in _sys.modules:
        _sys.argv = [_sys.argv[0]]
    ap = argparse.ArgumentParser(description="Extraction+fusion ERA5 v2.")
    ap.add_argument("--route", default=None)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--keep-raw", action="store_true",
                    help="Conserver les fichiers _atmos/_wave après fusion.")
    args, _ = ap.parse_known_args()

    data_dir = Path(args.data_dir)
    route_ids = [args.route] if args.route else list(ROUTES.keys())
    years, months = ERA5_CONFIG["years"], ERA5_CONFIG["months"]

    log.info(f"Extraction+fusion ERA5 v2 | data_dir={data_dir.resolve()}")
    stats = {}
    for rid in route_ids:
        if rid not in ROUTES:
            sys.exit(f"Route inconnue : {rid}")
        log.info("=" * 72); log.info(f"Route {rid}")
        s = {}
        for box in ROUTES[rid]["boxes"]:
            for year in years:
                for month in months:
                    r = process_month(rid, box["suffix"], year, month,
                                      data_dir, args.keep_raw)
                    s[r] = s.get(r, 0) + 1
        log.info(f"  {rid} : {s}")
        stats[rid] = s

    log.info(""); log.info("=" * 72); log.info("RÉSUMÉ EXTRACTION")
    agg = {}
    for rid, s in stats.items():
        for k, v in s.items():
            agg[k] = agg.get(k, 0) + v
    log.info(f"  {agg}")
    if agg.get("ok_no_wave"):
        log.warning(f"  {agg['ok_no_wave']} fichier(s) fusionné(s) SANS vagues "
                    f"→ relancez ERA5_download_v2.py --waves-only.")
    if agg.get("error_atmos") or agg.get("missing_atmos"):
        log.warning("  Fichiers atmos manquants/illisibles : "
                    "relancez ERA5_download_v2.py --atmos-only.")
    log.info("Étape suivante : ERA5_diagnostic_v2.py")
    log.info("=" * 72)


if __name__ == "__main__":
    main()
