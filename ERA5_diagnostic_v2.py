

import os
import sys
import csv
import calendar
import argparse
import logging
from pathlib import Path

import numpy as np

try:
    import xarray as xr
except Exception as exc:                                   # pragma: no cover
    sys.exit(f"xarray requis : pip install xarray netCDF4 h5netcdf  ({exc})")

try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = os.getcwd()
sys.path.insert(0, _SCRIPT_DIR)
from route_config_v2 import ROUTES, ERA5_CONFIG

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
CORE = ["u10", "v10", "msl", "t2m"]


def open_nc(fp):
    for eng in ["netcdf4", "h5netcdf", "scipy"]:
        try:
            return xr.open_dataset(str(fp), engine=eng)
        except Exception:
            continue
    return None


def resolve(ds, short):
    for a in VAR_ALIASES[short]:
        if a in ds.data_vars:
            return a
    return None


def grid_res(ds):
    def step(c):
        if c in ds.coords and ds[c].size > 1:
            return float(np.abs(np.diff(ds[c].values)).round(4)[0])
        return None
    return step("lat"), step("lon")


def check_file(fp):
    """Renvoie un dict de métriques pour un fichier fusionné, ou None."""
    ds = open_nc(fp)
    if ds is None:
        return {"file": fp.name, "ok": False, "reason": "illisible"}
    try:
        present = [s for s in VAR_ALIASES if resolve(ds, s)]
        time_n = int(ds.sizes.get("time", 0))
        lat_n  = int(ds.sizes.get("lat", 0))
        lon_n  = int(ds.sizes.get("lon", 0))
        dlat, dlon = grid_res(ds)
        nan = {}
        for s in present:
            v = resolve(ds, s)
            arr = ds[v].values
            nan[s] = float(np.isnan(arr).mean())
        # mois attendu depuis le nom de fichier
        parts = fp.stem.split("_")
        yr, mo = int(parts[-2]), int(parts[-1])
        exp_h = calendar.monthrange(yr, mo)[1] * 24
        return {"file": fp.name, "ok": True, "present": present,
                "time_n": time_n, "exp_h": exp_h, "time_ok": time_n == exp_h,
                "lat_n": lat_n, "lon_n": lon_n, "dlat": dlat, "dlon": dlon,
                "nan": nan,
                "core_ok": all(s in present for s in CORE),
                "has_swh": "swh" in present,
                "swh_all_nan": nan.get("swh", 1.0) >= 0.999}
    finally:
        ds.close()


def main():
    import sys as _sys
    if hasattr(_sys, "ps1") or "ipykernel" in _sys.modules:
        _sys.argv = [_sys.argv[0]]
    ap = argparse.ArgumentParser(description="Diagnostic ERA5 v2.")
    ap.add_argument("--route", default=None)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--sample", type=int, default=0,
                    help="N>0 : n'inspecter que N fichiers/boîte (rapide).")
    ap.add_argument("--report", default="ERA5_diagnostic_v2_report.csv")
    args, _ = ap.parse_known_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        sys.exit(f"Répertoire introuvable : {data_dir.resolve()}")
    route_ids = [args.route] if args.route else list(ROUTES.keys())
    years, months = ERA5_CONFIG["years"], ERA5_CONFIG["months"]

    rows = []
    global_ok = True
    log.info(f"Diagnostic ERA5 v2 | {data_dir.resolve()}")
    for rid in route_ids:
        if rid not in ROUTES:
            sys.exit(f"Route inconnue : {rid}")
        log.info("=" * 72); log.info(f"Route {rid}")
        for box in ROUTES[rid]["boxes"]:
            sfx = box["suffix"]
            folder = data_dir / rid / sfx
            expected = [(y, m) for y in years for m in months]
            present_files, missing = [], []
            for (y, m) in expected:
                fp = folder / f"ERA5_{rid}_{sfx}_{y}_{m}.nc"
                (present_files if fp.exists() else missing).append((fp, y, m))
            to_check = present_files[:args.sample] if args.sample else present_files

            n_swh_bad = n_time_bad = n_core_bad = 0
            swh_nan_vals = []
            res_seen = set()
            for fp, y, m in to_check:
                r = check_file(fp)
                r["route"], r["box"] = rid, sfx
                rows.append(r)
                if not r.get("ok"):
                    n_core_bad += 1; continue
                if not r["core_ok"]:    n_core_bad += 1
                if not r["time_ok"]:    n_time_bad += 1
                if r["swh_all_nan"]:    n_swh_bad += 1
                if "swh" in r["nan"]:   swh_nan_vals.append(r["nan"]["swh"])
                res_seen.add((r["dlat"], r["dlon"]))

            n = len(to_check)
            swh_med = float(np.median(swh_nan_vals)) if swh_nan_vals else None
            box_ok = (len(missing) == 0 and n_core_bad == 0 and n_swh_bad == 0
                      and n_time_bad == 0)
            global_ok = global_ok and box_ok
            log.info(f"  [{sfx}] fichiers {len(present_files)}/{len(expected)} "
                     f"| inspectés {n} | manquants {len(missing)}")
            log.info(f"      core_manquant={n_core_bad}  heures_KO={n_time_bad}  "
                     f"swh_100%NaN={n_swh_bad}  swh_NaN_médian="
                     f"{swh_med if swh_med is None else round(swh_med,3)}")
            log.info(f"      résolutions (dlat,dlon) vues : {sorted(res_seen)}")
            if missing:
                ex = [f'{y}-{m}' for _, y, m in missing[:6]]
                log.warning(f"      mois manquants ({len(missing)}) ex.: {ex}")
            if n_swh_bad:
                log.error(f"      ⚠ {n_swh_bad} fichier(s) swh 100 % NaN — la "
                          f"correction des vagues n'a PAS abouti pour cette boîte.")

    # Rapport CSV
    with open(args.report, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["route", "box", "file", "ok", "core_ok", "has_swh",
                    "swh_all_nan", "time_n", "exp_h", "time_ok",
                    "lat_n", "lon_n", "dlat", "dlon", "swh_nan_frac",
                    "present"])
        for r in rows:
            if not r.get("ok"):
                w.writerow([r.get("route"), r.get("box"), r["file"], False,
                            "", "", "", "", "", "", "", "", "", "", "",
                            r.get("reason", "")])
                continue
            w.writerow([r["route"], r["box"], r["file"], r["ok"], r["core_ok"],
                        r["has_swh"], r["swh_all_nan"], r["time_n"], r["exp_h"],
                        r["time_ok"], r["lat_n"], r["lon_n"], r["dlat"],
                        r["dlon"], round(r["nan"].get("swh", float("nan")), 4),
                        "|".join(r["present"])])

    # Vérification ciblée : boîte océan Indien de R1
    io_dir = data_dir / "R1_SHA_RTM" / "IndianOcean"
    log.info(""); log.info("=" * 72)
    if io_dir.exists() and any(io_dir.glob("ERA5_*_*.nc")):
        io_rows = [r for r in rows if r.get("box") == "IndianOcean"
                   and r.get("ok")]
        bad = sum(1 for r in io_rows if r["swh_all_nan"])
        log.info(f"Boîte océan Indien (R1) : présente, {len(io_rows)} fichiers "
                 f"inspectés, swh 100%NaN={bad} → "
                 f"{'OK' if bad == 0 else 'À CORRIGER'}")
    else:
        log.warning("Boîte océan Indien (R1) : AUCUN fichier fusionné trouvé "
                    "(lancez download_v2 + extract_v2 pour la boîte IndianOcean).")
    log.info(f"Rapport détaillé : {Path(args.report).resolve()}")
    log.info(f"VERDICT GLOBAL : {'✅ CONFORME' if global_ok else '⚠ ANOMALIES (voir ci-dessus)'}")
    log.info("=" * 72)


if __name__ == "__main__":
    main()
