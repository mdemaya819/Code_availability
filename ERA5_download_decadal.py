

import cdsapi
import os
import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime


    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    
    _SCRIPT_DIR = os.getcwd()
sys.path.insert(0, _SCRIPT_DIR)
from route_config import ROUTES, ERA5_CONFIG

# ── Configuration du logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ERA5_download.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── Répertoire de sortie ──────────────────────────────────────────────────────
OUTPUT_BASE = Path("ERA5_data")

# ── Paramètre : taille minimale de fichier valide (en octets) ─────────────────
MIN_FILE_SIZE_BYTES = 2 * 1024 * 1024   # 2 MB minimum


def build_output_path(route_id: str, box_suffix: str,
                      year: str, month: str) -> Path:
    """Construit le chemin de sortie du fichier NC."""
    folder = OUTPUT_BASE / route_id / box_suffix
    folder.mkdir(parents=True, exist_ok=True)
    filename = f"ERA5_{route_id}_{box_suffix}_{year}_{month}.nc"
    return folder / filename


def is_valid_file(filepath: Path) -> bool:
    """
    Vérifie qu'un fichier NC existe et a une taille minimale valide.
    Un fichier trop petit indique un téléchargement partiel ou une erreur.
    """
    return filepath.exists() and filepath.stat().st_size >= MIN_FILE_SIZE_BYTES


def download_one_month(
    client:     cdsapi.Client,
    route_id:   str,
    box:        dict,
    year:       str,
    month:      str,
    variables:  list,
    max_retry:  int = 3,
    retry_wait: int = 60,
) -> bool:
    """
    Télécharge un fichier ERA5 pour une route, une boîte, un mois.

    Paramètres
    ----------
    client     : instance cdsapi.Client
    route_id   : identifiant de la route (ex: "R1_SHA_RTM")
    box        : dict {"suffix": str, "area": [N, W, S, E]}
    year       : année (ex: "2020")
    month      : mois (ex: "03")
    variables  : liste des variables ERA5 à télécharger
    max_retry  : nombre maximal de tentatives en cas d'erreur réseau
    retry_wait : temps d'attente entre deux tentatives (secondes)

    Retourne
    --------
    True si le fichier est disponible (existant valide ou nouveau download)
    False en cas d'échec définitif
    """
    suffix   = box["suffix"]
    area     = box["area"]
    outfile  = build_output_path(route_id, suffix, year, month)

    # ── Skip si le fichier existe et est valide ────────────────────────────
    if is_valid_file(outfile):
        log.info(
            f"[SKIP] {outfile.name} "
            f"({outfile.stat().st_size / 1024**2:.1f} MB) — already exists"
        )
        return True

    log.info(
        f"[DOWNLOAD] {outfile.name} "
        f"| Area: N={area[0]} W={area[1]} S={area[2]} E={area[3]} "
        f"| {len(variables)} variables | {year}-{month}"
    )

    request = {
        "product_type": ERA5_CONFIG["product_type"],
        "variable":     variables,
        "year":         year,
        "month":        [month],
        "day":          ERA5_CONFIG["days"],
        "time":         ERA5_CONFIG["hours"],
        "area":         area,
        "grid":         ERA5_CONFIG["grid"],
        "format":       ERA5_CONFIG["format"],
    }

    # ── Tentatives avec retry ──────────────────────────────────────────────
    for attempt in range(1, max_retry + 1):
        try:
            client.retrieve(
                "reanalysis-era5-single-levels",
                request,
                str(outfile),
            )
            size_mb = outfile.stat().st_size / 1024**2
            log.info(f"  ✓ Done — {size_mb:.1f} MB")

            # Vérification post-download
            if not is_valid_file(outfile):
                log.warning(
                    f"  ⚠ File too small ({size_mb:.1f} MB) — "
                    f"possible incomplete download, will retry"
                )
                outfile.unlink(missing_ok=True)
                raise ValueError("File too small after download")

            return True

        except Exception as exc:
            log.error(f"  ✗ Attempt {attempt}/{max_retry} failed: {exc}")
            if attempt < max_retry:
                log.info(f"  → Waiting {retry_wait}s before retry...")
                time.sleep(retry_wait)
            else:
                log.error(
                    f"  ✗ DEFINITIVE FAILURE after {max_retry} attempts: "
                    f"{outfile.name}"
                )
                # Supprimer le fichier corrompu s'il existe
                outfile.unlink(missing_ok=True)
                return False

    return False


def download_route(
    client:     cdsapi.Client,
    route_id:   str,
    years:      list | None = None,
    months:     list | None = None,
    variables:  list | None = None,
) -> dict:
    """
    Télécharge tous les mois d'une route.

    Retourne un dict {"success": int, "skipped": int, "failed": int}
    """
    if route_id not in ROUTES:
        raise ValueError(
            f"Route '{route_id}' inconnue. "
            f"Disponibles : {list(ROUTES.keys())}"
        )

    route   = ROUTES[route_id]
    years   = years   or ERA5_CONFIG["years"]
    months  = months  or ERA5_CONFIG["months"]
    variables = variables or ERA5_CONFIG["vars_all"]

    log.info("=" * 70)
    log.info(f"Route : {route['name']}")
    log.info(f"Boxes : {[b['suffix'] for b in route['boxes']]}")
    log.info(f"Years : {years[0]} → {years[-1]}  ({len(years)} years)")
    log.info(f"Months: {months}")
    log.info(f"Vars  : {variables}")
    log.info("=" * 70)

    stats = {"success": 0, "skipped": 0, "failed": 0}

    for box in route["boxes"]:
        for year in years:
            for month in months:
                result = download_one_month(
                    client, route_id, box, year, month, variables
                )
                if result:
                    # Distinguer succès réel vs skip (fichier existant)
                    outfile = build_output_path(
                        route_id, box["suffix"], year, month
                    )
                    # Si le fichier existait avant (log SKIP), on compte skip
                    # Sinon on compte success
                    stats["success"] += 1
                else:
                    stats["failed"] += 1

    return stats


def print_summary(all_stats: dict) -> None:
    """Affiche le résumé global du téléchargement."""
    log.info("")
    log.info("=" * 70)
    log.info("DOWNLOAD SUMMARY")
    log.info("=" * 70)
    total_s = total_f = 0
    for route_id, stats in all_stats.items():
        log.info(
            f"  {route_id:<20} "
            f"OK={stats['success']:3d}  "
            f"FAILED={stats['failed']:2d}"
        )
        total_s += stats["success"]
        total_f += stats["failed"]
    log.info(f"  {'TOTAL':<20} OK={total_s:3d}  FAILED={total_f:2d}")
    log.info("=" * 70)
    if total_f > 0:
        log.warning(
            f"{total_f} file(s) failed. Re-run the script to retry "
            f"(successful files are skipped automatically)."
        )


# ── Point d'entrée ────────────────────────────────────────────────────────────
def main():
    # Reset sys.argv for Jupyter compatibility
    import sys as _sys
    if hasattr(_sys, "ps1") or "ipykernel" in _sys.modules:
        _sys.argv = [_sys.argv[0]]
    parser = argparse.ArgumentParser(
        description="ERA5 decadal download for Flettner rotor operational analysis"
    )
    parser.add_argument(
        "--route",
        type=str,
        default=None,
        help="Route ID to download (e.g. R1_SHA_RTM). "
             "If omitted, all 4 routes are downloaded.",
    )
    parser.add_argument(
        "--year",
        type=str,
        default=None,
        help="Single year to download (e.g. 2020). "
             "If omitted, all years 2015–2024 are downloaded.",
    )
    parser.add_argument(
        "--month",
        type=str,
        default=None,
        help="Single month to download (e.g. 01). "
             "If omitted, all 12 months are downloaded.",
    )
    parser.add_argument(
        "--wind-only",
        action="store_true",
        help="Download only wind variables (u10, v10). "
             "Skips MSLP, T2m, and wave height.",
    )
    args, _ = parser.parse_known_args()  # ignore Jupyter kernel args

    # Sélection des routes
    route_ids = (
        [args.route] if args.route else list(ROUTES.keys())
    )
    # Vérification
    for rid in route_ids:
        if rid not in ROUTES:
            sys.exit(
                f"ERROR: Unknown route '{rid}'. "
                f"Valid options: {list(ROUTES.keys())}"
            )

    # Sélection des années et mois
    years  = [args.year]  if args.year  else ERA5_CONFIG["years"]
    months = [args.month] if args.month else ERA5_CONFIG["months"]

    # Sélection des variables
    variables = (
        ERA5_CONFIG["vars_wind"]
        if args.wind_only
        else ERA5_CONFIG["vars_all"]
    )

    log.info(f"Starting ERA5 decadal download — {datetime.now():%Y-%m-%d %H:%M}")
    log.info(f"Routes : {route_ids}")
    log.info(f"Years  : {years}")
    log.info(f"Months : {months}")
    log.info(f"Vars   : {variables}")
    log.info(f"Output : {OUTPUT_BASE.resolve()}")
    log.info("")

    # Initialiser le client CDS
    try:
        client = cdsapi.Client()
    except Exception as exc:
        sys.exit(
            f"ERROR: Cannot initialise CDS API client.\n"
            f"Check your ~/.cdsapirc file.\nDetails: {exc}"
        )

    # Lancer les téléchargements
    all_stats = {}
    for route_id in route_ids:
        all_stats[route_id] = download_route(
            client, route_id, years, months, variables
        )

    print_summary(all_stats)


if __name__ == "__main__":
    main()
