"""
ERA5_build_ML_dataset.py
========================
Construction d'un jeu de données prêt pour l'apprentissage profond à partir
des champs ERA5 décennaux bruts (2015–2024) déjà téléchargés et extraits.

Ce script est INDÉPENDANT des travaux précédents (rotor Flettner) : il ne
contient aucune notion de navire, de cap, d'angle de vent apparent (AWA) ni
de puissance. Il produit un jeu de données métocéan eulérien, destiné à la
prévision spatio-temporelle multi-horizon (BiLSTM, TCN, CNN-BiLSTM-Attention,
Temporal Fusion Transformer, ConvLSTM…) et à l'optimisation par route.

Deux modes de sortie
--------------------
  mode "nodes" (par défaut, recommandé)
      Échantillonne K nœuds FIXES, régulièrement espacés le long du
      grand cercle de chaque corridor, et extrait la SÉRIE TEMPORELLE
      HORAIRE CONTINUE (2015–2024, ~87 600 pas) à chacun.
      → table « tidy » parquet : 1 ligne par (nœud, heure).
      → alimente directement BiLSTM / TCN / TFT / CNN-BiLSTM-Attention.

  mode "grid" (optionnel, pour ConvLSTM / modèles de champ)
      Exporte les champs horaires de chaque boîte, éventuellement
      dégrossis (coarsen), en NetCDF compressé PAR ANNÉE (pour éviter
      de charger 10 ans en mémoire). Volumineux : à utiliser avec un
      facteur de coarsen ≥ 2 et sur une seule route à la fois.

Sorties (dans --out, défaut « ERA5_ML_dataset/ ») :
  routes_nodes.parquet            table tidy multi-routes (mode nodes)
  <ROUTE>_nodes.parquet           table tidy par route
  dataset_manifest.json           schéma, nœuds, couverture, split
                                  temporel train/val/test, stats de
                                  normalisation calculées SUR LE TRAIN
                                  uniquement (anti-fuite), rapport NaN/lacunes
  grid/<ROUTE>_<BOX>_<year>.nc     (mode grid)

Conventions réutilisées de la chaîne existante
----------------------------------------------
  - route_config.ROUTES / ERA5_CONFIG / PHYSICS / ENSO_THRESHOLDS
  - arborescence ERA5_data/{route}/{box}/ERA5_{route}_{box}_{year}_{month}.nc
  - open_nc / normalise_ds / find_var / VAR_ALIASES / wind_to_U_WD / air_density
  - lecture ROBUSTE anti-fuite HDF : un fichier à la fois → interp → .load()
    → close() immédiat (jamais de concaténation paresseuse accumulée).

Ordre dans le workflow
----------------------
  1. ERA5_download_decadal.py        (fait)
  2. ERA5_extract_zips.py            (fait)
  3. ONI_download.py                 (fait → ONI_index_2015_2024.csv)
  4. ERA5_build_ML_dataset.py        ← CE SCRIPT (nouveau projet ML)

Usage
-----
    python ERA5_build_ML_dataset.py                         # 4 routes, mode nodes
    python ERA5_build_ML_dataset.py --route R2_NFK_HAM
    python ERA5_build_ML_dataset.py --spacing-km 200
    python ERA5_build_ML_dataset.py --mode grid --route R3_SIN_SYD --coarsen 4

Prérequis
---------
    pip install xarray netCDF4 h5netcdf numpy pandas pyarrow
"""

import os
import sys
import json
import math
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import xarray as xr
except Exception as exc:                                   # pragma: no cover
    sys.exit(f"xarray requis : pip install xarray netCDF4 h5netcdf  ({exc})")

# ── route_config (mêmes définitions que la chaîne de téléchargement) ──────────
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = os.getcwd()
sys.path.insert(0, _SCRIPT_DIR)
from route_config import ROUTES, ERA5_CONFIG, PHYSICS, ENSO_THRESHOLDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DATA_IN  = Path("ERA5_data")
ONI_FILE = Path("ONI_index_2015_2024.csv")

# Variables continues à normaliser (les angles/temps sont déjà bornés)
NORM_VARS = ["u10", "v10", "wspd", "t2m", "msl", "rho", "swh", "oni"]

VAR_ALIASES = {
    "u10": ["u10"],
    "v10": ["v10"],
    "t2m": ["t2m", "2m_temperature"],
    "msl": ["msl", "mean_sea_level_pressure"],
    "swh": ["swh", "significant_height_of_combined_wind_waves_and_swell"],
}

# ── E/S NetCDF (réutilisé à l'identique de ERA5_process_decadal.py) ───────────

def open_nc(filepath):
    for eng in ["netcdf4", "h5netcdf", "scipy"]:
        try:
            return xr.open_dataset(str(filepath), engine=eng)
        except Exception:
            continue
    return None


def normalise_ds(ds):
    if "valid_time" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    for dim in ["expver", "number"]:
        if dim in ds.dims:
            ds = ds.isel({dim: 0}).drop_vars(dim, errors="ignore")
    rename = {}
    if "latitude"  in ds.dims: rename["latitude"]  = "lat"
    if "longitude" in ds.dims: rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)
    if "lon" in ds.coords and float(ds.lon.max()) > 180:
        ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180)
        ds = ds.sortby("lon")
    if "time" in ds.dims:
        ds = ds.sortby("time")
    return ds


def find_var(ds, var_key):
    for alias in VAR_ALIASES.get(var_key, [var_key]):
        if alias in ds.data_vars:
            return alias
    return None


# ── Physique (générique, non liée au navire) ──────────────────────────────────

def wind_to_U_WD(u10, v10):
    U  = np.sqrt(u10**2 + v10**2)
    WD = (270.0 - np.degrees(np.arctan2(v10, u10))) % 360.0
    return U, WD


def air_density(T_K, msl_Pa):
    return msl_Pa / (PHYSICS["R_dry"] * T_K)


# ── Génération de nœuds fixes le long du grand cercle ─────────────────────────

def _latlon_to_vec(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return np.array([math.cos(la) * math.cos(lo),
                     math.cos(la) * math.sin(lo),
                     math.sin(la)])


def _vec_to_latlon(v):
    v = v / np.linalg.norm(v)
    lat = math.degrees(math.asin(np.clip(v[2], -1, 1)))
    lon = math.degrees(math.atan2(v[1], v[0]))
    return lat, lon


def _slerp(v1, v2, f):
    """Interpolation sphérique (gère correctement l'antiméridien, R4)."""
    dot = float(np.clip(np.dot(v1, v2), -1, 1))
    omega = math.acos(dot)
    if omega < 1e-9:
        return v1
    s = math.sin(omega)
    return (math.sin((1 - f) * omega) / s) * v1 + (math.sin(f * omega) / s) * v2


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2)
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def generate_nodes(route_id, spacing_km):
    """
    Renvoie une liste de nœuds FIXES {node_id, lat, lon, dist_km}
    régulièrement espacés (~spacing_km) le long du grand cercle reliant
    les waypoints de la route.
    """
    wpts = ROUTES[route_id]["waypoints"]
    # Distance cumulée et vecteurs des waypoints
    seg_d, vecs = [], [_latlon_to_vec(la, lo) for la, lo, _ in wpts]
    for i in range(len(wpts) - 1):
        seg_d.append(_haversine_km(wpts[i][0], wpts[i][1],
                                   wpts[i + 1][0], wpts[i + 1][1]))
    total = sum(seg_d)
    n_nodes = max(2, int(round(total / spacing_km)) + 1)
    targets = np.linspace(0, total, n_nodes)

    cum = np.concatenate([[0], np.cumsum(seg_d)])
    nodes = []
    for k, d in enumerate(targets):
        # Segment contenant la distance cible d
        j = int(np.searchsorted(cum, d, side="right") - 1)
        j = min(max(j, 0), len(seg_d) - 1)
        seg_len = seg_d[j] if seg_d[j] > 0 else 1.0
        f = (d - cum[j]) / seg_len
        lat, lon = _vec_to_latlon(_slerp(vecs[j], vecs[j + 1], float(np.clip(f, 0, 1))))
        nodes.append({"node_id": k, "lat": round(lat, 4),
                      "lon": round(lon, 4), "dist_km": round(float(d), 1)})
    return nodes


def _node_box(route_id, lat, lon):
    """Renvoie le suffixe de boîte ERA5 contenant (lat, lon), sinon None."""
    for box in ROUTES[route_id]["boxes"]:
        N, W, S, E = box["area"]
        if (S <= lat <= N) and (W <= lon <= E):
            return box["suffix"]
    return None


# ── Chargement ONI (réutilise ONI_index_2015_2024.csv) ────────────────────────

def load_oni():
    for cand in [ONI_FILE, Path(_SCRIPT_DIR) / ONI_FILE.name,
                 Path(os.getcwd()) / ONI_FILE.name]:
        if cand.exists():
            df = pd.read_csv(cand)
            if "ONI" in df.columns:
                omin, omax = float(df["ONI"].min()), float(df["ONI"].max())
                if omin < -4 or omax > 5:        # plage physique ONI
                    log.error(
                        f"ONI hors plage physique [{omin:.2f}, {omax:.2f}] : "
                        f"la colonne ONI est ERRONÉE. Relancez ONI_download.py "
                        f"AVANT ce script. → ONI/ENSO mis à NaN.")
                    return pd.DataFrame(
                        columns=["year", "month", "ONI", "ENSO_phase"])
                if "ENSO_phase" not in df.columns:
                    df["ENSO_phase"] = np.where(
                        df["ONI"] >= ENSO_THRESHOLDS["el_nino"], "El_Nino",
                        np.where(df["ONI"] <= ENSO_THRESHOLDS["la_nina"],
                                 "La_Nina", "Neutral"))
                log.info(f"ONI chargé : {len(df)} mois ({cand.name})")
                return df[["year", "month", "ONI", "ENSO_phase"]]
    log.warning("ONI introuvable → colonnes ONI/ENSO_phase = NaN.")
    return pd.DataFrame(columns=["year", "month", "ONI", "ENSO_phase"])


# ── Caractéristiques temporelles (covariables « connues du futur ») ───────────

def add_time_features(df):
    dt = pd.to_datetime(df["datetime"])
    hod = dt.dt.hour
    doy = dt.dt.dayofyear
    df["hod_sin"] = np.sin(2 * np.pi * hod / 24.0)
    df["hod_cos"] = np.cos(2 * np.pi * hod / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"]   = dt.dt.month
    return df


# ── Extraction d'une route en mode "nodes" ────────────────────────────────────

def extract_route_nodes(route_id, data_dir, spacing_km, years, months):
    """
    Construit la série horaire continue (2015–2024) à chaque nœud fixe.
    Lecture robuste : un fichier mensuel à la fois → interp aux nœuds de la
    boîte → .load() → close() immédiat (aucune fuite de handle HDF).
    """
    nodes = generate_nodes(route_id, spacing_km)
    # Affecter chaque nœud à une boîte ; signaler les nœuds hors couverture
    covered, uncovered = [], []
    for nd in nodes:
        sfx = _node_box(route_id, nd["lat"], nd["lon"])
        nd["box"] = sfx
        (covered if sfx else uncovered).append(nd)
    if uncovered:
        log.warning(f"  {route_id}: {len(uncovered)} nœud(s) hors de toute "
                    f"boîte ERA5 (lacune de couverture) → ignorés : "
                    f"{[n['node_id'] for n in uncovered]}")
    log.info(f"  {route_id}: {len(nodes)} nœuds générés "
             f"(~{spacing_km} km), {len(covered)} couverts")

    # Regrouper les nœuds par boîte
    by_box = {}
    for nd in covered:
        by_box.setdefault(nd["box"], []).append(nd)

    frames = []
    for box_suffix, box_nodes in by_box.items():
        lats = xr.DataArray([n["lat"] for n in box_nodes], dims="node")
        lons = xr.DataArray([n["lon"] for n in box_nodes], dims="node")
        nid  = [n["node_id"] for n in box_nodes]

        per_node_chunks = {k: [] for k in nid}
        for year in years:
            for month in months:
                fp = (data_dir / route_id / box_suffix /
                      f"ERA5_{route_id}_{box_suffix}_{year}_{month}.nc")
                if not fp.exists():
                    found = list(data_dir.rglob(
                        f"*{route_id}*{box_suffix}*{year}*{month}*.nc"))
                    if not found:
                        continue
                    fp = found[0]
                ds = open_nc(fp)
                if ds is None:
                    continue
                ds = normalise_ds(ds)
                if "lat" in ds.dims:
                    ds = ds.sortby("lat")     # interp exige lat croissante
                vu, vv = find_var(ds, "u10"), find_var(ds, "v10")
                if not vu or not vv:
                    ds.close(); continue
                vt, vm, vs = (find_var(ds, "t2m"), find_var(ds, "msl"),
                              find_var(ds, "swh"))
                try:
                    # Interpolation ponctuelle vectorisée à TOUS les nœuds
                    pt = ds.interp(lat=lats, lon=lons, method="linear").load()
                finally:
                    ds.close()                       # ← fermeture immédiate

                times = pd.DatetimeIndex(pt.time.values)
                u = np.asarray(pt[vu].values)        # (time, node)
                v = np.asarray(pt[vv].values)
                t2 = (np.asarray(pt[vt].values) if vt
                      else np.full_like(u, 288.15))
                ms = (np.asarray(pt[vm].values) if vm
                      else np.full_like(u, 101325.0))
                sw = (np.asarray(pt[vs].values) if vs
                      else np.full_like(u, np.nan))
                del pt

                wspd, wdir = wind_to_U_WD(u, v)
                rho = air_density(t2, ms)
                for col, node_id in enumerate(nid):
                    per_node_chunks[node_id].append(pd.DataFrame({
                        "datetime": times,
                        "u10": u[:, col], "v10": v[:, col],
                        "wspd": wspd[:, col],
                        "wdir_sin": np.sin(np.radians(wdir[:, col])),
                        "wdir_cos": np.cos(np.radians(wdir[:, col])),
                        "t2m": t2[:, col], "msl": ms[:, col],
                        "rho": rho[:, col], "swh": sw[:, col],
                    }))

        for nd in box_nodes:
            chunks = per_node_chunks[nd["node_id"]]
            if not chunks:
                continue
            d = (pd.concat(chunks, ignore_index=True)
                   .drop_duplicates("datetime")
                   .sort_values("datetime")
                   .reset_index(drop=True))
            d.insert(0, "route", route_id)
            d.insert(1, "node_id", nd["node_id"])
            d.insert(2, "node_lat", nd["lat"])
            d.insert(3, "node_lon", nd["lon"])
            d.insert(4, "dist_km", nd["dist_km"])
            frames.append(d)

    if not frames:
        return None, nodes
    out = pd.concat(frames, ignore_index=True)

    # Fusion ONI (mensuel → horaire) + phase ENSO
    oni = load_oni()
    out["year"]  = pd.to_datetime(out["datetime"]).dt.year
    out["month"] = pd.to_datetime(out["datetime"]).dt.month
    if len(oni):
        out = out.merge(oni, on=["year", "month"], how="left")
        out = out.rename(columns={"ONI": "oni", "ENSO_phase": "enso_phase"})
    else:
        out["oni"], out["enso_phase"] = np.nan, "Unknown"

    out = add_time_features(out)
    # Arrondis raisonnables
    for c, nd in [("u10", 4), ("v10", 4), ("wspd", 4), ("t2m", 3),
                  ("msl", 1), ("rho", 5), ("swh", 3), ("oni", 2)]:
        if c in out:
            out[c] = out[c].round(nd)
    return out, nodes


# ── Split temporel + statistiques de normalisation (anti-fuite) ───────────────

def temporal_split_stats(df, train_end, val_end):
    """
    Split par année : train ≤ train_end ; val = (train_end, val_end] ;
    test > val_end. Stats (moyenne/écart-type) calculées SUR LE TRAIN SEUL.
    """
    y = pd.to_datetime(df["datetime"]).dt.year
    train = df[y <= train_end]
    stats = {}
    for v in NORM_VARS:
        if v in df.columns and train[v].notna().any():
            stats[v] = {"mean": float(train[v].mean()),
                        "std": float(train[v].std() or 1.0)}
    split = {"train": [int(y.min()), int(train_end)],
             "val":   [int(train_end) + 1, int(val_end)],
             "test":  [int(val_end) + 1, int(y.max())]}
    return split, stats


# ── Utilitaire de fenêtrage (importable par le code d'entraînement) ───────────

def build_sequences(df_node, feature_cols, target_cols,
                    lookback=48, horizon=24, stride=1):
    """
    Construit des fenêtres glissantes (X, y) pour UN nœud (série régulière,
    triée par datetime). À appeler séparément par split pour éviter toute
    fuite temporelle.
      X : (n, lookback, len(feature_cols))
      y : (n, horizon,  len(target_cols))
    """
    F = df_node[feature_cols].to_numpy(dtype="float32")
    T = df_node[target_cols].to_numpy(dtype="float32")
    n = len(df_node) - lookback - horizon + 1
    if n <= 0:
        return (np.empty((0, lookback, len(feature_cols)), "float32"),
                np.empty((0, horizon, len(target_cols)), "float32"))
    idx = range(0, n, stride)
    X = np.stack([F[i:i + lookback] for i in idx])
    y = np.stack([T[i + lookback:i + lookback + horizon] for i in idx])
    return X, y


# ── Mode grille (optionnel, ConvLSTM / modèles de champ) ──────────────────────

def export_grid(route_id, data_dir, out_dir, years, months, coarsen):
    """Exporte les champs par boîte, coarsen×, en NetCDF compressé PAR ANNÉE."""
    gdir = out_dir / "grid"; gdir.mkdir(parents=True, exist_ok=True)
    for box in ROUTES[route_id]["boxes"]:
        sfx = box["suffix"]
        for year in years:
            monthly = []
            for month in months:
                fp = (data_dir / route_id / sfx /
                      f"ERA5_{route_id}_{sfx}_{year}_{month}.nc")
                if not fp.exists():
                    found = list(data_dir.rglob(
                        f"*{route_id}*{sfx}*{year}*{month}*.nc"))
                    if not found:
                        continue
                    fp = found[0]
                ds = open_nc(fp)
                if ds is None:
                    continue
                ds = normalise_ds(ds)
                if "lat" in ds.dims:
                    ds = ds.sortby("lat")
                if coarsen and coarsen > 1:
                    ds = ds.coarsen(lat=coarsen, lon=coarsen,
                                    boundary="trim").mean()
                monthly.append(ds.load())
                ds.close()
            if not monthly:
                continue
            year_ds = xr.concat(monthly, dim="time").sortby("time")
            for m in monthly:
                m.close()
            enc = {v: {"zlib": True, "complevel": 4}
                   for v in year_ds.data_vars}
            outfp = gdir / f"{route_id}_{sfx}_{year}.nc"
            year_ds.to_netcdf(outfp, encoding=enc)
            log.info(f"  grid → {outfp.name} "
                     f"({outfp.stat().st_size/1024**2:.1f} MB, "
                     f"coarsen×{coarsen})")
            year_ds.close()


# ── Pipeline principal ────────────────────────────────────────────────────────

def main():
    import sys as _sys
    if hasattr(_sys, "ps1") or "ipykernel" in _sys.modules:
        _sys.argv = [_sys.argv[0]]

    ap = argparse.ArgumentParser(
        description="Construit le jeu de données ML métocéan depuis ERA5 brut.")
    ap.add_argument("--route", default=None,
                    help="Route unique (ex. R2_NFK_HAM). Sinon les 4.")
    ap.add_argument("--mode", choices=["nodes", "grid"], default="nodes")
    ap.add_argument("--spacing-km", type=float, default=150.0,
                    help="Espacement des nœuds le long du corridor (mode nodes).")
    ap.add_argument("--coarsen", type=int, default=2,
                    help="Facteur de sous-échantillonnage spatial (mode grid).")
    ap.add_argument("--data-dir", default=str(DATA_IN))
    ap.add_argument("--out", default="ERA5_ML_dataset")
    ap.add_argument("--train-end", type=int, default=2021)
    ap.add_argument("--val-end", type=int, default=2022)
    args, _ = ap.parse_known_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    years    = ERA5_CONFIG["years"]
    months   = ERA5_CONFIG["months"]
    route_ids = [args.route] if args.route else list(ROUTES.keys())
    for rid in route_ids:
        if rid not in ROUTES:
            sys.exit(f"Route inconnue : {rid}. Options : {list(ROUTES)}")

    log.info(f"ERA5 → jeu de données ML | mode={args.mode} | "
             f"{datetime.now():%Y-%m-%d %H:%M}")
    log.info(f"Routes : {route_ids} | data_dir={data_dir.resolve()}")

    if args.mode == "grid":
        for rid in route_ids:
            log.info(f"=== {rid} (grille) ===")
            export_grid(rid, data_dir, out_dir, years, months, args.coarsen)
        log.info("Mode grille terminé.")
        return

    # ── mode nodes ────────────────────────────────────────────────────────────
    all_routes, manifest_nodes, coverage = [], {}, {}
    for rid in route_ids:
        log.info(f"=== {rid} (nodes) ===")
        df, nodes = extract_route_nodes(
            rid, data_dir, args.spacing_km, years, months)
        manifest_nodes[rid] = nodes
        if df is None or len(df) == 0:
            log.warning(f"  Aucune donnée produite pour {rid} "
                        f"(fichiers ERA5 absents ?).")
            coverage[rid] = {"rows": 0, "nodes_with_data": 0}
            continue
        # Rapport couverture / lacunes
        n_with = df["node_id"].nunique()
        hours_per_node = df.groupby("node_id")["datetime"].nunique()
        coverage[rid] = {
            "rows": int(len(df)),
            "nodes_with_data": int(n_with),
            "hours_per_node_min": int(hours_per_node.min()),
            "hours_per_node_max": int(hours_per_node.max()),
            "expected_hours_full_decade": 24 * 365 * 10,  # ordre de grandeur
            "nan_fraction": {c: float(df[c].isna().mean())
                             for c in ["wspd", "swh", "rho", "oni"]
                             if c in df},
        }
        fp = out_dir / f"{rid}_nodes.parquet"
        df.to_parquet(fp, index=False)
        log.info(f"  → {fp.name} : {len(df):,} lignes, {n_with} nœuds, "
                 f"{hours_per_node.min()}–{hours_per_node.max()} h/nœud")
        all_routes.append(df)

    if not all_routes:
        sys.exit("Aucune donnée extraite : vérifiez que ERA5_data/ contient "
                 "les fichiers NetCDF (étapes download + extract_zips).")

    full = pd.concat(all_routes, ignore_index=True)
    full.to_parquet(out_dir / "routes_nodes.parquet", index=False)

    split, stats = temporal_split_stats(full, args.train_end, args.val_end)

    feature_cols = ["u10", "v10", "wspd", "wdir_sin", "wdir_cos",
                    "t2m", "msl", "rho", "swh", "oni",
                    "hod_sin", "hod_cos", "doy_sin", "doy_cos"]
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "description": "Metocean decadal route-node dataset for deep "
                       "spatiotemporal forecasting (independent project).",
        "source": "ERA5 hourly 0.25deg, 2015-2024",
        "mode": "nodes",
        "spacing_km": args.spacing_km,
        "routes": route_ids,
        "n_rows_total": int(len(full)),
        "feature_columns": feature_cols,
        "candidate_targets": ["wspd", "swh"],
        "static_covariates": ["route", "node_id", "node_lat",
                              "node_lon", "dist_km"],
        "known_future_covariates": ["hod_sin", "hod_cos",
                                    "doy_sin", "doy_cos", "month"],
        "temporal_split": split,
        "normalization": {"method": "zscore_train_only", "stats": stats},
        "nodes_per_route": manifest_nodes,
        "coverage": coverage,
        "leakage_protocol": "Normalisation et fenêtrage par split ; "
                            "stats calculées sur le train uniquement ; "
                            "fenêtres ne traversant ni les bornes de split "
                            "ni les nœuds.",
        "files": {"combined": "routes_nodes.parquet",
                  "per_route": [f"{r}_nodes.parquet" for r in route_ids]},
    }
    with open(out_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    log.info("")
    log.info("=" * 64)
    log.info(f"TERMINÉ : {len(full):,} lignes | "
             f"{full['route'].nunique()} routes | "
             f"split {split['train']}/{split['val']}/{split['test']}")
    log.info(f"Sorties dans : {out_dir.resolve()}")
    log.info("  routes_nodes.parquet, <ROUTE>_nodes.parquet, "
             "dataset_manifest.json")
    log.info("=" * 64)


if __name__ == "__main__":
    main()
