#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONI_download.py
===============
Régénère ONI_index_2015_2024.csv (Oceanic Niño Index, NOAA/CPC, ERSSTv5).

CORRECTION DU BUG D'ORIGINE
---------------------------
Le fichier corrompu contenait la colonne TOTAL (SST absolue ~27 °C) au lieu de
la colonne ANOM (l'anomalie = l'ONI, dans [-3, +3] °C). Ce script lit
EXPLICITEMENT la colonne ANOM.

COUVERTURE
----------
L'ONI est mensuel et glissant sur 3 mois : DJF, JFM, FMA, MAM, AMJ, MJJ, JJA,
JAS, ASO, SON, OND, NDJ. Les 12 saisons par an couvrent donc TOUTES les saisons
(été, automne, hiver, printemps) — il n'y a pas de fichier "hiver seulement".

Source : https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
Format : colonnes  SEAS  YR  TOTAL  ANOM   (on utilise ANOM)

Classement ENSO officiel (CPC) : un épisode El Niño / La Niña est déclaré
lorsque l'ONI franchit ±0.5 °C pendant >= 5 saisons glissantes CONSÉCUTIVES.
On fournit aussi une classification simple par seuil (±0.5 °C) par mois.

Usage :
    python ONI_download.py                 # télécharge, sinon table de secours
    python ONI_download.py --offline       # force la table de secours embarquée
    python ONI_download.py --start 2015 --end 2024 --out ONI_index_2015_2024.csv
"""
import argparse, io, sys
import numpy as np
import pandas as pd

URLS = [
    "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
    "https://origin.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
]

SEASONS = ["DJF","JFM","FMA","MAM","AMJ","MJJ","JJA","JAS","ASO","SON","OND","NDJ"]
SEAS_TO_MONTH = {s: i+1 for i, s in enumerate(SEASONS)}  # DJF->1 (centré sur janvier)

# Table de secours : ONI officiel ERSSTv5 (anomalies, °C), 2015–2024.
# 12 valeurs/an dans l'ordre DJF..NDJ. (2024: derniers mois provisoires.)
FALLBACK = {
2015:[ 0.5, 0.5, 0.5, 0.7, 0.9, 1.2, 1.5, 1.9, 2.2, 2.4, 2.6, 2.6],
2016:[ 2.5, 2.1, 1.6, 0.9, 0.4,-0.1,-0.4,-0.5,-0.6,-0.7,-0.7,-0.6],
2017:[-0.3,-0.2, 0.1, 0.2, 0.3, 0.3, 0.1,-0.1,-0.4,-0.7,-0.8,-1.0],
2018:[-0.9,-0.9,-0.7,-0.5,-0.2, 0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 0.8],
2019:[ 0.7, 0.7, 0.7, 0.7, 0.5, 0.5, 0.3, 0.1, 0.2, 0.3, 0.5, 0.5],
2020:[ 0.5, 0.5, 0.4, 0.2,-0.1,-0.3,-0.4,-0.6,-0.9,-1.2,-1.3,-1.2],
2021:[-1.0,-0.9,-0.8,-0.7,-0.5,-0.4,-0.4,-0.5,-0.7,-0.8,-1.0,-1.0],
2022:[-1.0,-0.9,-1.0,-1.1,-1.0,-0.9,-0.8,-0.9,-1.0,-1.0,-0.9,-0.8],
2023:[-0.7,-0.4,-0.1, 0.2, 0.5, 0.8, 1.1, 1.3, 1.6, 1.8, 1.9, 2.0],
2024:[ 1.8, 1.5, 1.1, 0.7, 0.4, 0.2,-0.1,-0.3,-0.5,-0.6,-0.7,-0.6],
}


def _download():
    import urllib.request
    for url in URLS:
        try:
            print(f"[INFO] Téléchargement {url}", file=sys.stderr)
            with urllib.request.urlopen(url, timeout=30) as r:
                txt = r.read().decode("utf-8", "replace")
            df = pd.read_csv(io.StringIO(txt), sep=r"\s+")
            df.columns = [c.upper() for c in df.columns]
            if {"SEAS","YR","ANOM"}.issubset(df.columns):
                print(f"[INFO] OK ({len(df)} lignes)", file=sys.stderr)
                return df[["SEAS","YR","ANOM"]].copy()
        except Exception as e:
            print(f"[WARN] échec {url}: {e}", file=sys.stderr)
    return None


def _from_fallback():
    rows = []
    for yr, vals in FALLBACK.items():
        for s, v in zip(SEASONS, vals):
            rows.append({"SEAS": s, "YR": yr, "ANOM": v})
    return pd.DataFrame(rows)


def classify_official(g):
    """Épisodes ENSO officiels : >=5 saisons glissantes consécutives au-delà de ±0.5."""
    g = g.sort_values(["year", "month"]).reset_index(drop=True)
    a = g["ONI"].values
    phase = np.array(["Neutral"] * len(a), dtype=object)
    for sign, name, cmp in [(1, "El_Nino", lambda x: x >= 0.5),
                            (-1, "La_Nina", lambda x: x <= -0.5)]:
        i = 0
        while i < len(a):
            if cmp(a[i]):
                j = i
                while j < len(a) and cmp(a[j]):
                    j += 1
                if j - i >= 5:           # >=5 saisons consécutives
                    phase[i:j] = name
                i = j
            else:
                i += 1
    return phase


def build(df, start, end):
    df = df[(df["YR"] >= start) & (df["YR"] <= end)].copy()
    df["ONI"] = pd.to_numeric(df["ANOM"], errors="coerce")

    # --- VALIDATION (le bug d'origine : TOTAL au lieu d'ANOM) ---
    lo, hi = df["ONI"].min(), df["ONI"].max()
    if lo < -4 or hi > 5:
        raise ValueError(
            f"Valeurs ONI hors plage [{lo:.2f}, {hi:.2f}] — vous lisez probablement "
            f"la colonne TOTAL (SST absolue) au lieu d'ANOM.")
    df["season"] = df["SEAS"]
    df["month"]  = df["SEAS"].map(SEAS_TO_MONTH)
    df = df.rename(columns={"YR": "year"})
    # classification simple par seuil
    df["ENSO_phase"] = np.where(df["ONI"] >= 0.5, "El_Nino",
                        np.where(df["ONI"] <= -0.5, "La_Nina", "Neutral"))
    # classification officielle (épisodes)
    df["ENSO_official"] = classify_official(df)
    return df[["year","month","season","ONI","ENSO_phase","ENSO_official"]] \
             .sort_values(["year","month"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--end",   type=int, default=2024)
    ap.add_argument("--out",   default="ONI_index_2015_2024.csv")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    raw = None if args.offline else _download()
    src = "NOAA/CPC (live)"
    if raw is None:
        raw = _from_fallback(); src = "table de secours embarquée (ERSSTv5)"
        print(f"[WARN] Utilisation de la {src}. Vérifiez contre la source live.", file=sys.stderr)

    out = build(raw, args.start, args.end)
    out.to_csv(args.out, index=False)
    print(f"[OK] {args.out} écrit — {len(out)} lignes — source : {src}")
    print(f"     ONI ∈ [{out['ONI'].min():+.2f}, {out['ONI'].max():+.2f}] °C ; "
          f"phases : {out['ENSO_phase'].value_counts().to_dict()}")
    # Aperçu valeurs d'hiver (DJF) par an — doit coller à annual_summary.ONI_mean
    djf = out[out['season']=='DJF'][['year','ONI']]
    print("     DJF par an :", dict(zip(djf['year'], djf['ONI'])))
    return out


if __name__ == "__main__":
    main()
