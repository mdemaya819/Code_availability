
import argparse, json
import numpy as np
import pandas as pd

POSITIVE = {"wspd", "swh", "mwp"}        # variables physiquement positives


def _to_parquet_robust(df, out):
    """Écrit un parquet de façon SÛRE : écrit dans un fichier temporaire,
    vérifie qu'il est relisible (footer valide), puis renomme. Essaie pyarrow
    puis fastparquet puis pyarrow non compressé — ne laisse jamais de fichier
    corrompu (contourne un éventuel pyarrow 19.0.0 défaillant)."""
    import os
    import pandas as pd
    tmp = str(out) + ".tmp"
    last = None
    for kw in ({"engine": "pyarrow"},
               {"engine": "fastparquet"},
               {"engine": "pyarrow", "compression": None}):
        try:
            df.to_parquet(tmp, index=False, **kw)
            # vérification : relire l'en-tête avec le même moteur
            pd.read_parquet(tmp, columns=list(df.columns[:1]),
                            engine=kw["engine"])
            os.replace(tmp, out)
            return
        except Exception as e:
            last = e
            try:
                os.remove(tmp)
            except OSError:
                pass
    raise RuntimeError(
        "Écriture parquet impossible (pyarrow ET fastparquet ont échoué). "
        "Installez fastparquet et/ou pyarrow>=19.0.1 dans cet environnement. "
        f"Dernière erreur : {last}")


def build(manifest_path, out="synthetic_nodes.parquet",
          routes=("R2_NFK_HAM", "R1_SHA_RTM"), nodes_per_route=3, seed=0):
    m = json.load(open(manifest_path, encoding="utf-8"))
    st = m["normalization"]["stats"]
    feats = m["feature_columns"]
    yr0 = m["temporal_split"]["train"][0]
    yr1 = m["temporal_split"]["test"][1]
    times = pd.date_range(f"{yr0}-01-01", f"{yr1}-12-31 23:00", freq="h")
    T = len(times)
    rng = np.random.default_rng(seed)

    months = pd.period_range(f"{yr0}-01", f"{yr1}-12", freq="M")
    oni_m = {(p.year, p.month): float(np.clip(rng.normal(0.1, 0.9), -2.5, 2.8))
             for p in months}

    hod = times.hour.values; doy = times.dayofyear.values
    time_feats = {
        "hod_sin": np.sin(2*np.pi*hod/24),   "hod_cos": np.cos(2*np.pi*hod/24),
        "doy_sin": np.sin(2*np.pi*doy/365.25), "doy_cos": np.cos(2*np.pi*doy/365.25),
    }

    def gen(name):
        """Génère une colonne plausible selon son nom."""
        if name in time_feats:
            return time_feats[name].astype("float32")
        if name == "oni":
            return None                       # rempli ensuite via oni_m
        if name.endswith("_sin") or name.endswith("_cos"):
            ang = rng.uniform(0, 2*np.pi, T)   # direction aléatoire
            return (np.sin(ang) if name.endswith("_sin") else np.cos(ang)).astype("float32")
        s = st.get(name, {"mean": 0.0, "std": 1.0})
        x = rng.normal(s["mean"], s["std"], T).astype("float32")
        return np.abs(x) if name in POSITIVE else x

    rows = []
    for r in routes:
        nodes = [n for n in m["nodes_per_route"][r] if n.get("box")][:nodes_per_route]
        for nd in nodes:
            df = pd.DataFrame({
                "route": r, "node_id": nd["node_id"],
                "node_lat": nd["lat"], "node_lon": nd["lon"],
                "dist_km": nd["dist_km"], "datetime": times,
            })
            for f in feats:
                g = gen(f)
                if g is not None:
                    df[f] = g
            df["year"] = df["datetime"].dt.year
            df["month"] = df["datetime"].dt.month
            if "oni" in feats:
                df["oni"] = [oni_m[(y, mo)] for y, mo in zip(df.year, df.month)]
                df["enso_phase"] = np.where(df.oni >= 0.5, "El_Nino",
                                    np.where(df.oni <= -0.5, "La_Nina", "Neutral"))
            rows.append(df)
    out_df = pd.concat(rows, ignore_index=True)
    _to_parquet_robust(out_df, out)
    print(f"[synthetic] {out} : {len(out_df):,} lignes, "
          f"{out_df['route'].nunique()} routes, "
          f"{out_df.groupby('route')['node_id'].nunique().to_dict()} nœuds, "
          f"{T} pas/nœud ; colonnes={[c for c in feats if c in out_df.columns]}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="dataset_manifest.json")
    ap.add_argument("--out", default="synthetic_nodes.parquet")
    ap.add_argument("--nodes", type=int, default=3)
    a = ap.parse_args()
    build(a.manifest, a.out, nodes_per_route=a.nodes)
