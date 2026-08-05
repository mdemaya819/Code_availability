"""
mf/data.py — Couche données anti-fuite pour la prévision métocéan multi-horizon.

Garanties anti-fuite (vérifiées par tests/test_data_leakage.py) :
  1. Split STRICTEMENT par année (depuis le manifeste : train/val/test).
  2. Normalisation z-score calculée SUR LE TRAIN UNIQUEMENT (stats du
     manifeste), appliquée à l'identique à val/test.
  3. Fenêtrage effectué SÉPARÉMENT dans chaque bloc (route, node_id, split) :
     aucune fenêtre ne traverse une frontière de split ni une frontière de
     nœud (entrée ET cible restent dans le même bloc contigu).
  4. Variables auto-écartées si stat de normalisation absente OU NaN quasi
     total (ex. swh à 100 % NaN dans ce jeu) → jamais injectées en entrée.

Le module s'importe SANS torch ; la classe WindowDataset importe torch
paresseusement (uniquement si on l'instancie).
"""
from __future__ import annotations
import json
import glob
import os
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# ── Manifeste ─────────────────────────────────────────────────────────────────

def load_manifest(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_features(manifest, target, drop_extra=None):
    """
    Détermine la liste de features RÉELLEMENT utilisables :
      - part de manifest['feature_columns'] ;
      - retire toute variable continue sans stat de normalisation
        (ex. swh, 100 % NaN) ;
      - retire les NaN quasi totaux signalés dans coverage.nan_fraction ;
      - retire la cible des features (évite la fuite triviale d'identité).
    Les variables déjà bornées (sin/cos) n'ont pas besoin de stat.
    """
    feats = list(manifest["feature_columns"])
    stats = manifest["normalization"]["stats"]
    bounded = {"wdir_sin", "wdir_cos", "hod_sin", "hod_cos", "doy_sin",
               "doy_cos", "mwd_sin", "mwd_cos"}
    # NaN quasi total (sur n'importe quelle route)
    nan_bad = set()
    for cov in manifest.get("coverage", {}).values():
        for v, fr in (cov.get("nan_fraction") or {}).items():
            if fr is not None and fr >= 0.999:
                nan_bad.add(v)
    drop = set(drop_extra or []) | nan_bad
    usable = []
    for f in feats:
        if f == target:
            continue                      # la cible n'entre pas en feature
        if f in drop:
            continue
        if f in bounded or f in stats:    # bornée OU stat dispo
            usable.append(f)
    if target not in stats:
        raise ValueError(f"Cible '{target}' sans stat de normalisation "
                         f"(NaN ?). Cibles valides : {list(stats)}")
    return usable, sorted(drop)


class Scaler:
    """Z-score à partir des stats train-only du manifeste (jamais refit)."""
    def __init__(self, stats, cols):
        self.cols = list(cols)
        self.mean = np.array([stats[c]["mean"] if c in stats else 0.0
                              for c in cols], dtype="float64")
        self.std = np.array([stats[c]["std"] if c in stats else 1.0
                             for c in cols], dtype="float64")
        self.std[self.std == 0] = 1.0

    def transform(self, arr):                       # arr (..., len(cols))
        return (arr - self.mean) / self.std

    def inverse(self, arr):
        return arr * self.std + self.mean


class TargetScaler:
    def __init__(self, stats, target):
        self.mean = float(stats[target]["mean"])
        self.std = float(stats[target]["std"]) or 1.0

    def transform(self, y):
        return (y - self.mean) / self.std

    def inverse(self, y):
        return y * self.std + self.mean


# ── Chargement + découpage en blocs (route, node) × split ─────────────────────

def _year_mask(times, yr0, yr1):
    y = times.astype("datetime64[Y]").astype(int) + 1970
    return (y >= yr0) & (y <= yr1)


def _read_parquet_robust(path, columns):
    """
    Lit un parquet en contournant :
      - le bug pyarrow 19.0.0 (« histogram », Arrow #45283) ;
      - une lecture pyarrow défaillante sur fichier valide,
    via un repli AUTOMATIQUE sur fastparquet. Détecte aussi un fichier
    corrompu/incomplet (footer manquant) et le signale clairement.
    """
    import os as _os
    try:
        size = _os.path.getsize(path)
    except OSError:
        size = None
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception as exc:
        msg = str(exc).lower()
        recoverable = any(k in msg for k in (
            "histogram", "magic bytes", "could not open",
            "not a parquet", "corrupt"))
        if recoverable:
            try:
                return pd.read_parquet(path, columns=columns,
                                       engine="fastparquet")
            except Exception:
                pass
        hint = ""
        if size is not None and size < 1024:
            hint = (f"\nLe fichier ne fait que {size} octet(s) → il est "
                    f"probablement CORROMPU/incomplet (écriture interrompue ou "
                    f"ratée). Supprimez-le et régénérez-le.")
        elif "magic bytes" in msg or "not a parquet" in msg or "corrupt" in msg:
            hint = ("\nLe fichier semble CORROMPU (footer parquet invalide). "
                    "Supprimez-le et régénérez-le.")
        try:
            import pyarrow as _pa
            ver = _pa.__version__
        except Exception:
            ver = "inconnue"
        raise RuntimeError(
            f"Lecture parquet impossible : « {exc} ».{hint}\n"
            f"Vérifiez, dans CET environnement (puis redémarrez le noyau) :\n"
            f"  • pyarrow {ver} → installez pyarrow>=19.0.1 ET fastparquet ;\n"
            f"  • si le fichier est corrompu, supprimez-le et relancez la "
            f"génération du jeu de données.\nFichier : {path}") from exc


def load_blocks(parquet_paths, manifest, target="wspd",
                feature_cols=None, routes=None):
    """
    Charge les parquets et renvoie une liste de blocs :
      block = {route, node_id, split, F (T,Fdim) float32, y (T,) float32,
               times (T,) datetime64[ns], phase (T,) str}
    Un bloc = série CONTIGUË d'un nœud restreinte à un split (train/val/test).
    Rien n'est fenêtré ici ; le fenêtrage se fait par bloc (anti-fuite).
    """
    if feature_cols is None:
        feature_cols, _ = resolve_features(manifest, target)
    split = manifest["temporal_split"]
    needed = list(dict.fromkeys(
        feature_cols + [target, "route", "node_id", "datetime"]
        + (["enso_phase"] if True else [])))
    frames = []
    for p in parquet_paths:
        frames.append(_read_parquet_robust(p, [c for c in needed]))
    df = pd.concat(frames, ignore_index=True)
    if routes:
        df = df[df["route"].isin(routes)]
    df["datetime"] = pd.to_datetime(df["datetime"])
    if "enso_phase" not in df.columns:
        df["enso_phase"] = "Unknown"

    blocks = []
    for (route, node), g in df.groupby(["route", "node_id"], sort=False):
        g = g.sort_values("datetime")
        t = g["datetime"].values.astype("datetime64[ns]")
        F = g[feature_cols].to_numpy(dtype="float32")
        y = g[target].to_numpy(dtype="float32")
        ph = g["enso_phase"].to_numpy()
        for sname, (yr0, yr1) in split.items():
            m = _year_mask(t, yr0, yr1)
            if m.sum() == 0:
                continue
            ts, Fs, ys, phs = t[m], F[m], y[m], ph[m]
            # Découpe en segments HORAIRES STRICTEMENT CONTIGUS : aucune fenêtre
            # ne pourra enjamber un mois manquant (cas des trous de R1).
            if len(ts) > 1:
                dh = np.diff(ts).astype("timedelta64[h]").astype("int64")
                breaks = np.where(dh != 1)[0] + 1
                bounds = np.concatenate(([0], breaks, [len(ts)]))
            else:
                bounds = np.array([0, len(ts)])
            for a, c in zip(bounds[:-1], bounds[1:]):
                blocks.append({"route": route, "node_id": int(node),
                               "split": sname, "F": Fs[a:c], "y": ys[a:c],
                               "times": ts[a:c], "phase": phs[a:c]})
    return blocks, feature_cols


# ── Fenêtrage par bloc (vectorisé, sans traverser les frontières) ─────────────

def window_block(block, lookback, horizon, stride=1):
    """
    Renvoie pour UN bloc :
      X    (N,L,F) float32  — features (sans la cible)
      Y    (N,H)   float32  — cible sur l'horizon
      Yin  (N,L)   float32  — cible sur la fenêtre d'entrée (pour baselines)
      Ttar (N,H)   int64[ns]
      Tin_last (N,) int64[ns]
    Fenêtre i : entrée [i, i+L-1], cible [i+L, i+L+H-1] — entièrement DANS le
    bloc (un seul split, un seul nœud). Aucune fuite.
    """
    F, y, t = block["F"], block["y"], block["times"]
    T = len(y)
    N = T - lookback - horizon + 1
    if N <= 0:
        F0 = F.shape[1]
        return (np.empty((0, lookback, F0), "float32"),
                np.empty((0, horizon), "float32"),
                np.empty((0, lookback), "float32"),
                np.empty((0, horizon), "int64"),
                np.empty((0,), "int64"))
    swF = sliding_window_view(F, lookback, axis=0)          # (T-L+1, F, L)
    swF = np.moveaxis(swF, -1, 1)                           # (T-L+1, L, F)
    swyin = sliding_window_view(y, lookback)                # (T-L+1, L)
    swy = sliding_window_view(y, horizon)                   # (T-H+1, H)
    swt = sliding_window_view(t, horizon)                   # (T-H+1, H)
    idx = np.arange(0, N, stride)
    X = swF[idx]                                            # entrée: i..i+L-1
    Yin = swyin[idx]                                        # cible passée
    Y = swy[idx + lookback]                                 # cible: i+L..+H-1
    Ttar = swt[idx + lookback].astype("int64")
    Tin_last = t[idx + lookback - 1].astype("int64")
    return (X.astype("float32"), Y.astype("float32"),
            Yin.astype("float32"), Ttar, Tin_last)


# ── Dataset torch (fenêtrage paresseux, mémoire-léger) ────────────────────────

def make_torch_dataset(blocks, split, lookback, horizon, scaler_x,
                       scaler_y, target_idx, stride=1):
    """
    Dataset torch n'indexant QUE les blocs du split demandé.

    OPTIMISATION CLÉ : la normalisation (train-only) est appliquée UNE SEULE FOIS
    par bloc à la construction, pas à chaque fenêtre. __getitem__ ne fait plus
    qu'un découpage de tableaux déjà normalisés (× ~100 plus rapide que
    l'ancien code qui appelait scaler.transform à chaque fenêtre). Les fenêtres
    restent fabriquées à la volée (pas de matérialisation massive).
    L'anti-fuite est inchangé : mêmes stats train-only, mêmes bornes de bloc.
    """
    import torch
    from torch.utils.data import Dataset

    sub = [b for b in blocks if b["split"] == split]
    Fs, ys, idx = [], [], []
    n_drop = 0
    for bi, b in enumerate(sub):
        Fsc = scaler_x.transform(b["F"].astype("float64")).astype("float32")
        ysc = scaler_y.transform(b["y"].astype("float64")).astype("float32")
        # IMPUTATION des entrées manquantes → 0 (= moyenne train après z-score).
        # Indispensable : un NaN d'entrée (swh/mwp/mwd sur certains nœuds R1)
        # rendrait la prédiction, la perte, puis TOUS les poids NaN.
        Fsc = np.nan_to_num(Fsc, nan=0.0, posinf=0.0, neginf=0.0)
        Fs.append(np.ascontiguousarray(Fsc))
        ys.append(np.ascontiguousarray(ysc))
        T = len(ysc)
        N = T - lookback - horizon + 1
        for s in range(0, max(0, N), stride):
            # ignorer les fenêtres dont la CIBLE (horizon) contient un NaN
            if np.isnan(ysc[s + lookback:s + lookback + horizon]).any():
                n_drop += 1
                continue
            idx.append((bi, s))
    if n_drop:
        import logging
        logging.getLogger("mf.data").info(
            f"  [{split}] {n_drop} fenêtres ignorées (cible manquante)")

    class _WindowDataset(Dataset):
        def __len__(self):
            return len(idx)

        def __getitem__(self, k):
            bi, s = idx[k]
            X = Fs[bi][s:s + lookback]                         # (L,F) normalisé
            Y = ys[bi][s + lookback:s + lookback + horizon]    # (H,)  normalisé
            return torch.from_numpy(X), torch.from_numpy(Y)

    return _WindowDataset()


# ── Aide : découverte des parquets ────────────────────────────────────────────

def find_parquets(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, "*_nodes.parquet")))
    combined = os.path.join(data_dir, "routes_nodes.parquet")
    if os.path.exists(combined):
        return [combined]
    if not paths:
        raise FileNotFoundError(f"Aucun parquet dans {data_dir}")
    return paths
