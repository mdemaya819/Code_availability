"""
mf/baselines.py — Références obligatoires (tout gain DL est mesuré en skill
relatif à ces baselines). Toutes respectent l'anti-fuite : la climatologie
est apprise SUR LE TRAIN UNIQUEMENT ; persistance et naïf saisonnier
n'utilisent que le passé observé dans la fenêtre d'entrée.

Évaluation STREAMING par bloc (route, node, split) : on n'accumule jamais
toutes les fenêtres en mémoire (compatible 24 M lignes). On renvoie, pour le
split d'évaluation, le tableau des erreurs par fenêtre×horizon (utile pour le
test de Diebold-Mariano) — borné en mémoire car ~ N_windows × H.
"""
import numpy as np
import pandas as pd
from .data import window_block


# ── Climatologie apprise sur le train ─────────────────────────────────────────

def fit_climatology(blocks, target_is_y=True):
    """
    Moyenne climatologique de la cible par (mois, heure) sur TOUS les blocs
    'train'. Renvoie un dict {(month,hour): mean} + moyenne globale de repli.
    """
    sums, cnts = {}, {}
    g_sum, g_cnt = 0.0, 0
    for b in blocks:
        if b["split"] != "train":
            continue
        t = pd.DatetimeIndex(b["times"])
        key = list(zip(t.month.values, t.hour.values))
        y = b["y"]
        for k, v in zip(key, y):
            if np.isnan(v):
                continue
            sums[k] = sums.get(k, 0.0) + v
            cnts[k] = cnts.get(k, 0) + 1
            g_sum += v; g_cnt += 1
    clim = {k: sums[k] / cnts[k] for k in sums}
    g_mean = g_sum / max(g_cnt, 1)
    return clim, g_mean


def _clim_predict(clim, g_mean, Ttar):
    t = pd.DatetimeIndex(Ttar.reshape(-1))
    keys = list(zip(t.month.values, t.hour.values))
    out = np.array([clim.get(k, g_mean) for k in keys], dtype="float64")
    return out.reshape(Ttar.shape)


# ── Prédicteurs baseline au niveau fenêtre ────────────────────────────────────

def predict_persistence(Yin, horizon):
    """ŷ(t+k) = dernière valeur observée de la cible (Yin[:, -1])."""
    last = Yin[:, -1]
    return np.repeat(last[:, None], horizon, axis=1)


def predict_seasonal_daily(Yin, lookback, horizon):
    """
    Naïf saisonnier journalier : ŷ(t+k) = valeur 24 h avant l'instant cible,
    lue dans la fenêtre d'entrée (nécessite lookback >= 24). Sinon None.
    """
    if lookback < 24:
        return None
    cols = [lookback + k - 24 for k in range(horizon)]
    if min(cols) < 0:
        return None
    return Yin[:, cols]


# ── Évaluation streaming d'une baseline sur un split ──────────────────────────

def evaluate_baselines(blocks, split, lookback, horizon,
                       clim=None, g_mean=None, stride=1, max_keep=2_000_000):
    """
    Parcourt les blocs du split et calcule, pour chaque baseline, l'erreur
    (ŷ-y) par fenêtre×horizon + agrégats RMSE/MAE. Stockage borné.
    """
    names = ["persistence", "seasonal_daily", "climatology"]
    err = {n: [] for n in names}
    ytrue = {n: [] for n in names}
    kept = {n: 0 for n in names}
    for b in blocks:
        if b["split"] != split:
            continue
        X, Y, Yin, Ttar, _ = window_block(b, lookback, horizon, stride)
        if len(Y) == 0:
            continue
        preds = {
            "persistence": predict_persistence(Yin, horizon),
            "seasonal_daily": predict_seasonal_daily(Yin, lookback, horizon),
        }
        if clim is not None:
            preds["climatology"] = _clim_predict(clim, g_mean, Ttar)
        for n, p in preds.items():
            if p is None:
                continue
            e = p - Y
            if kept[n] < max_keep:
                err[n].append(e.astype("float32"))
                ytrue[n].append(Y.astype("float32"))
                kept[n] += e.shape[0]
    out = {}
    for n in names:
        if not err[n]:
            continue
        E = np.concatenate(err[n], axis=0)
        Yt = np.concatenate(ytrue[n], axis=0)
        out[n] = {"err": E, "y": Yt,
                  "rmse": float(np.sqrt(np.nanmean(E ** 2))),
                  "mae": float(np.nanmean(np.abs(E))),
                  "per_h_rmse": np.sqrt(np.nanmean(E ** 2, axis=0))}
    return out
