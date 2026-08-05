"""
mf/metrics.py — Métriques de prévision (déterministes + probabilistes).

Toutes les fonctions opèrent en UNITÉS PHYSIQUES (inverser la normalisation
avant d'évaluer). y_true / y_pred : (N, H) ou (N,).
"""
import numpy as np
from scipy import stats as _st


def _flat(a):
    return np.asarray(a, dtype="float64").reshape(-1)


def rmse(y_true, y_pred):
    e = _flat(y_true) - _flat(y_pred)
    return float(np.sqrt(np.mean(e ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(_flat(y_true) - _flat(y_pred))))


def mape(y_true, y_pred, eps=0.5):
    yt = _flat(y_true)
    m = np.abs(yt) > eps                       # évite la division par ~0 (vent faible)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((yt[m] - _flat(y_pred)[m]) / yt[m])) * 100)


def bias(y_true, y_pred):
    return float(np.mean(_flat(y_pred) - _flat(y_true)))


def per_horizon_rmse(y_true, y_pred):
    yt = np.asarray(y_true, "float64"); yp = np.asarray(y_pred, "float64")
    return np.sqrt(np.mean((yt - yp) ** 2, axis=0))      # (H,)


def skill_score(mse_model, mse_ref):
    """1 - MSE_model / MSE_ref. >0 : meilleur que la référence."""
    if mse_ref <= 0:
        return float("nan")
    return float(1.0 - mse_model / mse_ref)


# ── Probabiliste (quantiles) ──────────────────────────────────────────────────

def pinball_loss(y_true, q_pred, quantiles):
    """
    y_true (N,H) ; q_pred (N,H,Q) ; quantiles liste de Q niveaux.
    Renvoie la perte pinball moyenne (sur tous les points et quantiles).
    """
    yt = np.asarray(y_true, "float64")[..., None]      # (N,H,1)
    qp = np.asarray(q_pred, "float64")                 # (N,H,Q)
    q = np.asarray(quantiles, "float64")[None, None, :]
    e = yt - qp
    loss = np.maximum(q * e, (q - 1) * e)
    return float(np.nanmean(loss))


def crps_from_quantiles(y_true, q_pred, quantiles):
    """
    Approximation du CRPS par la perte pinball moyenne sur une grille de
    quantiles : CRPS ≈ 2 · moyenne_q pinball  (exact à la limite continue).
    """
    return 2.0 * pinball_loss(y_true, q_pred, quantiles)


def coverage(y_true, q_pred, quantiles, lower=0.1, upper=0.9):
    """Taux de couverture empirique de l'intervalle [lower, upper]."""
    q = list(quantiles)
    if lower not in q or upper not in q:
        return float("nan")
    lo = np.asarray(q_pred)[..., q.index(lower)]
    hi = np.asarray(q_pred)[..., q.index(upper)]
    yt = np.asarray(y_true, "float64")
    valid = np.isfinite(yt)
    if valid.sum() == 0:
        return float("nan")
    inside = (yt >= lo) & (yt <= hi)
    return float(inside[valid].mean())


# ── Test de Diebold-Mariano (comparaison de deux prévisions) ──────────────────

def diebold_mariano(e1, e2, h=1, power=2):
    """
    Compare deux séries d'erreurs (e1, e2) appariées. H0 : précision égale.
    Renvoie (DM_stat, p_value bilatérale). e1, e2 : (N,) ou (N,H) aplaties.
    """
    e1 = _flat(e1); e2 = _flat(e2)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    dbar = d.mean()
    # variance long-terme (Newey-West tronqué à h-1)
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    for lag in range(1, h):
        cov = np.mean((d[lag:] - dbar) * (d[:-lag] - dbar))
        var += 2 * (1 - lag / h) * cov
    if var <= 0:
        return float("nan"), float("nan")
    dm = dbar / np.sqrt(var / n)
    p = 2 * (1 - _st.norm.cdf(abs(dm)))
    return float(dm), float(p)
