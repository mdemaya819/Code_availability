
import numpy as np
import pandas as pd
from .data import window_block
from . import metrics as M


def _acc():
    return {"se": 0.0, "ae": 0.0, "n": 0}


def predict_and_score(model, blocks, feats, scaler_x, scaler_y, lookback,
                      horizon, quantiles=None, device=None, stride=1,
                      batch=4096, keep_err=1_000_000):
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    is_q = quantiles is not None and len(quantiles) > 1
    qmid = quantiles.index(0.5) if (is_q and 0.5 in quantiles) else None

    glob, by_route, by_phase = _acc(), {}, {}
    se_h = np.zeros(horizon); n_h = np.zeros(horizon)
    err_keep, q_keep, y_keep = [], [], []

    with torch.no_grad():
        for b in blocks:
            if b["split"] != "test":
                continue
            X, Y, Yin, Ttar, _ = window_block(b, lookback, horizon, stride)
            if len(Y) == 0:
                continue
            T = len(b["y"]); idx = np.arange(0, T - lookback - horizon + 1, stride)
            phase = b["phase"][idx + lookback - 1]
            Xs = scaler_x.transform(X.astype("float64")).astype("float32")
            Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
            preds = []
            for i in range(0, len(Xs), batch):
                xb = torch.from_numpy(Xs[i:i + batch]).to(device)
                out = model(xb).cpu().numpy()
                preds.append(out)
            P = np.concatenate(preds, axis=0)              # (N,H) or (N,H,Q)
            if is_q:
                Pq = scaler_y.inverse(P)                   # (N,H,Q) phys
                point = Pq[:, :, qmid] if qmid is not None else Pq.mean(-1)
                if len(q_keep) * horizon < keep_err:
                    q_keep.append(Pq.astype("float32")); y_keep.append(Y)
            else:
                point = scaler_y.inverse(P)                # (N,H) phys
            valid = np.isfinite(Y)                          # (N,H) cible présente
            e = np.where(valid, point - Y, 0.0)
            sq = e ** 2; ab = np.abs(e)
            glob["se"] += float(sq.sum()); glob["ae"] += float(ab.sum())
            glob["n"] += int(valid.sum())
            se_h += sq.sum(0); n_h += valid.sum(0)
            r = b["route"]; by_route.setdefault(r, _acc())
            by_route[r]["se"] += float(sq.sum())
            by_route[r]["ae"] += float(ab.sum()); by_route[r]["n"] += int(valid.sum())
            for ph in np.unique(phase):
                mph = phase == ph
                by_phase.setdefault(ph, _acc())
                by_phase[ph]["se"] += float(sq[mph].sum())
                by_phase[ph]["ae"] += float(ab[mph].sum())
                by_phase[ph]["n"] += int(valid[mph].sum())
            if sum(len(x) for x in err_keep) < keep_err:
                err_keep.append(e.astype("float32"))

    def fin(a):
        return {"rmse": float(np.sqrt(a["se"] / max(a["n"], 1))),
                "mae": float(a["ae"] / max(a["n"], 1)), "n": a["n"]}

    out = {"overall": fin(glob),
           "per_route": {r: fin(a) for r, a in by_route.items()},
           "per_enso_phase": {str(p): fin(a) for p, a in by_phase.items()},
           "per_horizon_rmse": np.sqrt(se_h / np.maximum(n_h, 1)).tolist(),
           "_err_sample": (np.concatenate(err_keep) if err_keep else None)}
    if is_q and q_keep:
        Q = np.concatenate(q_keep, 0); Yk = np.concatenate(y_keep, 0)
        out["crps"] = M.crps_from_quantiles(Yk, Q, quantiles)
        out["pinball"] = M.pinball_loss(Yk, Q, quantiles)
        if 0.1 in quantiles and 0.9 in quantiles:
            out["coverage_80"] = M.coverage(Yk, Q, quantiles, 0.1, 0.9)
    return out


def compare_table(model_res, baseline_res, ref="persistence"):
    """Construit un tableau skill du modèle et des baselines vs `ref`."""
    ref_mse = baseline_res[ref]["rmse"] ** 2
    rows = [("MODEL", model_res["overall"]["rmse"], model_res["overall"]["mae"],
             M.skill_score(model_res["overall"]["rmse"] ** 2, ref_mse))]
    for n, r in baseline_res.items():
        rows.append((n, r["rmse"], r["mae"],
                     M.skill_score(r["rmse"] ** 2, ref_mse)))
    return pd.DataFrame(rows, columns=["method", "rmse", "mae", f"skill_vs_{ref}"])
