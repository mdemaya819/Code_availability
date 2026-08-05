"""
run_experiments.py — Orchestration Phase 4.

Lance la MATRICE expérimentale complète et sauvegarde un JSON par exécution :
   cibles × modèles × graines  (+ baselines, une fois par cible)

Pour chaque exécution (cible, modèle, graine), enregistre :
  - métriques TEST : overall, per_route, per_enso_phase, per_horizon_rmse,
    crps, coverage_80 ;
  - skill vs persistance ; test de Diebold-Mariano (modèle vs persistance)
    avec correction Newey-West à l'horizon.

Tout respecte le protocole anti-fuite du pipeline (split par année, stats
train-only, fenêtrage par nœud/segment contigu).

────────────────────────────────────────────────────────────────────────────
UTILISATION DANS JUPYTER (recommandé)
────────────────────────────────────────────────────────────────────────────
Le notebook doit être ouvert DANS le dossier 'metocean_forecast' (celui qui
contient 'mf/'), ou le dataset doit être trouvable à côté / dans un parent.

  • Le plus simple — éditez le bloc CONFIG en bas du fichier puis, en cellule :
        %run run_experiments.py

  • Ou bien, sans rien éditer, appelez la fonction avec vos réglages :
        import run_experiments as R
        R.run_config(targets=["swh"], models=["tcn"], seeds=[0],
                     max_nodes_per_route=5, stride=24, epochs=5)

  • Aucun argument de ligne de commande n'est requis dans Jupyter : le script
    n'appelle PAS parse_args() en notebook (ce qui éviterait l'erreur due à
    l'argument « -f …kernel.json » injecté par le noyau).

UTILISATION EN TERMINAL (GPU recommandé pour la matrice complète) :
    python run_experiments.py --data-dir ERA5_ML_dataset \
        --manifest ERA5_ML_dataset/dataset_manifest.json \
        --targets wspd swh mwp --models tcn bilstm cnnbilstmattention tft \
        --seeds 0 1 2 --lookback 48 --horizon 24 --quantiles 0.1 0.5 0.9 \
        --epochs 60 --out-dir results

Puis :  python aggregate_results.py --results results
        (ou en notebook :  %run aggregate_results.py)
"""
import os
import sys
import glob
import json
import time
import argparse
import logging
from pathlib import Path

import numpy as np

def _locate_mf_root():
    """Trouve le dossier contenant le paquet 'mf/' (script, cwd, ou parents)."""
    cands = []
    try:
        cands.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:                       # cellule Jupyter : pas de __file__
        pass
    cands.append(os.getcwd())
    for base in list(cands):                # ajoute jusqu'à 3 niveaux de parents
        p = base
        for _ in range(3):
            p = os.path.dirname(p)
            cands.append(p)
    for c in cands:
        if os.path.isfile(os.path.join(c, "mf", "data.py")):
            return c
    return None


_ROOT = _locate_mf_root()
if _ROOT is None:
    raise SystemExit(
        "\nLe paquet 'mf' est introuvable depuis cet emplacement.\n"
        "Ce script doit tourner DANS le dossier 'metocean_forecast' (celui qui "
        "contient le sous-dossier 'mf/' et make_synthetic.py).\n"
        "Deux solutions :\n"
        "  • Ouvrez et exécutez le run_experiments.py situé DANS ce dossier "
        "(pas une copie collée ailleurs comme 'sanstitre0.py') ; OU\n"
        "  • Ajoutez au début du script (adaptez le chemin) :\n"
        "        import os ; os.chdir(r'C:\\\\chemin\\\\vers\\\\metocean_forecast')\n"
        f"Dossier courant : {os.getcwd()}")
sys.path.insert(0, _ROOT)
try:
    import torch  # noqa: F401  (requis par les modèles)
except ModuleNotFoundError:
    raise SystemExit(
        "\nPyTorch n'est pas installé dans cet environnement.\n"
        "  • CPU :  pip install torch\n"
        "  • GPU NVIDIA : https://pytorch.org/get-started/locally/  "
        "(ex. pip install torch --index-url "
        "https://download.pytorch.org/whl/cu124)\n")
from mf import data as D
from mf import baselines as B
from mf import metrics as Mx
from mf.models import build_model
from mf.train import train_model
from mf.evaluate import predict_and_score

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("phase4")


def set_seed(seed):
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_default(o):
    """Rend sérialisables les objets NumPy (ndarray, scalaires)."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"Type non sérialisable : {type(o).__name__}")


def _resolve_dataset(data_dir, manifest):
    """
    Localise dataset_manifest.json + les parquets, même si --data-dir/--manifest
    pointent au mauvais endroit (cas fréquent sous Spyder où l'on ne passe pas
    d'arguments). Cherche dans : le chemin donné, le dossier courant, _ROOT, et
    jusqu'à 3 niveaux de parents, sous des noms usuels. Renvoie (data_dir,
    manifest_path) ou (None, None).
    """
    if os.path.isfile(manifest):
        return (os.path.dirname(manifest) or "."), manifest
    cand = os.path.join(data_dir, "dataset_manifest.json")
    if os.path.isfile(cand):
        return data_dir, cand
    roots = [os.getcwd(), _ROOT]
    for r in list(roots):
        p = r
        for _ in range(3):
            p = os.path.dirname(p)
            roots.append(p)
    subdirs = [data_dir, "ERA5_ML_dataset", "ERA5_ML_dataset_v2",
               "dataset", "ML_dataset", "."]
    seen = set()
    for r in roots:
        if not r or r in seen:
            continue
        seen.add(r)
        for n in subdirs:
            cand = os.path.join(r, n, "dataset_manifest.json")
            if os.path.isfile(cand):
                return os.path.join(r, n), cand
    return None, None


def _parquets(data_dir):
    """Liste des parquets par nœud (un par route) ; repli sur le global."""
    pq = sorted(glob.glob(str(Path(data_dir) / "*_nodes.parquet")))
    pq = [p for p in pq if not p.endswith("routes_nodes.parquet")]
    if not pq:
        g = Path(data_dir) / "routes_nodes.parquet"
        if g.exists():
            pq = [str(g)]
    if not pq:
        sys.exit(f"Aucun parquet *_nodes.parquet sous {data_dir}.")
    return pq


def _subsample_nodes(blocks, k):
    """Garde au plus k nœuds (les premiers rencontrés) par route — pour des
    runs rapides sur CPU. N'altère pas le protocole (sous-ensemble de nœuds)."""
    order = {}
    for b in blocks:
        order.setdefault(b["route"], [])
        if b["node_id"] not in order[b["route"]]:
            order[b["route"]].append(b["node_id"])
    keep = {(r, nid) for r, ids in order.items() for nid in ids[:k]}
    return [b for b in blocks if (b["route"], b["node_id"]) in keep]


def paired_errors(model, blocks, feats, sx, sy, L, H, stride, quantiles):
    """
    Erreurs ponctuelles TEST alignées modèle vs persistance (mêmes fenêtres),
    pour Diebold-Mariano. Renvoie (err_model, err_persist) aplaties.
    """
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()
    is_q = quantiles is not None and len(quantiles) > 1
    qmid = quantiles.index(0.5) if (is_q and 0.5 in quantiles) else None
    em, ep = [], []
    with torch.no_grad():
        for b in blocks:
            if b["split"] != "test":
                continue
            X, Y, Yin, Ttar, _ = D.window_block(b, L, H, stride)
            if len(Y) == 0:
                continue
            Xs = sx.transform(X.astype("float64")).astype("float32")
            Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
            P = []
            for i in range(0, len(Xs), 4096):
                xb = torch.from_numpy(Xs[i:i + 4096]).to(dev)
                P.append(model(xb).cpu().numpy())
            P = np.concatenate(P, 0)
            if is_q:
                Pq = sy.inverse(P)
                point = Pq[:, :, qmid] if qmid is not None else Pq.mean(-1)
            else:
                point = sy.inverse(P)
            persist = B.predict_persistence(Yin, H)         # unités physiques
            valid = np.isfinite(Y) & np.isfinite(persist)   # cible + persist définis
            em.append(np.where(valid, point - Y, np.nan).ravel())
            ep.append(np.where(valid, persist - Y, np.nan).ravel())
    if not em:
        return np.array([]), np.array([])
    em = np.concatenate(em); ep = np.concatenate(ep)
    m = np.isfinite(em) & np.isfinite(ep)
    return em[m], ep[m]


def run(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    data_dir, manifest_path = _resolve_dataset(args.data_dir, args.manifest)
    if manifest_path is None:
        raise SystemExit(
            "\nJeu de données introuvable : ni le manifeste "
            f"« {args.manifest} » ni un dossier ERA5_ML_dataset/ valide.\n"
            "Indiquez les bons chemins, p. ex. (adaptez) :\n"
            "  --data-dir \"../ERA5_ML_dataset\" "
            "--manifest \"../ERA5_ML_dataset/dataset_manifest.json\"\n"
            "Sous Spyder : Exécution → Configuration par fichier → "
            "« Options en ligne de commande ». Sinon, déplacez le dossier "
            "ERA5_ML_dataset à côté de ce script.\n"
            f"Dossier courant : {os.getcwd()}")
    if (os.path.abspath(manifest_path) != os.path.abspath(args.manifest)):
        log.info(f"Jeu de données localisé automatiquement : {data_dir}")
        log.info(f"  manifeste : {manifest_path}")
    manifest = D.load_manifest(manifest_path)
    parquets = _parquets(data_dir)
    Q = list(args.quantiles) if args.quantiles else None
    L, H, st = args.lookback, args.horizon, args.stride
    log.info(f"Matrice : cibles={args.targets} modèles={args.models} "
             f"graines={args.seeds} | L={L} H={H} stride={st} epochs={args.epochs}")
    log.info(f"Parquets : {len(parquets)} fichier(s) | sortie={out.resolve()}")

    for target in args.targets:
        feats, dropped = D.resolve_features(manifest, target)
        log.info("=" * 72)
        log.info(f"CIBLE = {target} | {len(feats)} features | dropped={dropped}")
        blocks, feats = D.load_blocks(parquets, manifest, target=target,
                                      feature_cols=feats)
        if args.max_nodes_per_route:
            blocks = _subsample_nodes(blocks, args.max_nodes_per_route)
            log.info(f"  nœuds limités à {args.max_nodes_per_route}/route "
                     f"→ {len(blocks)} blocs")
        sx = D.Scaler(manifest["normalization"]["stats"], feats)
        sy = D.TargetScaler(manifest["normalization"]["stats"], target)
        clim, gmean = B.fit_climatology(blocks)

        # Baselines (une fois par cible) — ne garder que les scalaires (rmse/mae…)
        base = B.evaluate_baselines(blocks, "test", L, H, clim=clim,
                                    g_mean=gmean, stride=st)
        base = {k: {kk: float(vv) for kk, vv in v.items() if np.isscalar(vv)}
                for k, v in base.items()}
        json.dump({"kind": "baseline", "target": target, "results": base},
                  open(out / f"base_{target}.json", "w"), indent=2,
                  default=_json_default)
        log.info(f"  baselines: " +
                 " ".join(f"{k}={v['rmse']:.3f}" for k, v in base.items()))

        # Datasets construits UNE FOIS par cible (indépendants du modèle/graine)
        import torch as _torch
        dev = "cuda" if _torch.cuda.is_available() else "cpu"
        tr = D.make_torch_dataset(blocks, "train", L, H, sx, sy,
                                  target_idx=None, stride=st)
        va = D.make_torch_dataset(blocks, "val", L, H, sx, sy,
                                  target_idx=None, stride=st)
        log.info(f"  device={dev} | fenêtres train={len(tr):,} val={len(va):,} "
                 f"(stride={st})")
        if dev == "cpu" and len(tr) > 500_000:
            log.warning(f"  ⚠ CPU + {len(tr):,} fenêtres d'entraînement → très "
                        f"lent. Augmentez --stride, réduisez --max-nodes-per-route "
                        f"/ --epochs, ou utilisez un GPU.")

        for model_name in args.models:
            for seed in args.seeds:
                tag = f"{target}__{model_name}__seed{seed}"
                done = out / f"run_{tag}.json"
                if done.exists() and not args.force:
                    log.info(f"  [SKIP] {tag} (déjà fait)")
                    continue
                t0 = time.time()
                try:
                    set_seed(seed)
                    nq = len(Q) if Q else 1
                    model = build_model(model_name, n_features=len(feats),
                                        horizon=H, n_quantiles=nq)
                    model, best = train_model(model, tr, va, quantiles=Q,
                                              epochs=args.epochs,
                                              batch_size=args.batch,
                                              patience=args.patience,
                                              verbose=False)
                    res = predict_and_score(model, blocks, feats, sx, sy, L, H,
                                            quantiles=Q, stride=st)
                    # Diebold-Mariano vs persistance
                    em, ep = paired_errors(model, blocks, feats, sx, sy, L, H,
                                           st, Q)
                    dm, p = (Mx.diebold_mariano(em, ep, h=H)
                             if em.size else (float("nan"), float("nan")))
                    ref = base.get("persistence", {}).get("rmse")
                    skill = (Mx.skill_score(res["overall"]["rmse"] ** 2, ref ** 2)
                             if ref else None)
                    rec = {
                        "kind": "run", "target": target, "model": model_name,
                        "seed": seed, "best_val": best,
                        "overall": res["overall"],
                        "per_route": res["per_route"],
                        "per_enso_phase": res["per_enso_phase"],
                        "per_horizon_rmse": res["per_horizon_rmse"],
                        "crps": res.get("crps"), "pinball": res.get("pinball"),
                        "coverage_80": res.get("coverage_80"),
                        "skill_vs_persistence": skill,
                        "dm_vs_persistence": dm, "dm_pvalue": p,
                        "seconds": round(time.time() - t0, 1),
                    }
                    # ne pas sérialiser l'échantillon d'erreurs brut
                    json.dump(rec, open(done, "w"), indent=2, default=_json_default)
                    log.info(f"  [OK] {tag} | RMSE={res['overall']['rmse']:.3f} "
                             f"skill={skill:+.3f} DM={dm:+.2f} p={p:.3g} "
                             f"({rec['seconds']}s)")
                except Exception as exc:
                    log.error(f"  [FAIL] {tag} : {exc}")

    log.info("=" * 72)
    log.info(f"Terminé. JSON dans {out.resolve()} → agréger avec "
             f"aggregate_results.py --results {out}")


# ─────────────────── CONFIGURATION (éditable dans le notebook) ──────────────
# Modifiez ces valeurs puis exécutez «  %run run_experiments.py  »,
# OU appelez run_config(**réglages) sans rien éditer.
CONFIG = dict(
    data_dir="ERA5_ML_dataset",
    manifest="ERA5_ML_dataset/dataset_manifest.json",
    targets=["wspd", "swh", "mwp"],
    models=["tcn", "bilstm", "cnnbilstmattention", "tft"],
    seeds=[0, 1, 2],
    lookback=48,
    horizon=24,
    stride=12,                 # 12 = 1 fenêtre/12 h (stride=1 est trop lourd)
    quantiles=[0.1, 0.5, 0.9],
    epochs=60,
    batch=256,
    patience=8,
    out_dir="results",
    max_nodes_per_route=0,     # 0 = tous ; mettre 5 pour un essai rapide CPU
    force=False,
)


def run_config(**overrides):
    """Lance la matrice avec CONFIG, surchargé par les **overrides.
    Exemple notebook :
        run_config(targets=['swh'], models=['tcn'], seeds=[0],
                   max_nodes_per_route=5, stride=24, epochs=5)
    """
    from types import SimpleNamespace
    cfg = dict(CONFIG)
    cfg.update(overrides)
    return run(SimpleNamespace(**cfg))


def _running_in_notebook():
    """Vrai si on tourne dans un noyau IPython/Jupyter (ou Spyder)."""
    try:
        from IPython import get_ipython
        sh = get_ipython()
        return sh is not None and sh.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def main(argv=None):
    """Sûr en notebook ET en terminal.

    • En notebook (Jupyter/Spyder) : utilise CONFIG, SANS parse_args() — aucune
      erreur due à l'argument « -f …kernel.json » injecté par le noyau.
    • En terminal : les arguments de la ligne de commande surchargent CONFIG
      (les valeurs .json comme « --manifest … .json » sont bien préservées).
    """
    from types import SimpleNamespace
    cfg = dict(CONFIG)
    if _running_in_notebook():
        run(SimpleNamespace(**cfg))
        return
    argv = (sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        description="Phase 4 — matrice expérimentale.")
    ap.add_argument("--data-dir", default=cfg["data_dir"])
    ap.add_argument("--manifest", default=cfg["manifest"])
    ap.add_argument("--targets", nargs="+", default=cfg["targets"])
    ap.add_argument("--models", nargs="+", default=cfg["models"])
    ap.add_argument("--seeds", nargs="+", type=int, default=cfg["seeds"])
    ap.add_argument("--lookback", type=int, default=cfg["lookback"])
    ap.add_argument("--horizon", type=int, default=cfg["horizon"])
    ap.add_argument("--stride", type=int, default=cfg["stride"])
    ap.add_argument("--max-nodes-per-route", type=int,
                    default=cfg["max_nodes_per_route"])
    ap.add_argument("--quantiles", nargs="+", type=float,
                    default=cfg["quantiles"])
    ap.add_argument("--epochs", type=int, default=cfg["epochs"])
    ap.add_argument("--batch", type=int, default=cfg["batch"])
    ap.add_argument("--patience", type=int, default=cfg["patience"])
    ap.add_argument("--out-dir", default=cfg["out_dir"])
    ap.add_argument("--force", action="store_true", default=cfg["force"])
    args, _unknown = ap.parse_known_args(argv)
    run(args)


if __name__ == "__main__":
    main()
