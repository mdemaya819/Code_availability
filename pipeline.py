
import os, sys, json, argparse
def _locate_mf_root():
    """Trouve le dossier contenant 'mf/' (script, cwd, ou parents)."""
    cands = []
    try:
        cands.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    cands.append(os.getcwd())
    for base in list(cands):
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
        "Exécutez ce script DANS le dossier 'metocean_forecast' (qui "
        "contient 'mf/' et make_synthetic.py), ou ajoutez au début :\n"
        "    import os ; os.chdir(r'C:\\\\chemin\\\\vers\\\\metocean_forecast')\n"
        f"Dossier courant : {os.getcwd()}")
sys.path.insert(0, _ROOT)
try:
    import torch 
except ModuleNotFoundError:
    raise SystemExit(
        "\nPyTorch n'est pas installé dans cet environnement.\n"
        "  • CPU :  pip install torch\n"
        "  • GPU NVIDIA : https://pytorch.org/get-started/locally/  "
        "(ex. pip install torch --index-url "
        "https://download.pytorch.org/whl/cu124)\n")
import numpy as np
from mf import data as D, baselines as B, metrics as M



def _resolve_dataset(data_dir, manifest):
    """Localise dataset_manifest.json + parquets même si les chemins fournis
    sont erronés : cherche dans le chemin donné, le dossier courant, _ROOT et
    jusqu'à 3 niveaux de parents. Renvoie (data_dir, manifest_path) ou (None, None)."""
    if os.path.isfile(manifest):
        return (os.path.dirname(manifest) or "."), manifest
    cand = os.path.join(data_dir, "dataset_manifest.json")
    if os.path.isfile(cand):
        return data_dir, cand
    roots = [os.getcwd(), _ROOT]
    for r in list(roots):
        q = r
        for _ in range(3):
            q = os.path.dirname(q); roots.append(q)
    subdirs = [data_dir, "ERA5_ML_dataset", "ERA5_ML_dataset_v2", "dataset",
               "ML_dataset", "."]
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


def run(a):
    data_dir, manifest_path = _resolve_dataset(a.data_dir, a.manifest)
    if manifest_path is None:
        raise SystemExit(
            "\nJeu de donnees introuvable : ni le manifeste "
            f"\u00ab {a.manifest} \u00bb ni un dossier ERA5_ML_dataset/ valide.\n"
            "Renseignez data_dir/manifest dans CONFIG, ou placez le dossier "
            "ERA5_ML_dataset a cote de ce script.\n"
            f"Dossier courant : {os.getcwd()}")
    if os.path.abspath(manifest_path) != os.path.abspath(a.manifest):
        print(f"[data] jeu de donnees localise automatiquement : {data_dir}")
    man = D.load_manifest(manifest_path)
    feats, dropped = D.resolve_features(man, a.target)
    print(f"[data] target={a.target} | features={len(feats)} | dropped={dropped}")
    paths = D.find_parquets(data_dir)
    blocks, feats = D.load_blocks(paths, man, target=a.target,
                                  feature_cols=feats, routes=a.routes)
    print(f"[data] {len(blocks)} blocs (route,node,split)")

    clim, gm = B.fit_climatology(blocks)
    bres = B.evaluate_baselines(blocks, "test", a.lookback, a.horizon,
                                clim=clim, g_mean=gm, stride=a.stride)
    ref = bres["persistence"]["rmse"] ** 2
    report = {"config": vars(a), "features": feats, "dropped": dropped,
              "baselines": {n: {"rmse": r["rmse"], "mae": r["mae"],
                                "skill_vs_persistence":
                                    M.skill_score(r["rmse"] ** 2, ref)}
                            for n, r in bres.items()}}
    print("\n  Baseline        RMSE     MAE")
    for n, r in bres.items():
        print(f"  {n:15s} {r['rmse']:7.3f} {r['mae']:7.3f}")

    if not a.baselines_only:
        import torch as _torch
        from mf.models import build_model
        from mf.train import train_model
        from mf.evaluate import predict_and_score, compare_table
        sx = D.Scaler(man["normalization"]["stats"], feats)
        sy = D.TargetScaler(man["normalization"]["stats"], a.target)
        q = list(a.quantiles) if a.quantiles else None
        tr = D.make_torch_dataset(blocks, "train", a.lookback, a.horizon,
                                  sx, sy, None, a.stride)
        va = D.make_torch_dataset(blocks, "val", a.lookback, a.horizon,
                                  sx, sy, None, a.stride)
        dev = "cuda" if _torch.cuda.is_available() else "cpu"
        print(f"[train] device={dev} | fenetres train={len(tr):,} "
              f"val={len(va):,} (stride={a.stride})")
        if dev == "cpu" and len(tr) > 500_000:
            print(f"[train] CPU + {len(tr):,} fenetres -> tres lent ; augmentez "
                  f"stride, reduisez epochs, ou utilisez un GPU.")
        model = build_model(a.model, n_features=len(feats), horizon=a.horizon,
                            n_quantiles=len(q) if q else 1)
        model, best = train_model(model, tr, va, quantiles=q, epochs=a.epochs,
                                  batch_size=a.batch_size, lr=a.lr, seed=a.seed)
        res = predict_and_score(model, blocks, feats, sx, sy, a.lookback,
                                a.horizon, quantiles=q, stride=a.stride)
        res.pop("_err_sample", None)
        report["model"] = {"name": a.model, "best_val": best, "scores": res}
        print("\n", compare_table(
            {"overall": res["overall"]}, bres).to_string(index=False))

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n[OK] rapport -> {os.path.abspath(a.out)}")


# \u2500\u2500\u2500 CONFIGURATION \u2500\u2500\u2500
CONFIG = dict(
    data_dir="ERA5_ML_dataset",
    manifest="ERA5_ML_dataset/dataset_manifest.json",
    target="wspd",
    routes=None,
    model="tcn",                # tcn | bilstm | cnnbilstmattention | tft
    lookback=48,
    horizon=24,
    stride=12,                  
    quantiles=[0.1, 0.5, 0.9],
    epochs=60,
    batch_size=512,
    lr=1e-3,
    seed=0,
    baselines_only=False,
    out="results.json",
)


def run_config(**overrides):
    """Lance le pipeline avec CONFIG surcharge. Exemple notebook :
        import run_pipeline as P
        P.run_config(target='swh', model='tcn', epochs=10)
    """
    from types import SimpleNamespace
    cfg = dict(CONFIG); cfg.update(overrides)
    return run(SimpleNamespace(**cfg))


def _running_in_notebook():
    try:
        from IPython import get_ipython
        sh = get_ipython()
        return sh is not None and sh.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def main(argv=None):
    """Sûr en notebook ET en terminal.

    • En notebook (Jupyter/Spyder) : utilise CONFIG, SANS parse_args() — pas
      d'erreur sur l'argument « -f …kernel.json » du noyau.
    • En terminal : les arguments surchargent CONFIG ; une valeur .json comme
      « --out resultats.json » est correctement conservée.
    """
    from types import SimpleNamespace
    cfg = dict(CONFIG)
    if _running_in_notebook():
        run(SimpleNamespace(**cfg))
        return
    argv = (sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        description="Pipeline - un modele, donnees reelles.")
    ap.add_argument("--data-dir", default=cfg["data_dir"])
    ap.add_argument("--manifest", default=cfg["manifest"])
    ap.add_argument("--target", default=cfg["target"])
    ap.add_argument("--routes", nargs="*", default=cfg["routes"])
    ap.add_argument("--model", default=cfg["model"],
                    choices=["tcn", "bilstm", "cnnbilstmattention", "tft"])
    ap.add_argument("--lookback", type=int, default=cfg["lookback"])
    ap.add_argument("--horizon", type=int, default=cfg["horizon"])
    ap.add_argument("--stride", type=int, default=cfg["stride"])
    ap.add_argument("--quantiles", type=float, nargs="*",
                    default=cfg["quantiles"])
    ap.add_argument("--epochs", type=int, default=cfg["epochs"])
    ap.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    ap.add_argument("--lr", type=float, default=cfg["lr"])
    ap.add_argument("--seed", type=int, default=cfg["seed"])
    ap.add_argument("--baselines-only", action="store_true",
                    default=cfg["baselines_only"])
    ap.add_argument("--out", default=cfg["out"])
    args, _unknown = ap.parse_known_args(argv)
    run(args)


if __name__ == "__main__":
    main()
