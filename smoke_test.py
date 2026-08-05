
import os, sys
def _locate_mf_root():
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
    import torch  # noqa: F401  (requis par les modèles)
except ModuleNotFoundError:
    sys.exit(
        "\nPyTorch n'est pas installé dans cet environnement Python.\n"
        "Installez-le puis relancez :\n"
        "  • CPU (suffit pour ce test) :  pip install torch\n"
        "  • GPU NVIDIA (entraînement complet) : voir le sélecteur officiel\n"
        "    https://pytorch.org/get-started/locally/  (ex. :\n"
        "    pip install torch --index-url https://download.pytorch.org/whl/cu124)\n"
        "  • Conda (CPU) :  conda install pytorch cpuonly -c pytorch\n")
import numpy as np
from mf import data as D, baselines as B, metrics as M
from mf.models import build_model
from mf.train import train_model
from mf.evaluate import predict_and_score, compare_table
import make_synthetic


def main(manifest="dataset_manifest.json", lookback=48, horizon=12,
         stride=6, quantiles=(0.1, 0.5, 0.9), epochs=2):
    
    make_synthetic.build(manifest, "synthetic_nodes.parquet", nodes_per_route=2)
    man = D.load_manifest(manifest)
    target = "wspd"
    feats, dropped = D.resolve_features(man, target)
    print(f"features={feats}\ndropped={dropped}")
    blocks, feats = D.load_blocks(["synthetic_nodes.parquet"], man,
                                  target=target, feature_cols=feats)
    sx = D.Scaler(man["normalization"]["stats"], feats)
    sy = D.TargetScaler(man["normalization"]["stats"], target)
    quantiles = list(quantiles)

   
    clim, gm = B.fit_climatology(blocks)
    bres = B.evaluate_baselines(blocks, "test", lookback, horizon,
                                clim=clim, g_mean=gm, stride=stride)

    for name in ["tcn", "bilstm", "cnnbilstmattention", "tft"]:
        print(f"\n=== {name} ===")
        tr = D.make_torch_dataset(blocks, "train", lookback, horizon, sx, sy,
                                  target_idx=None, stride=stride)
        va = D.make_torch_dataset(blocks, "val", lookback, horizon, sx, sy,
                                  target_idx=None, stride=stride)
        model = build_model(name, n_features=len(feats), horizon=horizon,
                            n_quantiles=len(quantiles))
        model, best = train_model(model, tr, va, quantiles=quantiles,
                                  epochs=epochs, batch_size=256, patience=3,
                                  verbose=True)
        res = predict_and_score(model, blocks, feats, sx, sy, lookback,
                                horizon, quantiles=quantiles, stride=stride)
        print(compare_table(res, bres).to_string(index=False))
        print("per-route RMSE:", {r: round(v["rmse"], 3)
                                  for r, v in res["per_route"].items()})
        if "crps" in res:
            print(f"CRPS={res['crps']:.3f}  coverage80={res.get('coverage_80')}")
    print("\n==== SMOKE TEST TORCH : OK ====")


if __name__ == "__main__":
    main()
