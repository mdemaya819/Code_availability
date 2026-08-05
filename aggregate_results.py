"""
aggregate_results.py — Agrégation Phase 4.

Lit les JSON produits par run_experiments.py et produit des tables prêtes pour
publication :
  - results_overall.csv     : par (cible, modèle), moyenne±écart-type sur graines
                              de RMSE/MAE/CRPS/couverture/skill/Diebold-Mariano ;
  - results_per_route.csv   : RMSE par route ;
  - results_per_enso.csv    : RMSE par phase ENSO (El_Nino / La_Nina / Neutral) ;
  - baselines.csv           : RMSE/MAE des baselines par cible ;
  - report.md               : rapport lisible (tables Markdown par cible).

Usage :  python aggregate_results.py --results results --out report
"""
import os
import json
import glob
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


def _load(results_dir):
    runs, bases = [], {}
    for fp in glob.glob(str(Path(results_dir) / "*.json")):
        d = json.load(open(fp, encoding="utf-8"))
        if d.get("kind") == "run":
            runs.append(d)
        elif d.get("kind") == "baseline":
            bases[d["target"]] = d["results"]
    return runs, bases


def _ms(values):
    v = [x for x in values if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return (float("nan"), float("nan"), 0)
    return (float(np.mean(v)), float(np.std(v)), len(v))


def aggregate(results_dir, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    runs, bases = _load(results_dir)
    if not runs:
        raise SystemExit(f"Aucun run_*.json sous {results_dir}.")

    # ── overall par (cible, modèle) ───────────────────────────────────────────
    by = defaultdict(list)
    for r in runs:
        by[(r["target"], r["model"])].append(r)
    rows = []
    for (tgt, mdl), rs in sorted(by.items()):
        rmse_m, rmse_s, n = _ms([x["overall"]["rmse"] for x in rs])
        mae_m, _, _ = _ms([x["overall"]["mae"] for x in rs])
        sk_m, sk_s, _ = _ms([x.get("skill_vs_persistence") for x in rs])
        crps_m, _, _ = _ms([x.get("crps") for x in rs])
        cov_m, _, _ = _ms([x.get("coverage_80") for x in rs])
        dm_m, _, _ = _ms([x.get("dm_vs_persistence") for x in rs])
        pvals = [x.get("dm_pvalue") for x in rs
                 if x.get("dm_pvalue") is not None and not np.isnan(x.get("dm_pvalue"))]
        p_med = float(np.median(pvals)) if pvals else float("nan")
        frac_sig = float(np.mean([p < 0.05 for p in pvals])) if pvals else float("nan")
        rows.append({"target": tgt, "model": mdl, "n_seeds": n,
                     "rmse_mean": rmse_m, "rmse_std": rmse_s, "mae_mean": mae_m,
                     "skill_mean": sk_m, "skill_std": sk_s,
                     "crps_mean": crps_m, "coverage80_mean": cov_m,
                     "dm_mean": dm_m, "dm_p_median": p_med,
                     "dm_frac_sig_0.05": frac_sig})
    overall = pd.DataFrame(rows)
    overall.to_csv(out / "results_overall.csv", index=False)

    # ── par route ─────────────────────────────────────────────────────────────
    rr = defaultdict(lambda: defaultdict(list))
    for r in runs:
        for route, d in r.get("per_route", {}).items():
            rr[(r["target"], r["model"])][route].append(d["rmse"])
    rows = []
    for (tgt, mdl), routes in sorted(rr.items()):
        for route, vals in sorted(routes.items()):
            m, s, n = _ms(vals)
            rows.append({"target": tgt, "model": mdl, "route": route,
                         "rmse_mean": m, "rmse_std": s, "n_seeds": n})
    per_route = pd.DataFrame(rows)
    per_route.to_csv(out / "results_per_route.csv", index=False)

    # ── par phase ENSO ────────────────────────────────────────────────────────
    re = defaultdict(lambda: defaultdict(list))
    for r in runs:
        for ph, d in r.get("per_enso_phase", {}).items():
            re[(r["target"], r["model"])][ph].append(d["rmse"])
    rows = []
    for (tgt, mdl), phases in sorted(re.items()):
        for ph, vals in sorted(phases.items()):
            m, s, n = _ms(vals)
            rows.append({"target": tgt, "model": mdl, "enso_phase": ph,
                         "rmse_mean": m, "rmse_std": s, "n_seeds": n})
    per_enso = pd.DataFrame(rows)
    per_enso.to_csv(out / "results_per_enso.csv", index=False)

    # ── baselines ─────────────────────────────────────────────────────────────
    rows = []
    for tgt, res in sorted(bases.items()):
        for method, d in res.items():
            rows.append({"target": tgt, "method": method,
                         "rmse": d["rmse"], "mae": d["mae"]})
    baseline_df = pd.DataFrame(rows)
    baseline_df.to_csv(out / "baselines.csv", index=False)

    _write_report(out, overall, per_route, per_enso, baseline_df)
    print(f"[OK] Tables + rapport écrits dans {out.resolve()}")
    print(overall.to_string(index=False))
    return overall, per_route, per_enso, baseline_df


def _fmt(m, s):
    if m != m:  # NaN
        return "—"
    return f"{m:.3f} ± {s:.3f}" if (s == s and s > 0) else f"{m:.3f}"


def _write_report(out, overall, per_route, per_enso, baseline_df):
    lines = ["# Résultats — prévision multi-horizon des conditions météo-océaniques",
             "",
             "Moyenne ± écart-type sur les graines. Skill = 1 − MSE/MSE(persistance) ; "
             "DM = statistique de Diebold-Mariano vs persistance (DM<0 ⇒ modèle "
             "meilleur), p = p-value médiane.", ""]
    for tgt in sorted(overall["target"].unique()):
        lines += [f"## Cible : {tgt}", ""]
        sub = overall[overall.target == tgt].sort_values("rmse_mean")
        lines += ["| Modèle | RMSE | Skill vs persist. | CRPS | Couv. 80% | DM | p (méd.) |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['model']} | {_fmt(r['rmse_mean'], r['rmse_std'])} | "
                f"{_fmt(r['skill_mean'], r['skill_std'])} | "
                f"{'—' if r['crps_mean']!=r['crps_mean'] else f'{r['crps_mean']:.3f}'} | "
                f"{'—' if r['coverage80_mean']!=r['coverage80_mean'] else f'{r['coverage80_mean']:.3f}'} | "
                f"{'—' if r['dm_mean']!=r['dm_mean'] else f'{r['dm_mean']:+.2f}'} | "
                f"{'—' if r['dm_p_median']!=r['dm_p_median'] else f'{r['dm_p_median']:.3g}'} |")
        # baselines de la cible
        b = baseline_df[baseline_df.target == tgt]
        if not b.empty:
            lines += ["", "Baselines (RMSE) : " +
                      ", ".join(f"{x['method']} {x['rmse']:.3f}"
                                for _, x in b.iterrows()), ""]
    # Par route (RMSE du meilleur modèle par cible)
    lines += ["## RMSE par route (tous modèles)", ""]
    if not per_route.empty:
        piv = per_route.pivot_table(index=["target", "route"], columns="model",
                                    values="rmse_mean")
        lines += [piv.round(3).to_markdown(), ""]
    lines += ["## RMSE par phase ENSO (tous modèles)", ""]
    if not per_enso.empty:
        piv = per_enso.pivot_table(index=["target", "enso_phase"],
                                   columns="model", values="rmse_mean")
        lines += [piv.round(3).to_markdown(), ""]
    open(out / "report.md", "w", encoding="utf-8").write("\n".join(lines))


def _running_in_notebook():
    try:
        from IPython import get_ipython
        sh = get_ipython()
        return sh is not None and sh.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


# CONFIG éditable (utilisé en notebook ; voir aussi run_config ci-dessous)
CONFIG = dict(results="results", out="report")


def run_config(**overrides):
    """Agrège depuis un notebook : aggregate_results.run_config(results='results')."""
    cfg = dict(CONFIG); cfg.update(overrides)
    aggregate(cfg["results"], cfg["out"])


def main(argv=None):
    """Sûr en notebook (utilise CONFIG, pas de parse_args) ET en terminal."""
    cfg = dict(CONFIG)
    if _running_in_notebook():
        aggregate(cfg["results"], cfg["out"]); return
    import sys
    argv = (sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description="Phase 4 — agrégation des résultats.")
    ap.add_argument("--results", default=cfg["results"])
    ap.add_argument("--out", default=cfg["out"])
    a, _unknown = ap.parse_known_args(argv)
    aggregate(a.results, a.out)


if __name__ == "__main__":
    main()
