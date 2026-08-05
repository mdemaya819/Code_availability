# -*- coding: utf-8 -*-
"""
run_phase5.py — PHASE 5 : ROUTAGE SOUS INCERTITUDE, sur DONNÉES RÉELLES (ERA5).

Ce script exécute la Phase 5 de bout en bout et produit tous ses résultats
(optimisation vitesse/départ + valeur décisionnelle), directement à partir du
jeu de données ERA5 réel — exactement comme run_experiments.py / run_smoke.py :
il s'auto-localise, trouve le manifeste et les parquets, et tourne aussi bien
en terminal qu'en cellule Jupyter/Spyder (aucun crash d'argument).

Ce qu'il fait
-------------
Pour une route, sur la période de TEST :
  1. Géométrie exacte des tronçons depuis le manifeste (lat/lon/dist_km).
  2. Champ espace-temps de quantiles PRÉVUS q10/q50/q90 de wspd et swh,
     construit à partir des vraies séries ERA5 :
       • médiane = mélange skill-pondéré  obs ↔ climatologie mensuelle
         (skillful à court terme, climatologique au-delà) ;
       • dispersion croissant avec l'échéance, saturant à la variabilité
         climatologique (comportement mesuré en Phase 4).
     (Pour brancher de VRAIES prévisions quantiles d'un modèle entraîné, voir la
      NOTE dans mf/routing.build_scenario.)
  3. Optimisation (vitesse de service, délai de départ) minimisant le carburant
     ESPÉRÉ sous contrainte de sécurité P(swh>Hs_max) ≤ alpha et deadline —
     planificateur PROBABILISTE (quantiles) vs DÉTERMINISTE (médiane seule).
  4. BACKTEST : les deux plans sont RÉÉVALUÉS sur la VÉRITÉ ERA5, sur de
     nombreuses dates de départ → carburant réel et TAUX RÉEL de dépassement de
     sécurité. La différence = « valeur de l'incertitude ».

Résultats produits (dans out_dir, défaut « phase5/ ») :
  • phase5_<route>.json      — agrégats + détail par date de départ ;
  • Fig_phase5_<route>.png   — départ représentatif (houle rencontrée + bande
                               q10–q90 + seuil) et barres agrégées (taux de
                               violation réel et carburant réel, prob vs det).

Lancement
---------
  # terminal
  python run_phase5.py --route R1_SHA_RTM --hs-max 6.0 --alpha 0.10
  # Jupyter / Spyder : éditez CONFIG puis exécutez, ou
  import run_phase5 as P5
  P5.run_config(route="R2_NFK_HAM", n_departures=16)

Si aucun jeu de données n'est trouvé, le script bascule sur un scénario de
DÉMONSTRATION synthétique (aucune donnée requise), en le signalant clairement.
"""
from __future__ import annotations
import os
import sys
import json
import glob
import argparse
from pathlib import Path


# ── Localisation du paquet mf/ (script, cwd, parents) ─────────────────────────
def _locate_mf_root():
    cands = []
    try:
        cands.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:                       # cellule Jupyter : pas de __file__
        pass
    cands.append(os.getcwd())
    for base in list(cands):
        p = base
        for _ in range(3):
            p = os.path.dirname(p)
            cands.append(p)
    for c in cands:
        if os.path.isfile(os.path.join(c, "mf", "routing.py")):
            return c
    return None


_ROOT = _locate_mf_root()
if _ROOT is None:
    raise SystemExit(
        "\nLe paquet 'mf' est introuvable depuis cet emplacement.\n"
        "Exécutez ce run_phase5.py DANS le dossier 'metocean_forecast' (celui "
        "qui contient 'mf/'), ou ajoutez au début :\n"
        "        import os ; os.chdir(r'C:\\\\chemin\\\\vers\\\\metocean_forecast')\n"
        f"Dossier courant : {os.getcwd()}")
sys.path.insert(0, _ROOT)

import numpy as np
import importlib
from mf import routing as RT
from mf import data as D

# Jupyter / Spyder : si le noyau a déjà chargé d'ANCIENNES versions de ces modules
# lors d'une session précédente, « from mf import routing » renvoie la version en
# cache (sans les fonctions Phase 5 récentes) → AttributeError. On force le
# rechargement depuis le disque pour éviter cela, sans avoir à redémarrer le noyau.
importlib.reload(D)
importlib.reload(RT)
if not hasattr(RT, "build_route_geometry"):
    raise SystemExit(
        "\nLe fichier mf/routing.py chargé est une version ANCIENNE : il ne "
        "contient pas les fonctions de la Phase 5 sur données réelles "
        "(build_route_geometry, rolling_backtest, …).\n"
        "Causes possibles et solutions :\n"
        "  • Vous n'avez mis à jour que run_phase5.py : ré-extrayez "
        "l'INTÉGRALITÉ de metocean_forecast.zip en écrasant le dossier mf/.\n"
        "  • Le noyau a gardé une version en cache : Noyau → Redémarrer "
        "(Kernel → Restart), puis ré-exécutez.\n"
        f"  • Vérifiez le fichier réellement chargé : {getattr(RT, '__file__', '?')}")


# ── Résolveurs dataset / route_config (robustes, comme run_experiments) ───────
def _resolve_dataset(data_dir, manifest):
    """Localise (data_dir, manifest_path) ou (None, None)."""
    if manifest and os.path.isfile(manifest):
        return (os.path.dirname(manifest) or "."), manifest
    if data_dir:
        cand = os.path.join(data_dir, "dataset_manifest.json")
        if os.path.isfile(cand):
            return data_dir, cand
    roots = [os.getcwd(), _ROOT]
    for r in list(roots):
        p = r
        for _ in range(3):
            p = os.path.dirname(p)
            roots.append(p)
    subdirs = [data_dir or ".", "ERA5_ML_dataset", "ERA5_ML_dataset_v2",
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


def _find_route_config():
    """Rend route_config importable (cherché dans _ROOT/cwd/parents)."""
    for c in [_ROOT, os.getcwd(), os.path.dirname(_ROOT)]:
        if c and os.path.isfile(os.path.join(c, "route_config.py")) \
                and c not in sys.path:
            sys.path.insert(0, c)
    try:
        import importlib
        return importlib.import_module("route_config")
    except Exception:
        return None


def _ship_for_route(route):
    rc = _find_route_config()
    if rc is not None and route in getattr(rc, "ROUTES", {}):
        return RT.ship_from_route_meta(rc.ROUTES[route]), "route_config"
    return RT.Ship(), "défauts Ship()"


def _load_route_df(data_dir, route):
    """Charge le DataFrame d'UNE route (colonnes utiles) depuis le parquet
    par-route si présent, sinon depuis le parquet global."""
    need = ["route", "node_id", "datetime", "swh", "wspd", "u10", "v10"]
    per_route = os.path.join(data_dir, f"{route}_nodes.parquet")
    combined = os.path.join(data_dir, "routes_nodes.parquet")
    path = per_route if os.path.isfile(per_route) else combined
    if not os.path.isfile(path):
        pq = sorted(glob.glob(os.path.join(data_dir, "*_nodes.parquet")))
        if not pq:
            raise FileNotFoundError(f"Aucun parquet *_nodes.parquet sous {data_dir}")
        path = next((p for p in pq if os.path.basename(p).startswith(route)), pq[0])
    df = D._read_parquet_robust(path, need)
    if "route" in df.columns:
        df = df[df["route"] == route].copy()
    if len(df) == 0:
        raise ValueError(f"Aucune ligne pour la route {route} dans {path}")
    return df


# ── Configuration (éditable en notebook) ──────────────────────────────────────
CONFIG = dict(
    route="R1_SHA_RTM",
    mode="auto",               # auto | real | demo
    data_dir=None,             # None = auto-localisation
    manifest=None,             # None = auto-localisation
    results_dir=None,          # dossier des JSON de Phase 4 (calibration) ; None = auto
    sweep_hs=None,             # liste de seuils Hs_max → analyse de sensibilité (ex. [3,4,5,6])
    # sécurité / décision
    Hs_max=6.0,                # seuil hauteur de vagues [m]
    alpha=0.10,                # risque toléré P(swh>Hs_max) ≤ alpha
    eta_slack=1.15,            # deadline = eta_slack × durée à V_nom
    dep_max_h=48.0,            # délai de départ max exploré [h]
    v_lo_kn=None, v_hi_kn=None,  # None = auto (slow-steam .. surcharge moteur)
    # géométrie / backtest
    n_legs_max=0,              # 0 = tous les nœuds natifs (~150 km, échéance tactique fine)
    n_departures=40,           # nb de dates de départ backtestées (robustesse stat.)
    n_bootstrap=2000,          # rééchantillonnages bootstrap pour les IC 95 %
    real_gap_h=12.0,           # un tronçon ne compte que si une obs réelle est ≤ ce délai
    horizon_h=36.0,            # horizon d'anticipation (≈ échéance ML exploitable)
    # grille d'optimisation / Monte-Carlo
    n_v=11, n_dep=7, n_mc=500,
    # modèle de prévision (propriétés mesurées en Phase 4)
    tau_skill_h=36.0,          # décroissance du skill (mélange obs↔climato)
    tau_spread_h=60.0,         # croissance de la dispersion
    rel_err_short={"swh": 0.17, "wspd": 0.22},  # erreur rel. à court terme
    # validité des départs (trous de données)
    min_cov_frac=0.6, max_gap_h=168.0,
    seed=0,
    out_dir="phase5",
)


def _resolve_results(results_dir):
    """Localise un dossier contenant les JSON de Phase 4 (run_*.json)."""
    def _has(d): return d and os.path.isdir(d) and bool(
        glob.glob(os.path.join(d, "run_*.json")))
    if _has(results_dir):
        return results_dir
    roots = [os.getcwd(), _ROOT]
    for r in list(roots):
        p = r
        for _ in range(3):
            p = os.path.dirname(p); roots.append(p)
    for r in roots:
        for n in [results_dir or "", "results", "report", "phase4",
                  "ERA5_ML_dataset", "."]:
            cand = os.path.join(r, n) if n else r
            if _has(cand):
                return cand
    return None


# ── Cœur : Phase 5 sur données réelles ────────────────────────────────────────
def run_real(cfg, data_dir, manifest_path, manifest=None, skill="__load__"):
    if manifest is None:
        manifest = D.load_manifest(manifest_path)
    route = cfg["route"]
    if route not in manifest["nodes_per_route"]:
        raise SystemExit(f"Route « {route} » absente du manifeste. "
                         f"Disponibles : {list(manifest['nodes_per_route'])}")
    # calibration sur Phase 4 (RMSE par horizon × par route du meilleur modèle)
    if skill == "__load__":
        rdir = _resolve_results(cfg.get("results_dir"))
        skill = RT.load_phase4_skill(rdir) if rdir else None
        if skill:
            print("[phase5] calibration Phase 4 : " + ", ".join(
                f"{v} via {skill[v]['model'].upper()} "
                f"(RMSE {skill[v]['rmse_h'][0]:.2f}→{skill[v]['rmse_h'][-1]:.2f}, "
                f"couv {skill[v]['coverage']:.2f})" for v in skill))
        else:
            print("[phase5] AVERTISSEMENT : aucun dossier results/ trouvé → "
                  "incertitude paramétrique de repli (fournir --results pour "
                  "calibrer sur vos modèles).")
    cfg = {**cfg, "skill": skill}
    ship, ship_src = _ship_for_route(route)
    geom = RT.build_route_geometry(manifest["nodes_per_route"][route],
                                   n_legs_max=cfg["n_legs_max"])
    df = _load_route_df(data_dir, route)
    prep = RT.prepare_route_series(df, geom, manifest["temporal_split"])

    print(f"[phase5] route={route} | {len(geom['node_ids'])} tronçons | "
          f"{geom['leg_km'].sum():,.0f} km | navire ({ship_src}) "
          f"P_nom={ship.P_nom_kW:.0f} kW V_nom={ship.V_nom_kn:.0f} kn")
    print(f"[phase5] backtest à horizon glissant sur ≤ {cfg['n_departures']} "
          f"départs (test {prep['test_years'][0]}–{prep['test_years'][1]})…")

    res = RT.rolling_backtest(prep, ship, cfg)
    agg = res["aggregate"]

    pci = agg["prob_unsafe_leg_rate_ci"]; dci = agg["det_unsafe_leg_rate_ci"]
    fci = agg["fuel_overhead_pct_ci"]
    print(f"\n  Seuil sécurité Hs_max={cfg['Hs_max']} m | risque toléré "
          f"alpha={cfg['alpha']:.0%} | départs backtestés : {agg['n_departures']} "
          f"| tronçons observés prob/det : {agg['n_valid_legs_prob']}/"
          f"{agg['n_valid_legs_det']}")
    print(f"  {'politique':14s} {'fuel réel(t)':>12s} "
          f"{'tronçons dgx (IC95%)':>26s} {'voyages dgx':>12s} {'retard(h)':>10s}")
    print(f"  {'probabiliste':14s} {agg['prob_fuel_mean']:12.0f}   "
          f"{agg['prob_unsafe_leg_rate']:5.1%} "
          f"[{pci[0]:.1%}–{pci[1]:.1%}] {agg['prob_voyage_unsafe_rate']:10.0%} "
          f"{agg['prob_mean_delay_h']:10.1f}")
    print(f"  {'déterministe':14s} {agg['det_fuel_mean']:12.0f}   "
          f"{agg['det_unsafe_leg_rate']:5.1%} "
          f"[{dci[0]:.1%}–{dci[1]:.1%}] {agg['det_voyage_unsafe_rate']:10.0%} "
          f"{agg['det_mean_delay_h']:10.1f}")
    rc = agg["reduction_abs_ci"]; sig = "SIGNIFICATIVE" if agg["reduction_significant"] else "non significative"
    print(f"\n  → Valeur de l'incertitude : {agg['det_unsafe_leg_rate']:.1%} → "
          f"{agg['prob_unsafe_leg_rate']:.1%} de tronçons dangereux ; réduction "
          f"appariée {agg['reduction_abs']:+.2%} [IC95% {rc[0]:+.2%}; {rc[1]:+.2%}] "
          f"→ {sig}. Surcoût carburant "
          f"{agg['fuel_overhead_pct']:+.1f} % [IC95% {fci[0]:+.1f}; {fci[1]:+.1f}].")

    ex = agg.get("exposure", {})
    ex_txt = " ".join(f"{k.replace('frac_over_', '>')}={v:.1%}" for k, v in ex.items())
    print(f"  Exposition (disculpe un 0 %) : houle max observée "
          f"{agg.get('max_swh_obs', 0):.1f} m | tronçons {ex_txt}")

    out = cfg["out_dir"]; os.makedirs(out, exist_ok=True)
    payload = {"route": route, "ship_source": ship_src,
               "Hs_max": cfg["Hs_max"], "alpha": cfg["alpha"],
               "aggregate": agg, "departures": res["rows"]}
    jpath = os.path.join(out, f"phase5_{route}.json")
    json.dump(payload, open(jpath, "w"), indent=2, default=float)
    _make_real_figure(res, ship, cfg, out)
    print(f"\n[OK] Résultats + figure → {os.path.abspath(out)}")
    return res


def _make_real_figure(res, ship, cfg, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0, prob, det = res["representative"]
    agg = res["aggregate"]
    cum = prob["cum_km"]

    fig = plt.figure(figsize=(11, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.36, wspace=0.28)

    # (a) real wave height encountered per leg (prob vs det) — OBSERVED legs
    ax0 = fig.add_subplot(gs[0, :])
    pmask = prob.get("real_mask", np.ones_like(cum, bool))
    dmask = det.get("real_mask", np.ones_like(cum, bool))
    ax0.plot(cum, prob["swh_enc"], "-o", ms=3, color="#1b6ca8",
             label="probabilistic policy")
    ax0.plot(cum, det["swh_enc"], "-s", ms=3, color="#e08a1e",
             label="deterministic policy")
    ax0.axhline(cfg["Hs_max"], color="#b23a48", ls="--", lw=1.3,
                label=f"safety threshold {cfg['Hs_max']} m")
    # mark REAL threshold exceedances of BOTH policies
    pbad = pmask & (prob["swh_enc"] > cfg["Hs_max"])
    dbad = dmask & (det["swh_enc"] > cfg["Hs_max"])
    ax0.scatter(cum[dbad], det["swh_enc"][dbad], s=70, facecolors="none",
                edgecolors="#e08a1e", linewidths=1.6, zorder=5,
                label="exceedance — deterministic")
    ax0.scatter(cum[pbad], prob["swh_enc"][pbad], s=40, marker="D",
                facecolors="none", edgecolors="#1b6ca8", linewidths=1.6, zorder=6,
                label="exceedance — probabilistic")
    ax0.set_title(f"(a) Representative departure {RT.prep_name(t0)} — real wave "
                  f"height encountered (receding horizon)", loc="left",
                  fontweight="bold", fontsize=10.5)
    ax0.set_xlabel("Distance along route (km)")
    ax0.set_ylabel("Significant wave height (m)")
    ax0.legend(fontsize=8, ncol=2); ax0.grid(alpha=0.3)

    # (b) speed profile (prob vs det)
    axb = fig.add_subplot(gs[1, 0])
    axb.plot(cum, prob["v_used"], "-", color="#1b6ca8", label="probabilistic")
    axb.plot(cum, det["v_used"], "-", color="#e08a1e", label="deterministic")
    axb.set_title("(b) Chosen speed profile", loc="left",
                  fontweight="bold", fontsize=10)
    axb.set_xlabel("Distance (km)"); axb.set_ylabel("Speed (kn)")
    axb.legend(fontsize=8); axb.grid(alpha=0.3)

    # (c) aggregates: dangerous legs (± 95% CI) + fuel overhead
    axc = fig.add_subplot(gs[1, 1])
    x = np.arange(2)
    r = [agg["prob_unsafe_leg_rate"], agg["det_unsafe_leg_rate"]]
    pci = agg["prob_unsafe_leg_rate_ci"]; dci = agg["det_unsafe_leg_rate_ci"]
    yerr = np.array([[r[0] - pci[0], r[1] - dci[0]],
                     [pci[1] - r[0], dci[1] - r[1]]])
    yerr = np.clip(yerr, 0, None)
    bars = axc.bar(x, r, color=["#1b6ca8", "#e08a1e"], edgecolor="white",
                   yerr=yerr, capsize=4, error_kw=dict(lw=1.0, ecolor="#333"))
    for b, v in zip(bars, r):
        axc.text(b.get_x() + b.get_width() / 2, v, f" {v:.1%}",
                 ha="center", va="bottom", fontsize=9)
    axc.set_xticks(x); axc.set_xticklabels(["probabilistic", "deterministic"])
    axc.set_ylabel("dangerous legs (swh>Hs)")
    axc.set_ylim(0, max(0.02, max(dci[1], pci[1]) * 1.25))
    axc.set_title(f"(c) Safety — {agg['n_departures']} departures · fuel "
                  f"{agg['fuel_overhead_pct']:+.1f}%", loc="left",
                  fontweight="bold", fontsize=9.5)
    axc.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Phase 5 — uncertainty-aware receding-horizon routing, "
                 f"real data: {cfg['route']}", fontweight="bold",
                 fontsize=12.5)
    fpath = os.path.join(out, f"Fig_phase5_{cfg['route']}.png")
    fig.savefig(fpath, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[phase5] figure → {fpath}")


# ── Repli DÉMO (aucune donnée requise) ────────────────────────────────────────
def run_demo(cfg):
    print("[phase5] Aucun jeu de données trouvé → scénario de DÉMONSTRATION "
          "synthétique (fournissez --data-dir pour l'exécution sur vos données).")
    ship, ship_src = _ship_for_route(cfg["route"])
    n_legs, leg_km = 14, 1393.0
    total_km = n_legs * leg_km
    v_nom = ship.V_nom_kn * RT.KN2MS
    v_lo = cfg["v_lo_kn"] or round(0.75 * ship.V_nom_kn, 1)
    v_hi = cfg["v_hi_kn"] or round(ship.P_ovl_frac ** (1 / 3) * ship.V_nom_kn, 1)
    dep_max = 120.0
    t_max = dep_max + 1.05 * total_km * 1000.0 / (v_lo * RT.KN2MS) / 3600.0
    n_times = max(60, int(t_max / 12) + 1)
    sf = n_legs // 2
    storm_center = (sf + 0.5) * leg_km * 1000.0 / (0.855 * v_nom) / 3600.0
    fc = RT.synthesize_spacetime_forecast(
        cfg["route"] + " (démo)", n_legs=n_legs, leg_km=leg_km, t_max_h=t_max,
        n_times=n_times, storm_legs=(sf, sf + 1), storm_center_h=storm_center,
        storm_dur_h=120.0, storm_swh=6.2, storm_wspd=18.0, spread_growth=0.9,
        seed=cfg["seed"])
    eta = 1.003 * total_km * 1000.0 / v_nom / 3600.0
    r = RT.compare_planners(fc, ship, Hs_max=cfg["Hs_max"], alpha=cfg["alpha"],
                            eta_max_h=eta, v_lo_kn=v_lo, v_hi_kn=v_hi,
                            dep_max_h=dep_max, n_mc=max(cfg["n_mc"], 1500),
                            seed=cfg["seed"])
    pb, dt = r["probabiliste"], r["deterministe"]
    print(f"  probabiliste  V={pb['decision']['speed_kn']:.1f}kn "
          f"P(swh>Hs)={pb['reel']['p_exceed']:.0%} | "
          f"déterministe V={dt['decision']['speed_kn']:.1f}kn "
          f"P(swh>Hs)={dt['reel']['p_exceed']:.0%}")
    out = cfg["out_dir"]; os.makedirs(out, exist_ok=True)
    json.dump(r, open(os.path.join(out, f"phase5_demo_{cfg['route']}.json"), "w"),
              indent=2, default=float)
    print(f"[OK] démo → {os.path.abspath(out)}")
    return r


# ── Orchestration ─────────────────────────────────────────────────────────────
def run(cfg):
    cfg = dict(cfg)
    mode = cfg.get("mode", "auto")
    data_dir, manifest_path = (None, None)
    if mode in ("auto", "real"):
        data_dir, manifest_path = _resolve_dataset(cfg["data_dir"], cfg["manifest"])
    if mode == "real" and manifest_path is None:
        raise SystemExit(
            "Mode 'real' demandé mais aucun dataset_manifest.json trouvé.\n"
            "Passez --data-dir \"chemin/vers/ERA5_ML_dataset\" (ou "
            "--manifest …/dataset_manifest.json).")
    if manifest_path is None:
        return run_demo(cfg)
    if cfg.get("sweep_hs"):
        return run_sweep(cfg, data_dir, manifest_path, cfg["sweep_hs"])
    if str(cfg.get("route", "")).upper() in ("ALL", "TOUTES", "*"):
        return run_all_routes(cfg, data_dir, manifest_path)
    return run_real(cfg, data_dir, manifest_path)


def run_all_routes(cfg, data_dir, manifest_path):
    """Exécute la Phase 5 sur TOUTES les routes du manifeste et produit la figure
    consolidée inter-routes + un récapitulatif (la sortie « article »)."""
    manifest = D.load_manifest(manifest_path)
    rdir = _resolve_results(cfg.get("results_dir"))
    skill = RT.load_phase4_skill(rdir) if rdir else None
    if skill:
        print("[phase5] calibration Phase 4 (" + os.path.basename(rdir) + ") : "
              + ", ".join(f"{v}→{skill[v]['model'].upper()}" for v in skill)
              + " | bande q10–q90 = médiane ± 1.28·RMSE(h)·échelle_route")
    else:
        print("[phase5] AVERTISSEMENT : results/ introuvable → incertitude "
              "paramétrique de repli (passez --results pour calibrer).")
    routes = list(manifest["nodes_per_route"])
    out = cfg["out_dir"]; os.makedirs(out, exist_ok=True)
    summary = []
    for route in routes:
        print("\n" + "=" * 70 + f"\n  ROUTE {route}\n" + "=" * 70)
        rcfg = {**cfg, "route": route}
        try:
            res = run_real(rcfg, data_dir, manifest_path,
                           manifest=manifest, skill=skill)
        except Exception as e:                      # une route sans données ne bloque pas
            print(f"[phase5] route {route} ignorée : {e}")
            continue
        a = res["aggregate"]; a["route"] = route
        summary.append(a)
    if not summary:
        raise RuntimeError("Aucune route exploitable (données manquantes).")
    # correction de multiplicité Benjamini-Hochberg sur les 4 tests (une route)
    _apply_bh(summary, q=0.05)
    json.dump(summary, open(os.path.join(out, "phase5_summary_all_routes.json"),
                            "w"), indent=2, default=float)
    _write_summary_csv(summary, os.path.join(out, "phase5_summary_all_routes.csv"))
    _make_consolidated_figure(summary, cfg, out)
    print("\n" + "=" * 70)
    print("  RÉCAPITULATIF INTER-ROUTES (valeur de l'incertitude, * = signif. BH)")
    print(f"  {'route':14s} {'tronçons dgx prob→det':>22s} {'réduction (BH)':>18s} "
          f"{'surcoût fuel':>13s}")
    for a in summary:
        star = " *" if a.get("significant_bh") else ""
        print(f"  {a['route']:14s} "
              f"{a['prob_unsafe_leg_rate']:8.1%} → {a['det_unsafe_leg_rate']:<8.1%} "
              f"{a['reduction_abs']:+8.2%} (p={a.get('reduction_pvalue', 1):.3f}){star:>2s} "
              f"{a['fuel_overhead_pct']:+11.1f} %")
    print(f"\n[OK] Sortie article → {os.path.abspath(out)}")
    return summary


def _benjamini_hochberg(pvals, q=0.05):
    """Renvoie un tableau booléen : True si l'hypothèse est rejetée (significative)
    au taux de fausses découvertes q, procédure de Benjamini-Hochberg."""
    p = np.asarray(pvals, float)
    m = p.size
    if m == 0:
        return np.zeros(0, bool)
    order = np.argsort(p)
    thresh = (np.arange(1, m + 1) / m) * q
    passed = p[order] <= thresh
    kmax = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    reject = np.zeros(m, bool)
    if kmax > 0:
        reject[order[:kmax]] = True
    return reject


def _apply_bh(records, q=0.05, pkey="reduction_pvalue", outkey="significant_bh"):
    """Ajoute records[i][outkey] = significativité après correction BH sur la
    famille des p-values records[i][pkey]."""
    pv = [float(r.get(pkey, 1.0)) for r in records]
    rej = _benjamini_hochberg(pv, q=q)
    for r, s in zip(records, rej):
        r[outkey] = bool(s)
    return records


def _write_summary_csv(summary, path):
    import csv
    keys = ["route", "n_departures", "n_valid_legs_prob", "n_valid_legs_det",
            "max_swh_obs", "frac_legs_over_Hs",
            "prob_unsafe_leg_rate", "det_unsafe_leg_rate",
            "reduction_abs", "reduction_abs_ci", "reduction_pvalue",
            "reduction_significant", "significant_bh",
            "prob_voyage_unsafe_rate", "det_voyage_unsafe_rate",
            "prob_fuel_mean", "det_fuel_mean",
            "fuel_overhead_pct", "fuel_overhead_pct_ci",
            "unsafe_leg_reduction_pct"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for a in summary:
            row = {k: a.get(k) for k in keys}
            for ck in ("reduction_abs_ci", "fuel_overhead_pct_ci"):
                if isinstance(row.get(ck), (list, tuple)):
                    row[ck] = f"[{row[ck][0]:.4g}; {row[ck][1]:.4g}]"
            w.writerow(row)


def _make_consolidated_figure(summary, cfg, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    routes = [a["route"].split("_")[0] for a in summary]
    x = np.arange(len(routes)); w = 0.38
    prob_leg = [a["prob_unsafe_leg_rate"] for a in summary]
    det_leg = [a["det_unsafe_leg_rate"] for a in summary]
    fuel = [a["fuel_overhead_pct"] for a in summary]

    def _err(vals, ci_key):
        lo = [max(v - a[ci_key][0], 0) for v, a in zip(vals, summary)]
        hi = [max(a[ci_key][1] - v, 0) for v, a in zip(vals, summary)]
        return np.array([lo, hi])
    det_err = _err(det_leg, "det_unsafe_leg_rate_ci")
    prob_err = _err(prob_leg, "prob_unsafe_leg_rate_ci")
    fuel_err = _err(fuel, "fuel_overhead_pct_ci")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax0.bar(x - w / 2, det_leg, w, color="#e08a1e", edgecolor="white",
            yerr=det_err, capsize=3, error_kw=dict(lw=0.9, ecolor="#333"),
            label="deterministic (median only)")
    ax0.bar(x + w / 2, prob_leg, w, color="#1b6ca8", edgecolor="white",
            yerr=prob_err, capsize=3, error_kw=dict(lw=0.9, ecolor="#333"),
            label="probabilistic (quantiles)")
    # headroom for the significance markers
    ci_hi = [max(a["det_unsafe_leg_rate_ci"][1], a["prob_unsafe_leg_rate_ci"][1])
             for a in summary]
    ymax = max(ci_hi + [1e-3])
    ax0.set_ylim(0, ymax * 1.30)
    for xi, d, p, a, hi in zip(x, det_leg, prob_leg, summary, ci_hi):
        ax0.text(xi - w / 2, d, f"{d:.1%}", ha="center", va="bottom", fontsize=7)
        ax0.text(xi + w / 2, p, f"{p:.1%}", ha="center", va="bottom", fontsize=7)
        if a.get("significant_bh"):
            ax0.text(xi, hi + 0.05 * ymax, "*", ha="center", va="bottom",
                     fontsize=16, color="#2e7d32", fontweight="bold")
    ax0.set_xticks(x); ax0.set_xticklabels(routes)
    ax0.set_ylabel("real dangerous-leg rate (swh > Hs)")
    ax0.set_title("(a) Safety per route (95% CI)", loc="left",
                  fontweight="bold", fontsize=11)
    handles, labels = ax0.get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker="*", color="w",
                          markerfacecolor="#2e7d32", markersize=12,
                          label="significant reduction (BH, q<0.05)"))
    ax0.legend(handles=handles, fontsize=8, frameon=False, loc="upper right")
    ax0.grid(alpha=0.3, axis="y")

    bars = ax1.bar(x, fuel, 0.55, color="#3a9d5d", edgecolor="white",
                   yerr=fuel_err, capsize=3, error_kw=dict(lw=0.9, ecolor="#333"))
    for b, v in zip(bars, fuel):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:+.1f}%",
                 ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax1.axhline(0, color="#888", lw=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(routes)
    ax1.set_ylabel("probabilistic fuel overhead (%)")
    ax1.set_title("(b) Fuel cost of safety", loc="left",
                  fontweight="bold", fontsize=11)
    ax1.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Phase 5 — decision value of probabilistic forecasting, "
                 f"4 routes (Hs_max = {cfg['Hs_max']} m, α = {cfg['alpha']:.0%})",
                 fontweight="bold", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fpath = os.path.join(out, "Fig_phase5_consolidee.png")
    fig.savefig(fpath, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(fpath.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[phase5] figure consolidée → {fpath}")


def run_sweep(cfg, data_dir, manifest_path, hs_list):
    """Analyse de sensibilité au seuil Hs_max : pour chaque route et chaque seuil,
    backtest complet des deux politiques. Produit la figure et le tableau qui
    montrent (i) l'exposition réelle de chaque route (→ un 0 % est physique, pas
    un bug) et (ii) l'échéance à laquelle la valeur de l'incertitude émerge.
    Les données (prep) ne sont chargées qu'UNE fois par route et réutilisées pour
    tous les seuils."""
    manifest = D.load_manifest(manifest_path)
    rdir = _resolve_results(cfg.get("results_dir"))
    skill = RT.load_phase4_skill(rdir) if rdir else None
    if skill:
        print("[phase5] calibration Phase 4 : " + ", ".join(
            f"{v}→{skill[v]['model'].upper()}" for v in skill))
    routes = (list(manifest["nodes_per_route"])
              if str(cfg.get("route", "")).upper() in ("ALL", "TOUTES", "*")
              else [cfg["route"]])
    out = cfg["out_dir"]; os.makedirs(out, exist_ok=True)
    nominal = float(cfg["Hs_max"])          # seuil « opérationnel » (défaut 6 m)
    sweep = {}
    summary = []                            # agrégats COMPLETS au seuil nominal
    for route in routes:
        if route not in manifest["nodes_per_route"]:
            print(f"[phase5] route {route} absente du manifeste — ignorée."); continue
        print("\n" + "=" * 70 + f"\n  BALAYAGE Hs_max — ROUTE {route}\n" + "=" * 70)
        ship, ship_src = _ship_for_route(route)
        geom = RT.build_route_geometry(manifest["nodes_per_route"][route],
                                       n_legs_max=cfg["n_legs_max"])
        try:
            df = _load_route_df(data_dir, route)
            prep = RT.prepare_route_series(df, geom, manifest["temporal_split"])
        except Exception as e:
            print(f"[phase5] route {route} ignorée : {e}"); continue
        recs = []; res_nominal = None
        for hs in hs_list:
            cfg2 = {**cfg, "route": route, "Hs_max": float(hs), "skill": skill}
            try:
                res = RT.rolling_backtest(prep, ship, cfg2)
            except Exception as e:
                print(f"   Hs={hs} m ignoré : {e}"); continue
            a = res["aggregate"]
            if abs(float(hs) - nominal) < 1e-9:       # backtest au seuil nominal
                res_nominal = res
            recs.append(dict(
                Hs_max=float(hs), n_departures=a["n_departures"],
                prob=a["prob_unsafe_leg_rate"], det=a["det_unsafe_leg_rate"],
                prob_ci=a["prob_unsafe_leg_rate_ci"], det_ci=a["det_unsafe_leg_rate_ci"],
                reduction_abs=a["reduction_abs"], reduction_ci=a["reduction_abs_ci"],
                reduction_pvalue=a["reduction_pvalue"],
                significant=a["reduction_significant"],
                reduction_pct=a["unsafe_leg_reduction_pct"],
                fuel=a["fuel_overhead_pct"], fuel_ci=a["fuel_overhead_pct_ci"],
                max_swh=a["max_swh_obs"], exposure=a["exposure"]))
            sig = "signif." if a["reduction_significant"] else "n.s."
            rc = a["reduction_abs_ci"]
            print(f"   Hs={hs:>4.1f} m | det {a['det_unsafe_leg_rate']:5.1%} "
                  f"prob {a['prob_unsafe_leg_rate']:5.1%} | réduction appariée "
                  f"{a['reduction_abs']:+.2%} [{rc[0]:+.2%};{rc[1]:+.2%}] "
                  f"(p={a['reduction_pvalue']:.3f} {sig}) "
                  f"| fuel {a['fuel_overhead_pct']:+.1f}%")
        if not recs:
            continue
        sweep[route] = recs
        mx = recs[0]["max_swh"]; exp = recs[0]["exposure"]
        print(f"   → EXPOSITION {route} : houle max {mx:.1f} m | " + " ".join(
            f"{k.replace('frac_over_', '>')}={v:.1%}" for k, v in exp.items()))
        # backtest au seuil nominal (s'il n'était pas dans la liste balayée)
        if res_nominal is None:
            res_nominal = RT.rolling_backtest(
                prep, ship, {**cfg, "route": route, "Hs_max": nominal, "skill": skill})
        # figure PAR ROUTE au seuil nominal + collecte pour la consolidée
        _make_real_figure(res_nominal, ship,
                          {**cfg, "route": route, "Hs_max": nominal}, out)
        a_nom = res_nominal["aggregate"]; a_nom["route"] = route
        summary.append(a_nom)
    if not sweep:
        raise RuntimeError("Balayage vide (aucune route exploitable).")
    # (1) figures au seuil NOMINAL : consolidée inter-routes + récap (BH sur routes)
    if summary:
        _apply_bh(summary, q=0.05)
        json.dump(summary, open(os.path.join(out, "phase5_summary_all_routes.json"),
                                "w"), indent=2, default=float)
        _write_summary_csv(summary,
                           os.path.join(out, "phase5_summary_all_routes.csv"))
        _make_consolidated_figure(summary, {**cfg, "Hs_max": nominal}, out)
    # (2) figure de sensibilité : BH sur TOUTE la famille (routes × seuils)
    flat = [rec for recs in sweep.values() for rec in recs]
    _apply_bh(flat, q=0.05)
    n_bh = sum(1 for r in flat if r["significant_bh"])
    print(f"\n[phase5] Benjamini-Hochberg (q=0.05) sur {len(flat)} tests : "
          f"{n_bh} réduction(s) significative(s).")
    json.dump(sweep, open(os.path.join(out, "phase5_sweep_hs.json"), "w"),
              indent=2, default=float)
    _make_sweep_figure(sweep, cfg, out)
    print(f"\n[OK] Phase 5 complète (sensibilité + consolidée + par route) → "
          f"{os.path.abspath(out)}")
    return {"sweep": sweep, "summary": summary}


def _make_sweep_figure(sweep, cfg, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    routes = list(sweep)
    ncol = 2; nrow = int(np.ceil(len(routes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.6 * nrow), squeeze=False)
    for i, route in enumerate(routes):
        ax = axes[i // ncol][i % ncol]
        recs = sorted(sweep[route], key=lambda r: r["Hs_max"])
        hs = [r["Hs_max"] for r in recs]
        det = [r["det"] for r in recs]; prob = [r["prob"] for r in recs]
        dlo = [r["det_ci"][0] for r in recs]; dhi = [r["det_ci"][1] for r in recs]
        plo = [r["prob_ci"][0] for r in recs]; phi = [r["prob_ci"][1] for r in recs]
        ax.fill_between(hs, dlo, dhi, color="#e08a1e", alpha=0.18)
        ax.fill_between(hs, plo, phi, color="#1b6ca8", alpha=0.18)
        ax.plot(hs, det, "-s", color="#e08a1e", ms=4, label="deterministic")
        ax.plot(hs, prob, "-o", color="#1b6ca8", ms=4, label="probabilistic")
        # points where the PAIRED reduction is significant after BH correction
        sig_hs = [r["Hs_max"] for r in recs if r.get("significant_bh")]
        sig_y = [r["prob"] for r in recs if r.get("significant_bh")]
        if sig_hs:
            ax.scatter(sig_hs, sig_y, s=90, marker="*", color="#2e7d32",
                       zorder=6, label="significant reduction (BH, q<0.05)")
        mx = recs[0]["max_swh"]
        ax.axvline(mx, color="#888", ls=":", lw=1.1)
        ax.text(mx, ax.get_ylim()[1] * 0.92, f" max wave\n {mx:.1f} m",
                fontsize=7, color="#555", va="top")
        ax.set_title(f"{route.split('_')[0]} — {route}", loc="left",
                     fontweight="bold", fontsize=9.5)
        ax.set_xlabel("safety threshold Hs_max (m)")
        ax.set_ylabel("real dangerous-leg rate")
        ax.grid(alpha=0.3); ax.legend(fontsize=8, frameon=False)
    for j in range(len(routes), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Phase 5 — sensitivity to the Hs_max threshold: real exposure and "
                 f"decision value of uncertainty (α = {cfg['alpha']:.0%}, 95% CI)",
                 fontweight="bold", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fpath = os.path.join(out, "Fig_phase5_sweep_hs.png")
    fig.savefig(fpath, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(fpath.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[phase5] figure de sensibilité → {fpath}")


def run_config(**overrides):
    """Lance la Phase 5 avec CONFIG surchargé. Exemple notebook :
        import run_phase5 as P5
        P5.run_config(route="R2_NFK_HAM", n_departures=16, Hs_max=5.5)
    """
    cfg = dict(CONFIG); cfg.update(overrides)
    return run(cfg)


def _running_in_notebook():
    try:
        from IPython import get_ipython
        sh = get_ipython()
        return sh is not None and sh.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def main(argv=None):
    cfg = dict(CONFIG)
    if _running_in_notebook():
        return run(cfg)
    argv = (sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description="Phase 5 — routage sous incertitude "
                                             "(données réelles ERA5).")
    ap.add_argument("--route", default=cfg["route"])
    ap.add_argument("--mode", default=cfg["mode"], choices=["auto", "real", "demo"])
    ap.add_argument("--data-dir", default=cfg["data_dir"])
    ap.add_argument("--manifest", default=cfg["manifest"])
    ap.add_argument("--hs-max", type=float, default=cfg["Hs_max"])
    ap.add_argument("--alpha", type=float, default=cfg["alpha"])
    ap.add_argument("--eta-slack", type=float, default=cfg["eta_slack"])
    ap.add_argument("--dep-max-h", type=float, default=cfg["dep_max_h"])
    ap.add_argument("--n-legs-max", type=int, default=cfg["n_legs_max"])
    ap.add_argument("--n-departures", type=int, default=cfg["n_departures"])
    ap.add_argument("--n-mc", type=int, default=cfg["n_mc"])
    ap.add_argument("--results", default=cfg["results_dir"],
                    help="dossier des JSON de Phase 4 (calibration de l'incertitude)")
    ap.add_argument("--all-routes", action="store_true",
                    help="exécuter les 4 routes + figure consolidée (sortie article)")
    ap.add_argument("--sweep-hs", default=None,
                    help="seuils Hs_max séparés par des virgules (ex. '3,4,5,6') "
                         "→ analyse de sensibilité (disculpe les 0 %, α via --alpha)")
    ap.add_argument("--out-dir", default=cfg["out_dir"])
    a, _unknown = ap.parse_known_args(argv)
    sweep = None
    if a.sweep_hs:
        sweep = [float(x) for x in str(a.sweep_hs).replace(";", ",").split(",")
                 if x.strip()]
    cfg.update(route=("ALL" if a.all_routes else a.route), mode=a.mode,
               data_dir=a.data_dir, manifest=a.manifest, results_dir=a.results,
               sweep_hs=sweep, Hs_max=a.hs_max, alpha=a.alpha, eta_slack=a.eta_slack,
               dep_max_h=a.dep_max_h, n_legs_max=a.n_legs_max,
               n_departures=a.n_departures, n_mc=a.n_mc, out_dir=a.out_dir)
    return run(cfg)


if __name__ == "__main__":
    main()
