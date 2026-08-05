
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

G = 9.81
RHO_AIR = 1.225
RHO_SW = 1025.0
KN2MS = 0.514444
Z90 = 1.2815515655446004        # quantile normal à 90 %


# ── Navire ────────────────────────────────────────────────────────────────────

@dataclass
class Ship:
    name: str = "container"
    P_nom_kW: float = 12000.0
    V_nom_kn: float = 12.0
    SFOC_gkWh: float = 180.0
    CO2f: float = 3.17
    L: float = 200.0
    B: float = 32.0
    n_rotors: int = 4
    A_rotor: float = 30.0
    Ct_max: float = 8.0
    eta_prop: float = 0.65
    P_spin_frac: float = 0.10
    c_wave: float = 0.55
    P_min_frac: float = 0.15
    P_ovl_frac: float = 1.50        # surcharge max moteur (MCR / marge de mer)

    @property
    def V_nom(self):
        return self.V_nom_kn * KN2MS


def ship_from_route_meta(meta: dict) -> Ship:
    return Ship(name=meta.get("ship_type", "container"),
                P_nom_kW=float(meta.get("P_nom_kW", 12000.0)),
                V_nom_kn=float(meta.get("speed_kn", 12.0)),
                SFOC_gkWh=float(meta.get("SFOC_gkWh", meta.get("SFOC", 180.0))),
                CO2f=float(meta.get("CO2f", 3.17)))


# ── Puissances (kW) ───────────────────────────────────────────────────────────

def calm_water_power(V, ship: Ship):
    V = np.asarray(V, float)
    return ship.P_nom_kW * (V / ship.V_nom) ** 3


def added_wave_power(Hs, V, ship: Ship):
    Hs = np.asarray(Hs, float); V = np.asarray(V, float)
    R_aw = ship.c_wave * RHO_SW * G * (Hs ** 2) * (ship.B ** 2) / ship.L
    return R_aw * V / 1000.0


def rotor_net_power(U, awa_deg, V, ship: Ship, rho=RHO_AIR):
    """Puissance propulsive nette des rotors (kW, ≥0). Ct max au vent de travers."""
    U = np.asarray(U, float); awa = np.deg2rad(np.asarray(awa_deg, float))
    Ct = ship.Ct_max * np.sin(np.clip(awa, 0, np.pi))
    F = 0.5 * rho * ship.A_rotor * ship.n_rotors * U ** 2 * Ct
    P_prop = F * np.asarray(V, float) * ship.eta_prop / 1000.0
    return np.maximum(P_prop * (1.0 - ship.P_spin_frac), 0.0)


rotor_power_fn = rotor_net_power        # point d'entrée (brancher le GPR ici)


def engine_power(V, U, awa_deg, Hs, ship: Ship, rho=RHO_AIR):
    P = (calm_water_power(V, ship) + added_wave_power(Hs, V, ship)
         - rotor_power_fn(U, awa_deg, V, ship, rho))
    return np.clip(P, ship.P_min_frac * ship.P_nom_kW,
                   ship.P_ovl_frac * ship.P_nom_kW)


def fuel_rate_kg_h(V, U, awa_deg, Hs, ship: Ship, rho=RHO_AIR):
    return engine_power(V, U, awa_deg, Hs, ship, rho) * ship.SFOC_gkWh / 1000.0


# ── Reconstruction d'échantillons depuis des quantiles ────────────────────────

def _sigma_log(q10, q90):
    q10 = max(float(q10), 1e-6); q90 = max(float(q90), 1e-6)
    return float(np.clip(np.log(q90 / q10) / (2 * Z90), 1e-3, 1.5))


def sample_lognormal(q10, q50, q90, n, rng):
    mu = np.log(max(float(q50), 1e-6))
    return np.exp(rng.normal(mu, _sigma_log(q10, q90), size=n))


# ── Champ de prévision ESPACE-TEMPS (quantiles) ───────────────────────────────

@dataclass
class SpaceTimeForecast:
    """Quantiles de wspd et swh sur une grille (tronçon k, temps t_h).

    wspd_q, swh_q : tableaux (K, T, 3) = (q10, q50, q90).
    leg_km, awa_deg : (K,) ; times_h : (T,).
    """
    name: str
    leg_km: np.ndarray
    awa_deg: np.ndarray
    times_h: np.ndarray
    wspd_q: np.ndarray
    swh_q: np.ndarray

    @property
    def K(self):
        return len(self.leg_km)

    @property
    def total_km(self):
        return float(np.sum(self.leg_km))

    def _q_at(self, field, k, t_h):
        arr = field[k]                                  # (T,3)
        tt = self.times_h
        return (np.interp(t_h, tt, arr[:, 0]),
                np.interp(t_h, tt, arr[:, 1]),
                np.interp(t_h, tt, arr[:, 2]))

    def median_at(self, field_name, k, t_h):
        f = self.wspd_q if field_name == "wspd" else self.swh_q
        return self._q_at(f, k, t_h)[1]

    def sample_at(self, field_name, k, t_h, n, rng):
        f = self.wspd_q if field_name == "wspd" else self.swh_q
        return sample_lognormal(*self._q_at(f, k, t_h), n, rng)


# ── Temps d'arrivée par tronçon ───────────────────────────────────────────────

def leg_arrival_times(leg_km, V_ms, departure_h):
    """Instant (h) au MILIEU de chaque tronçon pour une vitesse constante."""
    hours = leg_km * 1000.0 / max(V_ms, 0.1) / 3600.0
    ends = departure_h + np.cumsum(hours)
    mids = ends - hours / 2.0
    return mids, float(ends[-1] - departure_h)


# ── Évaluation d'un plan (Monte-Carlo sur l'incertitude) ──────────────────────

def evaluate_plan(fc: SpaceTimeForecast, V_kn, departure_h, ship: Ship,
                  Hs_max, n_mc=1000, seed=0):
    rng = np.random.default_rng(seed)
    V = float(V_kn) * KN2MS
    mids, dur = leg_arrival_times(fc.leg_km, V, departure_h)
    fuel = np.zeros(n_mc); exceed = np.zeros(n_mc, bool)
    for k in range(fc.K):
        hours = fc.leg_km[k] * 1000.0 / V / 3600.0
        U = fc.sample_at("wspd", k, mids[k], n_mc, rng)
        Hs = fc.sample_at("swh", k, mids[k], n_mc, rng)
        fuel += fuel_rate_kg_h(V, U, fc.awa_deg[k], Hs, ship) * hours / 1000.0
        exceed |= (Hs > Hs_max)
    return {"speed_kn": float(V_kn), "departure_h": float(departure_h),
            "fuel_t_mean": float(fuel.mean()), "fuel_t_std": float(fuel.std()),
            "fuel_t_p90": float(np.percentile(fuel, 90)),
            "duration_h": float(dur), "arrival_h": float(departure_h + dur),
            "p_exceed": float(exceed.mean()),
            "co2_t_mean": float(fuel.mean() * ship.CO2f)}


def evaluate_plan_median(fc: SpaceTimeForecast, V_kn, departure_h, ship: Ship,
                         Hs_max):
    """Évaluation DÉTERMINISTE : médianes q50, dispersion ignorée."""
    V = float(V_kn) * KN2MS
    mids, dur = leg_arrival_times(fc.leg_km, V, departure_h)
    fuel = 0.0; exceed = False
    for k in range(fc.K):
        hours = fc.leg_km[k] * 1000.0 / V / 3600.0
        U = fc.median_at("wspd", k, mids[k])
        Hs = fc.median_at("swh", k, mids[k])
        fuel += fuel_rate_kg_h(V, U, fc.awa_deg[k], Hs, ship) * hours / 1000.0
        exceed = exceed or (Hs > Hs_max)
    return {"speed_kn": float(V_kn), "departure_h": float(departure_h),
            "fuel_t_mean": float(fuel), "fuel_t_std": 0.0,
            "fuel_t_p90": float(fuel), "duration_h": float(dur),
            "arrival_h": float(departure_h + dur),
            "p_exceed": 1.0 if exceed else 0.0,
            "co2_t_mean": float(fuel * ship.CO2f)}


# ── Optimisation (vitesse, départ) sous contraintes ───────────────────────────

def optimize_plan(fc: SpaceTimeForecast, ship: Ship, *, Hs_max=6.0, alpha=0.10,
                  eta_max_h=None, v_lo_kn=8.0, v_hi_kn=16.0, n_v=17,
                  dep_max_h=72.0, n_dep=13, n_mc=800, seed=0,
                  use_uncertainty=True, dep_penalty_t_per_h=0.02):
    """
    Balaie (vitesse × délai de départ) et renvoie le meilleur plan FAISABLE
    (P(exceed) ≤ alpha et arrivée ≤ eta_max_h) minimisant le carburant espéré
    (+ légère pénalité de retard de départ pour départager). Si aucun plan n'est
    faisable, renvoie le moins risqué.
    """
    speeds = np.linspace(v_lo_kn, v_hi_kn, n_v)
    deps = np.linspace(0.0, dep_max_h, n_dep)
    best = None
    for d in deps:
        for Vkn in speeds:
            ev = (evaluate_plan(fc, Vkn, d, ship, Hs_max, n_mc=n_mc, seed=seed)
                  if use_uncertainty else
                  evaluate_plan_median(fc, Vkn, d, ship, Hs_max))
            feasible = (ev["p_exceed"] <= alpha) and \
                       (eta_max_h is None or ev["arrival_h"] <= eta_max_h + 1e-9)
            cost = ev["fuel_t_mean"] + dep_penalty_t_per_h * d
            cand = {**ev, "feasible": feasible, "cost": cost}
            key = lambda c: (0 if c["feasible"] else 1,
                             c["p_exceed"] if not c["feasible"] else 0.0,
                             c["cost"])
            if best is None or key(cand) < key(best):
                best = cand
    return best


def compare_planners(fc: SpaceTimeForecast, ship: Ship, *, Hs_max=6.0,
                     alpha=0.10, eta_max_h=None, n_mc=1200, seed=0, **kw):
    prob = optimize_plan(fc, ship, Hs_max=Hs_max, alpha=alpha,
                         eta_max_h=eta_max_h, n_mc=n_mc, seed=seed,
                         use_uncertainty=True, **kw)
    det = optimize_plan(fc, ship, Hs_max=Hs_max, alpha=alpha,
                        eta_max_h=eta_max_h, n_mc=n_mc, seed=seed,
                        use_uncertainty=False, **kw)
    prob_real = evaluate_plan(fc, prob["speed_kn"], prob["departure_h"], ship,
                              Hs_max, n_mc=max(n_mc, 2000), seed=seed + 100)
    det_real = evaluate_plan(fc, det["speed_kn"], det["departure_h"], ship,
                             Hs_max, n_mc=max(n_mc, 2000), seed=seed + 100)
    return {
        "route": fc.name, "Hs_max": Hs_max, "alpha": alpha,
        "eta_max_h": eta_max_h,
        "probabiliste": {"decision": prob, "reel": prob_real},
        "deterministe": {"decision": det, "reel": det_real},
        "gain": {
            "d_speed_kn": det["speed_kn"] - prob["speed_kn"],
            "d_departure_h": det["departure_h"] - prob["departure_h"],
            "d_fuel_t": det_real["fuel_t_mean"] - prob_real["fuel_t_mean"],
            "d_p_exceed": det_real["p_exceed"] - prob_real["p_exceed"],
        },
    }


# ── Générateur de scénario espace-temps (démo & validation) ───────────────────

def synthesize_spacetime_forecast(name, n_legs=12, leg_km=1000.0,
                                  base_wspd=9.0, base_swh=2.6, t_max_h=520.0,
                                  n_times=53, storm_legs=(6, 7, 8),
                                  storm_center_h=210.0, storm_dur_h=60.0,
                                  storm_swh=7.5, storm_wspd=19.0,
                                  spread_growth=0.9, seed=0):
    """
    Champ (tronçon, temps) réaliste : conditions de base + TEMPÊTE TRANSITOIRE
    (houle forte sur `storm_legs` autour de storm_center_h). La dispersion des
    quantiles croît avec l'échéance t (propriété mesurée en Phase 4).

    Le même format (quantiles q10/q50/q90 par tronçon et par temps) est produit
    par les prévisions RÉELLES : agrégation, par nœud/tronçon et par heure cible,
    des quantiles de predict_and_score(..., quantiles=[0.1,0.5,0.9]).
    """
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, t_max_h, n_times)
    leg_km_arr = np.full(n_legs, leg_km, float)
    awa = rng.uniform(45, 135, n_legs)
    base_w = base_wspd * (0.85 + 0.3 * rng.random(n_legs))
    base_s = base_swh * (0.85 + 0.3 * rng.random(n_legs))
    wspd_q = np.zeros((n_legs, n_times, 3))
    swh_q = np.zeros((n_legs, n_times, 3))
    for k in range(n_legs):
        for j, t in enumerate(times):
            storm = 0.0
            if k in storm_legs:
                storm = np.exp(-0.5 * ((t - storm_center_h) /
                                       (storm_dur_h / 2.0)) ** 2)
            w50 = base_w[k] + (storm_wspd - base_w[k]) * storm
            s50 = base_s[k] + (storm_swh - base_s[k]) * storm
            f = 0.10 * (1.0 + spread_growth * t / t_max_h)
            wspd_q[k, j] = (w50 * np.exp(-Z90 * f), w50, w50 * np.exp(Z90 * f))
            swh_q[k, j] = (s50 * np.exp(-Z90 * (f + 0.03)), s50,
                           s50 * np.exp(Z90 * (f + 0.03)))
    return SpaceTimeForecast(name, leg_km_arr, awa, times, wspd_q, swh_q)


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 5 SUR DONNÉES RÉELLES (ERA5)
#  ---------------------------------------------------------------------------
#  Construit le champ espace-temps de quantiles à partir des vraies séries du
#  jeu ERA5, puis BACKTESTE les plans probabiliste et déterministe contre la
#  VÉRITÉ ERA5 sur plusieurs dates de départ → carburant réel et taux réel de
#  dépassement de sécurité (valeur décisionnelle de l'incertitude).
#
#  Modèle de prévision (sans ré-exécuter le réseau) : la prévision ML est
#  représentée par ses PROPRIÉTÉS MESURÉES en Phase 4 —
#    • médiane = mélange skill-pondéré  q50 = w(τ)·obs + (1−w(τ))·climato,
#      w(τ)=exp(−τ/τ_skill)  (skillful à court terme, climatologique au-delà) ;
#    • dispersion relative croissant avec l'échéance et SATURANT à la
#      variabilité climatologique :  σ_rel(k,τ) = r0 + (CV_k − r0)·(1−e^{−τ/τ_spread}).
#  La VÉRITÉ (évaluation) reste l'ERA5 observé. On peut brancher de vraies
#  prévisions quantiles en fournissant directement q10/q50/q90 (voir NOTE).
# ══════════════════════════════════════════════════════════════════════════════

_EPOCH = np.datetime64("1970-01-01T00:00:00")


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def initial_bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def relative_wind_angle_deg(bearing_deg, u10, v10):
    """Angle du vent par rapport au cap (0=de face, 90=travers, 180=arrière),
    dans [0,180]. Seul |sin| compte pour la poussée rotor (max au travers)."""
    br = np.radians(bearing_deg)
    hx, hy = np.sin(br), np.cos(br)            # cap : est=x, nord=y
    w = np.hypot(u10, v10) + 1e-9
    cosang = np.clip(hx * (u10 / w) + hy * (v10 / w), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def _hours_since_epoch(dt64):
    return (np.asarray(dt64, dtype="datetime64[s]") - _EPOCH) / np.timedelta64(1, "h")


def _clean_series(hours, *value_arrays):
    """Trie par heure et retire les points où une valeur est non finie."""
    hours = np.asarray(hours, float)
    order = np.argsort(hours)
    hours = hours[order]
    vals = [np.asarray(v, float)[order] for v in value_arrays]
    good = np.ones(hours.shape, bool)
    for v in vals:
        good &= np.isfinite(v)
    hours = hours[good]
    vals = [v[good] for v in vals]
    return (hours, *vals)


def build_route_geometry(nodes_meta, n_legs_max=20):
    """À partir de la liste de nœuds du manifeste (lat/lon/dist_km ordonnés),
    sous-échantillonne à ≤ n_legs_max tronçons et renvoie la géométrie.

    Renvoie dict : node_ids (K,), lat (K,), lon (K,), dist_km (K,),
    leg_km (K,)  [longueur du tronçon k = milieu→milieu, dernier = résiduel],
    bearing_deg (K,).
    """
    lat = np.array([n["lat"] for n in nodes_meta], float)
    lon = np.array([n["lon"] for n in nodes_meta], float)
    dist = np.array([n["dist_km"] for n in nodes_meta], float)
    nid = np.array([n["node_id"] for n in nodes_meta], int)
    K0 = len(nid)
    if n_legs_max and K0 > n_legs_max:
        idx = np.unique(np.linspace(0, K0 - 1, n_legs_max).round().astype(int))
    else:
        idx = np.arange(K0)
    lat, lon, dist, nid = lat[idx], lon[idx], dist[idx], nid[idx]
    K = len(nid)
    # longueur de tronçon = distance cumulée entre nœuds sélectionnés
    edge = np.diff(dist)                                    # (K-1,)
    leg_km = np.empty(K)
    leg_km[:-1] = edge
    leg_km[-1] = edge[-1] if len(edge) else 150.0          # dernier ~ précédent
    # cap de chaque tronçon (nœud k → k+1 ; dernier = précédent)
    brg = np.empty(K)
    brg[:-1] = initial_bearing_deg(lat[:-1], lon[:-1], lat[1:], lon[1:])
    brg[-1] = brg[-2] if K > 1 else 0.0
    return {"node_ids": nid, "lat": lat, "lon": lon, "dist_km": dist,
            "leg_km": leg_km, "bearing_deg": brg}


def _monthly_climatology(hours, month, value):
    """Moyenne par mois (1..12) ; renvoie (clim[12], moy_globale, std_globale)."""
    clim = np.full(12, np.nan)
    for m in range(1, 13):
        sel = month == m
        if sel.any():
            clim[m - 1] = np.nanmean(value[sel])
    gmean = np.nanmean(value) if np.isfinite(value).any() else 0.0
    gstd = np.nanstd(value) if np.isfinite(value).any() else 0.0
    # bouche-trous mensuels par la moyenne globale
    clim = np.where(np.isfinite(clim), clim, gmean)
    return clim, float(gmean), float(gstd)


def prepare_route_series(route_df, geom, split, clim_split="train"):
    """Prépare, par nœud sélectionné, les séries d'ÉVALUATION (test) et la
    CLIMATOLOGIE mensuelle (train) de swh/wspd, plus l'angle rotor moyen.

    route_df : DataFrame d'UNE route, colonnes
        node_id, datetime, swh, wspd, u10, v10.
    geom : sortie de build_route_geometry.
    split : manifest['temporal_split'] (dict split -> [yr0, yr1]).
    Renvoie dict prêt pour build_scenarios.
    """
    import pandas as pd
    df = route_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    yr = df["datetime"].dt.year.to_numpy()
    ev0, ev1 = split["test"]; cl0, cl1 = split[clim_split]
    K = len(geom["node_ids"])
    per_node = []
    awa = np.empty(K)
    raw = []
    for k, nid in enumerate(geom["node_ids"]):
        g = df[df["node_id"] == nid]
        gh = _hours_since_epoch(g["datetime"].values)
        yk = g["datetime"].dt.year.to_numpy()
        mo = g["datetime"].dt.month.to_numpy()
        swh = g["swh"].to_numpy(float); wspd = g["wspd"].to_numpy(float)
        u = g["u10"].to_numpy(float); v = g["v10"].to_numpy(float)
        # angle rotor moyen (vent moyen du nœud vs cap du tronçon)
        um = float(np.nanmean(u)) if np.isfinite(u).any() else 0.0
        vm = float(np.nanmean(v)) if np.isfinite(v).any() else 0.0
        awa[k] = relative_wind_angle_deg(geom["bearing_deg"][k], um, vm)
        # séries d'évaluation (test)
        te = (yk >= ev0) & (yk <= ev1)
        h_s, swh_s, wspd_s = _clean_series(gh[te], swh[te], wspd[te])
        # climatologie mensuelle (train)
        tc = (yk >= cl0) & (yk <= cl1)
        cs, cs_m, cs_sd = _monthly_climatology(gh[tc], mo[tc], swh[tc])
        cw, cw_m, cw_sd = _monthly_climatology(gh[tc], mo[tc], wspd[tc])
        clim_ok = (cs_m > 1e-6) and (cw_m > 1e-6)
        raw.append(dict(
            h_test=h_s, swh_test=swh_s, wspd_test=wspd_s,
            clim_swh=cs, clim_wspd=cw, clim_ok=clim_ok,
            cv_swh=(cs_sd / cs_m if cs_m > 1e-6 else 0.4),
            cv_wspd=(cw_sd / cw_m if cw_m > 1e-6 else 0.3)))
    # Climatologie de REPLI au niveau de la ROUTE (moyenne des nœuds valides) :
    # empêche qu'un nœud sans données d'entraînement (terre-masqué / lacune)
    # ait une climatologie nulle — cause de l'artefact swh=0.
    valid = [r for r in raw if r["clim_ok"]]
    if valid:
        g_swh = np.mean([r["clim_swh"] for r in valid], axis=0)
        g_wspd = np.mean([r["clim_wspd"] for r in valid], axis=0)
        g_cvs = float(np.mean([r["cv_swh"] for r in valid]))
        g_cvw = float(np.mean([r["cv_wspd"] for r in valid]))
    else:                                     # aucun nœud entraîné : valeurs types
        g_swh = np.full(12, 1.5); g_wspd = np.full(12, 8.0); g_cvs, g_cvw = 0.4, 0.3
    for r in raw:
        if not r["clim_ok"]:
            r["clim_swh"] = g_swh.copy(); r["clim_wspd"] = g_wspd.copy()
            r["cv_swh"], r["cv_wspd"] = g_cvs, g_cvw
        r["n_test"] = int(len(r["h_test"]))
        r.pop("clim_ok", None)
        per_node.append(r)
    return {"geom": geom, "awa_deg": awa, "per_node": per_node,
            "test_years": (ev0, ev1)}


def _grid_truth_clim(node, times_h, t0_h, months_grid):
    """Vérité (interp temporelle) et climatologie (par mois) sur la grille."""
    abs_h = t0_h + times_h
    if len(node["h_test"]) >= 2:
        swh_t = np.interp(abs_h, node["h_test"], node["swh_test"])
        wspd_t = np.interp(abs_h, node["h_test"], node["wspd_test"])
    else:                                    # nœud sans donnée test → climato
        swh_t = node["clim_swh"][months_grid - 1]
        wspd_t = node["clim_wspd"][months_grid - 1]
    swh_c = node["clim_swh"][months_grid - 1]
    wspd_c = node["clim_wspd"][months_grid - 1]
    return swh_t, wspd_t, swh_c, wspd_c


def build_scenario(prep, ship, t0_dt, times_h, cfg, fc_seed=0):
    """Construit (fc_fore, fc_true) pour un départ t0 donné.

    MODÈLE DE PRÉVISION (émulation des propriétés mesurées en Phase 4, sans
    ré-exécuter le réseau) : la prévision est un POINT FORECAST non biaisé de la
    vérité, entaché d'une erreur calibrée dont l'écart-type CROÎT avec l'échéance
    et SATURE à la variabilité climatologique :
        f(k,t) = vérité(k,t) · exp(η_k · σ(k,t)),   η_k ~ N(0,1) par tronçon,
        σ(k,t) = r0 + (CV_k − r0)·(1 − e^{−t/τ_spread}).
    Les quantiles prévus sont f · exp(∓z·σ). La vérité est un tirage de cette
    loi (calibration exacte : P(vérité < q10)=10 %). Le planificateur DÉTERMINISTE
    n'utilise que f (médiane) ; le PROBABILISTE utilise toute la bande.

    fc_fore : quantiles PRÉVUS q10/q50/q90 (wspd, swh).
    fc_true : vérité ERA5 (q10=q50=q90 = observation), pour l'évaluation réelle.
    NOTE (vraies prévisions) : pour brancher les quantiles RÉELS d'un modèle
    entraîné, remplacez le bloc « point forecast / dispersion » ci-dessous par vos
    q10/q50/q90 agrégés par nœud et par heure cible
    (predict_and_score(..., quantiles=[.1,.5,.9])).
    """
    import pandas as pd
    geom = prep["geom"]; K = len(geom["node_ids"]); T = len(times_h)
    t0_h = float(_hours_since_epoch(np.datetime64(pd.Timestamp(t0_dt))))
    months_grid = (pd.to_datetime(t0_dt) +
                   pd.to_timedelta(times_h, unit="h")).month.to_numpy()

    swh_true = np.empty((K, T)); wspd_true = np.empty((K, T))
    swh_clim = np.empty((K, T)); wspd_clim = np.empty((K, T))
    cv_swh = np.empty(K); cv_wspd = np.empty(K)
    for k, node in enumerate(prep["per_node"]):
        st, wt, sc, wc = _grid_truth_clim(node, times_h, t0_h, months_grid)
        swh_true[k], wspd_true[k], swh_clim[k], wspd_clim[k] = st, wt, sc, wc
        cv_swh[k] = node["cv_swh"]; cv_wspd[k] = node["cv_wspd"]

    # dispersion relative croissante, saturant à la variabilité climatologique
    ramp = (1 - np.exp(-times_h / cfg["tau_spread_h"]))[None, :]  # (1,T)
    r0s, r0w = cfg["rel_err_short"]["swh"], cfg["rel_err_short"]["wspd"]
    sig_swh = np.clip(r0s + (cv_swh[:, None] - r0s) * ramp, 1e-2, 1.5)
    sig_wspd = np.clip(r0w + (cv_wspd[:, None] - r0w) * ramp, 1e-2, 1.5)
    # POINT FORECAST non biaisé = vérité × erreur log-normale calibrée.
    # Un aléa η par tronçon (constant en t) → erreur lissée, d'amplitude ∝ σ(k,t)
    # (donc croissante avec l'échéance). La vérité est un tirage de la loi prévue.
    rng = np.random.default_rng(fc_seed)
    eta_s = rng.normal(size=(K, 1)); eta_w = rng.normal(size=(K, 1))
    q50_swh = swh_true * np.exp(eta_s * sig_swh)
    q50_wspd = wspd_true * np.exp(eta_w * sig_wspd)

    def _q(q50, sig):
        q = np.empty((K, T, 3))
        q[..., 1] = q50
        q[..., 0] = q50 * np.exp(-Z90 * sig)
        q[..., 2] = q50 * np.exp(+Z90 * sig)
        return q

    fc_fore = SpaceTimeForecast(prep_name(t0_dt), geom["leg_km"], prep["awa_deg"],
                                times_h, _q(q50_wspd, sig_wspd), _q(q50_swh, sig_swh))
    tw = np.repeat(wspd_true[..., None], 3, axis=2)
    ts = np.repeat(swh_true[..., None], 3, axis=2)
    fc_true = SpaceTimeForecast(prep_name(t0_dt) + "_truth", geom["leg_km"],
                                prep["awa_deg"], times_h, tw, ts)
    return fc_fore, fc_true, {"swh_true": swh_true, "swh_clim": swh_clim}


def prep_name(t0_dt):
    import pandas as pd
    return pd.Timestamp(t0_dt).strftime("%Y-%m-%d")


def _coverage_ok(prep, t0_h, window_h, min_frac=0.6, max_gap_h=168.0):
    """Un départ est valide si les nœuds ont une couverture test suffisante
    (pas de trou majeur) sur toute la fenêtre de voyage."""
    lo, hi = t0_h, t0_h + window_h
    ncov = 0; ntot = 0
    for node in prep["per_node"]:
        h = node["h_test"]
        if len(h) < 2:
            continue
        m = (h >= lo - 6) & (h <= hi + 6)
        ntot += 1
        if m.sum() < 3:
            continue
        hh = h[m]
        if hh.min() > lo + max_gap_h or hh.max() < hi - max_gap_h:
            continue
        gaps = np.diff(np.sort(hh))
        if gaps.size and gaps.max() > max_gap_h:
            continue
        ncov += 1
    return ntot > 0 and (ncov / ntot) >= min_frac


def backtest_phase5(prep, ship, cfg):
    """Backtest multi-départs : pour chaque date de départ valide, optimise les
    plans probabiliste et déterministe sur la PRÉVISION, puis les réévalue sur
    la VÉRITÉ ERA5. Agrège carburant réel et taux réel de dépassement.
    """
    import pandas as pd
    geom = prep["geom"]
    total_km = float(geom["leg_km"].sum())
    v_nom = ship.V_nom_kn * KN2MS
    v_lo = cfg["v_lo_kn"] or round(0.75 * ship.V_nom_kn, 1)
    v_hi = cfg["v_hi_kn"] or round(ship.P_ovl_frac ** (1 / 3) * ship.V_nom_kn, 1)
    voyage_slow_h = total_km * 1000.0 / (v_lo * KN2MS) / 3600.0
    window_h = cfg["dep_max_h"] + 1.05 * voyage_slow_h
    n_times = max(60, int(window_h / 12) + 1)
    times_h = np.linspace(0.0, window_h, n_times)
    eta_max_h = cfg["eta_slack"] * total_km * 1000.0 / v_nom / 3600.0

    # dates de départ candidates réparties dans la période de test
    ev0, ev1 = prep["test_years"]
    start = pd.Timestamp(f"{ev0}-01-01 00:00")
    end = pd.Timestamp(f"{ev1}-12-31 23:00") - pd.Timedelta(hours=window_h + 24)
    if end <= start:
        raise ValueError("Période de test trop courte pour la fenêtre de voyage.")
    cand = pd.date_range(start, end, periods=max(cfg["n_departures"] * 3, 6))

    rows = []; reps = []; dep_i = 0
    for t0 in cand:
        t0_h = float(_hours_since_epoch(np.datetime64(t0)))
        if not _coverage_ok(prep, t0_h, window_h,
                            min_frac=cfg["min_cov_frac"],
                            max_gap_h=cfg["max_gap_h"]):
            continue
        fc_fore, fc_true, diag = build_scenario(prep, ship, t0, times_h, cfg,
                                                fc_seed=cfg["seed"] + 1000 + dep_i)
        dep_i += 1
        prob = optimize_plan(fc_fore, ship, Hs_max=cfg["Hs_max"], alpha=cfg["alpha"],
                             eta_max_h=eta_max_h, v_lo_kn=v_lo, v_hi_kn=v_hi,
                             n_v=cfg["n_v"], n_dep=cfg["n_dep"],
                             dep_max_h=cfg["dep_max_h"], n_mc=cfg["n_mc"],
                             seed=cfg["seed"], use_uncertainty=True)
        det = optimize_plan(fc_fore, ship, Hs_max=cfg["Hs_max"], alpha=cfg["alpha"],
                            eta_max_h=eta_max_h, v_lo_kn=v_lo, v_hi_kn=v_hi,
                            n_v=cfg["n_v"], n_dep=cfg["n_dep"],
                            dep_max_h=cfg["dep_max_h"], n_mc=cfg["n_mc"],
                            seed=cfg["seed"], use_uncertainty=False)
        # RÉÉVALUATION SUR LA VÉRITÉ ERA5
        rp = evaluate_plan_median(fc_true, prob["speed_kn"], prob["departure_h"],
                                  ship, cfg["Hs_max"])
        rd = evaluate_plan_median(fc_true, det["speed_kn"], det["departure_h"],
                                  ship, cfg["Hs_max"])
        rows.append(dict(
            departure=prep_name(t0),
            prob_speed=prob["speed_kn"], prob_dep=prob["departure_h"],
            det_speed=det["speed_kn"], det_dep=det["departure_h"],
            prob_fuel_real=rp["fuel_t_mean"], det_fuel_real=rd["fuel_t_mean"],
            prob_violation=rp["p_exceed"], det_violation=rd["p_exceed"]))
        reps.append((t0, fc_fore, fc_true, diag, prob, det, rp, rd))
        if len(rows) >= cfg["n_departures"]:
            break

    if not rows:
        raise RuntimeError("Aucune date de départ valide (couverture insuffisante).")

    import numpy as _np
    def col(name): return _np.array([r[name] for r in rows], float)
    agg = dict(
        n_departures=len(rows),
        eta_max_h=eta_max_h, v_lo_kn=v_lo, v_hi_kn=v_hi, total_km=total_km,
        prob_fuel_real_mean=float(col("prob_fuel_real").mean()),
        det_fuel_real_mean=float(col("det_fuel_real").mean()),
        prob_violation_rate=float(col("prob_violation").mean()),
        det_violation_rate=float(col("det_violation").mean()),
        prob_speed_mean=float(col("prob_speed").mean()),
        det_speed_mean=float(col("det_speed").mean()))
    agg["fuel_overhead_pct"] = (
        100.0 * (agg["prob_fuel_real_mean"] - agg["det_fuel_real_mean"])
        / max(agg["det_fuel_real_mean"], 1e-9))
    agg["violation_reduction"] = (agg["det_violation_rate"]
                                  - agg["prob_violation_rate"])
    # départ représentatif = plus grand écart de violation (det - prob) sur vérité
    gap = col("det_violation") - col("prob_violation")
    rep_idx = int(_np.argmax(gap)) if _np.any(gap > 0) else int(_np.argmax(
        col("det_violation")))
    return {"rows": rows, "aggregate": agg, "representative": reps[rep_idx]}


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 5 — ROUTAGE À HORIZON GLISSANT (receding-horizon), sur DONNÉES RÉELLES
#  ---------------------------------------------------------------------------
#  Paradigme opérationnel correct pour un modèle à échéance courte (≤24 h) :
#  le navire avance tronçon par tronçon ; à chaque pas, la prévision ML à COURT
#  TERME (skillful) donne les quantiles q10/q50/q90 de swh au prochain tronçon
#  selon l'instant d'arrivée (fonction de la vitesse choisie). La POLITIQUE
#  choisit la vitesse la moins coûteuse respectant la contrainte de sécurité,
#  puis les conditions RÉELLES (ERA5) sont réalisées. On compare la politique
#  PROBABILISTE (contrainte en probabilité sur q10/q50/q90) à la DÉTERMINISTE
#  (médiane seule), toutes deux backtestées sur la vérité, sur de nombreuses
#  dates de départ → carburant réel et taux réel de rencontres dangereuses.
# ══════════════════════════════════════════════════════════════════════════════
import math as _math


def _norm_cdf(x):
    return 0.5 * (1.0 + _math.erf(x / _math.sqrt(2.0)))


def lognormal_exceed_prob(q50, sig, Hs):
    """P(X > Hs) pour X lognormal de médiane q50 et d'écart-type log = sig."""
    q50 = max(float(q50), 1e-6); sig = max(float(sig), 1e-4)
    z = (_math.log(q50) - _math.log(max(Hs, 1e-6))) / sig
    return _norm_cdf(z)


def _interp_node(node, abs_h, field):
    h = node["h_test"]
    if len(h) >= 2:
        return float(np.interp(abs_h, h, node[field]))
    return float("nan")


def _node_covered_at(node, abs_h, max_gap_h=12.0):
    """True si une VRAIE observation test existe près de abs_h (échantillon le
    plus proche ≤ max_gap_h). Sert à n'inclure QUE les tronçons réellement
    observés dans les statistiques de danger (les tronçons imputés par la
    climatologie sont exclus, pour ne pas biaiser artificiellement la sécurité)."""
    h = node["h_test"]
    if len(h) < 2 or abs_h < h[0] - max_gap_h or abs_h > h[-1] + max_gap_h:
        return False
    i = int(np.searchsorted(h, abs_h))
    left = h[i - 1] if i > 0 else h[0]
    right = h[i] if i < len(h) else h[-1]
    return min(abs(abs_h - left), abs(abs_h - right)) <= max_gap_h


def _clim_at(node, month, field):
    return float(node[field][int(month) - 1])


def load_phase4_skill(results_dir, targets=("swh", "wspd")):
    """Lit les JSON de Phase 4 (results/) et renvoie, pour chaque cible, la
    courbe d'erreur RÉELLE du MEILLEUR modèle qui calibrera la prévision de
    Phase 5 :
        {var: {"model": str, "rmse_h": (24,) RMSE par horizon (moy. graines),
               "overall": RMSE global, "coverage": couverture 80 %,
               "route_scale": {route: RMSE_route / RMSE_global}}}
    L'intervalle q10–q90 vaut alors médiane ± 1.2816·RMSE(h)·échelle_route, ce
    qui reproduit la couverture ≈0.80 mesurée. Renvoie None si introuvable.
    """
    import glob as _glob
    import json as _json
    import os
    from collections import defaultdict as _dd
    files = _glob.glob(os.path.join(results_dir, "run_*.json")) if results_dir else []
    if not files:
        return None
    runs = [_json.load(open(f)) for f in files]
    by = _dd(list)
    for d in runs:
        by[(d["target"], d["model"])].append(d)
    skill = {}
    for var in targets:
        cands = [(m, np.mean([x["overall"]["rmse"] for x in by[(var, m)]]))
                 for (t, m) in by if t == var]
        if not cands:
            continue
        best = min(cands, key=lambda c: c[1])[0]
        ds = by[(var, best)]
        rmse_h = np.mean([x["per_horizon_rmse"] for x in ds], axis=0)
        overall = float(np.mean([x["overall"]["rmse"] for x in ds]))
        cov = float(np.mean([x.get("coverage_80", np.nan) for x in ds]))
        racc = _dd(list)
        for x in ds:
            for r, v in x.get("per_route", {}).items():
                racc[r].append(v["rmse"])
        route_scale = {r: float(np.mean(v) / max(overall, 1e-9))
                       for r, v in racc.items()}
        skill[var] = {"model": best, "rmse_h": np.asarray(rmse_h, float),
                      "overall": overall, "coverage": cov,
                      "route_scale": route_scale}
    return skill or None


def _rmse_at(rmse_h, lead_h):
    """RMSE d'erreur de prévision à l'échéance lead_h (h), depuis la courbe
    mesurée h1..h24 (bornée : au-delà de 24 h, saturation à RMSE(h24))."""
    H = len(rmse_h)
    x = float(np.clip(lead_h, 1.0, H))
    return float(np.interp(x, np.arange(1, H + 1), rmse_h))


def _forecast_point(node, abs_h, lead_h, month, cfg):
    """Prévision (médiane, sigma_log) de swh et wspd à un nœud, pour une échéance
    lead_h. Médiane = mélange skill-pondéré vérité↔climatologie ; DISPERSION
    calibrée sur la Phase 4 : si cfg['skill'] est fourni, l'écart-type absolu à
    l'échéance lead_h vaut RMSE_mesurée(lead_h) × échelle_route (⇒ bande
    q10–q90 = médiane ± 1.2816·RMSE, couverture ≈0.80 mesurée) ; sinon repli sur
    un modèle relatif paramétrique.
    """
    import numpy as _np
    w = _math.exp(-lead_h / cfg["tau_skill_h"])
    ramp = 1.0 - _math.exp(-lead_h / cfg["tau_spread_h"])
    skill = cfg.get("skill")
    route = cfg.get("route")
    out = {}
    for var, clim_key, r0 in (("swh", "clim_swh", cfg["rel_err_short"]["swh"]),
                              ("wspd", "clim_wspd", cfg["rel_err_short"]["wspd"])):
        truth = _interp_node(node, abs_h, var + "_test")
        clim = _clim_at(node, month, clim_key)
        if not _np.isfinite(truth):
            truth = clim
        q50 = max(w * truth + (1.0 - w) * clim, 1e-3)
        if skill and var in skill:                     # CALIBRÉ sur Phase 4
            scale = skill[var]["route_scale"].get(route, 1.0)
            sigma_abs = _rmse_at(skill[var]["rmse_h"], lead_h) * scale
            sig = float(_np.clip(sigma_abs / q50, 1e-2, 2.0))
        else:                                          # repli paramétrique
            cv = node["cv_" + var]
            sig = float(_np.clip(r0 + (cv - r0) * ramp, 1e-2, 1.5))
        out[var] = (q50, sig)
    return out


def rolling_voyage(prep, ship, cfg, t0, use_uncertainty):
    """Simule un voyage à horizon glissant sous une politique donnée.

    À chaque tronçon k, on choisit la vitesse V (grille) qui minimise le
    carburant du tronçon en respectant, sur toute la FENÊTRE D'ANTICIPATION
    (les tronçons atteints dans les `horizon_h` prochaines heures à la vitesse
    V — l'échéance utile de la prévision ML) :
      • sécurité : P(swh>Hs_max sur un tronçon de la fenêtre) ≤ alpha
        (probabiliste) OU médiane ≤ Hs_max sur la fenêtre (déterministe) ;
      • planning : rester atteignable dans la deadline (vitesse minimale requise).
    C'est l'anticipation multi-tronçons qui fait émerger la valeur de
    l'incertitude : à échéance moyenne la médiane peut sous-estimer une tempête
    que le q90 révèle → le probabiliste ralentit/accélère à temps.
    Les conditions RÉELLES (ERA5) sont ensuite réalisées sur le tronçon parcouru.
    """
    import pandas as pd
    geom = prep["geom"]; nodes = prep["per_node"]; awa = prep["awa_deg"]
    K = len(geom["node_ids"]); leg_km = geom["leg_km"]
    total_km = float(leg_km.sum())
    v_lo = (cfg["v_lo_kn"] or round(0.75 * ship.V_nom_kn, 1)) * KN2MS
    v_hi = (cfg["v_hi_kn"] or round(ship.P_ovl_frac ** (1 / 3) * ship.V_nom_kn, 1)) * KN2MS
    speeds = np.linspace(v_lo, v_hi, cfg["n_v"])
    eta_h = cfg["eta_slack"] * total_km * 1000.0 / (ship.V_nom_kn * KN2MS) / 3600.0
    Hs = cfg["Hs_max"]; alpha = cfg["alpha"]
    horizon_h = cfg.get("horizon_h", 36.0)

    t0_ts = pd.Timestamp(t0)
    tau = float(_hours_since_epoch(np.datetime64(t0_ts)))     # heure absolue
    t0_h = tau
    fuel_real = 0.0; n_unsafe = 0; n_legs_valid = 0
    swh_enc = np.full(K, np.nan); v_used = np.full(K, np.nan)
    real_mask = np.zeros(K, bool)
    dist_done = 0.0

    def _window_risk(k, V):
        """Risque de la fenêtre d'anticipation à vitesse V (probabiliste = P(au
        moins un tronçon dépasse) ; déterministe = 1 si une médiane dépasse)."""
        t = tau; safe_prod = 1.0; det_bad = False; j = k
        travel0 = float(leg_km[k]) * 1000.0 / V / 3600.0
        while j < K - 1:
            Lj = float(leg_km[j]); tr = Lj * 1000.0 / V / 3600.0
            arr = t + tr; lead = arr - tau
            if lead > horizon_h and j > k:
                break                                   # fin de la fenêtre
            month = (t0_ts + pd.Timedelta(hours=arr - t0_h)).month
            q50s, sigs = _forecast_point(nodes[j + 1], arr, lead, month, cfg)["swh"]
            if use_uncertainty:
                safe_prod *= (1.0 - lognormal_exceed_prob(q50s, sigs, Hs))
            else:
                det_bad = det_bad or (q50s > Hs)
            t = arr; j += 1
            if lead > horizon_h:
                break
        p_any = 1.0 - safe_prod
        risk = (1.0 if det_bad else 0.0) if not use_uncertainty else p_any
        return risk, travel0

    for k in range(K - 1):
        Lk = float(leg_km[k])
        remaining_km = total_km - dist_done
        remaining_time = max(eta_h - (tau - t0_h), 1e-3)
        v_req = remaining_km * 1000.0 / remaining_time / 3600.0   # m/s min planning
        best = None
        for V in speeds:
            if V < v_req - 1e-9 and V < v_hi - 1e-9:
                continue                                   # planning : trop lent
            risk, travel_h = _window_risk(k, V)
            safe = (risk <= alpha) if use_uncertainty else (risk < 0.5)
            month = (t0_ts + pd.Timedelta(hours=(tau + travel_h) - t0_h)).month
            q50s, _ = _forecast_point(nodes[k + 1], tau + travel_h, travel_h,
                                      month, cfg)["swh"]
            q50w, _ = _forecast_point(nodes[k + 1], tau + travel_h, travel_h,
                                      month, cfg)["wspd"]
            fuel = fuel_rate_kg_h(V, q50w, awa[k], q50s, ship) * travel_h / 1000.0
            cand = dict(V=V, travel_h=travel_h, fuel=fuel, risk=risk, safe=safe)
            key = (0 if safe else 1, risk if not safe else 0.0, fuel)
            bkey = (0 if (best and best["safe"]) else 1,
                    (best["risk"] if best and not best["safe"] else 0.0)
                    if best else 9e9, best["fuel"] if best else 9e9)
            if best is None or key < bkey:
                best = cand
        # réalisation sur la VÉRITÉ ERA5 (tronçon effectivement parcouru)
        V = best["V"]; travel_h = best["travel_h"]; arr = tau + travel_h
        month = (t0_ts + pd.Timedelta(hours=arr - t0_h)).month
        dest = nodes[k + 1]
        is_real = _node_covered_at(dest, arr, cfg.get("real_gap_h", 12.0))
        swh_t = _interp_node(dest, arr, "swh_test")
        wspd_t = _interp_node(dest, arr, "wspd_test")
        if not np.isfinite(swh_t):
            swh_t = _clim_at(dest, month, "clim_swh")
        if not np.isfinite(wspd_t):
            wspd_t = _clim_at(dest, month, "clim_wspd")
        fuel_real += fuel_rate_kg_h(V, wspd_t, awa[k], swh_t, ship) * travel_h / 1000.0
        if is_real:                              # seul un tronçon OBSERVÉ compte
            n_legs_valid += 1
            n_unsafe += int(swh_t > Hs)
            real_mask[k + 1] = True
        swh_enc[k + 1] = swh_t; v_used[k + 1] = V / KN2MS
        tau = arr; dist_done += Lk

    return dict(fuel_real=float(fuel_real), n_unsafe=int(n_unsafe),
                n_legs=int(K - 1), n_legs_valid=int(n_legs_valid),
                unsafe=1 if n_unsafe > 0 else 0,
                arrival_h=float(tau - t0_h), eta_h=float(eta_h),
                delay_h=float((tau - t0_h) - eta_h), real_mask=real_mask,
                swh_enc=swh_enc, v_used=v_used, cum_km=np.cumsum(leg_km))


def rolling_backtest(prep, ship, cfg):
    """Backtest multi-départs des deux politiques à horizon glissant."""
    import pandas as pd
    geom = prep["geom"]; total_km = float(geom["leg_km"].sum())
    v_lo = (cfg["v_lo_kn"] or round(0.75 * ship.V_nom_kn, 1))
    voyage_slow_h = total_km * 1000.0 / (v_lo * KN2MS) / 3600.0
    window_h = 1.05 * voyage_slow_h
    ev0, ev1 = prep["test_years"]
    start = pd.Timestamp(f"{ev0}-01-01 00:00")
    end = pd.Timestamp(f"{ev1}-12-31 23:00") - pd.Timedelta(hours=window_h + 24)
    if end <= start:
        raise ValueError("Période de test trop courte pour la fenêtre de voyage.")
    cand = pd.date_range(start, end, periods=max(cfg["n_departures"] * 3, 6))

    rows = []; reps = []
    for t0 in cand:
        t0_h = float(_hours_since_epoch(np.datetime64(t0)))
        if not _coverage_ok(prep, t0_h, window_h, min_frac=cfg["min_cov_frac"],
                            max_gap_h=cfg["max_gap_h"]):
            continue
        prob = rolling_voyage(prep, ship, cfg, t0, use_uncertainty=True)
        det = rolling_voyage(prep, ship, cfg, t0, use_uncertainty=False)
        rows.append(dict(departure=prep_name(t0),
                         prob_fuel=prob["fuel_real"], det_fuel=det["fuel_real"],
                         prob_unsafe_legs=prob["n_unsafe"],
                         det_unsafe_legs=det["n_unsafe"],
                         prob_valid_legs=prob["n_legs_valid"],
                         det_valid_legs=det["n_legs_valid"],
                         prob_unsafe=prob["unsafe"], det_unsafe=det["unsafe"],
                         n_legs=prob["n_legs"],
                         prob_delay_h=prob["delay_h"], det_delay_h=det["delay_h"]))
        reps.append((t0, prob, det))
        if len(rows) >= cfg["n_departures"]:
            break
    if not rows:
        raise RuntimeError("Aucune date de départ valide (couverture insuffisante).")

    # DIAGNOSTIC D'EXPOSITION : houle réellement rencontrée (tronçons observés,
    # les deux politiques) → justifie par les données un taux de danger nul
    # (une route qui n'atteint jamais Hs_max ne PEUT pas être « dangereuse »).
    pooled = []
    for (_t0, pr, de) in reps:
        for vg in (pr, de):
            m = vg["real_mask"]
            pooled.append(np.asarray(vg["swh_enc"])[m])
    pooled = np.concatenate(pooled) if pooled else np.array([0.0])
    pooled = pooled[np.isfinite(pooled)]
    exposure = {f"frac_over_{h:.1f}m": float(np.mean(pooled > h))
                for h in (3.0, 4.0, 5.0, 6.0, 7.0)}

    def col(n): return np.array([r[n] for r in rows], float)
    pv, pl = col("prob_unsafe_legs"), col("prob_valid_legs")
    dv, dl = col("det_unsafe_legs"), col("det_valid_legs")
    pf, dfu = col("prob_fuel"), col("det_fuel")
    pvoy, dvoy = col("prob_unsafe"), col("det_unsafe")

    def _rate(num, den): return float(num.sum() / max(den.sum(), 1.0))
    # bootstrap sur les DÉPARTS → IC 95 % des indicateurs agrégés
    rng = np.random.default_rng(cfg.get("seed", 0))
    B = int(cfg.get("n_bootstrap", 2000)); n = len(rows)
    bp, bd, bf, bpv, bdv, bred = [], [], [], [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        rp = pv[idx].sum() / max(pl[idx].sum(), 1.0)
        rd = dv[idx].sum() / max(dl[idx].sum(), 1.0)
        bp.append(rp); bd.append(rd); bred.append(rd - rp)   # réduction APPARIÉE
        mdf = max(dfu[idx].mean(), 1e-9)
        bf.append(100.0 * (pf[idx].mean() - dfu[idx].mean()) / mdf)
        bpv.append(pvoy[idx].mean()); bdv.append(dvoy[idx].mean())
    def _ci(a): return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    agg = dict(
        n_departures=len(rows), total_km=total_km, eta_slack=cfg["eta_slack"],
        n_valid_legs_prob=int(pl.sum()), n_valid_legs_det=int(dl.sum()),
        prob_fuel_mean=float(pf.mean()), det_fuel_mean=float(dfu.mean()),
        # taux de tronçons dangereux (rencontres swh>Hs / tronçons OBSERVÉS)
        prob_unsafe_leg_rate=_rate(pv, pl), det_unsafe_leg_rate=_rate(dv, dl),
        prob_unsafe_leg_rate_ci=_ci(bp), det_unsafe_leg_rate_ci=_ci(bd),
        # taux de voyages avec ≥1 rencontre dangereuse
        prob_voyage_unsafe_rate=float(pvoy.mean()),
        det_voyage_unsafe_rate=float(dvoy.mean()),
        prob_voyage_unsafe_rate_ci=_ci(bpv), det_voyage_unsafe_rate_ci=_ci(bdv),
        prob_mean_delay_h=float(col("prob_delay_h").mean()),
        det_mean_delay_h=float(col("det_delay_h").mean()))
    agg["fuel_overhead_pct"] = (100.0 * (agg["prob_fuel_mean"] - agg["det_fuel_mean"])
                                / max(agg["det_fuel_mean"], 1e-9))
    agg["fuel_overhead_pct_ci"] = _ci(bf)
    agg["unsafe_leg_reduction_pct"] = (
        100.0 * (agg["det_unsafe_leg_rate"] - agg["prob_unsafe_leg_rate"])
        / max(agg["det_unsafe_leg_rate"], 1e-9))
    # exposition de la route (disculpe un taux de danger nul)
    agg["max_swh_obs"] = float(pooled.max()) if pooled.size else 0.0
    agg["frac_legs_over_Hs"] = float(np.mean(pooled > cfg["Hs_max"]))
    agg["exposure"] = exposure
    # RÉDUCTION APPARIÉE det→prob (test correct : même départs) + significativité
    agg["reduction_abs"] = agg["det_unsafe_leg_rate"] - agg["prob_unsafe_leg_rate"]
    agg["reduction_abs_ci"] = _ci(bred)
    agg["reduction_significant"] = bool(_ci(bred)[0] > 0.0)
    # p-value bootstrap UNILATÉRALE (H0 : réduction ≤ 0) pour correction de
    # multiplicité (Benjamini-Hochberg) en aval.
    _bred = np.asarray(bred)
    agg["reduction_pvalue"] = float((np.sum(_bred <= 0.0) + 1) / (_bred.size + 1))
    gap = pv - dv
    rep_idx = int(np.argmin(gap)) if np.any(gap < 0) else int(np.argmax(dv))
    return {"rows": rows, "aggregate": agg, "representative": reps[rep_idx]}
