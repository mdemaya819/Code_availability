"""
route_config.py
===============
Configuration centralisée des 4 routes commerciales et de leurs
boîtes géographiques ERA5.

Utilisé par tous les scripts de téléchargement et de traitement.

Auteur : Likeufack Mdemaya F.A. et al.
         Jiangsu University of Science and Technology
Article: Decadal ERA5-Coupled Probabilistic Assessment of Surrogate-Adaptive
         Flettner Rotor Performance — Ocean Engineering (soumission)
"""

# ── Définition des routes ─────────────────────────────────────────────────────
# Chaque route est décrite par :
#   id       : identifiant court
#   name     : nom complet
#   distance : distance approximative en km
#   speed_kn : vitesse de service en nœuds
#   P_nom_kW : puissance nominale du moteur principal en kW
#   SFOC     : consommation spécifique en g/kWh (HFO)
#   CO2f     : facteur d'émission CO2 en tCO2/t_fuel (MARPOL Annex VI)
#   DWT      : port en lourd en tonnes (pour calcul CII)
#   ship_type: type de navire (pour référence CII MEPC.337(76))
#   waypoints: liste de (lat, lon, nom_segment)
#   boxes    : boîtes géographiques ERA5 [N, W, S, E]
#              (une route peut nécessiter plusieurs boîtes)

ROUTES = {

    # ──────────────────────────────────────────────────────────────────────────
    "R1_SHA_RTM": {
        "id":         "R1_SHA_RTM",
        "name":       "Shanghai → Rotterdam (Trans-Eurasian, via Suez)",
        "distance_km": 19500,
        "speed_kn":    12.0,
        "P_nom_kW":    12000,
        "SFOC_gkWh":   180,
        "CO2f":        3.17,
        "DWT":         50000,
        "ship_type":   "container",
        "waypoints": [
            (31.40, 121.90, "Shanghai departure"),
            (25.00, 123.00, "East China Sea"),
            (22.30, 114.20, "South China Sea North"),
            (13.00, 109.50, "South China Sea South"),
            ( 1.25, 103.82, "Singapore / Malacca East"),
            (-1.50, 104.50, "Malacca West"),
            ( 5.50,  98.70, "Andaman Sea"),
            ( 8.00,  87.00, "Indian Ocean East"),
            ( 5.00,  73.00, "Indian Ocean Central"),
            (11.00,  51.00, "Gulf of Aden"),
            (12.75,  43.60, "Red Sea South"),
            (22.00,  37.20, "Red Sea Central"),
            (29.90,  32.57, "Suez Canal"),
            (31.20,  32.30, "Mediterranean East"),
            (35.00,  23.00, "Mediterranean Central"),
            (37.00,  11.00, "Mediterranean West"),
            (36.00,  -5.60, "Strait of Gibraltar"),
            (43.00, -10.00, "Bay of Biscay"),
            (47.50,  -4.50, "Bay of Biscay North"),
            (51.30,   2.90, "English Channel"),
            (52.00,   4.50, "North Sea"),
            (51.92,   4.50, "Rotterdam"),
        ],
        # Deux boîtes : Asie/Océan Indien + Mer Rouge/Europe
        "boxes": [
            {"suffix": "Asia_IO",    "area": [35,  95, -5,  130]},
            {"suffix": "RedSea_EU",  "area": [60, -15, 10,   50]},
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    "R2_NFK_HAM": {
        "id":         "R2_NFK_HAM",
        "name":       "Norfolk → Hamburg (North Atlantic)",
        "distance_km": 9200,
        "speed_kn":    12.0,
        "P_nom_kW":    8000,
        "SFOC_gkWh":   185,
        "CO2f":        3.17,
        "DWT":         35000,
        "ship_type":   "container",
        "waypoints": [
            (36.90, -76.30, "Norfolk departure"),
            (38.00, -70.00, "US Eastern Seaboard"),
            (40.00, -60.00, "Mid-Atlantic West"),
            (42.00, -45.00, "Mid-Atlantic Central"),
            (45.00, -35.00, "Mid-Atlantic East"),
            (48.00, -25.00, "Azores region"),
            (50.00, -15.00, "Eastern Atlantic"),
            (52.00,  -5.00, "Celtic Sea"),
            (54.00,   3.00, "North Sea South"),
            (53.55,   9.97, "Hamburg"),
        ],
        "boxes": [
            {"suffix": "NAtlantic", "area": [62, -80, 30, 15]},
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    "R3_SIN_SYD": {
        "id":         "R3_SIN_SYD",
        "name":       "Singapore → Sydney (Indo-Pacific)",
        "distance_km": 8300,
        "speed_kn":    12.0,
        "P_nom_kW":    8000,
        "SFOC_gkWh":   185,
        "CO2f":        3.17,
        "DWT":         35000,
        "ship_type":   "container",
        "waypoints": [
            ( 1.25, 103.82, "Singapore"),
            (-5.00, 108.00, "Java Sea"),
            (-8.60, 115.20, "Bali Strait"),
            (-9.50, 130.00, "Timor Sea"),
            (-10.00,141.00, "Torres Strait approach"),
            (-15.00,147.00, "Coral Sea North"),
            (-20.00,152.00, "Coral Sea Central"),
            (-28.00,154.00, "Tasman Sea North"),
            (-33.87,151.21, "Sydney"),
        ],
        "boxes": [
            {"suffix": "IndoPacific", "area": [10, 100, -38, 160]},
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    "R4_SHA_LAX": {
        "id":         "R4_SHA_LAX",
        "name":       "Shanghai → Los Angeles (Trans-Pacific)",
        "distance_km": 10300,
        "speed_kn":    12.0,
        "P_nom_kW":    10000,
        "SFOC_gkWh":   182,
        "CO2f":        3.17,
        "DWT":         45000,
        "ship_type":   "container",
        "waypoints": [
            (31.40, 121.90, "Shanghai departure"),
            (32.00, 135.00, "East China Sea"),
            (35.00, 145.00, "NW Pacific"),
            (40.00, 155.00, "North Pacific West"),
            (42.00, 165.00, "North Pacific Central-West"),
            (43.00, 175.00, "North Pacific Central"),
            (42.00,-175.00, "North Pacific Central-East"),
            (40.00,-165.00, "North Pacific East"),
            (37.00,-150.00, "NE Pacific"),
            (35.00,-135.00, "NE Pacific South"),
            (33.73,-118.26, "Los Angeles"),
        ],
        # Trans-Pacific nécessite deux boîtes (traversée de 180°)
        "boxes": [
            {"suffix": "WPacific", "area": [50, 115,  25,  180]},
            {"suffix": "EPacific", "area": [50,-180,  25, -110]},
        ],
    },
}

# ── Paramètres ERA5 communs ───────────────────────────────────────────────────
ERA5_CONFIG = {
    "product_type": "reanalysis",

    # Variables obligatoires (vent)
    "vars_wind": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ],

    # Variables complémentaires
    "vars_complementary": [
        "mean_sea_level_pressure",       # densité de l'air variable
        "2m_temperature",                # densité de l'air variable
        "significant_height_of_combined_wind_waves_and_swell",  # routage
    ],

    # Toutes les variables (wind + complémentaires)
    "vars_all": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
        "2m_temperature",
        "significant_height_of_combined_wind_waves_and_swell",
    ],

    # Résolution spatiale 0.25° × 0.25° (native ERA5)
    "grid": [0.25, 0.25],

    # Format de sortie
    "format": "netcdf",

    # Heures : toutes les heures (0h–23h)
    "hours": [
        "00:00", "01:00", "02:00", "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00", "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00", "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00", "21:00", "22:00", "23:00",
    ],

    # Jours : tous les jours (31 jours max, ERA5 gère automatiquement les mois courts)
    "days": [f"{d:02d}" for d in range(1, 32)],

    # Période décennale
    "years":  [str(y) for y in range(2015, 2025)],   # 2015–2024
    "months": [f"{m:02d}" for m in range(1, 13)],     # Jan–Déc
}

# ── Correspondances ENSO ──────────────────────────────────────────────────────
ENSO_THRESHOLDS = {
    "el_nino":  +0.5,   # ONI ≥ +0.5°C pendant ≥ 5 mois consécutifs
    "la_nina":  -0.5,   # ONI ≤ -0.5°C pendant ≥ 5 mois consécutifs
    "neutral_lo": -0.5,
    "neutral_hi": +0.5,
}

# ── Constantes physiques ──────────────────────────────────────────────────────
PHYSICS = {
    "R_dry":  287.05,   # constante gaz sec [J/(kg·K)]
    "rho_std": 1.225,   # densité air standard [kg/m³]
    "A_rotor": 30.0,    # surface de référence rotor [m²] (H=10m, D=3m)
    "D_full":   3.0,    # diamètre pleine échelle [m]
    "H_full":  10.0,    # hauteur pleine échelle [m]
    "Vship":    6.17,   # vitesse navire [m/s] = 12 nœuds
}
