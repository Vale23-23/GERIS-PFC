"""
FDCA - Fire Detection and Characterization Algorithm
Constants and configuration values based on NOAA ABI ATBD v2.7
"""

import numpy as np

# ── Physical constants ──────────────────────────────────────────────────────
H_PLANCK   = 6.62607015e-34   # Planck constant [J·s]
K_BOLTZ    = 1.380649e-23     # Boltzmann constant [J/K]
C_LIGHT    = 299_792_458.0    # Speed of light [m/s]
SIGMA_SB   = 5.67e-8          # Stefan-Boltzmann [W·m⁻²·K⁻⁴]

# ── ABI band central wavelengths [m] ────────────────────────────────────────
LAMBDA = {
    2:  0.64e-6,
    7:  3.90e-6,
    13: 10.35e-6,
    14: 11.20e-6,
    15: 12.30e-6,
}

# ── Sensor thresholds ───────────────────────────────────────────────────────
SAT_TEMP_CH7  = 411.86        # Channel 7 saturation temperature [K]
SAT_TEMP_CH14 = 340.0         # Channel 14 saturation temperature [K]
SAT_BUFF      = 5.0           # Saturation buffer [K]
FPT_THRESHOLD = 90.0          # Focal Plane Temperature threshold [K]
MIN_BT        = 200.0         # Minimum usable brightness temperature [K]

# Effective saturation thresholds for the "saturated flag" test (sat - 0.1 K)
SAT_FLAG_CH7  = SAT_TEMP_CH7  - 0.10   # 411.76 K
SAT_FLAG_CH14 = SAT_TEMP_CH14 - 0.10   # 339.90 K

# ── Geometry thresholds ─────────────────────────────────────────────────────
MAX_LOCAL_ZENITH   = 80.0     # Max satellite zenith angle [deg]
MAX_SZA_DAYLIGHT   = 85.0     # Solar zenith for daylight flag [deg]
GLINT_THRESHOLD    = 10.0     # Sun glint / sub-solar block-out [deg]


# __ others

CH_DIFF = 2 #2 K 
BRIGHTNESS_THRESHOLD = 273 #K

# ── Cloud thresholds ────────────────────────────────────────────────────────
CLOUD_BT14_THRESH        = 270.0
CLOUD_BT7_BT14_NEG       = -4.0
CLOUD_BT7_BT14_POS       = 20.0      # mask 210 condition
CLOUD_BT7_FOR_POS        = 285.0     # mask 210: T3.9 < 285 when diff > 20
CLOUD_ALBEDO_THRESH       = 0.38
CLOUD_BT15_THRESH        = 265.0
CLOUD_BT14_BT15_NEG      = -4.0
CLOUD_BT14_BT15_POS      = 60.0

# ── Background window ───────────────────────────────────────────────────────
BKG_WINDOW_INIT  = 5          # Initial half-width (11×11)
BKG_WINDOW_STEP  = 5          # Expansion step each iteration
BKG_MAX_ITER     = 10         # Maximum expansions → max 111×111
BKG_VALID_FRAC   = 0.20       # Required fraction of valid pixels

BKG_WARM_BT7_NIGHT   = 310.0
BKG_WARM_SOLAR_COEF  = 25.0
BKG_COLD_THRESH      = 270.0
BKG_MIN_VIS          = 1
BKG_MAX_ALBEDO       = 0.38

# ── Fire threshold base values ───────────────────────────────────────────────
BT7_MIN_NIGHT            = 285.0
BT7_MIN_SOLAR_COEF       = 15.0
BT7_REFL_THRESH_NIGHT    = 315.0
BT7_REFL_THRESH_SOLAR    = 5.0

# ── Corrections ─────────────────────────────────────────────────────────────
CLOUD_ADJ_ALBEDO_LOW    = 0.025
CLOUD_ADJ_ALBEDO_HIGH   = 0.07
CLOUD_ADJ_BT7_COEF      = 10.0
CLOUD_ADJ_BT14_COEF     = 30.0
CLOUD_ADJ_BT7_FIXED     = 0.7
CLOUD_ADJ_BT14_FIXED    = 2.1

DIFFRAC_CH7_SUB  = 0.15       # Diffraction subtraction coef channel 7
DIFFRAC_CH7_DIV  = 0.85
DIFFRAC_CH14_SUB = 0.30
DIFFRAC_CH14_DIV = 0.70

# ── FRP ─────────────────────────────────────────────────────────────────────
# Antes (calibrado para W/m²/sr/µm): FRP_MIR_A = 3.0e-9

# Después (ajustado para W/m²/sr/m, que es lo que devuelve planck_rad):
FRP_MIR_A = 3.0e-3          # MIR approximation constant [W·m⁻²·sr⁻¹·μm⁻¹·K⁻⁴]
MIN_PIXEL_AREA = 4.0          # Minimum pixel area to recompute [km²]

# ── Dozier ───────────────────────────────────────────────────────────────────
DOZIER_P_UPPER    = 1.0
DOZIER_P_LOWER    = 1e-6
DOZIER_BISECT_N   = 15
DOZIER_NEWTON_MAX = 50
# DOZIER_NEWTON_TOL = 1e-20 # Antes, imposible con radiancias ~1e7
# Después (tolerancia relativa al orden de las radiancias):
DOZIER_NEWTON_TOL = 1e-6
MIN_FIRE_TEMP     = 400.0     # Minimum valid fire temperature [K]
MAX_SURF_TEMP     = 350.0     # Maximum surface temperature [K]

# ── Temporal filter ──────────────────────────────────────────────────────────
TEMPORAL_WINDOW_H  = 12       # Hours to look back
TEMPORAL_PIXEL_RAD = 1        # Pixel search radius for temporal match

# ── Ecosystem / land mask codes ──────────────────────────────────────────────
MODIS_WATER_CODES = {7, 6, 0, 3, 5}
UMD_WATER_CODE    = 0
DESERT_BRIGHT     = 2
USGS_SEA_WATER    = 15
USGS_COAST_FRINGE = {80, 85}
USGS_INLAND_WATER = {14, 73, 74, 75}

# ── Fire mask codes (Table 3.11) ─────────────────────────────────────────────
class FireMask:
    INIT           = -99
    NON_PROCESSED  = 0
    PROCESSED      = 10
    SATURATED      = 11
    CLOUD_CONTAM   = 12
    HIGH_PROB      = 13
    MED_PROB       = 14
    LOW_PROB       = 15
    SPACE          = 40
    ZENITH_BLOCK   = 50
    GLINT_BLOCK    = 60
    FIRE_FREE      = 100
    TOO_COLD       = 201
    MISS_CH7       = 120
    MISS_CH14      = 121
    SAT_CH7        = 123
    SAT_CH14       = 124
    NEG_RAD        = 125
    UNUS_CH7       = 126
    UNUS_CH14      = 127
    BAD_ECOSYSTEM  = 150
    SEA_WATER      = 151
    COAST_FRINGE   = 152
    INLAND_WATER   = 153
    NO_BACKGROUND  = 170
    CONV_ERROR     = 180
    CLOUD_BT14     = 200
    CLOUD_BT7_BT14_NEG = 205
    CLOUD_BT7_BT14_POS = 210
    CLOUD_ALBEDO   = 215
    CLOUD_BT15     = 220
    CLOUD_BT14_BT15_NEG = 225
    CLOUD_BT14_BT15_POS = 230
    ALONG_SCAN_NIGHT = 240
    ALONG_SCAN_DAY   = 245

    # Temporally filtered (Part II)
    TEMP_PROCESSED  = 30
    TEMP_SATURATED  = 31
    TEMP_CLOUD      = 32
    TEMP_HIGH       = 33
    TEMP_MED        = 34
    TEMP_LOW        = 35

# ── FailChar codes (Table 3.5) ───────────────────────────────────────────────
class FailChar:
    NONE    = -1
    F1      = 1   # BT7-BT14 within std dev or Refl check failed
    F2      = 2   # BT7-BkgBT7 within std dev or Refl check failed
    F3      = 3   # Adjusted BTs below thresholds
    F4      = 4   # BT14 adj differs from obs by < 0.25 K
    F5      = 5   # BT7 adj < 2 K
    F6      = 6   # Fire temp < 400 K (non-glint)
    F7      = 7   # Saturated pixel
    F8      = 8   # Potential sun glint (albedo > 0.25 or diff > 0.07)
    F9      = 9   # Glint + fire temp < 400 K
    F10     = 10  # BT14 adj < 0.25 K AND cloudy AND BT7 adj > 10 K
    F11     = 11  # Cloud/fog edge (Part II)
