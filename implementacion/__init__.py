"""
FDCA – GOES-R ABI Fire Detection and Characterization Algorithm
Python implementation based on NOAA NESDIS ATBD v2.7 (October 2020)

Corrected and extended from the original pseudocode, including:
  - Full Part II (temporal filtering, fire categorization)
  - Histogram vs. statistical background selection (ATBD 3.4.2.5)
  - Fixed bisection logarithmic midpoint formula (ATBD eq. 3.1)
  - Proper temperature→radiance conversion before TPW correction
  - Correct std_dev_reflb thresholds (floor/ceiling order)
  - Full std_dev_reflb_max computation
  - FRP formula with MIR constant (ATBD eq. 3.4)
  - Cloud mask 210 (BT7-BT14 > 20 K AND BT7 < 285 K)
  - Out-of-bounds pixels count in background window denominator
  - Solar correction without double-emissivity division

Public API
----------
    from fdca import run_fdca, FDCAInput, FDCAOutput
"""

from .fdca import run_fdca, FDCAInput, FDCAOutput
from .part1 import FireCandidate
from .constants import FireMask, FailChar
from .planck import planck_rad, planck_temp

__all__ = [
    "run_fdca",
    "FDCAInput",
    "FDCAOutput",
    "FireCandidate",
    "FireMask",
    "FailChar",
    "planck_rad",
    "planck_temp",
]
