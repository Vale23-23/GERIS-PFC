"""
state.py

Management of the "previous fire mask" state mask (ATBD 3.3.3 / Table 3.10) between successive FDCA runs.
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from .constants import FireMask, TEMPORAL_FILTER_CODE_OFFSET


# "Base" Part II codes that represent a fire detection in the current run (10, 11, 13, 14, 15). 
# Their already-filtered equivalents (30-35) are also included: if a pixel has been carrying a
# detected fire for several runs and remains active, its timestamp must continue to be updated
# so that the temporal filter keeps working going forward.

FIRE_CODES_FOR_STATE_UPDATE = (
    FireMask.PROCESSED, FireMask.SATURATED, FireMask.CLOUD_CONTAM,
    FireMask.HIGH_PROB, FireMask.MED_PROB, FireMask.LOW_PROB,
    FireMask.PROCESSED + TEMPORAL_FILTER_CODE_OFFSET,
    FireMask.SATURATED + TEMPORAL_FILTER_CODE_OFFSET,
    FireMask.CLOUD_CONTAM + TEMPORAL_FILTER_CODE_OFFSET,
    FireMask.HIGH_PROB + TEMPORAL_FILTER_CODE_OFFSET,
    FireMask.MED_PROB + TEMPORAL_FILTER_CODE_OFFSET,
    FireMask.LOW_PROB + TEMPORAL_FILTER_CODE_OFFSET,
)

EPOCH_ATBD = datetime(2001, 1, 1)

def scene_datetime_to_epoch(scene_dt: datetime) -> float:
    """Convert a scene UTC datetime to 'current_epoch' (seconds since
    2001-01-01), as expected by run_part2.
    """
    return (scene_dt - EPOCH_ATBD).total_seconds()

class PreviousFireMaskStore:
    """
    Persistence wrapper for the prev_fire_mask array consumed by
    run_part2. It lives outside the Part I/Part II cycle: it is
    loaded once when the pipeline starts and saved after each
    processed scene.
    """

    def __init__(self, shape: tuple[int, int], path: str | Path):
        self.shape = shape
        self.path = Path(path)
        self.data = self._load_or_init()

    def _load_or_init(self) -> np.ndarray:
        if self.path.exists():
            arr = np.load(self.path)

            if arr.shape != self.shape:
                raise ValueError(
                    f"Saved prev_fire_mask has shape {arr.shape}, "
                    f"but {self.shape} was expected. Did the crop/region change?"
                )

            return arr

        # 0 (or negative) = "no previous detection";
        # run_part2 already filters using last_t > 0
        return np.zeros(self.shape, dtype=np.float64)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.path, self.data)

    def update(self, fire_mask: np.ndarray, current_epoch: float) -> None:
        """
        Called AFTER run_part2, with the final fire_mask for that scene.
        Overwrites with current_epoch all pixels marked as fire
        (base or already-filtered) in this run.
        """
        is_fire = np.isin(fire_mask, FIRE_CODES_FOR_STATE_UPDATE)

        self.data = np.where(is_fire, current_epoch, self.data)