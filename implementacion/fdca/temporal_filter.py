# Create file: fdca/temporal_filter.py

"""
Temporal filter for FDCA - loads previous 12 hours of fire detections
"""

import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Callable
import logging

logger = logging.getLogger(__name__)
EPOCH_ATBD = datetime(2001, 1, 1)
FIRE_CODES = (10, 11, 12, 13, 14, 15, 30, 31, 32, 33, 34, 35)


def _epoch_seconds(dt: datetime) -> int:
    """Convert a naive UTC scene datetime to seconds since 2001-01-01."""
    # Do not use datetime.timestamp(): for naive datetimes it depends on the
    # host timezone, while FDCA's epoch is defined in UTC.
    return int((dt - EPOCH_ATBD).total_seconds())


class TemporalFilter:
    """
    Manages temporal filtering by loading fire detections from previous 12 hours.
    """
    
    def __init__(
        self,
        data_root: str,
        region: str,
        timestamp: str,
        shape: Tuple[int, int],
        lookback_hours: int = 12,
        download_callback: Optional[Callable] = None,
    ):
        """
        Initialize temporal filter.
        
        Parameters
        ----------
        data_root : str
            Root directory for data
        region : str
            Region name (e.g., 'uruguay')
        timestamp : str
            Current timestamp in YYYYMMDD_HHMM format
        shape : tuple
            Shape of the image (rows, cols)
        lookback_hours : int
            Hours to look back for previous fires (default: 12)
        download_callback : callable
            Function to call to download missing data
        """
        self.data_root = Path(data_root)
        self.region = region
        self.timestamp = timestamp
        self.shape = shape
        self.lookback_hours = lookback_hours
        self.download_callback = download_callback
        
        # Parse current time
        self.current_dt = datetime.strptime(timestamp, "%Y%m%d_%H%M")
        
    def get_previous_timestamps(self, interval_hours: int = 1) -> List[str]:
        """Generate fallback timestamps for optional downloads.

        Local masks are discovered by their actual timestamps in
        :meth:`discover_previous_scenes`; this helper is only used when a
        caller explicitly permits downloading missing scenes.
        """
        if interval_hours <= 0:
            raise ValueError("interval_hours debe ser positivo")
        timestamps = []
        cursor = self.current_dt - timedelta(hours=self.lookback_hours)
        end = self.current_dt
        step = timedelta(hours=interval_hours)
        while cursor < end:
            timestamps.append(cursor.strftime("%Y%m%d_%H%M"))
            cursor += step
        return timestamps
    
    def discover_previous_scenes(self) -> List[Tuple[str, Path]]:
        """Find every locally available mask inside the lookback interval.

        Operational ABI scenes are not guaranteed to be exactly hourly (and
        this repository contains timestamps such as ``...0550``). Enumerating
        the filenames avoids silently dropping valid detections because of a
        guessed two-hour grid.
        """
        mask_dir = self.data_root / self.region / "ABI-L2-FDCF-Mask"
        cutoff = self.current_dt - timedelta(hours=self.lookback_hours)
        available = []
        if not mask_dir.exists():
            return available
        for mask_path in mask_dir.glob("*.npy"):
            try:
                scene_dt = datetime.strptime(mask_path.stem, "%Y%m%d_%H%M")
            except ValueError:
                continue
            if cutoff <= scene_dt < self.current_dt:
                available.append((mask_path.stem, mask_path))
        return sorted(available, key=lambda item: item[0])

    def find_available_scenes(self, timestamps: List[str]) -> List[Tuple[str, Path]]:
        """Find a requested subset of timestamps available locally."""
        mask_dir = self.data_root / self.region / "ABI-L2-FDCF-Mask"
        available = []
        for ts in timestamps:
            mask_path = mask_dir / f"{ts}.npy"
            if mask_path.exists():
                available.append((ts, mask_path))
        return available
    
    def load_previous_fires(self) -> np.ndarray:
        """
        Load fire detections from previous 12 hours.
        
        Returns
        -------
        np.ndarray
            Mask where each pixel contains the timestamp (seconds since 2001)
            of the most recent fire detection, or 0 if no fire was detected.
        """
        # Initialize state mask (0 = no previous fire)
        state_mask = np.zeros(self.shape, dtype=np.int64)
        
        # Use every local scene in the interval. The fallback hourly list is
        # only for optional downloads when a caller explicitly enables them.
        available = self.discover_previous_scenes()
        timestamps = self.get_previous_timestamps(interval_hours=1)
        logger.info(
            f"Found {len(available)} available previous scenes locally "
            f"within the last {self.lookback_hours} hours"
        )

        if self.download_callback:
            known = {ts for ts, _ in available}
            missing = [ts for ts in timestamps if ts not in known]
            logger.info(f"Attempting to download {len(missing)} fallback scenes")
            for ts in missing:
                try:
                    self.download_callback(ts)
                except Exception as e:
                    logger.warning(f"Failed to download {ts}: {e}")
            # A downloader may resolve a requested timestamp to a nearby scene;
            # rediscover files so the load phase uses the actual filename.
            available = self.discover_previous_scenes()
        
        # Load fire masks for available scenes
        for ts, mask_path in available:
            try:
                # Load and convert mask
                mask_raw = np.load(mask_path)
                # Convert int8 -> uint8 -> int32 to recover proper codes
                mask = mask_raw.astype(np.uint8).astype(np.int32)
                
                if mask.shape != self.shape:
                    logger.warning(
                        f"Skipping {ts}: mask shape {mask.shape} != expected "
                        f"{self.shape}"
                    )
                    continue

                # Find fire pixels (codes 10-15 or 30-35)
                fire_mask = np.isin(mask, FIRE_CODES)

                # Convert timestamp to seconds since 2001 without host-TZ
                # dependence.
                dt = datetime.strptime(ts, "%Y%m%d_%H%M")
                epoch_seconds = _epoch_seconds(dt)
                
                # Update state mask (keep the most recent detection)
                # If multiple detections, the most recent timestamp wins
                update_mask = (fire_mask) & (epoch_seconds > state_mask)
                state_mask[update_mask] = epoch_seconds
                
                logger.debug(f"Loaded {fire_mask.sum()} fires from {ts}")
                
            except Exception as e:
                logger.warning(f"Error loading mask from {ts}: {e}")
                continue
        
        # Apply temporal window: only keep fires within the lookback window
        current_epoch = _epoch_seconds(self.current_dt)
        cutoff_epoch = current_epoch - self.lookback_hours * 3600
        stale_mask = (state_mask > 0) & (state_mask < cutoff_epoch)
        state_mask[stale_mask] = 0
        
        total_fires = (state_mask > 0).sum()
        logger.info(f"Loaded {total_fires} previous fires within {self.lookback_hours} hours")
        
        return state_mask