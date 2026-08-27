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
        """
        Generate timestamps for previous N hours at regular intervals.
        
        Parameters
        ----------
        interval_hours : int
            Time interval between timestamps (default: 1 hour)
        """
        timestamps = []
        for hours_back in range(self.lookback_hours, 0, -interval_hours):
            prev_dt = self.current_dt - timedelta(hours=hours_back)
            # Round to nearest hour
            prev_dt = prev_dt.replace(minute=0, second=0, microsecond=0)
            ts = prev_dt.strftime("%Y%m%d_%H%M")
            timestamps.append(ts)
        return timestamps
    
    def find_available_scenes(self, timestamps: List[str]) -> List[Tuple[str, Path]]:
        """Find which timestamps are available locally."""
        available = []
        for ts in timestamps:
            mask_path = self.data_root / self.region / "ABI-L2-FDCF-Mask" / f"{ts}.npy"
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
        
        # Get timestamps to check (every 2 hours to reduce downloads)
        timestamps = self.get_previous_timestamps(interval_hours=2)
        logger.info(f"Checking {len(timestamps)} previous timestamps")
        
        # Find available scenes
        available = self.find_available_scenes(timestamps)
        logger.info(f"Found {len(available)} available previous scenes locally")
        
        # If download callback is provided, try to download missing scenes
        if self.download_callback and len(available) < len(timestamps):
            missing = set(timestamps) - set([ts for ts, _ in available])
            logger.info(f"Attempting to download {len(missing)} missing scenes")
            for ts in missing:
                try:
                    self.download_callback(ts)
                    mask_path = self.data_root / self.region / "ABI-L2-FDCF-Mask" / f"{ts}.npy"
                    if mask_path.exists():
                        available.append((ts, mask_path))
                        logger.debug(f"Successfully downloaded {ts}")
                except Exception as e:
                    logger.warning(f"Failed to download {ts}: {e}")
        
        # Load fire masks for available scenes
        for ts, mask_path in available:
            try:
                # Load and convert mask
                mask_raw = np.load(mask_path)
                # Convert int8 -> uint8 -> int32 to recover proper codes
                mask = mask_raw.astype(np.uint8).astype(np.int32)
                
                # Find fire pixels (codes 10-15 or 30-35)
                fire_codes = [10, 11, 12, 13, 14, 15, 30, 31, 32, 33, 34, 35]
                fire_mask = np.isin(mask, fire_codes)
                
                # Convert timestamp to seconds since 2001
                dt = datetime.strptime(ts, "%Y%m%d_%H%M")
                epoch_seconds = int(dt.timestamp()) - 978307200  # 2001-01-01
                
                # Update state mask (keep the most recent detection)
                # If multiple detections, the most recent timestamp wins
                update_mask = (fire_mask) & (epoch_seconds > state_mask)
                state_mask[update_mask] = epoch_seconds
                
                logger.debug(f"Loaded {fire_mask.sum()} fires from {ts}")
                
            except Exception as e:
                logger.warning(f"Error loading mask from {ts}: {e}")
                continue
        
        # Apply temporal window: only keep fires within the lookback window
        current_epoch = int(self.current_dt.timestamp()) - 978307200
        cutoff_epoch = current_epoch - self.lookback_hours * 3600
        stale_mask = (state_mask > 0) & (state_mask < cutoff_epoch)
        state_mask[stale_mask] = 0
        
        total_fires = (state_mask > 0).sum()
        logger.info(f"Loaded {total_fires} previous fires within {self.lookback_hours} hours")
        
        return state_mask