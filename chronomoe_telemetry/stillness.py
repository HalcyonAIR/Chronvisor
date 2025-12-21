"""
Stillness: Fast/medium/slow detectors.

Stillness detection at multiple timescales.
All thresholds are priors (designed, not learned).
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class StillnessThresholds:
    """Thresholds for detecting stillness at different timescales."""
    fast: float = 0.01      # Token-scale motion threshold
    medium: float = 0.02    # Arc-scale progress threshold
    slow: float = 0.005     # Long-horizon regime threshold


@dataclass
class StillnessFlags:
    """Stillness detection flags at different timescales."""
    fast: bool = False
    medium: bool = False
    slow: bool = False

    def any_triggered(self) -> bool:
        """True if any stillness flag is set."""
        return self.fast or self.medium or self.slow

    def all_triggered(self) -> bool:
        """True if all stillness flags are set."""
        return self.fast and self.medium and self.slow

    def count_triggered(self) -> int:
        """Count how many flags are set."""
        return sum([self.fast, self.medium, self.slow])


class StillnessDetector:
    """
    Detects stillness at multiple timescales.

    Stillness is not learned. It is detected by measuring absence of motion
    across different time windows, using engineered thresholds.

    Fast stillness: |dPC| < ε_fast for 1+ steps
    Medium stillness: |dPC| < ε_medium for 3+ steps
    Slow stillness: |dPC| < ε_slow for 5+ steps
    """

    def __init__(self, thresholds: StillnessThresholds = None):
        """
        Args:
            thresholds: Stillness thresholds at each timescale
        """
        self.thresholds = thresholds or StillnessThresholds()

        # Counters: how long motion has been below threshold
        self.counters = {
            'fast': 0,
            'medium': 0,
            'slow': 0,
        }

    def update(
        self,
        d_pc1: float,
        d_pc2: float,
    ) -> StillnessFlags:
        """
        Update stillness counters and return flags.

        Args:
            d_pc1: Change in PC1 (momentum)
            d_pc2: Change in PC2 (exploration)

        Returns:
            StillnessFlags indicating which timescales are still
        """
        import numpy as np

        # Magnitude of motion in PC space
        d_pc_mag = np.sqrt(d_pc1**2 + d_pc2**2)

        # Update counters for each timescale
        # Fast: PC1 only (momentum flatline)
        if abs(d_pc1) < self.thresholds.fast:
            self.counters['fast'] += 1
        else:
            self.counters['fast'] = 0

        # Medium: total motion magnitude
        if d_pc_mag < self.thresholds.medium:
            self.counters['medium'] += 1
        else:
            self.counters['medium'] = 0

        # Slow: very small total motion
        if d_pc_mag < self.thresholds.slow:
            self.counters['slow'] += 1
        else:
            self.counters['slow'] = 0

        # Set flags based on counter thresholds
        flags = StillnessFlags(
            fast=self.counters['fast'] >= 1,
            medium=self.counters['medium'] >= 3,
            slow=self.counters['slow'] >= 5,
        )

        return flags

    def reset(self):
        """Reset all counters (between sessions)."""
        self.counters = {'fast': 0, 'medium': 0, 'slow': 0}

    def get_counters(self) -> dict:
        """Get current counter values (for debugging)."""
        return self.counters.copy()
