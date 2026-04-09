"""Time-based decay for dance intensity with automatic cleanup."""

import math
from datetime import datetime
from typing import Any, Dict, List

from bee_swarm_protocol.dance_propagation import Dance, DancePropagator


class DanceDecay:
    """
    Applies exponential time-based decay to dance intensity.

    Formula: intensity * e^(-rate * time)

    Auto-cleans dances whose intensity falls below the expiry threshold.
    """

    def __init__(
        self,
        propagator: DancePropagator,
        decay_rate: float = 0.1,
        expiry_threshold: float = 0.05,
    ):
        self._propagator = propagator
        self._decay_rate = decay_rate
        self._expiry_threshold = expiry_threshold

    def apply_decay(self, dance: Dance, time_elapsed: float) -> Dance:
        """
        Apply decay to a dance based on elapsed time.

        Args:
            dance: The dance to decay
            time_elapsed: Seconds since dance creation

        Returns:
            Updated dance with decayed intensity
        """
        base = dance.original_intensity
        decayed = base * math.exp(-self._decay_rate * time_elapsed)
        dance.decayed_intensity = max(0.0, decayed)

        if dance.decayed_intensity < self._expiry_threshold:
            if dance.expiry_time is None:
                dance.expiry_time = datetime.utcnow()

        return dance

    def set_decay_rate(self, rate: float) -> None:
        """Set the decay rate (must be positive)."""
        if rate < 0:
            raise ValueError("Decay rate must be positive")
        self._decay_rate = rate

    def cleanup_expired_dances(self) -> List[str]:
        """Remove all expired dances. Returns list of removed dance IDs."""
        removed = []
        for dance_id, dance in list(self._propagator._active_dances.items()):
            time_elapsed = self._get_time_elapsed(dance)
            self.apply_decay(dance, time_elapsed)
            if dance.is_expired or dance.decayed_intensity < self._expiry_threshold:
                self._propagator.clear_dance(dance_id)
                removed.append(dance_id)
        return removed

    def _get_time_elapsed(self, dance: Dance) -> float:
        """Seconds since dance creation."""
        if dance.signal.timestamp is None:
            return 0.0
        return (datetime.utcnow() - dance.signal.timestamp).total_seconds()

    def get_decay_status(self, dance: Dance) -> Dict[str, Any]:
        """Get decay status for a dance."""
        time_elapsed = self._get_time_elapsed(dance)
        return {
            "dance_id": dance.dance_id,
            "original_intensity": dance.original_intensity,
            "current_intensity": dance.decayed_intensity,
            "time_elapsed": time_elapsed,
            "decay_rate": self._decay_rate,
            "is_expired": dance.is_expired,
            "remaining_fraction": (
                dance.decayed_intensity / dance.original_intensity
                if dance.original_intensity > 0
                else 0.0
            ),
        }
