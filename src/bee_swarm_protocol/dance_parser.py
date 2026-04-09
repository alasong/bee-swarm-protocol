"""Dance language parser — converts discovery signals into weighted dances."""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from bee_swarm_protocol.dance_signal import DanceSignal


@dataclass
class PatternWeights:
    """Accumulated weights for each pattern type."""

    weights: Dict[str, float] = field(default_factory=dict)
    support_counts: Dict[str, int] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.utcnow)

    def get_weight(self, pattern_type: str) -> float:
        """Get the weight for a pattern type."""
        return self.weights.get(pattern_type, 0.0)

    def get_support_count(self, pattern_type: str) -> int:
        """Get the number of agents supporting a pattern."""
        return self.support_counts.get(pattern_type, 0)


class DanceLanguageParser:
    """
    Parses discovery signals into dance signals and accumulates weights.

    **Dance Intensity Formula:**
        intensity = 0.4 * confidence + 0.3 * impact + 0.3 * novelty

    **Pattern Weight Accumulation:**
        weight = avg_intensity * log(agent_count + 1)

    This rewards both quality (avg intensity) and quantity (number of agents).
    """

    def __init__(self, intensity_threshold: float = 0.5):
        """
        Args:
            intensity_threshold: Minimum intensity to consider a dance valid
        """
        self.intensity_threshold = intensity_threshold
        self._dances: List[DanceSignal] = []
        self._pattern_weights = PatternWeights()
        self._dance_counter = 0

    def parse_discovery(self, discovery: Dict[str, Any]) -> Optional[DanceSignal]:
        """
        Parse a discovery signal into a dance signal.

        Args:
            discovery: Dict with confidence, impact, novelty, pattern, agent_id

        Returns:
            DanceSignal if intensity exceeds threshold, None otherwise
        """
        confidence = discovery.get("confidence", 0.5)
        impact = discovery.get("impact", 0.5)
        novelty = discovery.get("novelty", 0.5)

        intensity = 0.4 * confidence + 0.3 * impact + 0.3 * novelty

        if intensity < self.intensity_threshold:
            return None

        self._dance_counter += 1
        duration = intensity * 10  # Higher intensity = longer dance

        pattern = discovery.get("pattern", {})
        direction = self._extract_direction(pattern)

        dance = DanceSignal(
            dance_id=f"dance_{self._dance_counter}",
            agent_id=discovery.get("agent_id", "unknown"),
            intensity=intensity,
            direction=direction,
            duration=duration,
            pattern=pattern,
        )

        self._dances.append(dance)
        self._update_pattern_weights(dance)

        return dance

    def _extract_direction(self, pattern: Dict[str, Any]) -> str:
        """Extract direction (pattern type) from pattern data."""
        if "type" in pattern:
            return pattern["type"]
        if "direction" in pattern:
            return pattern["direction"]
        return "unknown"

    def _update_pattern_weights(self, dance: DanceSignal) -> None:
        """Update accumulated weights for the pattern."""
        direction = dance.direction
        current_weight = self._pattern_weights.weights.get(direction, 0.0)
        current_count = self._pattern_weights.support_counts.get(direction, 0)

        new_count = current_count + 1
        new_weight = (current_weight * current_count + dance.intensity) / new_count

        self._pattern_weights.weights[direction] = new_weight
        self._pattern_weights.support_counts[direction] = new_count
        self._pattern_weights.last_update = datetime.utcnow()

    def accumulate_dances(self) -> Dict[str, float]:
        """
        Calculate accumulated weights for all patterns.

        Returns:
            Dict mapping pattern types to accumulated weights
        """
        accumulated = {}
        for direction, avg_intensity in self._pattern_weights.weights.items():
            support_count = self._pattern_weights.support_counts.get(direction, 1)
            weight = avg_intensity * math.log(support_count + 1)
            accumulated[direction] = weight
        return accumulated

    def get_pattern_weights(self) -> PatternWeights:
        """Get current pattern weights."""
        return self._pattern_weights

    def get_dances(self, limit: int = 100) -> List[DanceSignal]:
        """Get recent dances."""
        return self._dances[-limit:]

    def clear_dances(self) -> None:
        """Clear stored dances (but keep accumulated weights)."""
        self._dances.clear()

    def reset_weights(self) -> None:
        """Reset all pattern weights."""
        self._pattern_weights = PatternWeights()

    def get_top_patterns(self, n: int = 5) -> List[tuple]:
        """Get top N patterns by weight."""
        weights = self.accumulate_dances()
        return sorted(weights.items(), key=lambda x: x[1], reverse=True)[:n]
