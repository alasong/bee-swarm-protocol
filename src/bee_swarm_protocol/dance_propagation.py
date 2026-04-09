"""Dance propagation — distance-based signal spreading with intensity decay."""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from bee_swarm_protocol.dance_signal import DanceSignal


@dataclass
class AgentLocation:
    """Agent location for distance-based propagation."""

    agent_id: str
    x: float
    y: float
    zone: str = "default"


@dataclass
class Response:
    """Response to a dance from an agent."""

    response_id: str
    dance_id: str
    agent_id: str
    response_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    attention_level: float = 0.0


@dataclass
class Dance:
    """
    Extended dance structure for propagation.

    Wraps DanceSignal with propagation metadata.
    """

    signal: DanceSignal
    location: Optional[AgentLocation] = None
    propagation_radius: float = 100.0
    expiry_time: Optional[datetime] = None
    responses: List[Response] = field(default_factory=list)
    decayed_intensity: float = 0.0

    @property
    def dance_id(self) -> str:
        return self.signal.dance_id

    @property
    def original_intensity(self) -> float:
        return self.signal.intensity

    @property
    def is_expired(self) -> bool:
        if self.expiry_time is None:
            return False
        return datetime.utcnow() > self.expiry_time


class DancePropagator:
    """
    Propagates dances to agents based on distance and intensity.

    - Distance-based propagation: only nearby agents see the dance
    - Intensity-based attention: high intensity = more attention
    - Inverse-square-like intensity decay with distance
    """

    def __init__(
        self,
        default_radius: float = 100.0,
        intensity_multiplier: float = 50.0,
        min_radius: float = 10.0,
        max_radius: float = 500.0,
    ):
        self.default_radius = default_radius
        self.intensity_multiplier = intensity_multiplier
        self.min_radius = min_radius
        self.max_radius = max_radius

        self._agent_locations: Dict[str, AgentLocation] = {}
        self._active_dances: Dict[str, Dance] = {}
        self._visibility_cache: Dict[str, List[str]] = {}

    def register_agent(self, location: AgentLocation) -> None:
        """Register an agent's location."""
        self._agent_locations[location.agent_id] = location

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the registry."""
        self._agent_locations.pop(agent_id, None)
        self._visibility_cache.pop(agent_id, None)

    def update_agent_location(self, agent_id: str, x: float, y: float) -> None:
        """Update an agent's location."""
        if agent_id in self._agent_locations:
            loc = self._agent_locations[agent_id]
            loc.x = x
            loc.y = y

    def propagate(self, dance: Dance, target_agents: List[str]) -> Dict[str, float]:
        """
        Send dance to specified agents within range.

        Returns:
            Dict mapping agent_id to received intensity
        """
        results = {}
        for agent_id in target_agents:
            if agent_id not in self._agent_locations:
                continue
            distance = self._calculate_distance(
                dance.location, self._agent_locations[agent_id]
            )
            radius = self._get_effective_radius(dance)
            if distance > radius:
                continue
            received = self._calculate_received_intensity(dance, distance, radius)
            results[agent_id] = received

        self._active_dances[dance.dance_id] = dance
        return results

    def broadcast(self, dance: Dance) -> Dict[str, float]:
        """Broadcast dance to all registered agents."""
        return self.propagate(dance, list(self._agent_locations.keys()))

    def get_visible_dances(self, agent_id: str) -> List[Dance]:
        """Get dances visible to an agent, sorted by intensity (highest first)."""
        if agent_id not in self._agent_locations:
            return []

        agent_loc = self._agent_locations[agent_id]
        visible = []

        for dance in self._active_dances.values():
            if dance.is_expired:
                continue
            distance = self._calculate_distance(dance.location, agent_loc)
            radius = self._get_effective_radius(dance)
            if distance <= radius:
                received = self._calculate_received_intensity(dance, distance, radius)
                dance.decayed_intensity = received
                visible.append(dance)

        visible.sort(key=lambda d: d.decayed_intensity, reverse=True)
        return visible

    def _calculate_distance(self, loc1: Optional[AgentLocation], loc2: AgentLocation) -> float:
        """Euclidean distance between two locations."""
        if loc1 is None:
            return float("inf")
        return math.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2)

    def _get_effective_radius(self, dance: Dance) -> float:
        """Effective propagation radius based on intensity."""
        intensity_radius = dance.original_intensity * self.intensity_multiplier
        radius = max(self.min_radius, min(self.max_radius, intensity_radius))
        if dance.propagation_radius > 0:
            radius = dance.propagation_radius
        return radius

    def _calculate_received_intensity(
        self, dance: Dance, distance: float, radius: float
    ) -> float:
        """Intensity decreases with distance (inverse square-like decay)."""
        if distance == 0:
            base = (
                dance.decayed_intensity
                if dance.decayed_intensity > 0
                else dance.original_intensity
            )
            return base
        decay_factor = 1.0 - (distance / radius) ** 2
        decay_factor = max(0.0, decay_factor)
        base = dance.decayed_intensity if dance.decayed_intensity > 0 else dance.original_intensity
        return base * decay_factor

    def clear_dance(self, dance_id: str) -> None:
        """Remove a dance from active dances."""
        self._active_dances.pop(dance_id, None)

    def get_all_active_dances(self) -> List[Dance]:
        """Get all active (non-expired) dances."""
        return [d for d in self._active_dances.values() if not d.is_expired]
