"""Dance propagation — distance-based signal spreading with intensity decay."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from agent_message_bus import MessageBus

from bee_swarm_protocol.dance_signal import DanceSignal


class DistanceMetric(ABC):
    """Base class for configurable distance metrics."""

    @abstractmethod
    def calculate(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate distance between two points."""


class EuclideanMetric(DistanceMetric):
    """Standard Euclidean distance (default)."""

    def calculate(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class ManhattanMetric(DistanceMetric):
    """Manhattan (taxicab) distance: |x1-x2| + |y1-y2|."""

    def calculate(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return abs(x1 - x2) + abs(y1 - y2)


class ChebyshevMetric(DistanceMetric):
    """Chebyshev distance: max(|x1-x2|, |y1-y2|)."""

    def calculate(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return max(abs(x1 - x2), abs(y1 - y2))


def _build_metric(metric: str) -> DistanceMetric:
    """Factory: build a metric from a name string."""
    metric_map: Dict[str, DistanceMetric] = {
        "euclidean": EuclideanMetric(),
        "manhattan": ManhattanMetric(),
        "chebyshev": ChebyshevMetric(),
    }
    if metric.lower() not in metric_map:
        raise ValueError(
            f"Unknown distance metric: '{metric}'. "
            f"Valid options: {list(metric_map.keys())}"
        )
    return metric_map[metric.lower()]


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
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
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
        return datetime.now(timezone.utc) > self.expiry_time


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
        distance_metric: Optional[DistanceMetric | str] = None,
        bus: Optional["MessageBus"] = None,
        source_agent_id: str = "dance_propagator",
    ):
        self.default_radius = default_radius
        self.intensity_multiplier = intensity_multiplier
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.bus = bus
        self.source_agent_id = source_agent_id

        if isinstance(distance_metric, str):
            self._distance_metric = _build_metric(distance_metric)
        else:
            self._distance_metric = distance_metric or EuclideanMetric()

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
        """Distance between two locations using the configured metric."""
        if loc1 is None:
            return float("inf")
        return self._distance_metric.calculate(loc1.x, loc1.y, loc2.x, loc2.y)

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

    async def async_propagate(self, dance: Dance, target_agents: List[str]) -> Dict[str, float]:
        """Async variant of propagate."""
        return self.propagate(dance, target_agents)

    async def async_broadcast(self, dance: Dance) -> Dict[str, float]:
        """Async variant of broadcast. Publishes to bus if available."""
        results = self.propagate(dance, list(self._agent_locations.keys()))
        if self.bus is not None:
            await self._bus_broadcast_dance(dance)
        return results

    async def _bus_broadcast_dance(self, dance: Dance) -> None:
        """Publish dance signal to the message bus as a broadcast."""
        await self.bus.broadcast(
            from_agent=self.source_agent_id,
            message={
                "type": "dance",
                "dance_id": dance.dance_id,
                "agent_id": dance.signal.agent_id,
                "intensity": dance.original_intensity,
                "direction": dance.signal.direction,
                "duration": dance.signal.duration,
                "pattern": dance.signal.pattern,
            },
        )

    async def async_get_visible_dances(self, agent_id: str) -> List[Dance]:
        """Async variant of get_visible_dances."""
        return self.get_visible_dances(agent_id)

    def set_distance_metric(self, metric: DistanceMetric | str) -> None:
        """Change the distance metric at runtime."""
        if isinstance(metric, str):
            self._distance_metric = _build_metric(metric)
        else:
            self._distance_metric = metric

    @property
    def distance_metric(self) -> DistanceMetric:
        """Return the current distance metric."""
        return self._distance_metric
