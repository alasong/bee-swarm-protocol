"""Signal aggregation — collects agent signals into system state snapshots."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from agent_message_bus import MessageBus


@dataclass
class AgentStateInfo:
    """Information about an agent's state."""

    agent_id: str
    state: str
    progress: float = 0.0
    current_task_id: Optional[str] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DiscoveryInfo:
    """Information about a discovery signal."""

    discovery_id: str
    discovery_type: str
    pattern: Dict[str, Any]
    confidence: float
    impact: float
    novelty: float
    dance_intensity: float
    agent_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TaskRequestInfo:
    """Information about a task request."""

    request_id: str
    agent_id: str
    capabilities: List[str]
    resource_availability: Dict[str, float]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SystemStateSnapshot:
    """
    Complete snapshot of the system state.

    Primary output of the SignalAggregator, used by coordination
    components for decision making.
    """

    snapshot_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_agents: int = 0
    idle_agents: int = 0
    executing_agents: int = 0
    exploring_agents: int = 0
    blocked_agents: int = 0
    agent_states: Dict[str, AgentStateInfo] = field(default_factory=dict)
    pending_discoveries: List[DiscoveryInfo] = field(default_factory=list)
    discovery_count: int = 0
    pending_task_requests: List[TaskRequestInfo] = field(default_factory=list)
    task_request_count: int = 0
    avg_resource_usage: Dict[str, float] = field(default_factory=dict)
    system_load: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp.isoformat(),
            "total_agents": self.total_agents,
            "idle_agents": self.idle_agents,
            "executing_agents": self.executing_agents,
            "exploring_agents": self.exploring_agents,
            "blocked_agents": self.blocked_agents,
            "discovery_count": self.discovery_count,
            "task_request_count": self.task_request_count,
            "system_load": self.system_load,
        }


class SignalAggregator:
    """
    Aggregates signals from the Agent layer.

    - Collect state reports from agents
    - Collect discovery signals and task requests
    - Generate system state snapshots

    Optional MessageBus integration: when a bus is provided, signals are
    also published to the bus for cross-agent communication.
    """

    def __init__(
        self,
        snapshot_interval_ms: int = 1000,
        bus: Optional["MessageBus"] = None,
        source_agent_id: str = "aggregator",
    ):
        self.snapshot_interval_ms = snapshot_interval_ms
        self.bus = bus
        self.source_agent_id = source_agent_id
        self._state_reports: Dict[str, AgentStateInfo] = {}
        self._discoveries: List[DiscoveryInfo] = []
        self._task_requests: List[TaskRequestInfo] = []
        self._last_snapshot_time: Optional[datetime] = None
        self._snapshot_counter = 0

    def receive_state_report(self, signal: Dict[str, Any]) -> None:
        """Receive and process a state report signal."""
        agent_id = signal.get("agent_id")
        if not agent_id:
            return
        self._state_reports[agent_id] = AgentStateInfo(
            agent_id=agent_id,
            state=signal.get("state", "idle"),
            progress=signal.get("progress", 0.0),
            current_task_id=signal.get("current_task_id"),
            resource_usage=signal.get("resource_usage", {}),
        )

    async def async_receive_state_report(self, signal: Dict[str, Any]) -> None:
        """Async variant of receive_state_report. Publishes to bus if available."""
        agent_id = signal.get("agent_id")
        if not agent_id:
            return
        self._state_reports[agent_id] = AgentStateInfo(
            agent_id=agent_id,
            state=signal.get("state", "idle"),
            progress=signal.get("progress", 0.0),
            current_task_id=signal.get("current_task_id"),
            resource_usage=signal.get("resource_usage", {}),
        )
        if self.bus is not None:
            await self.bus.broadcast(
                from_agent=self.source_agent_id,
                message={
                    "type": "state_report",
                    "agent_id": agent_id,
                    "state": signal.get("state", "idle"),
                    "progress": signal.get("progress", 0.0),
                },
            )

    def receive_discovery(self, signal: Dict[str, Any]) -> None:
        """Receive and process a discovery signal."""
        discovery_id = signal.get("signal_id") or f"disc_{len(self._discoveries)}"
        confidence = signal.get("confidence", 0.5)
        impact = signal.get("impact", 0.5)
        novelty = signal.get("novelty", 0.5)
        dance_intensity = 0.4 * confidence + 0.3 * impact + 0.3 * novelty

        self._discoveries.append(DiscoveryInfo(
            discovery_id=discovery_id,
            discovery_type=signal.get("discovery_type", "unknown"),
            pattern=signal.get("pattern", {}),
            confidence=confidence,
            impact=impact,
            novelty=novelty,
            dance_intensity=dance_intensity,
            agent_id=signal.get("agent_id", "unknown"),
        ))

    async def async_receive_discovery(self, signal: Dict[str, Any]) -> None:
        """Async variant of receive_discovery. Publishes to bus if available."""
        discovery_id = signal.get("signal_id") or f"disc_{len(self._discoveries)}"
        confidence = signal.get("confidence", 0.5)
        impact = signal.get("impact", 0.5)
        novelty = signal.get("novelty", 0.5)
        dance_intensity = 0.4 * confidence + 0.3 * impact + 0.3 * novelty

        self._discoveries.append(DiscoveryInfo(
            discovery_id=discovery_id,
            discovery_type=signal.get("discovery_type", "unknown"),
            pattern=signal.get("pattern", {}),
            confidence=confidence,
            impact=impact,
            novelty=novelty,
            dance_intensity=dance_intensity,
            agent_id=signal.get("agent_id", "unknown"),
        ))
        if self.bus is not None:
            await self.bus.broadcast(
                from_agent=self.source_agent_id,
                message={
                    "type": "discovery",
                    "discovery_id": discovery_id,
                    "discovery_type": signal.get("discovery_type", "unknown"),
                    "confidence": confidence,
                    "impact": impact,
                    "novelty": novelty,
                    "agent_id": signal.get("agent_id", "unknown"),
                },
            )

    def receive_task_request(self, signal: Dict[str, Any]) -> None:
        """Receive and process a task request signal."""
        request_id = signal.get("signal_id") or f"req_{len(self._task_requests)}"
        agent_id = signal.get("agent_id")
        if not agent_id:
            return
        self._task_requests.append(TaskRequestInfo(
            request_id=request_id,
            agent_id=agent_id,
            capabilities=signal.get("capabilities", []),
            resource_availability=signal.get("resource_availability", {}),
        ))

    async def async_receive_task_request(self, signal: Dict[str, Any]) -> None:
        """Async variant of receive_task_request. Publishes to bus if available."""
        request_id = signal.get("signal_id") or f"req_{len(self._task_requests)}"
        agent_id = signal.get("agent_id")
        if not agent_id:
            return
        self._task_requests.append(TaskRequestInfo(
            request_id=request_id,
            agent_id=agent_id,
            capabilities=signal.get("capabilities", []),
            resource_availability=signal.get("resource_availability", {}),
        ))
        if self.bus is not None:
            await self.bus.broadcast(
                from_agent=self.source_agent_id,
                message={
                    "type": "task_request",
                    "request_id": request_id,
                    "agent_id": agent_id,
                    "capabilities": signal.get("capabilities", []),
                },
            )

    def generate_snapshot(self) -> SystemStateSnapshot:
        """Generate a system state snapshot from collected signals."""
        self._snapshot_counter += 1
        state_counts = defaultdict(int)
        for info in self._state_reports.values():
            state_counts[info.state] += 1

        resource_totals = defaultdict(float)
        resource_counts = defaultdict(int)
        for info in self._state_reports.values():
            for resource, usage in info.resource_usage.items():
                resource_totals[resource] += usage
                resource_counts[resource] += 1

        avg_resource_usage = {
            r: resource_totals[r] / resource_counts[r]
            for r in resource_totals
            if resource_counts[r] > 0
        }

        total = len(self._state_reports)
        busy = state_counts.get("executing", 0) + state_counts.get("exploring", 0)
        system_load = busy / total if total > 0 else 0.0

        snapshot = SystemStateSnapshot(
            snapshot_id=f"snap_{self._snapshot_counter}",
            total_agents=total,
            idle_agents=state_counts.get("idle", 0),
            executing_agents=state_counts.get("executing", 0),
            exploring_agents=state_counts.get("exploring", 0),
            blocked_agents=state_counts.get("blocked", 0),
            agent_states=dict(self._state_reports),
            pending_discoveries=list(self._discoveries),
            discovery_count=len(self._discoveries),
            pending_task_requests=list(self._task_requests),
            task_request_count=len(self._task_requests),
            avg_resource_usage=avg_resource_usage,
            system_load=system_load,
        )

        self._last_snapshot_time = datetime.now(timezone.utc)
        self._discoveries.clear()
        self._task_requests.clear()
        return snapshot

    def get_agent_state(self, agent_id: str) -> Optional[AgentStateInfo]:
        """Get the current state of a specific agent."""
        return self._state_reports.get(agent_id)

    def get_all_agent_states(self) -> Dict[str, AgentStateInfo]:
        """Get all agent states."""
        return dict(self._state_reports)

    def clear_discoveries(self) -> None:
        """Clear pending discoveries."""
        self._discoveries.clear()

    def clear_task_requests(self) -> None:
        """Clear pending task requests."""
        self._task_requests.clear()

    async def async_generate_snapshot(self) -> SystemStateSnapshot:
        """Async variant of generate_snapshot."""
        return self.generate_snapshot()
