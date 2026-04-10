"""BeeAgent runtime loop — base agent class and concrete implementations."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from agent_message_bus import MessageBus

from bee_swarm_protocol.aggregator import SignalAggregator, SystemStateSnapshot
from bee_swarm_protocol.consensus import ConsensusAlgorithm, ConsensusResult
from bee_swarm_protocol.dance_parser import DanceLanguageParser
from bee_swarm_protocol.goal_emergence import GoalEmerger, SystemGoal

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Roles that a BeeAgent can play."""

    EXPLORER = "explorer"
    EXECUTOR = "executor"
    VALIDATOR = "validator"
    COORDINATOR = "coordinator"


@dataclass
class AgentState:
    """Current state of a BeeAgent."""

    agent_id: str
    role: str
    status: str = "idle"  # idle, running, stopped, error
    message_count: int = 0
    last_activity: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BeeAgent(ABC):
    """
    Base agent class that runs a continuous message-processing loop.

    Subclasses must implement ``process_dance`` to define how the agent
    responds to incoming dance signals.

    Lifecycle::

        agent = ExplorerAgent("exp_1", bus=my_bus)
        await agent.start()   # begins the async run loop
        # ... work happens ...
        await agent.stop()    # graceful shutdown

    Args:
        agent_id: Unique identifier for this agent.
        role: The agent's role in the swarm.
        bus: Optional message bus for cross-agent communication.
    """

    def __init__(
        self,
        agent_id: str,
        role: AgentRole | str,
        bus: Optional["MessageBus"] = None,
    ):
        self.agent_id = agent_id
        self.role = role.value if isinstance(role, AgentRole) else role
        self.bus = bus
        self._state = AgentState(agent_id=agent_id, role=self.role)
        self._running = False
        self._task: Optional[asyncio.Task] = None

    @property
    def is_running(self) -> bool:
        """Whether the agent loop is currently active."""
        return self._running

    @property
    def state(self) -> AgentState:
        """Current agent state snapshot."""
        return self._state

    async def start(self) -> None:
        """Start the agent's message processing loop."""
        if self._running:
            logger.warning("Agent %s is already running", self.agent_id)
            return

        self._running = True
        self._state.status = "running"
        logger.info("Agent %s (%s) started", self.agent_id, self.role)

        if self.bus is not None:
            await self._register_with_bus()

        self._task = asyncio.create_task(self._run_loop())

    async def _run_loop(self) -> None:
        """Internal message processing loop."""
        try:
            while self._running:
                try:
                    if self.bus is not None:
                        message = await self._receive_with_timeout(timeout=1.0)
                        if message is not None:
                            await self._handle_message(message)
                    else:
                        # No bus: poll-based mode with sleep
                        await asyncio.sleep(0.5)
                except asyncio.TimeoutError:
                    # Normal — no messages available
                    pass
                except Exception:
                    logger.exception(
                        "Error in agent %s run loop", self.agent_id
                    )
                    self._state.status = "error"
        except asyncio.CancelledError:
            pass
        finally:
            self._state.status = "stopped"
            logger.info("Agent %s stopped", self.agent_id)

    async def _receive_with_timeout(self, timeout: float) -> Optional[Dict[str, Any]]:
        """Receive a message from the bus with a timeout."""
        if self.bus is None:
            return None
        try:
            return await asyncio.wait_for(
                self.bus.receive(self.agent_id),
                timeout=timeout,
            )
        except (asyncio.TimeoutError, Exception):
            return None

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Dispatch an incoming message to the appropriate handler."""
        self._state.message_count += 1
        self._state.last_activity = datetime.now(timezone.utc)

        msg_type = message.get("type", "")
        if msg_type == "dance_signal":
            dance_signal = message.get("signal", message)
            await self._process_dance_signal(dance_signal)
        elif msg_type == "goal":
            await self._handle_goal(message)
        elif msg_type == "consensus":
            await self._handle_consensus(message)
        else:
            await self.process_custom_message(message)

    async def _process_dance_signal(self, dance_signal: Dict[str, Any]) -> None:
        """Route a dance signal through process_dance."""
        await self.process_dance(dance_signal)

    async def _handle_goal(self, message: Dict[str, Any]) -> None:
        """Handle an incoming goal message. Subclasses may override."""
        logger.debug(
            "Agent %s received goal: %s",
            self.agent_id,
            message.get("goal_id"),
        )

    async def _handle_consensus(self, message: Dict[str, Any]) -> None:
        """Handle an incoming consensus result. Subclasses may override."""
        logger.debug(
            "Agent %s received consensus: %s",
            self.agent_id,
            message.get("consensus_id"),
        )

    @abstractmethod
    async def process_dance(self, dance_signal: Dict[str, Any]) -> None:
        """
        Process an incoming dance signal.

        Must be implemented by subclasses to define agent-specific behavior.

        Args:
            dance_signal: The dance signal to process.
        """

    async def process_custom_message(self, message: Dict[str, Any]) -> None:
        """
        Handle a message that is not a recognized type.

        Default implementation logs a debug message. Subclasses may override.

        Args:
            message: The unrecognized message.
        """
        logger.debug(
            "Agent %s received unrecognized message type: %s",
            self.agent_id,
            message.get("type", "unknown"),
        )

    async def send_dance(self, signal: Dict[str, Any]) -> None:
        """
        Publish a dance signal to the message bus.

        If no bus is configured, the signal is logged but not sent.

        Args:
            signal: The dance signal to publish.
        """
        envelope = {
            "type": "dance_signal",
            "from_agent": self.agent_id,
            "signal": signal,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if self.bus is not None:
            await self.bus.send(
                from_agent=self.agent_id,
                to_agent="broadcast",
                message=envelope,
            )
        logger.debug("Agent %s published dance signal", self.agent_id)

    async def stop(self) -> None:
        """Graceful shutdown of the agent."""
        if not self._running:
            return

        self._running = False
        logger.info("Agent %s shutting down", self.agent_id)

        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self.bus is not None:
            await self._unregister_from_bus()

    async def _register_with_bus(self) -> None:
        """Register this agent with the message bus."""
        try:
            await self.bus.register_agent(self.agent_id, self.role)
        except Exception:
            logger.warning("Failed to register agent %s with bus", self.agent_id)

    async def _unregister_from_bus(self) -> None:
        """Unregister this agent from the message bus."""
        try:
            await self.bus.unregister_agent(self.agent_id)
        except Exception:
            logger.warning("Failed to unregister agent %s from bus", self.agent_id)


class ExplorerAgent(BeeAgent):
    """
    Explorer agent that periodically generates discovery signals.

    Scans the environment, parses discoveries through DanceLanguageParser,
    and broadcasts resulting dance signals to the swarm.

    Args:
        agent_id: Unique identifier.
        bus: Optional message bus.
        scan_interval: Seconds between discovery scans (default 5.0).
    """

    def __init__(
        self,
        agent_id: str,
        bus: Optional["MessageBus"] = None,
        scan_interval: float = 5.0,
    ):
        super().__init__(agent_id, AgentRole.EXPLORER, bus=bus)
        self.scan_interval = scan_interval
        self.parser = DanceLanguageParser()
        self._scan_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the explorer and its periodic discovery scanner."""
        await super().start()
        if self._running:
            self._scan_task = asyncio.create_task(self._periodic_discovery())

    async def stop(self) -> None:
        """Stop the explorer and cancel the discovery scanner."""
        if self._scan_task is not None:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
            self._scan_task = None
        await super().stop()

    async def _periodic_discovery(self) -> None:
        """Periodically generate and broadcast discovery signals."""
        try:
            while self._running:
                discoveries = self._generate_discoveries()
                for discovery in discoveries:
                    dance = self.parser.parse_discovery(discovery)
                    if dance is not None:
                        await self.send_dance({
                            "dance_id": dance.dance_id,
                            "agent_id": dance.agent_id,
                            "intensity": dance.intensity,
                            "direction": dance.direction,
                            "duration": dance.duration,
                            "pattern": dance.pattern,
                        })
                await asyncio.sleep(self.scan_interval)
        except asyncio.CancelledError:
            pass

    def _generate_discoveries(self) -> List[Dict[str, Any]]:
        """
        Generate discovery signals from the current environment.

        Default implementation returns an empty list. Subclasses should
        override to provide actual discovery logic.
        """
        return []

    async def process_dance(self, dance_signal: Dict[str, Any]) -> None:
        """
        Explorer processes dances from other explorers.

        Accumulates weights internally so the explorer can correlate
        its own discoveries with signals from peers.
        """
        intensity = dance_signal.get("intensity", 0.0)
        # Update the parser with this peer signal for weight tracking
        self.parser.parse_discovery({
            "confidence": intensity,
            "impact": intensity,
            "novelty": dance_signal.get("pattern", {}).get("novelty", 0.5),
            "pattern": dance_signal.get("pattern", {}),
            "agent_id": dance_signal.get("from_agent", "peer"),
        })

    def get_top_patterns(self, n: int = 5) -> List[tuple]:
        """Get the top-n accumulated dance patterns."""
        return self.parser.get_top_patterns(n)


class ExecutorAgent(BeeAgent):
    """
    Executor agent that acts on dance signals via consensus.

    Receives dance signals from explorers, uses ConsensusAlgorithm
    to form consensus, and takes action based on the result.

    Args:
        agent_id: Unique identifier.
        bus: Optional message bus.
        consensus_threshold: Minimum weight for strong consensus (default 0.7).
    """

    def __init__(
        self,
        agent_id: str,
        bus: Optional["MessageBus"] = None,
        consensus_threshold: float = 0.7,
    ):
        super().__init__(agent_id, AgentRole.EXECUTOR, bus=bus)
        self.consensus = ConsensusAlgorithm(threshold=consensus_threshold)
        self._accumulated_weights: Dict[str, float] = {}
        self._actions_taken: List[Dict[str, Any]] = []

    async def process_dance(self, dance_signal: Dict[str, Any]) -> None:
        """
        Accumulate a dance signal and attempt consensus formation.

        When enough signals accumulate, form consensus and take action.
        """
        direction = dance_signal.get("direction", "unknown")
        intensity = dance_signal.get("intensity", 0.0)

        self._accumulated_weights[direction] = (
            self._accumulated_weights.get(direction, 0.0) + intensity
        ) / 2  # running average

        if len(self._accumulated_weights) >= 1:
            result = await self.consensus.async_form_consensus(
                pattern_weights=dict(self._accumulated_weights),
            )
            await self._act_on_consensus(result)

    async def _act_on_consensus(self, result: ConsensusResult) -> None:
        """Take action based on a consensus result."""
        action = {
            "consensus_id": result.consensus_id,
            "consensus_type": result.consensus_type.value,
            "pattern": result.pattern,
            "confidence": result.confidence,
            "agent_id": self.agent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._actions_taken.append(action)

        if result.is_strong:
            logger.info(
                "Executor %s acting on strong consensus: %s (confidence=%.2f)",
                self.agent_id,
                result.pattern,
                result.confidence,
            )
            await self.send_dance({
                "type": "action",
                "action": "execute",
                "pattern": result.pattern,
                "confidence": result.confidence,
                "executor": self.agent_id,
            })
        elif result.requires_more_exploration:
            logger.debug(
                "Executor %s needs more exploration for consensus %s",
                self.agent_id,
                result.consensus_id,
            )
            await self.send_dance({
                "type": "signal",
                "action": "explore_more",
                "pattern": result.pattern,
                "executor": self.agent_id,
            })

    async def process_custom_message(self, message: Dict[str, Any]) -> None:
        """Handle custom messages — e.g. direct task assignments."""
        msg_type = message.get("type", "")
        if msg_type == "task":
            await self._execute_task(message)
        else:
            await super().process_custom_message(message)

    async def _execute_task(self, task: Dict[str, Any]) -> None:
        """Execute a directly assigned task."""
        task_id = task.get("task_id", "unknown")
        logger.info("Executor %s executing task %s", self.agent_id, task_id)

    def get_actions(self) -> List[Dict[str, Any]]:
        """Get all actions taken by this executor."""
        return list(self._actions_taken)

    def get_consensus_history(self, limit: int = 100) -> List[ConsensusResult]:
        """Get recent consensus history."""
        return self.consensus.get_consensus_history(limit)


class CoordinatorAgent(BeeAgent):
    """
    Coordinator agent that aggregates signals and emerges goals.

    Collects all dance signals via SignalAggregator, uses GoalEmerger
    to derive system-level goals, and broadcasts goals to the swarm.

    Args:
        agent_id: Unique identifier.
        bus: Optional message bus.
    """

    def __init__(
        self,
        agent_id: str,
        bus: Optional["MessageBus"] = None,
    ):
        super().__init__(agent_id, AgentRole.COORDINATOR, bus=bus)
        self.aggregator = SignalAggregator(bus=bus, source_agent_id=agent_id)
        self.goal_emerger = GoalEmerger()
        self._emitted_goals: List[SystemGoal] = []

    async def process_dance(self, dance_signal: Dict[str, Any]) -> None:
        """
        Process an incoming dance signal by feeding it to the aggregator
        and checking for emergent goals.
        """
        self.aggregator.receive_discovery({
            "signal_id": dance_signal.get("dance_id"),
            "discovery_type": dance_signal.get("direction", "unknown"),
            "pattern": dance_signal.get("pattern", {}),
            "confidence": dance_signal.get("intensity", 0.0),
            "impact": dance_signal.get("intensity", 0.0),
            "novelty": dance_signal.get("pattern", {}).get("novelty", 0.5),
            "agent_id": dance_signal.get("agent_id", "unknown"),
        })

        self.aggregator.receive_state_report({
            "agent_id": dance_signal.get("agent_id", "unknown"),
            "state": "exploring",
            "progress": dance_signal.get("intensity", 0.0),
        })

        snapshot = self.aggregator.generate_snapshot()
        goals = self._emerge_goals_from_snapshot(snapshot)
        for goal in goals:
            await self._broadcast_goal(goal)

    def _emerge_goals_from_snapshot(
        self, snapshot: SystemStateSnapshot
    ) -> List[SystemGoal]:
        """Run goal emergence on a system state snapshot."""
        goals = self.goal_emerger.emerge_goals(snapshot)
        self._emitted_goals.extend(goals)
        return goals

    async def _broadcast_goal(self, goal: SystemGoal) -> None:
        """Broadcast a goal to the swarm."""
        await self.send_dance({
            "type": "goal",
            "goal_id": goal.goal_id,
            "goal_type": goal.goal_type.value,
            "description": goal.description,
            "priority": goal.priority.value,
            "parameters": goal.parameters,
        })
        logger.info(
            "Coordinator %s broadcast goal: %s (%s)",
            self.agent_id,
            goal.goal_id,
            goal.goal_type.value,
        )

    async def process_custom_message(self, message: Dict[str, Any]) -> None:
        """Handle custom messages — e.g. state reports."""
        msg_type = message.get("type", "")
        if msg_type == "state_report":
            self.aggregator.receive_state_report(message)
        elif msg_type == "task_request":
            self.aggregator.receive_task_request(message)
        else:
            await super().process_custom_message(message)

    def get_snapshot(self) -> Optional[SystemStateSnapshot]:
        """Generate and return a current system state snapshot."""
        return self.aggregator.generate_snapshot()

    def get_active_goals(self) -> List[SystemGoal]:
        """Get all active goals."""
        return self.goal_emerger.get_active_goals()

    def get_all_goals(self) -> List[SystemGoal]:
        """Get all goals ever emitted."""
        return list(self._emitted_goals)
