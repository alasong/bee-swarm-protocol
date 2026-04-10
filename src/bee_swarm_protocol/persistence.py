"""Persistence and checkpoint support for swarm state.

This module provides:
- SwarmStateCheckpoint: serializes the complete swarm state
- StatePersister: saves/loads checkpoints to JSON or in-memory
- SwarmCheckpointMixin: mixin for checkpoint/restore on existing classes
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from bee_swarm_protocol.dance_parser import PatternWeights


# ---------------------------------------------------------------------------
# SwarmStateCheckpoint
# ---------------------------------------------------------------------------


@dataclass
class SwarmStateCheckpoint:
    """A serializable snapshot of the complete swarm state."""

    checkpoint_id: str
    timestamp: str

    # Pattern weights and support counts (from DanceLanguageParser)
    pattern_weights: Dict[str, float] = field(default_factory=dict)
    pattern_support_counts: Dict[str, int] = field(default_factory=dict)
    dance_count: int = 0

    # Consensus history (last N entries, stored as dicts)
    consensus_history: List[Dict[str, Any]] = field(default_factory=list)
    consensus_threshold: float = 0.7
    consensus_min_participants: int = 1

    # Active goals and their status (from GoalEmerger)
    active_goals: List[Dict[str, Any]] = field(default_factory=list)
    all_goals: List[Dict[str, Any]] = field(default_factory=list)
    goal_counter: int = 0

    # System state snapshot (from SignalAggregator)
    system_state: Optional[Dict[str, Any]] = None

    # Raw component state for full restore
    aggregator_state: Dict[str, Any] = field(default_factory=dict)
    parser_state: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        parser=None,
        consensus=None,
        emerger=None,
        aggregator=None,
        snapshot=None,
    ) -> "SwarmStateCheckpoint":
        """Build a checkpoint from live component instances.

        Each argument is optional — omit components you don't have.
        """
        pattern_weights: Dict[str, float] = {}
        pattern_support_counts: Dict[str, int] = {}
        dance_count = 0
        parser_state: Dict[str, Any] = {}

        if parser is not None:
            pattern_weights = parser.accumulate_dances()
            pw = getattr(parser, "_pattern_weights", None)
            if pw is not None:
                pattern_support_counts = dict(getattr(pw, "support_counts", {}))
            dance_count = len(getattr(parser, "_dances", []))
            # Capture full parser state for restore
            parser_state = {
                "intensity_threshold": getattr(parser, "intensity_threshold", 0.5),
            }

        consensus_history: List[Dict[str, Any]] = []
        consensus_threshold = 0.7
        consensus_min_participants = 1

        if consensus is not None:
            raw_history = consensus.get_consensus_history(limit=1000)
            consensus_history = [_serialise_consensus_result(c) for c in raw_history]
            consensus_threshold = getattr(consensus, "threshold", 0.7)
            consensus_min_participants = getattr(consensus, "min_participants", 1)

        active_goals: List[Dict[str, Any]] = []
        all_goals: List[Dict[str, Any]] = []
        goal_counter = 0

        if emerger is not None:
            active_goals = [g.to_dict() for g in emerger.get_active_goals()]
            all_goals = [g.to_dict() for g in emerger.get_all_goals()]
            goal_counter = getattr(emerger, "_goal_counter", 0)

        system_state: Optional[Dict[str, Any]] = None
        aggregator_state: Dict[str, Any] = {}

        if snapshot is not None:
            system_state = snapshot.to_dict()

        if aggregator is not None:
            all_agent_states = aggregator.get_all_agent_states()
            aggregator_state = {
                "snapshot_interval_ms": getattr(aggregator, "snapshot_interval_ms", 1000),
                "snapshot_counter": getattr(aggregator, "_snapshot_counter", 0),
                "agent_count": len(all_agent_states),
                "agent_states": {
                    aid: {
                        "agent_id": info.agent_id,
                        "state": info.state,
                        "progress": info.progress,
                        "current_task_id": info.current_task_id,
                        "resource_usage": info.resource_usage,
                    }
                    for aid, info in all_agent_states.items()
                },
            }

        return cls(
            checkpoint_id=f"ckpt_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            pattern_weights=pattern_weights,
            pattern_support_counts=pattern_support_counts,
            dance_count=dance_count,
            consensus_history=consensus_history,
            consensus_threshold=consensus_threshold,
            consensus_min_participants=consensus_min_participants,
            active_goals=active_goals,
            all_goals=all_goals,
            goal_counter=goal_counter,
            system_state=system_state,
            aggregator_state=aggregator_state,
            parser_state=parser_state,
        )


def _serialise_consensus_result(result) -> Dict[str, Any]:
    """Convert a ConsensusResult to a JSON-serialisable dict."""
    d = {
        "consensus_id": result.consensus_id,
        "consensus_type": result.consensus_type.value,
        "pattern": result.pattern,
        "confidence": result.confidence,
        "participating_agents": result.participating_agents,
        "requires_more_exploration": result.requires_more_exploration,
        "details": result.details,
        "timestamp": result.timestamp.isoformat() if result.timestamp else None,
    }
    return d


# ---------------------------------------------------------------------------
# StatePersister
# ---------------------------------------------------------------------------


class StatePersister:
    """Handles saving and loading SwarmStateCheckpoints."""

    def __init__(self):
        self._memory_store: List[SwarmStateCheckpoint] = []

    # -- File persistence --

    def save(self, checkpoint: SwarmStateCheckpoint, path: str) -> None:
        """Serialize a checkpoint to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(asdict(checkpoint), f, indent=2)

    def load(self, path: str) -> SwarmStateCheckpoint:
        """Deserialize a SwarmStateCheckpoint from a JSON file."""
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SwarmStateCheckpoint(**data)

    # -- In-memory persistence --

    def save_to_memory(self, checkpoint: SwarmStateCheckpoint) -> None:
        """Store a checkpoint in the in-memory list."""
        self._memory_store.append(checkpoint)

    def get_latest(self) -> Optional[SwarmStateCheckpoint]:
        """Return the most recently saved in-memory checkpoint, or None."""
        if self._memory_store:
            return self._memory_store[-1]
        return None

    def get_all(self) -> List[SwarmStateCheckpoint]:
        """Return all in-memory checkpoints."""
        return list(self._memory_store)

    def clear_memory(self) -> None:
        """Clear the in-memory store."""
        self._memory_store.clear()


# ---------------------------------------------------------------------------
# SwarmCheckpointMixin
# ---------------------------------------------------------------------------


class SwarmCheckpointMixin:
    """Mixin that adds checkpoint() and restore(checkpoint) methods.

    Designed to be combined with SignalAggregator, ConsensusAlgorithm,
    and/or GoalEmerger via multiple inheritance.

    Usage example:
        class CheckpointedAggregator(SwarmCheckpointMixin, SignalAggregator):
            pass
    """

    def _get_sub_components(self) -> Dict[str, Any]:
        """Return a dict of named sub-components for checkpoint creation.

        Subclasses should override this to expose their related objects
        (parser, consensus, emerger, aggregator, snapshot).
        """
        return {}

    def checkpoint(self) -> SwarmStateCheckpoint:
        """Capture current state as a SwarmStateCheckpoint."""
        components = self._get_sub_components()
        return SwarmStateCheckpoint.create(**components)

    def restore(self, cp: SwarmStateCheckpoint) -> None:
        """Restore state from a SwarmStateCheckpoint.

        Clears existing state and resets counters to checkpoint values.
        """
        components = self._get_sub_components()

        # Restore parser state
        parser = components.get("parser")
        if parser is not None:
            if hasattr(parser, "_dances"):
                parser._dances.clear()
            if hasattr(parser, "_pattern_weights"):
                parser._pattern_weights = PatternWeights()
                for pattern_type, weight in cp.pattern_weights.items():
                    count = cp.pattern_support_counts.get(pattern_type, 1)
                    if count > 0:
                        parser._pattern_weights.weights[pattern_type] = weight
                        parser._pattern_weights.support_counts[pattern_type] = count
            if hasattr(parser, "_dance_counter"):
                parser._dance_counter = cp.dance_count
            if hasattr(parser, "intensity_threshold"):
                parser.intensity_threshold = cp.parser_state.get(
                    "intensity_threshold", 0.5
                )

        # Restore consensus state
        consensus = components.get("consensus")
        if consensus is not None:
            if hasattr(consensus, "_consensus_history"):
                consensus._consensus_history.clear()
            if hasattr(consensus, "threshold"):
                consensus.threshold = cp.consensus_threshold
            if hasattr(consensus, "min_participants"):
                consensus.min_participants = cp.consensus_min_participants
            if hasattr(consensus, "_consensus_counter"):
                consensus._consensus_counter = len(cp.consensus_history)

        # Restore emerger state
        emerger = components.get("emerger")
        if emerger is not None:
            if hasattr(emerger, "_goals"):
                emerger._goals.clear()
            if hasattr(emerger, "_goal_counter"):
                emerger._goal_counter = cp.goal_counter

        # Restore aggregator state
        aggregator = components.get("aggregator")
        if aggregator is not None:
            if hasattr(aggregator, "_state_reports"):
                aggregator._state_reports.clear()
            if hasattr(aggregator, "_discoveries"):
                aggregator._discoveries.clear()
            if hasattr(aggregator, "_task_requests"):
                aggregator._task_requests.clear()
            if hasattr(aggregator, "_snapshot_counter"):
                aggregator._snapshot_counter = cp.aggregator_state.get(
                    "snapshot_counter", 0
                )
