"""Tests for persistence module."""

import json
import tempfile
from pathlib import Path

import pytest

from bee_swarm_protocol import (
    ConsensusAlgorithm,
    ConsensusType,
    DanceLanguageParser,
    GoalEmerger,
    GoalPriority,
    SignalAggregator,
    SystemNeed,
    SystemStateSnapshot,
)
from bee_swarm_protocol.dance_parser import PatternWeights
from bee_swarm_protocol.persistence import (
    StatePersister,
    SwarmCheckpointMixin,
    SwarmStateCheckpoint,
)


# --- SwarmStateCheckpoint ---


class TestSwarmStateCheckpoint:
    def test_create_empty(self):
        cp = SwarmStateCheckpoint.create()
        assert cp.checkpoint_id.startswith("ckpt_")
        assert cp.pattern_weights == {}
        assert cp.consensus_history == []
        assert cp.active_goals == []
        assert cp.system_state is None

    def test_create_with_parser(self):
        parser = DanceLanguageParser(intensity_threshold=0.3)
        parser.parse_discovery({
            "agent_id": "a", "confidence": 0.9, "impact": 0.8, "novelty": 0.7,
            "pattern": {"type": "anomaly"},
        })
        parser.parse_discovery({
            "agent_id": "b", "confidence": 0.8, "impact": 0.7, "novelty": 0.6,
            "pattern": {"type": "anomaly"},
        })

        cp = SwarmStateCheckpoint.create(parser=parser)
        assert "anomaly" in cp.pattern_weights
        assert cp.pattern_support_counts.get("anomaly", 0) == 2
        assert cp.dance_count == 2
        assert cp.parser_state["intensity_threshold"] == 0.3

    def test_create_with_consensus(self):
        consensus = ConsensusAlgorithm(threshold=0.6, min_participants=2)
        consensus.form_consensus({"a": 0.8}, {"a": 3})
        consensus.form_consensus({"b": 0.3})

        cp = SwarmStateCheckpoint.create(consensus=consensus)
        assert len(cp.consensus_history) == 2
        assert cp.consensus_history[0]["pattern"] == "a"
        assert cp.consensus_history[0]["consensus_type"] == "strong"
        assert cp.consensus_threshold == 0.6
        assert cp.consensus_min_participants == 2

    def test_create_with_emerger(self):
        emerger = GoalEmerger()
        need = SystemNeed(
            need_type="resource_optimization",
            description="High load",
            priority=GoalPriority.HIGH,
            metrics={"load": 0.9},
        )
        goal = emerger.generate_goal(need)
        emerger.complete_goal(goal.goal_id)

        cp = SwarmStateCheckpoint.create(emerger=emerger)
        assert len(cp.active_goals) == 0  # completed, not active
        assert len(cp.all_goals) == 1
        assert cp.goal_counter == 1

    def test_create_with_aggregator(self):
        agg = SignalAggregator()
        agg.receive_state_report({"agent_id": "a1", "state": "idle"})
        agg.receive_state_report({"agent_id": "a2", "state": "executing"})

        cp = SwarmStateCheckpoint.create(aggregator=agg)
        assert cp.aggregator_state["agent_count"] == 2
        assert "a1" in cp.aggregator_state["agent_states"]
        assert "a2" in cp.aggregator_state["agent_states"]

    def test_create_with_snapshot(self):
        snap = SystemStateSnapshot(
            snapshot_id="s1",
            total_agents=10,
            idle_agents=5,
            executing_agents=3,
            exploring_agents=1,
            blocked_agents=1,
            discovery_count=2,
            task_request_count=1,
            system_load=0.4,
        )
        cp = SwarmStateCheckpoint.create(snapshot=snap)
        assert cp.system_state["total_agents"] == 10
        assert cp.system_state["system_load"] == pytest.approx(0.4)

    def test_create_with_all_components(self):
        parser = DanceLanguageParser(intensity_threshold=0.3)
        parser.parse_discovery({
            "agent_id": "a", "confidence": 0.9, "impact": 0.8, "novelty": 0.7,
            "pattern": {"type": "anomaly"},
        })

        consensus = ConsensusAlgorithm(threshold=0.6)
        consensus.form_consensus({"anomaly": 0.8}, {"anomaly": 1})

        agg = SignalAggregator()
        agg.receive_state_report({"agent_id": "a1", "state": "idle"})
        snap = agg.generate_snapshot()

        emerger = GoalEmerger()

        cp = SwarmStateCheckpoint.create(
            parser=parser,
            consensus=consensus,
            emerger=emerger,
            aggregator=agg,
            snapshot=snap,
        )
        assert "anomaly" in cp.pattern_weights
        assert len(cp.consensus_history) == 1
        assert cp.system_state is not None
        assert cp.aggregator_state["agent_count"] == 1


# --- StatePersister ---


class TestStatePersister:
    def test_save_and_load_file(self):
        cp = SwarmStateCheckpoint.create()
        persister = StatePersister()

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "nested" / "checkpoint.json")
            persister.save(cp, path)
            loaded = persister.load(path)

            assert loaded.checkpoint_id == cp.checkpoint_id
            assert loaded.timestamp == cp.timestamp

    def test_save_and_load_with_data(self):
        parser = DanceLanguageParser(intensity_threshold=0.3)
        parser.parse_discovery({
            "agent_id": "a", "confidence": 0.9, "impact": 0.8, "novelty": 0.7,
            "pattern": {"type": "anomaly"},
        })
        cp = SwarmStateCheckpoint.create(parser=parser)
        persister = StatePersister()

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "cp.json")
            persister.save(cp, path)
            loaded = persister.load(path)

            assert loaded.pattern_weights["anomaly"] == pytest.approx(
                cp.pattern_weights["anomaly"]
            )
            assert loaded.pattern_support_counts["anomaly"] == 1

    def test_memory_store(self):
        persister = StatePersister()
        cp1 = SwarmStateCheckpoint.create()
        cp2 = SwarmStateCheckpoint.create()

        persister.save_to_memory(cp1)
        persister.save_to_memory(cp2)

        assert persister.get_latest().checkpoint_id == cp2.checkpoint_id
        assert len(persister.get_all()) == 2

    def test_get_latest_empty(self):
        persister = StatePersister()
        assert persister.get_latest() is None

    def test_clear_memory(self):
        persister = StatePersister()
        persister.save_to_memory(SwarmStateCheckpoint.create())
        persister.clear_memory()
        assert persister.get_latest() is None
        assert persister.get_all() == []

    def test_save_and_load_consensus_history(self):
        consensus = ConsensusAlgorithm(threshold=0.7)
        consensus.form_consensus({"pattern_a": 0.85})
        consensus.form_consensus({"pattern_b": 0.2})
        cp = SwarmStateCheckpoint.create(consensus=consensus)

        persister = StatePersister()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "cp.json")
            persister.save(cp, path)
            loaded = persister.load(path)

        assert len(loaded.consensus_history) == 2
        assert loaded.consensus_history[0]["consensus_type"] == "strong"
        assert loaded.consensus_history[1]["consensus_type"] == "none"

    def test_file_json_valid(self):
        """Verify saved files contain valid JSON."""
        cp = SwarmStateCheckpoint.create()
        persister = StatePersister()

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "cp.json")
            persister.save(cp, path)
            with open(path, "r") as f:
                data = json.load(f)
            assert "checkpoint_id" in data
            assert "pattern_weights" in data


# --- SwarmCheckpointMixin ---


class TestSwarmCheckpointMixin:
    def test_checkpoint_empty_mixin(self):
        """Mixin with no sub-components returns an empty checkpoint."""

        class EmptyMixinClass(SwarmCheckpointMixin):
            pass

        obj = EmptyMixinClass()
        cp = obj.checkpoint()
        assert isinstance(cp, SwarmStateCheckpoint)
        assert cp.pattern_weights == {}

    def test_checkpoint_with_components(self):
        """Mixin that exposes sub-components."""

        class MixinWithComponents(SwarmCheckpointMixin):
            def __init__(self):
                self.parser = DanceLanguageParser(intensity_threshold=0.3)
                self.consensus = ConsensusAlgorithm(threshold=0.6)

            def _get_sub_components(self):
                return {
                    "parser": self.parser,
                    "consensus": self.consensus,
                }

        obj = MixinWithComponents()
        obj.parser.parse_discovery({
            "agent_id": "a", "confidence": 0.9, "impact": 0.8, "novelty": 0.7,
            "pattern": {"type": "anomaly"},
        })
        obj.consensus.form_consensus({"anomaly": 0.8}, {"anomaly": 1})

        cp = obj.checkpoint()
        assert "anomaly" in cp.pattern_weights
        assert len(cp.consensus_history) == 1

    def test_restore_clears_parser_state(self):
        class MixinWithParser(SwarmCheckpointMixin):
            def __init__(self):
                self.parser = DanceLanguageParser(intensity_threshold=0.3)

            def _get_sub_components(self):
                return {"parser": self.parser}

        obj = MixinWithParser()
        obj.parser.parse_discovery({
            "agent_id": "a", "confidence": 0.9, "impact": 0.8, "novelty": 0.7,
            "pattern": {"type": "anomaly"},
        })
        assert len(obj.parser._dances) == 1

        cp_before = obj.checkpoint()

        # Add more data
        obj.parser.parse_discovery({
            "agent_id": "b", "confidence": 0.8, "impact": 0.7, "novelty": 0.6,
            "pattern": {"type": "noise"},
        })
        assert len(obj.parser._dances) == 2

        # Restore to checkpoint (state is restored to checkpoint values)
        obj.restore(cp_before)
        assert len(obj.parser._dances) == 0  # dances cleared (not serialized)
        assert obj.parser._dance_counter == 1  # restored to cp.dance_count

    def test_restore_clears_consensus_state(self):
        class MixinWithConsensus(SwarmCheckpointMixin):
            def __init__(self):
                self.consensus = ConsensusAlgorithm(threshold=0.5)

            def _get_sub_components(self):
                return {"consensus": self.consensus}

        obj = MixinWithConsensus()
        obj.consensus.form_consensus({"a": 0.8})
        obj.consensus.form_consensus({"b": 0.3})
        assert len(obj.consensus._consensus_history) == 2

        cp_before = obj.checkpoint()

        obj.consensus.form_consensus({"c": 0.9})
        assert len(obj.consensus._consensus_history) == 3

        obj.restore(cp_before)
        assert len(obj.consensus._consensus_history) == 0  # history cleared
        assert obj.consensus._consensus_counter == 2  # restored to len(consensus_history)

    def test_restore_clears_emerger_state(self):
        class MixinWithEmerger(SwarmCheckpointMixin):
            def __init__(self):
                self.emerger = GoalEmerger()

            def _get_sub_components(self):
                return {"emerger": self.emerger}

        obj = MixinWithEmerger()
        need = SystemNeed(
            need_type="coordination",
            description="fix",
            priority=GoalPriority.MEDIUM,
            metrics={},
        )
        obj.emerger.generate_goal(need)
        assert len(obj.emerger._goals) == 1

        cp_before = obj.checkpoint()

        obj.emerger.generate_goal(need)
        assert len(obj.emerger._goals) == 2

        obj.restore(cp_before)
        assert len(obj.emerger._goals) == 0
        assert obj.emerger._goal_counter == 1  # restored from checkpoint

    def test_restore_clears_aggregator_state(self):
        class MixinWithAggregator(SwarmCheckpointMixin):
            def __init__(self):
                self.aggregator = SignalAggregator()

            def _get_sub_components(self):
                return {"aggregator": self.aggregator}

        obj = MixinWithAggregator()
        obj.aggregator.receive_state_report({"agent_id": "a1", "state": "idle"})
        obj.aggregator.generate_snapshot()

        cp_before = obj.checkpoint()

        obj.aggregator.receive_state_report({"agent_id": "a2", "state": "executing"})
        assert len(obj.aggregator._state_reports) == 2

        obj.restore(cp_before)
        assert len(obj.aggregator._state_reports) == 0
        assert obj.aggregator._snapshot_counter == cp_before.aggregator_state["snapshot_counter"]

    def test_mixin_with_multiple_inheritance_aggregator(self):
        """Mixin works with SignalAggregator via multiple inheritance."""

        class CheckpointedAggregator(SwarmCheckpointMixin, SignalAggregator):
            def _get_sub_components(self):
                return {"aggregator": self}

        agg = CheckpointedAggregator()
        agg.receive_state_report({"agent_id": "a1", "state": "idle"})
        agg.receive_state_report({"agent_id": "a2", "state": "executing"})

        cp = agg.checkpoint()
        assert cp.aggregator_state["agent_count"] == 2

        agg.receive_state_report({"agent_id": "a3", "state": "blocked"})
        assert len(agg._state_reports) == 3

        agg.restore(cp)
        assert len(agg._state_reports) == 0

    def test_mixin_with_multiple_inheritance_consensus(self):
        """Mixin works with ConsensusAlgorithm via multiple inheritance."""

        class CheckpointedConsensus(SwarmCheckpointMixin, ConsensusAlgorithm):
            def _get_sub_components(self):
                return {"consensus": self}

        algo = CheckpointedConsensus(threshold=0.7)
        algo.form_consensus({"a": 0.9})
        algo.form_consensus({"b": 0.2})

        cp = algo.checkpoint()
        assert len(cp.consensus_history) == 2

        algo.form_consensus({"c": 0.8})
        assert len(algo._consensus_history) == 3

        algo.restore(cp)
        assert len(algo._consensus_history) == 0
        assert algo._consensus_counter == 2  # restored from checkpoint

    def test_mixin_with_multiple_inheritance_emerger(self):
        """Mixin works with GoalEmerger via multiple inheritance."""

        class CheckpointedEmerger(SwarmCheckpointMixin, GoalEmerger):
            def _get_sub_components(self):
                return {"emerger": self}

        emerger = CheckpointedEmerger()
        need = SystemNeed(
            need_type="resource_optimization",
            description="High load",
            priority=GoalPriority.HIGH,
            metrics={"load": 0.9},
        )
        goal = emerger.generate_goal(need)

        cp = emerger.checkpoint()
        assert len(cp.all_goals) == 1

        emerger.generate_goal(need)
        assert len(emerger._goals) == 2

        emerger.restore(cp)
        assert len(emerger._goals) == 0
        assert emerger._goal_counter == 1  # restored from checkpoint
