"""Tests for bee-swarm-protocol."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from bee_swarm_protocol import (
    AgentLocation,
    ConsensusAlgorithm,
    ConsensusType,
    Dance,
    DanceDecay,
    DanceLanguageParser,
    DancePropagator,
    DanceResponseHandler,
    DanceSignal,
    DanceVisualizer,
    GoalEmerger,
    GoalPriority,
    GoalType,
    SignalAggregator,
    SystemNeed,
    SystemStateSnapshot,
)

# --- DanceSignal ---


class TestDanceSignal:
    def test_creation(self):
        signal = DanceSignal(
            dance_id="dance_1",
            agent_id="explorer",
            intensity=0.8,
            direction="anomaly",
            duration=8.0,
            pattern={"type": "anomaly"},
        )
        assert signal.dance_id == "dance_1"
        assert signal.intensity == 0.8
        assert signal.direction == "anomaly"


# --- DanceLanguageParser ---


class TestDanceLanguageParser:
    def test_parse_high_quality_discovery(self):
        parser = DanceLanguageParser()
        dance = parser.parse_discovery({
            "agent_id": "explorer_1",
            "confidence": 0.9,
            "impact": 0.8,
            "novelty": 0.7,
            "pattern": {"type": "anomaly"},
        })
        assert dance is not None
        assert dance.intensity == pytest.approx(0.4 * 0.9 + 0.3 * 0.8 + 0.3 * 0.7)
        assert dance.direction == "anomaly"
        assert dance.agent_id == "explorer_1"

    def test_low_quality_filtered(self):
        parser = DanceLanguageParser(intensity_threshold=0.7)
        dance = parser.parse_discovery({
            "agent_id": "x",
            "confidence": 0.2,
            "impact": 0.1,
            "novelty": 0.1,
            "pattern": {"type": "noise"},
        })
        assert dance is None

    def test_accumulate_weights(self):
        parser = DanceLanguageParser()
        for i in range(3):
            parser.parse_discovery({
                "agent_id": f"agent_{i}",
                "confidence": 0.8,
                "impact": 0.7,
                "novelty": 0.6,
                "pattern": {"type": "pattern_a"},
            })
        weights = parser.accumulate_dances()
        assert "pattern_a" in weights
        # weight = avg_intensity * log(count + 1)
        expected = weights["pattern_a"]
        assert expected > 0

    def test_top_patterns(self):
        parser = DanceLanguageParser()
        parser.parse_discovery({
            "agent_id": "a", "confidence": 0.9, "impact": 0.9, "novelty": 0.9,
            "pattern": {"type": "strong"},
        })
        parser.parse_discovery({
            "agent_id": "b", "confidence": 0.3, "impact": 0.2, "novelty": 0.2,
            "pattern": {"type": "weak"},
        })
        top = parser.get_top_patterns(1)
        assert top[0][0] == "strong"

    def test_clear_and_reset(self):
        parser = DanceLanguageParser()
        parser.parse_discovery({
            "agent_id": "a", "confidence": 0.9, "impact": 0.8, "novelty": 0.7,
            "pattern": {"type": "t"},
        })
        assert len(parser.get_dances()) == 1
        parser.clear_dances()
        assert len(parser.get_dances()) == 0

        parser.reset_weights()
        assert parser.accumulate_dances() == {}


# --- DancePropagator ---


class TestDancePropagator:
    def test_register_and_propagate(self):
        prop = DancePropagator()
        loc1 = AgentLocation("sender", x=100, y=100)
        loc2 = AgentLocation("receiver", x=120, y=100)
        prop.register_agent(loc1)
        prop.register_agent(loc2)

        signal = DanceSignal("d1", "sender", intensity=0.8, direction="test", duration=5.0)
        dance = Dance(signal=signal, location=loc1)

        results = prop.propagate(dance, ["receiver"])
        assert "receiver" in results
        assert results["receiver"] > 0

    def test_out_of_range(self):
        prop = DancePropagator()
        prop.register_agent(AgentLocation("a", x=0, y=0))
        prop.register_agent(AgentLocation("b", x=500, y=500))

        signal = DanceSignal("d1", "a", intensity=0.3, direction="x", duration=3.0)
        dance = Dance(signal=signal, location=AgentLocation("a", x=0, y=0))

        results = prop.propagate(dance, ["b"])
        assert "b" not in results  # too far

    def test_broadcast(self):
        prop = DancePropagator()
        for i in range(5):
            prop.register_agent(AgentLocation(f"agent_{i}", x=i * 10, y=0))

        signal = DanceSignal("d1", "sender", intensity=0.9, direction="bcast", duration=9.0)
        dance = Dance(signal=signal, location=AgentLocation("sender", x=0, y=0))

        results = prop.broadcast(dance)
        # All agents within range due to high intensity
        assert len(results) >= 1

    def test_visible_dances_sorted(self):
        prop = DancePropagator()
        prop.register_agent(AgentLocation("observer", x=50, y=50))

        for i in range(3):
            loc = AgentLocation(f"src_{i}", x=50, y=50)
            signal = DanceSignal(
                f"d{i}", f"src_{i}",
                intensity=0.3 + i * 0.2, direction="t", duration=5.0,
            )
            dance = Dance(signal=signal, location=loc)
            prop.propagate(dance, ["observer"])

        visible = prop.get_visible_dances("observer")
        assert len(visible) == 3
        # Sorted by intensity descending
        assert visible[0].decayed_intensity >= visible[-1].decayed_intensity


# --- DanceResponseHandler ---


class TestDanceResponseHandler:
    def test_respond_and_aggregate(self):
        prop = DancePropagator()
        prop.register_agent(AgentLocation("agent_1", x=50, y=50))
        handler = DanceResponseHandler(prop)

        signal = DanceSignal("d1", "src", intensity=0.8, direction="t", duration=5.0)
        dance = Dance(signal=signal, location=AgentLocation("src", x=50, y=50))
        prop.propagate(dance, ["agent_1"])

        handler.respond_to_dance("agent_1", dance, {"type": "agree", "confidence": 0.9})
        responses = handler.get_responses("d1")
        assert len(responses) == 1

        agg = handler.get_aggregated_response("d1")
        assert agg["count"] == 1
        assert agg["average_confidence"] == pytest.approx(0.9)


# --- DanceDecay ---


class TestDanceDecay:
    def test_decay_reduces_intensity(self):
        prop = DancePropagator()
        decay = DanceDecay(prop, decay_rate=0.1)

        signal = DanceSignal("d1", "src", intensity=1.0, direction="t", duration=10.0)
        dance = Dance(signal=signal, location=AgentLocation("src", x=0, y=0))
        prop.propagate(dance, [])

        decay.apply_decay(dance, time_elapsed=10.0)
        expected = 1.0 * math.exp(-0.1 * 10.0)
        assert dance.decayed_intensity == pytest.approx(expected, abs=0.01)

    def test_cleanup_expired(self):
        prop = DancePropagator()
        decay = DanceDecay(prop, decay_rate=1.0, expiry_threshold=0.05)

        signal = DanceSignal("d1", "src", intensity=0.1, direction="t", duration=1.0)
        dance = Dance(signal=signal, location=AgentLocation("src", x=0, y=0))
        # Manually set timestamp to 10 seconds ago
        dance.signal.timestamp = datetime.now(timezone.utc) - timedelta(seconds=10)
        prop.propagate(dance, [])

        removed = decay.cleanup_expired_dances()
        assert "d1" in removed

    def test_invalid_decay_rate(self):
        prop = DancePropagator()
        decay = DanceDecay(prop)
        with pytest.raises(ValueError):
            decay.set_decay_rate(-1.0)


# --- DanceVisualizer ---


class TestDanceVisualizer:
    def test_visualize_produces_output(self):
        prop = DancePropagator()
        viz = DanceVisualizer(prop, grid_width=10, grid_height=5)

        signal = DanceSignal("d1", "src", intensity=0.9, direction="t", duration=9.0)
        dance = Dance(
            signal=signal,
            location=AgentLocation("src", x=250, y=250, zone="center"),
        )
        prop.propagate(dance, [])

        output = viz.visualize([dance])
        assert "+" in output
        assert "X" in output  # high intensity symbol

    def test_heat_map(self):
        prop = DancePropagator()
        viz = DanceVisualizer(prop)

        signal = DanceSignal("d1", "src", intensity=0.7, direction="anomaly", duration=7.0)
        dance = Dance(
            signal=signal,
            location=AgentLocation("src", x=100, y=100, zone="zone_a"),
        )
        prop.propagate(dance, [])

        heat = viz.get_heat_map()
        assert "zone:zone_a" in heat
        assert "type:anomaly" in heat


# --- SignalAggregator ---


class TestSignalAggregator:
    def test_state_reports_and_snapshot(self):
        agg = SignalAggregator()
        agg.receive_state_report({
            "agent_id": "agent_1",
            "state": "executing",
            "progress": 0.5,
            "resource_usage": {"cpu": 0.6, "memory": 0.4},
        })
        agg.receive_state_report({
            "agent_id": "agent_2",
            "state": "idle",
            "progress": 0.0,
        })

        snapshot = agg.generate_snapshot()
        assert snapshot.total_agents == 2
        assert snapshot.executing_agents == 1
        assert snapshot.idle_agents == 1
        assert snapshot.system_load == pytest.approx(0.5)

    def test_discovery_signals(self):
        agg = SignalAggregator()
        agg.receive_discovery({
            "signal_id": "disc_1",
            "agent_id": "explorer",
            "discovery_type": "pattern",
            "confidence": 0.9,
            "impact": 0.8,
            "novelty": 0.7,
            "pattern": {"type": "anomaly"},
        })
        snapshot = agg.generate_snapshot()
        assert snapshot.discovery_count == 1
        assert len(snapshot.pending_discoveries) == 1

    def test_task_requests(self):
        agg = SignalAggregator()
        agg.receive_task_request({
            "signal_id": "req_1",
            "agent_id": "executor",
            "capabilities": ["compute"],
            "resource_availability": {"cpu": 0.3},
        })
        snapshot = agg.generate_snapshot()
        assert snapshot.task_request_count == 1

    def test_snapshot_to_dict(self):
        agg = SignalAggregator()
        agg.receive_state_report({"agent_id": "a1", "state": "idle"})
        snap = agg.generate_snapshot()
        d = snap.to_dict()
        assert "total_agents" in d
        assert "system_load" in d


# --- ConsensusAlgorithm ---


class TestConsensusAlgorithm:
    def test_strong_consensus(self):
        algo = ConsensusAlgorithm(threshold=0.7)
        result = algo.form_consensus(
            pattern_weights={"anomaly": 0.85, "normal": 0.3},
            agent_counts={"anomaly": 5, "normal": 2},
        )
        assert result.consensus_type == ConsensusType.STRONG
        assert result.pattern == "anomaly"
        assert result.is_strong

    def test_weak_consensus(self):
        algo = ConsensusAlgorithm(threshold=0.7)
        result = algo.form_consensus(
            pattern_weights={"anomaly": 0.4},
        )
        assert result.consensus_type == ConsensusType.WEAK
        assert result.requires_more_exploration

    def test_no_consensus(self):
        algo = ConsensusAlgorithm(threshold=0.7)
        result = algo.form_consensus(pattern_weights={"a": 0.1})
        assert result.consensus_type == ConsensusType.NONE

    def test_empty_weights(self):
        algo = ConsensusAlgorithm()
        result = algo.form_consensus({})
        assert result.consensus_type == ConsensusType.NONE
        assert result.requires_more_exploration

    def test_consensus_rate(self):
        algo = ConsensusAlgorithm(threshold=0.5)
        algo.form_consensus({"a": 0.8})  # strong
        algo.form_consensus({"a": 0.3})  # weak (>= 0.25)
        algo.form_consensus({"a": 0.1})  # none
        rate = algo.get_consensus_rate()
        assert rate == pytest.approx(2 / 3)


# --- GoalEmerger ---


class TestGoalEmerger:
    def _make_snapshot(self, **overrides):
        defaults = {
            "snapshot_id": "snap_1",
            "total_agents": 5,
            "idle_agents": 3,
            "executing_agents": 1,
            "exploring_agents": 1,
            "blocked_agents": 0,
            "discovery_count": 0,
            "task_request_count": 0,
            "system_load": 0.4,
        }
        defaults.update(overrides)
        return SystemStateSnapshot(**defaults)

    def test_identify_high_load_need(self):
        emerger = GoalEmerger()
        snapshot = self._make_snapshot(system_load=0.9)
        needs = emerger.identify_needs(snapshot)
        assert any(n.need_type == "resource_optimization" for n in needs)

    def test_identify_task_capacity_need(self):
        emerger = GoalEmerger()
        snapshot = self._make_snapshot(idle_agents=1, task_request_count=5)
        needs = emerger.identify_needs(snapshot)
        assert any(n.need_type == "task_capacity" for n in needs)

    def test_identify_blocked_agents_need(self):
        emerger = GoalEmerger()
        snapshot = self._make_snapshot(blocked_agents=2)
        needs = emerger.identify_needs(snapshot)
        assert any(n.need_type == "coordination" for n in needs)

    def test_generate_goal(self):
        emerger = GoalEmerger()
        need = SystemNeed(
            need_type="resource_optimization",
            description="High load",
            priority=GoalPriority.HIGH,
            metrics={"load": 0.9},
        )
        goal = emerger.generate_goal(need)
        assert goal.goal_type == GoalType.RESOURCE_OPTIMIZATION
        assert goal.priority == GoalPriority.HIGH

    def test_decompose_goal(self):
        emerger = GoalEmerger()
        need = SystemNeed(
            need_type="task_capacity",
            description="Not enough agents",
            priority=GoalPriority.HIGH,
            metrics={},
        )
        goal = emerger.generate_goal(need)
        tasks = emerger.decompose_goal(goal)
        assert len(tasks) >= 2
        assert goal.sub_tasks == tasks

    def test_emerge_goals_full(self):
        emerger = GoalEmerger()
        snapshot = self._make_snapshot(
            system_load=0.9, idle_agents=1, task_request_count=5, blocked_agents=2
        )
        goals = emerger.emerge_goals(snapshot)
        assert len(goals) >= 2
        # All goals should have sub-tasks
        for goal in goals:
            assert len(goal.sub_tasks) > 0

    def test_complete_goal(self):
        emerger = GoalEmerger()
        need = SystemNeed(
            need_type="coordination", description="fix", priority=GoalPriority.MEDIUM, metrics={}
        )
        goal = emerger.generate_goal(need)
        assert emerger.complete_goal(goal.goal_id)
        assert not emerger.complete_goal("nonexistent")
        assert goal.status == "completed"

    def test_get_active_goals(self):
        emerger = GoalEmerger()
        need = SystemNeed(
            need_type="coordination", description="fix", priority=GoalPriority.MEDIUM, metrics={}
        )
        goal = emerger.generate_goal(need)
        active = emerger.get_active_goals()
        assert len(active) == 1
        emerger.complete_goal(goal.goal_id)
        assert len(emerger.get_active_goals()) == 0
