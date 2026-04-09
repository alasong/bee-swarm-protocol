"""
Example: Bee-inspired swarm coordination pipeline.

Demonstrates the full emergence loop:
  Discovery → Dance → Propagation → Consensus → Goal Emergence
"""

from bee_swarm_protocol import (
    AgentLocation,
    ConsensusAlgorithm,
    Dance,
    DanceDecay,
    DanceLanguageParser,
    DancePropagator,
    DanceResponseHandler,
    DanceSignal,
    GoalEmerger,
    SignalAggregator,
)


def main():
    print("=== Bee Swarm Protocol Demo ===\n")

    # --- Step 1: Signal Aggregator collects state ---
    agg = SignalAggregator()
    for agent_id, state in [
        ("explorer_1", "exploring"),
        ("explorer_2", "exploring"),
        ("executor_1", "executing"),
        ("executor_2", "idle"),
        ("validator_1", "idle"),
    ]:
        agg.receive_state_report({"agent_id": agent_id, "state": state})

    # --- Step 2: Discovery signals → Dance ---
    parser = DanceLanguageParser(intensity_threshold=0.4)
    discoveries = [
        {"agent_id": "explorer_1", "confidence": 0.9, "impact": 0.8, "novelty": 0.7,
         "pattern": {"type": "anomaly"}},
        {"agent_id": "explorer_2", "confidence": 0.85, "impact": 0.75, "novelty": 0.65,
         "pattern": {"type": "anomaly"}},
        {"agent_id": "explorer_1", "confidence": 0.3, "impact": 0.2, "novelty": 0.1,
         "pattern": {"type": "noise"}},
    ]

    print("1. Discovery → Dance")
    for disc in discoveries:
        dance = parser.parse_discovery(disc)
        if dance:
            print(f"   {disc['agent_id']}: intensity={dance.intensity:.2f}, "
                  f"direction={dance.direction}")
        else:
            print(f"   {disc['agent_id']}: filtered (below threshold)")

    # --- Step 3: Dance propagation ---
    prop = DancePropagator()
    prop.register_agent(AgentLocation("coordinator", x=100, y=100))
    prop.register_agent(AgentLocation("executor_1", x=120, y=100))
    prop.register_agent(AgentLocation("validator_1", x=200, y=100))

    # Create a dance from the accumulated pattern
    top = parser.get_top_patterns(1)
    if top:
        pattern, weight = top[0]
        signal = DanceSignal("d_main", "coordinator", intensity=min(weight, 1.0),
                             direction=pattern, duration=weight * 10)
        dance = Dance(signal=signal, location=AgentLocation("coordinator", x=100, y=100))

        print(f"\n2. Dance propagation (pattern={pattern}, weight={weight:.2f})")
        results = prop.broadcast(dance)
        for agent_id, intensity in results.items():
            print(f"   {agent_id}: received intensity={intensity:.3f}")

    # --- Step 4: Response aggregation ---
    handler = DanceResponseHandler(prop)
    for agent_id in ["executor_1", "validator_1"]:
        handler.respond_to_dance(agent_id, dance, {"type": "ack", "confidence": 0.8})

    agg_resp = handler.get_aggregated_response("d_main")
    print(f"\n3. Response aggregation: {agg_resp['count']} agents, "
          f"avg confidence={agg_resp['average_confidence']:.2f}")

    # --- Step 5: Consensus ---
    algo = ConsensusAlgorithm(threshold=0.6)
    weights = parser.accumulate_dances()
    result = algo.form_consensus(weights, parser.get_pattern_weights().support_counts)

    print(f"\n4. Consensus: type={result.consensus_type.value}, "
          f"pattern={result.pattern}, confidence={result.confidence:.2f}")

    # --- Step 6: Goal Emergence ---
    agg.receive_discovery({
        "agent_id": "explorer_1",
        "discovery_type": "anomaly",
        "confidence": 0.9,
        "impact": 0.8,
        "novelty": 0.7,
        "pattern": {"type": "anomaly"},
    })
    snapshot = agg.generate_snapshot()

    emerger = GoalEmerger()
    goals = emerger.emerge_goals(snapshot, consensus_result=result)

    print(f"\n5. Emerged goals ({len(goals)}):")
    for goal in goals:
        print(f"   [{goal.priority.value}] {goal.description}")
        print(f"      sub-tasks: {goal.sub_tasks}")

    # --- Summary ---
    print(f"\n=== System State ===")
    print(f"  Agents: {snapshot.total_agents} "
          f"(idle={snapshot.idle_agents}, executing={snapshot.executing_agents}, "
          f"exploring={snapshot.exploring_agents})")
    print(f"  Load: {snapshot.system_load:.1%}")
    print(f"  Top patterns: {parser.get_top_patterns(3)}")
    print(f"  Consensus rate: {algo.get_consensus_rate():.0%}")
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
