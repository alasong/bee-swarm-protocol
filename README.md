# Bee Swarm Protocol

[![CI](https://github.com/fiagent/bee-swarm-protocol/actions/workflows/ci.yml/badge.svg)](https://github.com/fiagent/bee-swarm-protocol/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/bee-swarm-protocol/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Bee-inspired swarm communication and coordination protocol for multi-agent systems.

## Features

- **Dance Language** — Inspired by honeybee waggle dance: encode discovery quality as intensity, direction as pattern type
- **Distance-Based Propagation** — Signals decay with distance (inverse-square law), agents receive based on proximity
- **Exponential Time Decay** — Dance signals naturally fade over time with configurable decay rate
- **Consensus Formation** — Accumulate dance weights across agents to reach STRONG/WEAK/NONE consensus on patterns
- **Goal Emergence** — System goals auto-emerge from state analysis (load, idle agents, blocked agents, pending discoveries)
- **Signal Aggregation** — Unified snapshot of system state: agent states, discoveries, task requests
- **ASCII Visualization** — Grid-based dance visualization with heat map by zone/type

## Installation

```bash
pip install bee-swarm-protocol
```

Optional messaging integration:

```bash
pip install bee-swarm-protocol[messaging]
```

## Quick Start

```python
from bee_swarm_protocol import (
    DanceLanguageParser,
    DancePropagator,
    AgentLocation,
    DanceSignal,
    Dance,
    SignalAggregator,
    ConsensusAlgorithm,
    GoalEmerger,
)

# 1. Parse discoveries into dance signals
parser = DanceLanguageParser()
dance = parser.parse_discovery({
    "agent_id": "explorer_1",
    "confidence": 0.9,
    "impact": 0.8,
    "novelty": 0.7,
    "pattern": {"type": "anomaly"},
})
print(f"Dance intensity: {dance.intensity:.2f}")  # 0.84

# 2. Propagate dance to nearby agents
prop = DancePropagator()
prop.register_agent(AgentLocation("coordinator", x=100, y=100))
prop.register_agent(AgentLocation("executor_1", x=120, y=100))
results = prop.broadcast(dance)
# {'executor_1': 0.735}  # intensity decays with distance

# 3. Form consensus from accumulated weights
weights = parser.accumulate_dances()
algo = ConsensusAlgorithm(threshold=0.6)
result = algo.form_consensus(weights, {"anomaly": 5})
print(f"Consensus: {result.consensus_type.value}")  # strong

# 4. Emerge goals from system state
agg = SignalAggregator()
agg.receive_state_report({"agent_id": "agent_1", "state": "idle"})
agg.receive_state_report({"agent_id": "agent_2", "state": "executing"})
snapshot = agg.generate_snapshot()
emerger = GoalEmerger()
goals = emerger.emerge_goals(snapshot, consensus_result=result)
for goal in goals:
    print(f"[{goal.priority.value}] {goal.description}")
```

Run the full demo:

```bash
python examples/swarm_coordination.py
```

## Architecture

```
Discovery → DanceLanguageParser → DanceSignal
                                       ↓
                              DancePropagator (distance decay)
                                       ↓
                              DanceResponseHandler (aggregation)
                                       ↓
                              ConsensusAlgorithm (STRONG/WEAK/NONE)
                                       ↓
                              GoalEmerger (system goals)
```

### Module Structure

| Module | Purpose |
|--------|---------|
| `DanceSignal` | Data class for dance signals with timestamp |
| `DanceLanguageParser` | Convert discoveries to dances with intensity scoring |
| `DancePropagator` | Distance-based signal propagation with inverse-square decay |
| `DanceResponseHandler` | Aggregate agent responses to dance signals |
| `DanceDecay` | Exponential time decay with auto-cleanup |
| `DanceVisualizer` | ASCII grid visualization |
| `SignalAggregator` | Collect state reports, discoveries, task requests |
| `ConsensusAlgorithm` | Pattern-based consensus formation |
| `GoalEmerger` | Auto-generate system goals from state + consensus |

## API Reference

### Dance Intensity

```
intensity = 0.4 * confidence + 0.3 * impact + 0.3 * novelty
```

### Pattern Weight Accumulation

```
weight = avg_intensity * log(agent_count + 1)
```

### Distance Propagation

```
received_intensity = original_intensity / (1 + distance^2 / 100)
```

### Time Decay

```
decayed_intensity = original_intensity * e^(-decay_rate * time_elapsed)
```

## Development

```bash
# Clone and install
git clone https://github.com/fiagent/bee-swarm-protocol.git
cd bee-swarm-protocol
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/
```

## License

MIT
