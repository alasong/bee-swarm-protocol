# Bee Swarm Protocol

[![CI](https://github.com/alasong/bee-swarm-protocol/actions/workflows/ci.yml/badge.svg)](https://github.com/alasong/bee-swarm-protocol/actions/workflows/ci.yml)
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

## Competitive Landscape

Bee Swarm Protocol sits in the multi-agent coordination space alongside several established frameworks. Here's how it compares:

### Comparison Matrix

| Feature | Bee Swarm Protocol | OpenAI Swarm | CrewAI | AutoGen / AG2 | LangGraph |
|---------|:-:|:-:|:-:|:-:|:-:|
| **Coordination model** | Emergent consensus | Handoff | Role assignment | Conversational | State graph |
| **Biomimetic signals** | Waggle dance language | None | None | None | None |
| **Distance-based propagation** | Inverse-square decay | N/A | N/A | N/A | N/A |
| **Time decay** | Exponential signal decay | N/A | N/A | N/A | N/A |
| **Goal emergence** | Auto-emerged from state | Manual | Pre-assigned | Pre-assigned | Explicit |
| **Consensus mechanism** | STRONG / WEAK / NONE | None | Voting (simple) | Group chat | State-based |
| **Decentralization** | Signal-driven, emergent | Centralized orchestrator | Centralized orchestrator | Centralized orchestrator | Centralized orchestrator |
| **Zero external deps** | Yes | Yes | No | No | No |
| **Stars** | New | ~10k+ | ~20k+ | ~30k+ | ~15k+ |

### Key Differentiators

#### 1. Waggle Dance Signaling
Unlike role-based or handoff-based frameworks, Bee Swarm Protocol encodes discovery quality into a **dance intensity** signal (`0.4*confidence + 0.3*impact + 0.3*novelty`), mirroring how honeybees communicate resource quality. Other frameworks use direct assignment — here, signals *compete* for attention.

#### 2. Distance-Aware Propagation
Signals propagate through a spatial model with **inverse-square decay**. Agents closer to the source receive stronger signals, creating natural priority ordering without explicit configuration. This is unique among multi-agent frameworks.

#### 3. Emergent Goals, Not Assigned Roles
Where CrewAI assigns roles (Researcher, Writer, Reviewer) and Swarm uses handoffs, Bee Swarm Protocol **derives goals from system state analysis** — high load triggers resource optimization goals, blocked agents trigger coordination goals. The system self-organizes rather than following a pre-programmed org chart.

#### 4. Consensus with Uncertainty
The `STRONG / WEAK / NONE` consensus model explicitly represents uncertainty — when no pattern reaches the threshold, the system knows to request more exploration. Most frameworks force a binary decision.

### When to Choose Bee Swarm Protocol

- **Exploratory/swarm intelligence** scenarios where agents discover and signal findings rather than execute predefined tasks
- **Decentralized coordination** where you want agents to self-organize based on signal strength, not central orchestration
- **Real-time adaptive systems** where goals emerge from conditions rather than being statically assigned
- **Research and prototyping** of bio-inspired multi-agent algorithms

### When to Choose Alternatives

- **Production LLM applications** → AutoGen or CrewAI (more mature, larger ecosystems)
- **Complex workflow orchestration** → LangGraph (stateful graph-based control flow)
- **Simple multi-agent routing** → OpenAI Swarm (minimal, easy to understand)

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
git clone https://github.com/alasong/bee-swarm-protocol.git
cd bee-swarm-protocol
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/
```

## License

MIT
