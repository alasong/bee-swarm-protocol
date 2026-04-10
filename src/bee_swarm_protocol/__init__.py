"""
bee-swarm-protocol: Bee-inspired swarm communication and coordination.

Inspired by honeybee waggle dance, this library provides:
- Dance language for discovery quality signaling
- Distance-based propagation with intensity decay
- Consensus formation from accumulated dance weights
- System-level goal emergence from aggregated signals

Zero external dependencies.

Example:
    >>> from bee_swarm_protocol import DanceSignal, DanceLanguageParser
    >>> parser = DanceLanguageParser()
    >>> dance = parser.parse_discovery({
    ...     "agent_id": "explorer_1",
    ...     "confidence": 0.9,
    ...     "impact": 0.8,
    ...     "novelty": 0.7,
    ...     "pattern": {"type": "anomaly"},
    ... })
    >>> print(dance.intensity)
    0.81
"""

from bee_swarm_protocol.aggregator import (
    AgentStateInfo,
    DiscoveryInfo,
    SignalAggregator,
    SystemStateSnapshot,
    TaskRequestInfo,
)
from bee_swarm_protocol.bee_agent import (
    AgentRole,
    AgentState,
    BeeAgent,
    CoordinatorAgent,
    ExecutorAgent,
    ExplorerAgent,
)
from bee_swarm_protocol.consensus import (
    ByzantineFaultTolerance,
    ConsensusAlgorithm,
    ConsensusResult,
    ConsensusType,
    WeightedConsensusAlgorithm,
)
from bee_swarm_protocol.dance_decay import DanceDecay
from bee_swarm_protocol.dance_parser import DanceLanguageParser, PatternWeights
from bee_swarm_protocol.dance_propagation import (
    AgentLocation,
    ChebyshevMetric,
    Dance,
    DancePropagator,
    DistanceMetric,
    EuclideanMetric,
    ManhattanMetric,
    Response,
)
from bee_swarm_protocol.dance_response import DanceResponseHandler
from bee_swarm_protocol.dance_signal import DanceSignal
from bee_swarm_protocol.dance_visualizer import DanceVisualizer
from bee_swarm_protocol.goal_emergence import (
    GoalEmerger,
    GoalPriority,
    GoalType,
    SystemGoal,
    SystemNeed,
)
from bee_swarm_protocol.llm_agent import (
    DashScopeAgent,
    LLMBeeAgent,
    LLMDanceParser,
    SwarmLLMConfig,
)
from bee_swarm_protocol.llm_integration import DanceLLMAnalyzer, DashScopeClient, LLMResponse

__version__ = "0.1.0"

__all__ = [
    # Dance core
    "DanceSignal",
    "DanceLanguageParser",
    "PatternWeights",
    # Dance propagation
    "DancePropagator",
    "AgentLocation",
    "Dance",
    "Response",
    "DistanceMetric",
    "EuclideanMetric",
    "ManhattanMetric",
    "ChebyshevMetric",
    # Dance handling
    "DanceResponseHandler",
    "DanceDecay",
    "DanceVisualizer",
    # Signal aggregation
    "SignalAggregator",
    "SystemStateSnapshot",
    "AgentStateInfo",
    "DiscoveryInfo",
    "TaskRequestInfo",
    # Consensus
    "ConsensusAlgorithm",
    "ConsensusResult",
    "ConsensusType",
    "ByzantineFaultTolerance",
    "WeightedConsensusAlgorithm",
    # Goal emergence
    "GoalEmerger",
    "SystemGoal",
    "SystemNeed",
    "GoalType",
    "GoalPriority",
    # LLM integration
    "DashScopeClient",
    "DanceLLMAnalyzer",
    "LLMResponse",
    # LLM agents
    "SwarmLLMConfig",
    "DashScopeAgent",
    "LLMDanceParser",
    "LLMBeeAgent",
    # BeeAgent runtime
    "AgentRole",
    "AgentState",
    "BeeAgent",
    "ExplorerAgent",
    "ExecutorAgent",
    "CoordinatorAgent",
]
