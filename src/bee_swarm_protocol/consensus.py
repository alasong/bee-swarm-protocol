"""Consensus formation from accumulated dance weights."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math
import statistics
from typing import Any, Dict, List, Optional, Set


class ConsensusType(str, Enum):
    """Types of consensus."""

    STRONG = "strong"  # Above threshold, high confidence
    WEAK = "weak"  # Below threshold but leading pattern exists
    NONE = "none"  # No clear leading pattern


@dataclass
class ConsensusResult:
    """Result of a consensus formation process."""

    consensus_id: str
    consensus_type: ConsensusType
    pattern: Optional[str]
    confidence: float
    participating_agents: int
    requires_more_exploration: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_strong(self) -> bool:
        return self.consensus_type == ConsensusType.STRONG

    @property
    def has_consensus(self) -> bool:
        return self.consensus_type != ConsensusType.NONE


class ConsensusAlgorithm:
    """
    Forms consensus from dance language weights.

    Process:
    1. Accumulate dance signals into pattern weights
    2. Find the best (highest-weighted) pattern
    3. Check if it reaches the threshold for strong consensus
    """

    def __init__(self, threshold: float = 0.7, min_participants: int = 1):
        """
        Args:
            threshold: Minimum weight for strong consensus
            min_participants: Minimum number of participating agents
        """
        self.threshold = threshold
        self.min_participants = min_participants
        self._consensus_history: List[ConsensusResult] = []
        self._consensus_counter = 0

    def form_consensus(
        self,
        pattern_weights: Dict[str, float],
        agent_counts: Optional[Dict[str, int]] = None,
    ) -> ConsensusResult:
        """
        Form consensus from pattern weights.

        Args:
            pattern_weights: Weights for each pattern type
            agent_counts: Number of agents supporting each pattern

        Returns:
            ConsensusResult with consensus decision
        """
        self._consensus_counter += 1
        consensus_id = f"cons_{self._consensus_counter}"

        if not pattern_weights:
            result = ConsensusResult(
                consensus_id=consensus_id,
                consensus_type=ConsensusType.NONE,
                pattern=None,
                confidence=0.0,
                participating_agents=0,
                requires_more_exploration=True,
            )
            self._consensus_history.append(result)
            return result

        best_pattern = max(pattern_weights, key=lambda k: pattern_weights[k])
        best_weight = pattern_weights[best_pattern]
        total_agents = sum(agent_counts.values()) if agent_counts else 1

        if best_weight >= self.threshold and total_agents >= self.min_participants:
            consensus_type = ConsensusType.STRONG
            requires_more = False
        elif best_weight >= self.threshold * 0.5:
            consensus_type = ConsensusType.WEAK
            requires_more = True
        else:
            consensus_type = ConsensusType.NONE
            requires_more = True

        result = ConsensusResult(
            consensus_id=consensus_id,
            consensus_type=consensus_type,
            pattern=best_pattern,
            confidence=best_weight,
            participating_agents=total_agents,
            requires_more_exploration=requires_more,
            details={"threshold": self.threshold, "all_weights": pattern_weights},
        )

        self._consensus_history.append(result)
        return result

    def get_consensus_history(self, limit: int = 100) -> List[ConsensusResult]:
        """Get recent consensus history."""
        return self._consensus_history[-limit:]

    def get_last_consensus(self) -> Optional[ConsensusResult]:
        """Get the most recent consensus."""
        if self._consensus_history:
            return self._consensus_history[-1]
        return None

    def clear_history(self) -> None:
        """Clear consensus history."""
        self._consensus_history.clear()

    def get_consensus_rate(self) -> float:
        """Fraction of successful consensus (strong + weak) vs total."""
        if not self._consensus_history:
            return 0.0
        successful = sum(
            1 for c in self._consensus_history if c.consensus_type != ConsensusType.NONE
        )
        return successful / len(self._consensus_history)

    async def async_form_consensus(
        self,
        pattern_weights: Dict[str, float],
        agent_counts: Optional[Dict[str, int]] = None,
    ) -> ConsensusResult:
        """Async variant of form_consensus."""
        return self.form_consensus(pattern_weights, agent_counts)


class ByzantineFaultTolerance:
    """
    Detects and tolerates faulty (Byzantine) agents in consensus.

    Uses median-based voting instead of average to resist outlier manipulation.
    Agents whose votes deviate beyond 2 standard deviations from the median
    are flagged as suspected Byzantine agents.
    """

    def __init__(self, max_byzantine: int = 0):
        """
        Args:
            max_byzantine: Maximum number of faulty agents the system can tolerate.
        """
        self.max_byzantine = max_byzantine
        self._suspected_agents: Set[str] = set()

    def detect_byzantine(
        self, agent_votes: Dict[str, float]
    ) -> List[str]:
        """
        Detect suspected Byzantine agents from their votes.

        Uses Median Absolute Deviation (MAD) for robust outlier detection.
        An agent is suspected if their vote deviates more than 2 scaled MAD
        units from the median of all votes.

        Args:
            agent_votes: Mapping of agent_id to their vote value.

        Returns:
            List of suspected Byzantine agent IDs.
        """
        if len(agent_votes) < 3:
            return []

        values = list(agent_votes.values())
        median_val = statistics.median(values)

        abs_deviations = [abs(v - median_val) for v in values]
        mad = statistics.median(abs_deviations)
        if mad == 0:
            # If MAD is 0, fall back to mean absolute deviation
            mean_dev = sum(abs_deviations) / len(abs_deviations)
            if mean_dev == 0:
                return []
            threshold = 2 * mean_dev
        else:
            # Scaled MAD (factor ~1.4826 for consistency with stdev on normal data)
            threshold = 2 * 1.4826 * mad

        suspected = [
            agent_id
            for agent_id, vote in agent_votes.items()
            if abs(vote - median_val) > threshold
        ]
        self._suspected_agents.update(suspected)
        return suspected

    def compute_fault_tolerant_consensus(
        self,
        agent_votes: Dict[str, float],
        pattern_weights: Dict[str, float],
        agent_counts: Optional[Dict[str, int]] = None,
    ) -> ConsensusResult:
        """
        Compute consensus using median-based aggregation, excluding Byzantine agents.

        The median of agent votes is used instead of average, making the result
        resistant to outlier manipulation by faulty agents.

        Args:
            agent_votes: Mapping of agent_id to their vote value.
            pattern_weights: Weights for each pattern type.
            agent_counts: Number of agents supporting each pattern.

        Returns:
            ConsensusResult with the fault-tolerant consensus decision.
        """
        if not agent_votes:
            return ConsensusResult(
                consensus_id="bft_0",
                consensus_type=ConsensusType.NONE,
                pattern=None,
                confidence=0.0,
                participating_agents=0,
                requires_more_exploration=True,
                details={"byzantine_agents": [], "excluded_count": 0},
            )

        suspected = self.detect_byzantine(agent_votes)
        is_tolerance_ok = len(suspected) <= self.max_byzantine

        clean_votes = {
            aid: v for aid, v in agent_votes.items() if aid not in suspected
        }
        if not clean_votes:
            clean_votes = agent_votes

        median_vote = statistics.median(clean_votes.values())
        total_agents = sum(agent_counts.values()) if agent_counts else len(agent_votes)
        excluded_count = len(agent_votes) - len(clean_votes)

        if is_tolerance_ok and median_vote >= 0.7 and total_agents >= 1:
            consensus_type = ConsensusType.STRONG
            requires_more = False
        elif is_tolerance_ok and median_vote >= 0.35:
            consensus_type = ConsensusType.WEAK
            requires_more = True
        else:
            consensus_type = ConsensusType.NONE
            requires_more = True

        best_pattern = None
        if pattern_weights:
            best_pattern = max(pattern_weights, key=lambda k: pattern_weights[k])

        result = ConsensusResult(
            consensus_id=f"bft_{hash(frozenset(agent_votes.items())) % 10000}",
            consensus_type=consensus_type,
            pattern=best_pattern,
            confidence=median_vote,
            participating_agents=total_agents - excluded_count,
            requires_more_exploration=requires_more,
            details={
                "byzantine_agents": suspected,
                "excluded_count": excluded_count,
                "median_vote": median_vote,
            },
        )
        return result

    def get_suspected_agents(self) -> Set[str]:
        """Return all agents ever suspected as Byzantine."""
        return self._suspected_agents.copy()

    def clear_suspected(self) -> None:
        """Clear the set of suspected Byzantine agents."""
        self._suspected_agents.clear()

    def is_within_tolerance(self, suspected_count: Optional[int] = None) -> bool:
        """
        Check if the number of suspected agents is within tolerance.

        Args:
            suspected_count: Override count to check. If None, uses current count.
        """
        count = suspected_count if suspected_count is not None else len(self._suspected_agents)
        return count <= self.max_byzantine


class WeightedConsensusAlgorithm(ConsensusAlgorithm):
    """
    Extends ConsensusAlgorithm with reputation-based agent weighting.

    Each agent's vote is multiplied by their reputation weight.
    Agents who consistently agree with consensus gain weight over time.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        min_participants: int = 1,
        agent_weights: Optional[Dict[str, float]] = None,
        weight_decay_rate: float = 0.05,
        weight_increment: float = 0.02,
        max_weight: float = 2.0,
        min_weight: float = 0.1,
    ):
        """
        Args:
            threshold: Minimum weight for strong consensus.
            min_participants: Minimum number of participating agents.
            agent_weights: Initial reputation-based weights per agent.
            weight_decay_rate: How much weight decreases for disagreeing agents.
            weight_increment: How much weight increases for agreeing agents.
            max_weight: Upper bound on agent weight.
            min_weight: Lower bound on agent weight.
        """
        super().__init__(threshold=threshold, min_participants=min_participants)
        self.agent_weights: Dict[str, float] = agent_weights or {}
        self.weight_decay_rate = weight_decay_rate
        self.weight_increment = weight_increment
        self.max_weight = max_weight
        self.min_weight = min_weight

    def _get_agent_weight(self, agent_id: str) -> float:
        """Get the current weight for an agent, defaulting to 1.0."""
        return self.agent_weights.get(agent_id, 1.0)

    def _update_weights(self, agent_votes: Dict[str, float], consensus_value: float) -> None:
        """
        Adjust agent weights based on agreement with consensus.

        Agents whose vote is closer to the consensus value gain weight;
        agents further away lose weight.
        """
        for agent_id, vote in agent_votes.items():
            current = self._get_agent_weight(agent_id)
            deviation = abs(vote - consensus_value)
            if deviation < 0.2:
                new_weight = current + self.weight_increment
            else:
                new_weight = current - self.weight_decay_rate
            self.agent_weights[agent_id] = max(self.min_weight, min(self.max_weight, new_weight))

    def form_consensus(
        self,
        pattern_weights: Dict[str, float],
        agent_counts: Optional[Dict[str, int]] = None,
    ) -> ConsensusResult:
        """
        Form consensus using weighted agent votes.

        Each agent's contribution is multiplied by their reputation weight.
        Weights are updated after consensus is reached.

        Args:
            pattern_weights: Weights for each pattern type.
            agent_counts: Number of agents supporting each pattern.

        Returns:
            ConsensusResult with consensus decision.
        """
        self._consensus_counter += 1
        consensus_id = f"wcons_{self._consensus_counter}"

        if not pattern_weights:
            result = ConsensusResult(
                consensus_id=consensus_id,
                consensus_type=ConsensusType.NONE,
                pattern=None,
                confidence=0.0,
                participating_agents=0,
                requires_more_exploration=True,
            )
            self._consensus_history.append(result)
            return result

        # Apply agent weights to pattern_weights
        weighted_patterns: Dict[str, float] = {}
        for pattern, weight in pattern_weights.items():
            weighted_patterns[pattern] = weight * self._get_agent_weight(pattern)

        best_pattern = max(weighted_patterns, key=lambda k: weighted_patterns[k])
        best_weight = weighted_patterns[best_pattern]
        total_agents = sum(agent_counts.values()) if agent_counts else 1

        if best_weight >= self.threshold and total_agents >= self.min_participants:
            consensus_type = ConsensusType.STRONG
            requires_more = False
        elif best_weight >= self.threshold * 0.5:
            consensus_type = ConsensusType.WEAK
            requires_more = True
        else:
            consensus_type = ConsensusType.NONE
            requires_more = True

        result = ConsensusResult(
            consensus_id=consensus_id,
            consensus_type=consensus_type,
            pattern=best_pattern,
            confidence=best_weight,
            participating_agents=total_agents,
            requires_more_exploration=requires_more,
            details={
                "threshold": self.threshold,
                "all_weights": pattern_weights,
                "weighted_patterns": weighted_patterns,
                "agent_weights": dict(self.agent_weights),
            },
        )

        # Update weights based on how well each agent agreed with consensus
        self._update_weights(pattern_weights, best_weight)

        self._consensus_history.append(result)
        return result

    def set_agent_weight(self, agent_id: str, weight: float) -> None:
        """Manually set an agent's reputation weight."""
        self.agent_weights[agent_id] = max(self.min_weight, min(self.max_weight, weight))

    def get_agent_weight(self, agent_id: str) -> float:
        """Get an agent's current reputation weight."""
        return self._get_agent_weight(agent_id)

    def get_weights_summary(self) -> Dict[str, float]:
        """Return a copy of all agent weights."""
        return dict(self.agent_weights)
