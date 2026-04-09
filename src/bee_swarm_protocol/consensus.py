"""Consensus formation from accumulated dance weights."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


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
    timestamp: datetime = field(default_factory=datetime.utcnow)

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

        best_pattern = max(pattern_weights, key=pattern_weights.get)
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
