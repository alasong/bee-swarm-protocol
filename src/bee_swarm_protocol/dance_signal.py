"""DanceSignal data model — the fundamental unit of swarm communication."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict


@dataclass
class DanceSignal:
    """
    A dance signal represents the quality of a discovery.

    Analogous to honeybee waggle dance:
    - **intensity**: How vigorously the bee dances (discovery quality, 0–1)
    - **direction**: Where the resource is (pattern type/direction)
    - **duration**: How long the dance lasts (importance)

    Attributes:
        dance_id: Unique identifier
        agent_id: Agent that produced this dance
        intensity: Discovery quality score (0.0 – 1.0)
        direction: Pattern type or direction string
        duration: Relative duration (higher = more important)
        pattern: Optional pattern metadata
        timestamp: Creation time
    """

    dance_id: str
    agent_id: str
    intensity: float  # 0.0 – 1.0
    direction: str  # pattern type or direction
    duration: float  # relative duration
    pattern: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
