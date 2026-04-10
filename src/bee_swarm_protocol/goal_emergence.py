"""Goal emergence from aggregated signals and consensus."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class GoalType(str, Enum):
    """Types of system goals."""

    RESOURCE_OPTIMIZATION = "resource_optimization"
    TASK_CAPACITY = "task_capacity"
    KNOWLEDGE_GAP = "knowledge_gap"
    EXPLORATION = "exploration"
    COORDINATION = "coordination"
    SAFETY = "safety"


class GoalPriority(str, Enum):
    """Goal priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SystemGoal:
    """A system-level goal generated through emergence."""

    goal_id: str
    goal_type: GoalType
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: GoalPriority = GoalPriority.MEDIUM
    deadline: Optional[datetime] = None
    derived_from: Optional[str] = None  # Consensus ID
    sub_tasks: List[str] = field(default_factory=list)
    status: str = "pending"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "goal_type": self.goal_type.value,
            "description": self.description,
            "parameters": self.parameters,
            "priority": self.priority.value,
            "status": self.status,
        }


@dataclass
class SystemNeed:
    """An identified system need."""

    need_type: str
    description: str
    priority: GoalPriority
    metrics: Dict[str, Any] = field(default_factory=dict)


class GoalEmerger:
    """
    Emerges system goals from state analysis and consensus.

    Process:
    1. Analyze system state snapshot
    2. Identify system needs (resource, capacity, coordination, knowledge)
    3. Combine with consensus results
    4. Generate specific goals with sub-task decomposition
    """

    HIGH_LOAD_THRESHOLD = 0.8
    LOW_IDLE_THRESHOLD = 0.2
    HIGH_PENDING_TASKS_THRESHOLD = 10

    def __init__(self):
        self._goals: List[SystemGoal] = []
        self._goal_counter = 0

    def identify_needs(self, snapshot) -> List[SystemNeed]:
        """Identify system needs from a state snapshot."""
        needs = []

        if snapshot.system_load >= self.HIGH_LOAD_THRESHOLD:
            needs.append(SystemNeed(
                need_type="resource_optimization",
                description="System load is high, resource optimization needed",
                priority=GoalPriority.HIGH,
                metrics={"system_load": snapshot.system_load},
            ))

        if snapshot.task_request_count > snapshot.idle_agents:
            needs.append(SystemNeed(
                need_type="task_capacity",
                description="More tasks than available idle agents",
                priority=GoalPriority.HIGH,
                metrics={
                    "pending_requests": snapshot.task_request_count,
                    "idle_agents": snapshot.idle_agents,
                },
            ))

        if snapshot.blocked_agents > 0:
            needs.append(SystemNeed(
                need_type="coordination",
                description=f"{snapshot.blocked_agents} agents are blocked",
                priority=GoalPriority.MEDIUM,
                metrics={"blocked_count": snapshot.blocked_agents},
            ))

        if snapshot.discovery_count > 5:
            needs.append(SystemNeed(
                need_type="knowledge_gap",
                description="Many pending discoveries to process",
                priority=GoalPriority.LOW,
                metrics={"discovery_count": snapshot.discovery_count},
            ))

        return needs

    def generate_goal(
        self,
        need: SystemNeed,
        consensus_result=None,
    ) -> SystemGoal:
        """Generate a system goal from a need."""
        self._goal_counter += 1
        goal_type_map = {
            "resource_optimization": GoalType.RESOURCE_OPTIMIZATION,
            "task_capacity": GoalType.TASK_CAPACITY,
            "knowledge_gap": GoalType.KNOWLEDGE_GAP,
            "coordination": GoalType.COORDINATION,
            "exploration": GoalType.EXPLORATION,
        }
        goal_type = goal_type_map.get(need.need_type, GoalType.COORDINATION)

        parameters = need.metrics.copy()
        derived_from = None
        if consensus_result and consensus_result.has_consensus:
            derived_from = consensus_result.consensus_id
            parameters["consensus_pattern"] = consensus_result.pattern

        goal = SystemGoal(
            goal_id=f"goal_{self._goal_counter}",
            goal_type=goal_type,
            description=need.description,
            parameters=parameters,
            priority=need.priority,
            derived_from=derived_from,
        )
        self._goals.append(goal)
        return goal

    def decompose_goal(self, goal: SystemGoal) -> List[str]:
        """Decompose a goal into sub-tasks."""
        decomposition_map = {
            GoalType.RESOURCE_OPTIMIZATION: self._decompose_resource_optimization,
            GoalType.TASK_CAPACITY: self._decompose_task_capacity,
            GoalType.KNOWLEDGE_GAP: self._decompose_knowledge_gap,
            GoalType.COORDINATION: self._decompose_coordination,
        }
        handler = decomposition_map.get(goal.goal_type)
        if handler:
            goal.sub_tasks = handler(goal)
        return goal.sub_tasks

    def _decompose_resource_optimization(self, goal: SystemGoal) -> List[str]:
        return [
            f"{goal.goal_id}_analyze_resources",
            f"{goal.goal_id}_identify_targets",
            f"{goal.goal_id}_apply_optimizations",
        ]

    def _decompose_task_capacity(self, goal: SystemGoal) -> List[str]:
        return [
            f"{goal.goal_id}_scale_capacity",
            f"{goal.goal_id}_balance_distribution",
        ]

    def _decompose_knowledge_gap(self, goal: SystemGoal) -> List[str]:
        return [
            f"{goal.goal_id}_process_discoveries",
            f"{goal.goal_id}_update_knowledge",
        ]

    def _decompose_coordination(self, goal: SystemGoal) -> List[str]:
        return [
            f"{goal.goal_id}_diagnose_blocked",
            f"{goal.goal_id}_resolve_blockages",
        ]

    def emerge_goals(self, snapshot, consensus_result=None) -> List[SystemGoal]:
        """Full goal emergence: needs → goals → decomposition."""
        needs = self.identify_needs(snapshot)
        goals = []
        for need in needs:
            goal = self.generate_goal(need, consensus_result)
            self.decompose_goal(goal)
            goals.append(goal)
        return goals

    def get_active_goals(self) -> List[SystemGoal]:
        """Get all active (pending) goals."""
        return [g for g in self._goals if g.status == "pending"]

    def get_all_goals(self) -> List[SystemGoal]:
        """Get all goals."""
        return list(self._goals)

    def complete_goal(self, goal_id: str) -> bool:
        """Mark a goal as completed."""
        for goal in self._goals:
            if goal.goal_id == goal_id:
                goal.status = "completed"
                return True
        return False

    async def async_identify_needs(self, snapshot) -> List[SystemNeed]:
        """Async variant of identify_needs."""
        return self.identify_needs(snapshot)

    async def async_generate_goal(
        self, need: SystemNeed, consensus_result=None
    ) -> SystemGoal:
        """Async variant of generate_goal."""
        return self.generate_goal(need, consensus_result)

    async def async_decompose_goal(self, goal: SystemGoal) -> List[str]:
        """Async variant of decompose_goal."""
        return self.decompose_goal(goal)

    async def async_emerge_goals(self, snapshot, consensus_result=None) -> List[SystemGoal]:
        """Async variant of emerge_goals."""
        return self.emerge_goals(snapshot, consensus_result)

    async def async_complete_goal(self, goal_id: str) -> bool:
        """Async variant of complete_goal."""
        return self.complete_goal(goal_id)
