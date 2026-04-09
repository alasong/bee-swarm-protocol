"""Dance response handling — agent responses, aggregation, and attention tracking."""

from collections import defaultdict
from typing import Any, Callable, Dict, List

from bee_swarm_protocol.dance_propagation import Dance, DancePropagator, Response


class DanceResponseHandler:
    """
    Handles agent responses to dances.

    - Response handler registration per agent
    - Response aggregation (count, attention, confidence)
    - Attention tracking based on dance intensity
    """

    def __init__(self, propagator: DancePropagator):
        self._propagator = propagator
        self._response_handlers: Dict[str, Callable] = {}
        self._response_counter = 0
        self._responses_by_dance: Dict[str, List[Response]] = defaultdict(list)

    def register_response_handler(self, agent_id: str, handler: Callable) -> None:
        """Register a callable to handle dance responses for an agent."""
        self._response_handlers[agent_id] = handler

    def unregister_response_handler(self, agent_id: str) -> None:
        """Remove a response handler."""
        self._response_handlers.pop(agent_id, None)

    def respond_to_dance(
        self, agent_id: str, dance: Dance, response: Dict[str, Any]
    ) -> Response:
        """
        Respond to a dance from an agent.

        Returns:
            The created Response object
        """
        self._response_counter += 1
        response_id = f"response_{self._response_counter}"

        attention_level = 0.0
        visible_dances = self._propagator.get_visible_dances(agent_id)
        for vd in visible_dances:
            if vd.dance_id == dance.dance_id:
                attention_level = vd.decayed_intensity
                break

        resp = Response(
            response_id=response_id,
            dance_id=dance.dance_id,
            agent_id=agent_id,
            response_data=response,
            attention_level=attention_level,
        )

        self._responses_by_dance[dance.dance_id].append(resp)
        dance.responses.append(resp)

        if agent_id in self._response_handlers:
            try:
                self._response_handlers[agent_id](resp)
            except Exception:
                pass

        return resp

    def get_responses(self, dance_id: str) -> List[Response]:
        """Get all responses for a dance, sorted by attention (highest first)."""
        responses = self._responses_by_dance.get(dance_id, [])
        return sorted(responses, key=lambda r: r.attention_level, reverse=True)

    def get_aggregated_response(self, dance_id: str) -> Dict[str, Any]:
        """Aggregate responses for a dance."""
        responses = self.get_responses(dance_id)
        if not responses:
            return {"count": 0, "average_attention": 0.0, "data": {}}

        total_attention = sum(r.attention_level for r in responses)
        avg_attention = total_attention / len(responses)

        response_types: Dict[str, int] = defaultdict(int)
        for resp in responses:
            resp_type = resp.response_data.get("type", "unknown")
            response_types[resp_type] += 1

        confidences = [
            r.response_data.get("confidence", 0.0)
            for r in responses
            if "confidence" in r.response_data
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "count": len(responses),
            "average_attention": avg_attention,
            "response_types": dict(response_types),
            "average_confidence": avg_confidence,
            "agents": [r.agent_id for r in responses],
        }

    def clear_responses(self, dance_id: str) -> None:
        """Clear all responses for a dance."""
        self._responses_by_dance.pop(dance_id, None)
