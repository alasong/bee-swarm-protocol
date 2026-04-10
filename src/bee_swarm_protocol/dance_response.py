"""Dance response handling — agent responses, aggregation, and attention tracking."""

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from agent_message_bus import MessageBus

from bee_swarm_protocol.dance_propagation import Dance, DancePropagator, Response


class DanceResponseHandler:
    """
    Handles agent responses to dances.

    - Response handler registration per agent (callback-based, backward compatible)
    - Bus-based response reception (when bus is provided)
    - Response aggregation (count, attention, confidence)
    - Attention tracking based on dance intensity
    """

    def __init__(
        self,
        propagator: DancePropagator,
        bus: Optional["MessageBus"] = None,
        source_agent_id: str = "response_handler",
    ):
        self._propagator = propagator
        self._response_handlers: Dict[str, Callable] = {}
        self._response_counter = 0
        self._responses_by_dance: Dict[str, List[Response]] = defaultdict(list)
        self.bus = bus
        self.source_agent_id = source_agent_id

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

    async def async_respond_to_dance(
        self, agent_id: str, dance: Dance, response: Dict[str, Any]
    ) -> Response:
        """Async variant of respond_to_dance. Publishes to bus if available."""
        resp = self.respond_to_dance(agent_id, dance, response)
        if self.bus is not None:
            await self.bus.send(
                from_agent=agent_id,
                to_agent=self.source_agent_id,
                message={
                    "type": "dance_response",
                    "dance_id": dance.dance_id,
                    "agent_id": agent_id,
                    "response_id": resp.response_id,
                    "response_data": response,
                    "attention_level": resp.attention_level,
                },
            )
        return resp

    async def receive_response_from_bus(
        self, agent_id: str, timeout: float = 0.0
    ) -> Optional[Response]:
        """
        Receive a response from the message bus for a specific agent.

        The caller must ensure the bus has agents registered.
        Returns a Response if a message is available, None otherwise.
        """
        if self.bus is None:
            return None
        msg = await self.bus.receive(agent_id, timeout=timeout)
        if msg is None:
            return None
        content = msg.content
        dance_id = content.get("dance_id", "")
        resp = Response(
            response_id=content.get("response_id", msg.message_id),
            dance_id=dance_id,
            agent_id=content.get("agent_id", agent_id),
            response_data=content.get("response_data", {}),
            attention_level=content.get("attention_level", 0.0),
        )
        self._responses_by_dance[dance_id].append(resp)
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

    async def async_get_responses(self, dance_id: str) -> List[Response]:
        """Async variant of get_responses."""
        return self.get_responses(dance_id)

    async def async_get_aggregated_response(self, dance_id: str) -> Dict[str, Any]:
        """Async variant of get_aggregated_response."""
        return self.get_aggregated_response(dance_id)
