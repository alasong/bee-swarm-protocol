"""LLM agent module — DashScope-powered intelligence for swarm agents.

Provides:
- SwarmLLMConfig: Configuration for DashScope LLM access
- DashScopeAgent: LLM client wrapper supporting multiple models
- LLMDanceParser: LLM-enhanced dance language parser
- LLMBeeAgent: LLM-driven bee agent with optional message bus
"""

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from agent_message_bus import MessageBus
except ImportError:
    MessageBus = None

try:
    import dashscope
    from dashscope import Generation
except ImportError:
    dashscope = None
    Generation = None

from bee_swarm_protocol.dance_parser import DanceLanguageParser
from bee_swarm_protocol.dance_signal import DanceSignal


@dataclass
class SwarmLLMConfig:
    """Configuration for DashScope LLM access.

    Attributes:
        api_key: DashScope API key (reads DASHSCOPE_PRO env var, falls back to DASHSCOPE_API_KEY)
        model: Model name (default: qwen-max)
        base_url: API base URL (default: DashScope default)
        temperature: Sampling temperature (0.0–2.0, default: 0.7)
        max_tokens: Maximum output tokens (default: 2048)
    """

    api_key: Optional[str] = None
    model: str = "qwen-max"
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get(
                "DASHSCOPE_PRO",
                os.environ.get("DASHSCOPE_API_KEY", ""),
            )

    @property
    def has_api_key(self) -> bool:
        """Check if an API key is configured."""
        return bool(self.api_key)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config (masking the API key)."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "has_api_key": self.has_api_key,
        }


class DashScopeAgent:
    """Wrapper around the DashScope API for text completion and chat.

    Supports multiple models (qwen-turbo, qwen-plus, qwen-max, etc.) and
    provides a clean interface for system prompts and user messages.

    Example:
        >>> config = SwarmLLMConfig(model="qwen-plus")
        >>> agent = DashScopeAgent(config)
        >>> resp = agent.chat("Hello!")
        >>> print(resp.content)
    """

    SUPPORTED_MODELS = [
        "qwen-turbo",
        "qwen-plus",
        "qwen-max",
        "qwen-max-longcontext",
        "qwen-plus-latest",
    ]

    def __init__(self, config: Optional[SwarmLLMConfig] = None):
        self._config = config or SwarmLLMConfig()
        self._history: List[Dict[str, str]] = []

    @property
    def config(self) -> SwarmLLMConfig:
        return self._config

    @property
    def model(self) -> str:
        return self._config.model

    def switch_model(self, model: str) -> None:
        """Switch to a different model."""
        self._config.model = model

    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a chat message and return the response text.

        Args:
            user_message: The user's message content
            system_prompt: Optional system prompt override
            model: Optional model override for this call
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override

        Returns:
            Response text content

        Raises:
            RuntimeError: If dashscope package is not installed or API fails
        """
        if dashscope is None:
            raise RuntimeError(
                "dashscope package is required. Install with: pip install dashscope"
            )

        messages: List[Dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_message})

        return self._call_api(messages, model, temperature, max_tokens)

    def chat_with_history(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Chat while maintaining conversation history.

        Args:
            user_message: The user's message
            system_prompt: System prompt (only used if history is empty)
            model: Optional model override
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override

        Returns:
            Response text content
        """
        if not self._history and system_prompt:
            self._history.append({"role": "system", "content": system_prompt})

        self._history.append({"role": "user", "content": user_message})

        response_text = self._call_api(self._history, model, temperature, max_tokens)

        self._history.append({"role": "assistant", "content": response_text})

        return response_text

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Single-turn completion (no history).

        Args:
            prompt: The prompt text
            system_prompt: Optional system prompt
            model: Optional model override
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override

        Returns:
            Completion text
        """
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self._call_api(messages, model, temperature, max_tokens)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return list(self._history)

    def _call_api(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Internal: call the DashScope API."""
        client_model = model or self._config.model
        client_temp = temperature if temperature is not None else self._config.temperature
        client_max = max_tokens if max_tokens is not None else self._config.max_tokens

        response = Generation.call(
            model=client_model,
            messages=messages,
            temperature=client_temp,
            max_tokens=client_max,
            api_key=self._config.api_key,
            result_format="message",
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"DashScope API error: {response.status_code} - {response.message}"
            )

        output = response.output
        choices = output.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            return message.get("content", "")

        return ""


class LLMDanceParser(DanceLanguageParser):
    """Extends DanceLanguageParser with LLM-powered enrichment.

    Enhances discovery signals by:
    - Generating natural language descriptions for dance patterns
    - Auto-categorizing discovery patterns
    - Enriching dance signals with LLM-generated context

    LLM calls are optional — falls back to base behavior when
    dashscope is unavailable or no config is provided.

    Example:
        >>> config = SwarmLLMConfig(model="qwen-plus")
        >>> agent = DashScopeAgent(config)
        >>> parser = LLMDanceParser(agent=agent)
        >>> dance = parser.parse_discovery({...})
        >>> print(dance.pattern.get("llm_description"))
    """

    DEFAULT_ENRICH_PROMPT = (
        "Given a discovery signal from a swarm agent, provide a concise "
        "natural language description of what this signal might mean. "
        "Also categorize it into one of: anomaly, optimization, opportunity, "
        "threat, coordination, exploration. Respond with JSON: "
        '{{"description": "...", "category": "..."}}'
    )

    def __init__(
        self,
        agent: Optional[DashScopeAgent] = None,
        enable_enrichment: bool = True,
        intensity_threshold: float = 0.5,
    ):
        super().__init__(intensity_threshold=intensity_threshold)
        self._llm_agent = agent
        self._enable_enrichment = enable_enrichment and agent is not None

    @property
    def llm_enabled(self) -> bool:
        return self._enable_enrichment

    def parse_discovery(self, discovery: Dict[str, Any]) -> Optional[DanceSignal]:
        """Parse a discovery signal, optionally enriched by LLM.

        Args:
            discovery: Dict with confidence, impact, novelty, pattern, agent_id

        Returns:
            DanceSignal if intensity exceeds threshold, None otherwise
        """
        dance = super().parse_discovery(discovery)
        if dance is None:
            return None

        if self._enable_enrichment:
            self._enrich_dance(dance, discovery)

        return dance

    def _enrich_dance(
        self, dance: DanceSignal, discovery: Dict[str, Any]
    ) -> None:
        """Enrich a dance signal with LLM-generated metadata."""
        signal_summary = json.dumps({
            "intensity": dance.intensity,
            "direction": dance.direction,
            "agent_id": dance.agent_id,
            "pattern_keys": list(dance.pattern.keys()),
        })

        prompt = (
            f"Analyze this swarm discovery signal:\n{signal_summary}\n\n"
            "Provide a JSON object with 'description' (1-2 sentence natural language "
            "interpretation) and 'category' (one of: anomaly, optimization, "
            "opportunity, threat, coordination, exploration)."
        )

        try:
            result = self._llm_agent.complete(
                prompt=prompt,
                system_prompt=self.DEFAULT_ENRICH_PROMPT,
                temperature=0.3,
                max_tokens=256,
            )

            enrichment = json.loads(result)
            dance.pattern["llm_description"] = enrichment.get("description", "")
            dance.pattern["llm_category"] = enrichment.get("category", "")
            dance.pattern["llm_enriched"] = True
        except (json.JSONDecodeError, RuntimeError, KeyError):
            dance.pattern["llm_enriched"] = False

    def categorize_patterns(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Use LLM to categorize top patterns and explain them.

        Returns:
            List of dicts with pattern name, weight, llm_category, llm_explanation
        """
        top = self.get_top_patterns(top_n)
        if not top or not self._enable_enrichment:
            return [{"pattern": p, "weight": w} for p, w in top]

        weights = dict(top)
        prompt = (
            f"Given these dance pattern weights: {json.dumps(weights)}, "
            f"categorize each pattern and provide a brief explanation. "
            f"Return JSON array of objects with 'pattern', 'category', and 'explanation'."
        )

        try:
            result = self._llm_agent.complete(
                prompt=prompt,
                system_prompt=(
                    "You are a swarm intelligence analyst. Categorize dance patterns "
                    "and explain their significance. Return a JSON array."
                ),
                temperature=0.3,
                max_tokens=512,
            )

            categories = json.loads(result)
            enriched = []
            for entry in categories:
                enriched.append({
                    "pattern": entry.get("pattern", ""),
                    "weight": weights.get(entry.get("pattern", ""), 0.0),
                    "llm_category": entry.get("category", ""),
                    "llm_explanation": entry.get("explanation", ""),
                })
            return enriched
        except (json.JSONDecodeError, RuntimeError):
            return [{"pattern": p, "weight": w} for p, w in top]


class LLMBeeAgent:
    """A bee agent that uses LLM to decide responses to dance signals.

    Receives dance signals (optionally via message bus), uses an LLM to
    decide how to respond, and sends responses back.

    Example:
        >>> config = SwarmLLMConfig(model="qwen-turbo")
        >>> agent = DashScopeAgent(config)
        >>> bee = LLMBeeAgent("worker_1", llm_agent=agent)
        >>> response = bee.receive_dance({
        ...     "dance_id": "d1",
        ...     "agent_id": "explorer",
        ...     "intensity": 0.8,
        ...     "direction": "anomaly",
        ...     "pattern": {"type": "anomaly"},
        ... })
        >>> print(response.get("action"))
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a worker bee in a swarm intelligence system. "
        "You receive dance signals from explorer bees and must decide "
        "how to respond. Available actions: investigate, ignore, "
        "recruit_others, report_upstream. Respond with JSON: "
        '{"action": "...", "reason": "...", "confidence": 0.0-1.0}'
    )

    def __init__(
        self,
        agent_id: str,
        llm_agent: Optional[DashScopeAgent] = None,
        message_bus: Optional[Any] = None,
        system_prompt: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self._llm_agent = llm_agent
        self._message_bus = message_bus
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._dance_history: List[Dict[str, Any]] = []
        self._response_log: List[Dict[str, Any]] = []

    @property
    def llm_enabled(self) -> bool:
        return self._llm_agent is not None

    def receive_dance(self, dance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a dance signal and return an LLM-informed decision.

        Args:
            dance_data: Dict containing dance signal info

        Returns:
            Response dict with action, reason, confidence
        """
        self._dance_history.append(dance_data)

        if self._llm_agent is not None:
            response = self._decide_via_llm(dance_data)
        else:
            response = self._decide_rule_based(dance_data)

        response["agent_id"] = self.agent_id
        response["dance_id"] = dance_data.get("dance_id", "")
        response["timestamp"] = datetime.now(timezone.utc).isoformat()

        self._response_log.append(response)

        if self._message_bus is not None:
            self._publish_response(response)

        return response

    def receive_dance_signal(self, signal: DanceSignal) -> Dict[str, Any]:
        """Process a DanceSignal object directly.

        Args:
            signal: A DanceSignal instance

        Returns:
            Response dict with action, reason, confidence
        """
        dance_data = {
            "dance_id": signal.dance_id,
            "agent_id": signal.agent_id,
            "intensity": signal.intensity,
            "direction": signal.direction,
            "duration": signal.duration,
            "pattern": signal.pattern,
        }
        return self.receive_dance(dance_data)

    def get_response_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent response decisions."""
        return self._response_log[-limit:]

    def get_dance_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent received dance signals."""
        return self._dance_history[-limit:]

    def _decide_via_llm(self, dance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to decide how to respond to a dance signal."""
        signal_summary = json.dumps(dance_data, indent=2)
        prompt = (
            f"A dance signal was received:\n{signal_summary}\n\n"
            "Decide how to respond. Return JSON with 'action' (one of: "
            "investigate, ignore, recruit_others, report_upstream), "
            "'reason' (brief explanation), and 'confidence' (0.0-1.0)."
        )

        try:
            result = self._llm_agent.complete(
                prompt=prompt,
                system_prompt=self._system_prompt,
                temperature=0.5,
                max_tokens=256,
            )
            response = json.loads(result)
            response.setdefault("action", "investigate")
            response.setdefault("reason", "")
            response.setdefault("confidence", 0.5)
            response["decision_method"] = "llm"
            return response
        except (json.JSONDecodeError, RuntimeError, KeyError):
            return self._decide_rule_based(dance_data)

    def _decide_rule_based(self, dance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback rule-based decision when LLM is unavailable."""
        intensity = dance_data.get("intensity", 0.5)
        direction = dance_data.get("direction", "unknown")

        if intensity >= 0.7:
            action = "investigate"
            reason = f"High intensity ({intensity:.2f}) in {direction}"
            confidence = intensity
        elif intensity >= 0.4:
            action = "recruit_others"
            reason = f"Moderate intensity ({intensity:.2f}), gather more data"
            confidence = 0.5
        else:
            action = "ignore"
            reason = f"Low intensity ({intensity:.2f}), not worth acting on"
            confidence = 0.3

        return {
            "action": action,
            "reason": reason,
            "confidence": confidence,
            "decision_method": "rule_based",
        }

    def _publish_response(self, response: Dict[str, Any]) -> None:
        """Publish response to message bus if available."""
        if MessageBus is None:
            return

        try:
            topic = f"bee_response.{self.agent_id}"
            bus = MessageBus()
            bus.publish(topic, {
                "id": str(uuid.uuid4()),
                "agent_id": self.agent_id,
                "response": response,
            })
        except Exception:
            pass
