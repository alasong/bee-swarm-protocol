"""LLM integration via DashScope API (Aliyun Bailian / 阿里云百炼).

Uses the OpenAI-compatible interface of DashScope to enable:
- Dance signal analysis with natural language summaries
- Pattern interpretation via LLM reasoning
- Goal description generation from system state
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import dashscope
    from dashscope import Generation
except ImportError:
    dashscope = None
    Generation = None


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw_response: Optional[Dict[str, Any]] = None


class DashScopeClient:
    """Thin wrapper around DashScope's OpenAI-compatible API."""

    DEFAULT_MODEL = "qwen-plus"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model or self.DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("DASHSCOPE_PRO") or os.environ.get("DASHSCOPE_API_KEY", "")
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Send a chat completion request."""
        if dashscope is None:
            raise RuntimeError(
                "dashscope package is required. Install with: pip install dashscope"
            )

        client_model = model or self.model
        response = Generation.call(
            model=client_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self.api_key,
            result_format="message",
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"DashScope API error: {response.status_code} - {response.message}"
            )

        output = response.output
        usage = response.usage or {}
        choices = output.get("choices", [])
        content = ""
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")

        return LLMResponse(
            content=content,
            model=client_model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            raw_response=output,
        )


class DanceLLMAnalyzer:
    """
    Uses DashScope LLM to analyze dance signals and patterns.

    - Natural language interpretation of dance patterns
    - Anomaly explanation and context
    - Strategic recommendations based on consensus results
    """

    ANALYSIS_SYSTEM_PROMPT = """You are an expert in swarm intelligence and multi-agent coordination.
Analyze dance signals from a bee-inspired swarm protocol and provide insightful interpretations.
Be concise and actionable."""

    def __init__(self, client: Optional[DashScopeClient] = None):
        self._client = client or DashScopeClient()
        self._analysis_history: List[LLMResponse] = []

    async def analyze_pattern(
        self,
        pattern_weights: Dict[str, float],
        dance_count: int = 0,
        context: Optional[str] = None,
    ) -> LLMResponse:
        """Use LLM to interpret accumulated dance pattern weights."""
        weights_str = json.dumps(pattern_weights, indent=2)
        prompt = (
            f"Analyze the following dance pattern weights from a swarm system:\n\n"
            f"Pattern Weights:\n{weights_str}\n"
            f"Total Dances: {dance_count}\n"
        )
        if context:
            prompt += f"\nContext: {context}\n"
        prompt += (
            "\nProvide: 1) What patterns are dominant and why, "
            "2) What this suggests about agent behavior, "
            "3) Recommended next action."
        )

        messages = [
            {"role": "system", "content": self.ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = await self._chat(messages)
        self._analysis_history.append(response)
        return response

    async def explain_anomaly(
        self, signal_data: Dict[str, Any], history: Optional[List[Dict[str, Any]]] = None
    ) -> LLMResponse:
        """Use LLM to provide explanation for an anomaly signal."""
        signal_str = json.dumps(signal_data, indent=2)
        prompt = (
            f"An anomaly signal was detected in the swarm system:\n\n{signal_str}\n\n"
            "Explain what this anomaly might indicate in a multi-agent system "
            "and suggest investigation steps."
        )

        messages = [
            {"role": "system", "content": self.ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = await self._chat(messages)
        self._analysis_history.append(response)
        return response

    async def recommend_from_goals(
        self, goals: List[Dict[str, Any]], system_state: Dict[str, Any]
    ) -> LLMResponse:
        """Use LLM to recommend priority ordering and strategy for active goals."""
        goals_str = json.dumps(goals, indent=2)
        state_str = json.dumps(system_state, indent=2)
        prompt = (
            f"System State:\n{state_str}\n\n"
            f"Active Goals:\n{goals_str}\n\n"
            "Recommend: 1) Priority ordering of these goals, "
            "2) Which goals should be pursued in parallel vs sequentially, "
            "3) Risk factors to monitor."
        )

        messages = [
            {"role": "system", "content": self.ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = await self._chat(messages)
        self._analysis_history.append(response)
        return response

    async def _chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Internal chat wrapper."""
        return self._client.chat(messages)

    def get_analysis_history(self, limit: int = 100) -> List[LLMResponse]:
        """Get recent analysis history."""
        return self._analysis_history[-limit:]

    def clear_history(self) -> None:
        """Clear analysis history."""
        self._analysis_history.clear()
