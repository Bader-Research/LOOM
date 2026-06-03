"""
loom.llm
========

A small LLM abstraction. The agents use it for two things: decomposing a
natural-language goal into subgoals (Orchestrator) and rendering the final
report (Interpreter). Every performance-critical or correctness-critical
decision in LOOM (typing, cost-model selection, telemetry reflection,
invariant checking) is real deterministic code, not the LLM, so the system is
fully functional with the rule-based fallback and merely *nicer* with a model.

* ``AnthropicLLM`` -- uses the ``anthropic`` SDK and ``ANTHROPIC_API_KEY``.
* ``RuleBasedLLM`` -- deterministic keyword intent parser; no network, no key.
"""
from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Dict, List


class LLMClient(ABC):
    @abstractmethod
    def decompose(self, goal: str) -> List[Dict]:
        """Return an ordered list of analytical subgoals, each a dict like
        {"intent": "community", "params": {...}}."""

    @abstractmethod
    def narrate(self, context: str) -> str:
        """Render a short natural-language report from a textual context."""


# Recognized analytical intents and the keywords that signal them.
_INTENTS = [
    ("community",            ["communit", "cluster", "module", "group"]),
    ("motif_density",        ["motif", "triangle", "three-neuron", "3-node", "dense"]),
    ("connected_components", ["connected component", "components", "reachab"]),
    ("centrality",           ["central", "betweenness", "influential", "important vertex"]),
    ("traversal",            ["shortest path", "bfs", "breadth", "distance", "hops"]),
    ("truss",                ["truss", "k-core", "core"]),
]


class RuleBasedLLM(LLMClient):
    """Deterministic, offline. Good enough to drive and to test the pipeline."""

    def decompose(self, goal: str) -> List[Dict]:
        g = goal.lower()
        found: List[str] = []
        for intent, kws in _INTENTS:
            if any(k in g for k in kws):
                found.append(intent)
        # sensible default if nothing matched
        if not found:
            found = ["community"]
        # honor a natural ordering: structure first, then per-structure metrics
        order = {"connected_components": 0, "community": 1, "truss": 1,
                 "centrality": 2, "traversal": 2, "motif_density": 3}
        found = sorted(dict.fromkeys(found), key=lambda i: order.get(i, 5))
        # extract a time budget if mentioned ("within 30 seconds", "interactive")
        budget = 60.0
        mb = re.search(r"(\d+)\s*(second|sec|s)\b", g)
        if mb:
            budget = float(mb.group(1))
        elif "interactive" in g:
            budget = 20.0
        return [{"intent": i, "params": {"time_budget_s": budget}} for i in found]

    def narrate(self, context: str) -> str:
        # The context is already a structured, human-readable digest; pass through
        # with a light framing line. (A model would prose-ify this further.)
        return "Summary of the completed analysis:\n" + context


class AnthropicLLM(LLMClient):  # pragma: no cover - requires network + key
    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 1024):
        try:
            import anthropic
        except Exception as exc:
            raise RuntimeError("pip install anthropic to use AnthropicLLM") from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("set ANTHROPIC_API_KEY to use AnthropicLLM")
        self._client = anthropic.Anthropic()
        self._fallback = RuleBasedLLM()
        self.model = model
        self.max_tokens = max_tokens

    def _complete(self, system: str, user: str) -> str:
        msg = self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            system=system, messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

    def decompose(self, goal: str) -> List[Dict]:
        system = (
            "You plan large-scale graph-analytics workflows on Arachne. "
            "Given a user goal, output ONLY a JSON array of subgoals. Each item is "
            '{"intent": <one of community|motif_density|connected_components|'
            'centrality|traversal|truss>, "params": {"time_budget_s": <number>}}. '
            "Order structural steps before per-structure metrics. No prose.")
        try:
            raw = self._complete(system, goal)
            data = json.loads(re.sub(r"```(json)?", "", raw).strip())
            assert isinstance(data, list) and data
            return data
        except Exception:
            return self._fallback.decompose(goal)

    def narrate(self, context: str) -> str:
        system = ("You are the Interpreter agent in a graph-analytics system. "
                  "Write a concise, plain-language report for a domain scientist. "
                  "No markdown headers. Do not use em dashes.")
        try:
            return self._complete(system, context)
        except Exception:
            return self._fallback.narrate(context)
