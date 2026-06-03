"""
loom.blackboard
===============

The shared blackboard. It holds the current GW-IR, the executed values keyed by
node, the latest telemetry per node, the open subgoals, and a growing
provenance record. Because the shared artifact (the GW-IR) is typed rather than
textual, conflicting agent proposals are reconciled by type checking and cost
comparison rather than by another round of free-form debate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .gwir import GWIR
from .telemetry import Telemetry


@dataclass
class ProvenanceEntry:
    node: int
    op: str
    params: dict
    telemetry: Optional[Telemetry] = None
    event: str = "execute"   # execute | reflect | verify | revise


@dataclass
class Subgoal:
    intent: str
    params: dict = field(default_factory=dict)
    closed: bool = False


class Blackboard:
    def __init__(self) -> None:
        self.gwir = GWIR()
        self.results: Dict[int, Any] = {}              # node id -> executed value
        self.telemetry: Dict[int, Telemetry] = {}      # node id -> latest telemetry
        self.provenance: List[ProvenanceEntry] = []
        self.subgoals: List[Subgoal] = []
        self.last_graph_node: Optional[int] = None     # convenience anchor for "the graph"
        self.notes: List[str] = []

    # --- subgoals --------------------------------------------------------- #
    def open_subgoal(self) -> Optional[Subgoal]:
        for s in self.subgoals:
            if not s.closed:
                return s
        return None

    # --- results ---------------------------------------------------------- #
    def value(self, node: int) -> Any:
        return self.results.get(node)

    def record(self, node: int, op: str, params: dict, tel: Telemetry, event: str = "execute"):
        if tel is not None:
            self.telemetry[node] = tel
        self.provenance.append(ProvenanceEntry(node, op, dict(params), tel, event))

    def log(self, msg: str):
        self.notes.append(msg)

    # --- provenance rendering -------------------------------------------- #
    def provenance_text(self) -> str:
        lines = []
        for e in self.provenance:
            t = f"  [{e.telemetry.summary()}]" if e.telemetry else ""
            lines.append(f"  n{e.node} {e.op} ({e.event}){t}")
        return "\n".join(lines)
