"""
loom.agents
===========

The role-specialized agent team. Each agent reads and writes the shared GW-IR on
the blackboard. Coordination follows a blackboard + contract-net pattern: the
Orchestrator announces a subgoal, specialists return typed *bids* (GW-IR
fragments annotated with confidence and estimated cost), and the Orchestrator
commits the strongest valid one. Because the shared artifact is typed, bids are
reconciled by type checking and cost rather than free-form debate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from .backend import ArachneBackend, GraphData
from .blackboard import Blackboard, Subgoal
from .gwir import Artifact, ArtifactType
from .llm import LLMClient
from .telemetry import CostModel, GraphStats, Telemetry


@dataclass
class Bid:
    label: str
    confidence: float
    est_cost: float
    apply: Callable[[Blackboard], List[int]]   # mutates GW-IR, returns added node ids
    bidder: str


class Agent:
    def __init__(self, name: str):
        self.name = name


# --------------------------------------------------------------------------- #
class OrchestratorAgent(Agent):
    """Decomposes the goal, seeds subgoals, runs contract-net rounds, decides done."""

    def __init__(self, llm: LLMClient):
        super().__init__("Orchestrator")
        self.llm = llm

    def seed(self, bb: Blackboard, goal: str) -> None:
        bb.subgoals.append(Subgoal("graph_ready", {"src": "connectome.pq"}))
        for sg in self.llm.decompose(goal):
            bb.subgoals.append(Subgoal(sg["intent"], sg.get("params", {})))
        bb.log(f"Orchestrator decomposed goal into: "
               f"{[s.intent for s in bb.subgoals]}")

    def commit(self, bids: List[Bid]) -> Optional[Bid]:
        if not bids:
            return None
        # highest confidence, then lowest estimated cost (contract-net award)
        return sorted(bids, key=lambda b: (-b.confidence, b.est_cost))[0]


# --------------------------------------------------------------------------- #
class DataEngineerAgent(Agent):
    """Builds the graph and is the sole issuer of server requests (executor)."""

    def __init__(self, backend: ArachneBackend):
        super().__init__("DataEngineer")
        self.backend = backend

    def bid(self, sg: Subgoal, bb: Blackboard) -> Optional[Bid]:
        if sg.intent != "graph_ready":
            return None

        def apply(bb: Blackboard) -> List[int]:
            g = bb.gwir
            n_load = g.add("load", params={"src": sg.params.get("src", "edges.pq")},
                           note="DataEngineer")
            n_build = g.add("build_graph", inputs=[n_load], note="DataEngineer")
            n_sym = g.add("symmetrize", inputs=[n_build], note="DataEngineer")
            n_cc = g.add("largest_cc", inputs=[n_sym], note="DataEngineer")
            bb.last_graph_node = n_cc
            return [n_load, n_build, n_sym, n_cc]

        return Bid("ingest+build+symmetrize+largest_cc", 0.95, 0.1, apply, self.name)

    # ---- execution (issues the actual backend / server call) ------------- #
    def execute(self, node, bb: Blackboard):
        op = node.op
        be = self.backend
        if op == "load":
            return be.load_edges(node.params.get("src", "edges.pq"))
        if op == "build_graph":
            return be.build_graph(bb.value(node.inputs[0]))
        if op == "symmetrize":
            return be.symmetrize(bb.value(node.inputs[0]))
        if op == "largest_cc":
            return be.largest_cc(bb.value(node.inputs[0]))
        if op == "connected_components":
            return be.connected_components(bb.value(node.inputs[0]))
        if op == "community":
            return be.community(bb.value(node.inputs[0]), node.params.get("variant", "label_propagation"))
        if op == "triangle_count":
            return be.triangle_count(bb.value(node.inputs[0]))
        if op == "per_block_triangles":
            g = bb.value(node.inputs[0]); part = bb.value(node.inputs[1])
            return be.per_block_triangles(g, part, node.params.get("strategy", "global"))
        if op == "k_truss":
            return be.k_truss(bb.value(node.inputs[0]), int(node.params.get("k", 3)))
        raise NotImplementedError(f"DataEngineer cannot execute op '{op}'")


# --------------------------------------------------------------------------- #
class AlgorithmStrategistAgent(Agent):
    """Selects kernels and variants for analytical subgoals via the cost model."""

    def __init__(self, cost: CostModel):
        super().__init__("AlgorithmStrategist")
        self.cost = cost

    def _graph_stats(self, bb: Blackboard) -> Optional[GraphStats]:
        g = bb.value(bb.last_graph_node) if bb.last_graph_node is not None else None
        return g.stats() if isinstance(g, GraphData) else GraphStats(0, 0, 0, 0.0)

    def bid(self, sg: Subgoal, bb: Blackboard) -> Optional[Bid]:
        anchor = bb.last_graph_node
        if anchor is None and sg.intent in ("community", "motif_density",
                                            "connected_components", "centrality",
                                            "traversal", "truss"):
            return None  # graph not ready yet

        stats = self._graph_stats(bb)
        budget = float(sg.params.get("time_budget_s", 60.0))

        if sg.intent == "connected_components":
            def apply(bb):
                nid = bb.gwir.add("connected_components", inputs=[anchor], note=self.name)
                return [nid]
            return Bid("connected_components", 0.9,
                       self.cost.estimate("connected_components", stats), apply, self.name)

        if sg.intent == "community":
            variant = self.cost.recommend_community_variant(stats, budget, target_quality=0.5)

            def apply(bb):
                nid = bb.gwir.add("community", inputs=[anchor],
                                  params={"variant": variant}, note=self.name)
                bb.last_partition_node = nid
                return [nid]
            return Bid(f"community/{variant}", 0.9,
                       self.cost.estimate("community", stats, variant), apply, self.name)

        if sg.intent == "motif_density":
            # needs a partition; reuse the latest, else add a community step first
            def apply(bb):
                added = []
                part_node = getattr(bb, "last_partition_node", None)
                if part_node is None:
                    variant = self.cost.recommend_community_variant(stats, budget, 0.5)
                    part_node = bb.gwir.add("community", inputs=[anchor],
                                            params={"variant": variant}, note=self.name)
                    bb.last_partition_node = part_node
                    added.append(part_node)
                pbt = bb.gwir.add("per_block_triangles", inputs=[anchor, part_node],
                                  params={"strategy": "global"}, note=self.name)
                added.append(pbt)
                return added
            return Bid("per_block_triangles", 0.85,
                       self.cost.estimate("per_block_triangles", stats), apply, self.name)

        if sg.intent == "truss":
            def apply(bb):
                nid = bb.gwir.add("k_truss", inputs=[anchor],
                                  params={"k": int(sg.params.get("k", 3))}, note=self.name)
                return [nid]
            return Bid("k_truss", 0.8, self.cost.estimate("k_truss", stats), apply, self.name)

        if sg.intent == "centrality":
            def apply(bb):
                return [bb.gwir.add("betweenness", inputs=[anchor], note=self.name)]
            return Bid("betweenness", 0.7, self.cost.estimate("betweenness", stats), apply, self.name)

        if sg.intent == "traversal":
            def apply(bb):
                return [bb.gwir.add("bfs", inputs=[anchor], params={"source": 0}, note=self.name)]
            return Bid("bfs", 0.7, self.cost.estimate("bfs", stats), apply, self.name)

        return None

    def alternative_for(self, node, bb: Blackboard) -> bool:
        """Invoked when a result fails verification: pick a different variant."""
        if node.op == "community":
            cur = node.params.get("variant")
            node.params["variant"] = "louvain" if cur != "louvain" else "label_propagation"
            node.executed = False
            bb.log(f"AlgorithmStrategist switched community variant to "
                   f"{node.params['variant']} after verification failure")
            return True
        return False


# --------------------------------------------------------------------------- #
class PerformanceEngineerAgent(Agent):
    """Reads telemetry and proposes concrete GW-IR edits to fix inefficiency."""

    def __init__(self):
        super().__init__("PerformanceEngineer")

    def reflect(self, node, tel: Telemetry, bb: Blackboard) -> bool:
        """Return True if a re-run is requested after an edit."""
        if tel is None or not tel.is_inefficient():
            return False
        if node.op == "per_block_triangles" and node.params.get("strategy") != "within_community":
            node.params["strategy"] = "within_community"
            node.executed = False
            bb.log(f"PerformanceEngineer: high comm/imbalance on n{node.nid} "
                   f"(imbalance={tel.load_imbalance:.2f}); restricting motif search to "
                   f"within-community edges and batching, then re-running")
            bb.record(node.nid, node.op, node.params, tel, event="reflect")
            return True
        bb.log(f"PerformanceEngineer: noted inefficiency on n{node.nid} "
               f"({tel.summary()}); no automatic remedy available")
        return False


# --------------------------------------------------------------------------- #
class VerifierAgent(Agent):
    """Static GW-IR validation plus invariant checks on results."""

    def __init__(self):
        super().__init__("Verifier")

    def check_results(self, bb: Blackboard):
        from . import verify as V
        invs = []
        # graph identity
        if bb.last_graph_node is not None:
            invs.append(V.degree_sum(bb.value(bb.last_graph_node)))
        # partition coverage + triangle cross-check
        for nid, node in bb.gwir.nodes.items():
            if not node.executed:
                continue
            if node.op in ("community", "connected_components"):
                invs.append(V.partition_covers(bb.value(nid)))
            if node.op == "per_block_triangles":
                graph = bb.value(node.inputs[0])
                invs.append(V.per_block_le_global(bb.value(nid), graph))
            if node.op == "k_truss":
                invs.append(V.k_truss_property(bb.value(nid), int(node.params.get("k", 3))))
        return invs


# --------------------------------------------------------------------------- #
class InterpreterAgent(Agent):
    """Renders verified results and provenance into a plain-language report."""

    def __init__(self, llm: LLMClient):
        super().__init__("Interpreter")
        self.llm = llm

    def render(self, bb: Blackboard, goal: str) -> str:
        lines = [f"Goal: {goal}", ""]

        # community / motif ranking, if present
        part_node = getattr(bb, "last_partition_node", None)
        pbt_node = next((nid for nid, n in bb.gwir.nodes.items()
                         if n.op == "per_block_triangles" and n.executed), None)
        if part_node is not None and bb.value(part_node) is not None:
            labels = bb.value(part_node)
            if isinstance(labels, np.ndarray):
                ids, sizes = np.unique(labels, return_counts=True)
                size_of = dict(zip(ids.tolist(), sizes.tolist()))
                lines.append(f"Found {len(ids)} communities "
                             f"(largest has {int(sizes.max())} vertices).")
                if pbt_node is not None and isinstance(bb.value(pbt_node), dict):
                    per = bb.value(pbt_node)
                    ranked = sorted(per.items(),
                                    key=lambda kv: kv[1] / max(size_of.get(kv[0], 1), 1),
                                    reverse=True)[:5]
                    lines.append("Top communities by internal triangle density:")
                    for rank, (b, tri) in enumerate(ranked, 1):
                        sz = size_of.get(b, 0)
                        dens = tri / max(sz, 1)
                        lines.append(f"  {rank}. community {b}: {tri} internal triangles, "
                                     f"{sz} vertices (density {dens:.2f})")

        lines.append("")
        lines.append("Provenance (replayable record):")
        lines.append(bb.provenance_text())
        digest = "\n".join(lines)
        return self.llm.narrate(digest)
