"""
loom.telemetry
==============

Structured runtime telemetry returned by the backend after each kernel, plus a
lightweight analytic cost model. Generic agentic reflection critiques text; in
LOOM the Performance Engineer reflects over these numeric signals, and the
Algorithm Strategist grounds kernel selection in the cost model rather than in
an LLM's generic prior.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Telemetry:
    """Runtime signals for a single kernel execution."""
    kernel: str
    variant: Optional[str] = None
    time_s: float = 0.0
    n: int = 0                       # vertices
    m: int = 0                       # edges (undirected count where applicable)
    load_imbalance: float = 0.0      # coefficient of variation of per-locale work
    comm_volume_mb: float = 0.0      # inter-locale communication
    peak_mem_gb: float = 0.0         # per-locale high-water mark
    locales: int = 1

    def is_inefficient(self, imbalance_thresh: float = 0.5,
                       comm_ratio_thresh: float = 0.75) -> bool:
        """Heuristic the Performance Engineer uses to decide whether to re-plan."""
        comm_ratio = self.comm_volume_mb / max(self.m / 1e3, 1.0)
        return self.load_imbalance > imbalance_thresh or comm_ratio > comm_ratio_thresh

    def summary(self) -> str:
        v = f"/{self.variant}" if self.variant else ""
        return (f"{self.kernel}{v}: t={self.time_s:.2f}s "
                f"imbalance={self.load_imbalance:.2f} comm={self.comm_volume_mb:.1f}MB "
                f"mem={self.peak_mem_gb:.2f}GB locales={self.locales}")


@dataclass
class GraphStats:
    """Cheap structural statistics Arachne can report without a full pass."""
    n: int
    m: int
    max_degree: int
    avg_degree: float
    est_diameter: int = 0


class CostModel:
    """A deliberately simple analytic model. Its value is that it grounds choices
    in graph structure and runtime cost, not that it is exact."""

    # ---- algorithm selection -------------------------------------------------

    def recommend_community_variant(self, stats: GraphStats,
                                    time_budget_s: float = 60.0,
                                    target_quality: float = 0.5) -> str:
        """Label propagation is cheap and communication-light but lower quality;
        Louvain is higher quality but heavier. Choose by edge count, budget, and
        the requested quality (cf. the Arachne community-detection study)."""
        heavy_cost = self._louvain_cost(stats)
        if heavy_cost <= time_budget_s and target_quality >= 0.4:
            return "louvain"
        return "label_propagation"

    def recommend_bfs_direction(self, frontier: int, unvisited: int) -> str:
        """Bottom-up wins when the frontier is large relative to the work it
        would save scanning unvisited vertices."""
        return "bottom_up" if frontier > 0.5 * max(unvisited, 1) else "top_down"

    # ---- rough cost estimates (seconds) -------------------------------------

    def _louvain_cost(self, s: GraphStats) -> float:
        # ~ O(m log n) with a constant; scaled to "seconds" for demonstration.
        import math
        return 2.0e-7 * s.m * max(math.log2(max(s.n, 2)), 1.0)

    def _label_prop_cost(self, s: GraphStats) -> float:
        return 6.0e-8 * s.m

    def estimate(self, op: str, stats: GraphStats, variant: Optional[str] = None) -> float:
        if op == "community":
            return (self._louvain_cost(stats) if variant == "louvain"
                    else self._label_prop_cost(stats))
        if op in ("triangle_count", "per_block_triangles"):
            return 3.0e-7 * stats.m * max(stats.avg_degree, 1.0)
        if op in ("connected_components", "bfs"):
            return 5.0e-8 * (stats.n + stats.m)
        if op == "betweenness":
            return 1.0e-7 * stats.n * stats.m / max(stats.n, 1)
        return 1.0e-8 * (stats.n + stats.m)
