"""
loom.verify
===========

Invariant-based verification. At billion-edge scale there is rarely a
ground-truth answer, so the Verifier checks identities that must hold for any
correct result, and cross-checks a quantity by recomputing it with an
independent method on a sample. These functions operate on the simulation
backend's native values (GraphData, NumPy partitions). On the real Arkouda
backend the same identities are expressed as server-side reductions; the checks
here degrade to "skipped" rather than crash if they cannot introspect a value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .backend import GraphData


@dataclass
class Invariant:
    name: str
    ok: bool
    detail: str
    skipped: bool = False


def degree_sum(graph) -> Invariant:
    """For an undirected graph, sum of degrees equals twice the edge count."""
    if not isinstance(graph, GraphData):
        return Invariant("degree_sum", True, "skipped (non-introspectable graph)", skipped=True)
    deg = graph.degrees()
    total = int(deg.sum())
    expected = 2 * graph.m
    ok = total == expected
    return Invariant("degree_sum", ok, f"sum(deg)={total}, 2|E|={expected}")


def simple_symmetric(graph) -> Invariant:
    """An undirected graph's arc list must be symmetric, with no duplicate arcs
    and no self-loops. (Under the both-directions representation, degree_sum
    holds for any even-length arc list; this check is the one that can fail.)"""
    if not isinstance(graph, GraphData) or graph.directed:
        return Invariant("simple_symmetric", True, "skipped", skipped=True)
    n = graph.n
    fwd = graph.src.astype(np.int64) * n + graph.dst
    rev = graph.dst.astype(np.int64) * n + graph.src
    ufwd = np.unique(fwd)
    dup = len(fwd) - len(ufwd)
    loops = int(np.sum(graph.src == graph.dst))
    asym = len(np.setxor1d(ufwd, np.unique(rev)))
    ok = dup == 0 and loops == 0 and asym == 0
    return Invariant("simple_symmetric", ok,
                     f"duplicate arcs={dup}, self-loops={loops}, unmatched arcs={asym}")


def partition_covers(labels, n: Optional[int] = None) -> Invariant:
    """A community/component labeling must assign every vertex exactly once."""
    if not isinstance(labels, np.ndarray):
        return Invariant("partition_covers", True, "skipped", skipped=True)
    n = n if n is not None else len(labels)
    if len(labels) != n:
        return Invariant("partition_covers", False, f"labels length {len(labels)} != n {n}")
    if np.any(labels < 0):
        return Invariant("partition_covers", False, "some vertices unassigned (label < 0)")
    return Invariant("partition_covers", True,
                     f"{n} vertices each assigned to one of {len(np.unique(labels))} blocks")


def components_monotone(before_count: int, after_count: int) -> Invariant:
    """Adding edges cannot increase the number of connected components."""
    ok = after_count <= before_count
    return Invariant("components_monotone", ok,
                     f"components before={before_count}, after={after_count}")


def per_block_le_global(per_block: Dict[int, int], graph) -> Invariant:
    """Internal-only triangle totals cannot exceed the global triangle count.

    The global count is recomputed here by an independent routine (not the
    backend op that produced per_block), realizing the paper's cross-estimator
    spot check."""
    if not isinstance(graph, GraphData):
        return Invariant("triangles_cross_check", True, "skipped", skipped=True)
    internal_total = int(sum(per_block.values()))
    global_total = _independent_global_triangles(graph)
    ok = internal_total <= global_total
    return Invariant("triangles_cross_check", ok,
                     f"sum(internal per-block)={internal_total} <= global={global_total}")


def per_block_matches_independent(per_block: Dict[int, int], graph, labels,
                                  max_block: int = 4096) -> Invariant:
    """Two-sided cross-check: each block's internal triangle count must equal an
    independent recount, trace(A_b^3)/6 on the block's induced subgraph.
    Unlike per_block_le_global this also catches undercounts, zeroed results,
    and dropped blocks. Blocks larger than max_block are skipped (dense
    recount); on the Arkouda backend this becomes a server-side reduction."""
    if not isinstance(graph, GraphData) or not isinstance(labels, np.ndarray):
        return Invariant("triangles_two_sided", True, "skipped", skipped=True)
    bad, skipped = [], 0
    for b in np.unique(labels):
        idx = np.where(labels == b)[0]
        if len(idx) > max_block:
            skipped += 1
            continue
        pos = -np.ones(graph.n, dtype=np.int64)
        pos[idx] = np.arange(len(idx))
        m = (pos[graph.src] >= 0) & (pos[graph.dst] >= 0)
        A = np.zeros((len(idx), len(idx)))
        A[pos[graph.src[m]], pos[graph.dst[m]]] = 1.0
        expected = int(round(float(((A @ A) * A).sum()) / 6.0))
        got = int(per_block.get(int(b), 0))
        if got != expected:
            bad.append((int(b), got, expected))
    detail = f"{len(bad)} block(s) disagree with independent recount"
    if bad:
        detail += f", e.g. block {bad[0][0]}: reported {bad[0][1]}, recount {bad[0][2]}"
    if skipped:
        detail += f"; {skipped} large block(s) skipped"
    return Invariant("triangles_two_sided", not bad, detail)


def k_truss_property(truss: GraphData, k: int) -> Invariant:
    """Every edge of a k-truss lies in at least k-2 triangles."""
    if not isinstance(truss, GraphData):
        return Invariant("k_truss_property", True, "skipped", skipped=True)
    adj = [frozenset(s) for s in truss.adjacency()]
    bad = 0
    for u in range(truss.n):
        for v in adj[u]:
            if v > u and len(adj[u] & adj[v]) < (k - 2):
                bad += 1
    return Invariant("k_truss_property", bad == 0, f"{bad} edge(s) below k-2 triangles")


def _independent_global_triangles(g: GraphData) -> int:
    adj = [frozenset(s) for s in g.adjacency()]
    total = 0
    for u in range(g.n):
        for v in adj[u]:
            if v > u:
                total += len(adj[u] & adj[v])
    return total // 3


def report(invs: List[Invariant]) -> str:
    out = []
    for i in invs:
        mark = "skip" if i.skipped else ("PASS" if i.ok else "FAIL")
        out.append(f"  [{mark}] {i.name}: {i.detail}")
    return "\n".join(out)
