"""Tests for invariant checks, the telemetry reflection loop, and end-to-end run."""
import numpy as np

from loom import LOOM, SimulationBackend
from loom import verify as V
from loom.backend import GraphData


def _toy_graph():
    # triangle 0-1-2 plus a pendant 3-0, symmetric edge list
    src = np.array([0, 1, 0, 2, 1, 2, 0, 3])
    dst = np.array([1, 0, 2, 0, 2, 1, 3, 0])
    return GraphData(n=4, src=src, dst=dst, directed=False)


def test_degree_sum_invariant():
    g = _toy_graph()
    inv = V.degree_sum(g)
    assert inv.ok, inv.detail


def test_partition_covers_detects_unassigned():
    good = np.array([0, 0, 1, 1])
    assert V.partition_covers(good, 4).ok
    bad = np.array([0, -1, 1, 1])           # vertex 1 unassigned
    assert not V.partition_covers(bad, 4).ok


def test_triangle_cross_check_bound():
    g = _toy_graph()
    # one community containing the triangle -> internal triangles = 1
    part = np.array([0, 0, 0, 0])
    per = {0: 1}
    inv = V.per_block_le_global(per, g)
    assert inv.ok, inv.detail
    # a fabricated over-count must be caught (internal cannot exceed global)
    inv_bad = V.per_block_le_global({0: 5}, g)
    assert not inv_bad.ok


def test_end_to_end_runs_and_reflects():
    backend = SimulationBackend(n=600, blocks=4, p_in=0.06, p_out=0.001, seed=3)
    result = LOOM(backend=backend).run(
        "find communities and rank by triangle density, interactive", budget=25)
    # the workflow produced a valid, fully typed GW-IR
    from loom import validate
    assert validate(result.gwir).ok
    # all reported invariants passed
    assert all(i.ok for i in result.invariants)
    # the performance-engineer reflection loop fired and switched the strategy
    pbt = [n for n in result.gwir.nodes.values() if n.op == "per_block_triangles"]
    assert pbt and pbt[0].params.get("strategy") == "within_community"
    # provenance recorded the before (global) and after (within_community) runs
    events = [(e.op, e.event) for e in result.blackboard.provenance]
    assert ("per_block_triangles", "reflect") in events
    assert result.report  # interpreter produced something
