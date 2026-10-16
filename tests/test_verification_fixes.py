"""Regression tests for the verification-layer fixes found by fault injection
(see eval/README.md). Each test fails on commit 05e17d4 and passes after the fix."""
import numpy as np
import pytest

from loom import GWIR, validate
from loom import verify as V
from loom.backend import GraphData, SimulationBackend


def _plan(variant="louvain", strategy="global", community_on_lcc=True):
    g = GWIR()
    n_load = g.add("load", params={"src": "x.pq"})
    n_build = g.add("build_graph", inputs=[n_load])
    n_sym = g.add("symmetrize", inputs=[n_build])
    n_cc = g.add("largest_cc", inputs=[n_sym])
    params = {} if variant is None else {"variant": variant}
    n_com = g.add("community", inputs=[n_cc if community_on_lcc else n_sym], params=params)
    n_pbt = g.add("per_block_triangles", inputs=[n_cc, n_com], params={"strategy": strategy})
    return g, dict(load=n_load, build=n_build, sym=n_sym, cc=n_cc, com=n_com, pbt=n_pbt)


# ---- static validation --------------------------------------------------- #
def test_duplicate_input_is_a_type_error_not_a_cycle():
    g, n = _plan()
    g.node(n["pbt"]).inputs = [n["com"], n["com"]]   # partition fed to the graph port
    vr = validate(g)
    assert not vr.ok and vr.node == n["pbt"]
    assert "cycle" not in vr.message


def test_cycle_is_rejected_not_raised():
    g, n = _plan()
    g.node(n["sym"]).inputs = [n["cc"]]
    vr = validate(g)                                  # must not raise
    assert not vr.ok and "cycle" in vr.message


@pytest.mark.parametrize("variant", ["spectral", None])
def test_invalid_or_missing_variant_rejected(variant):
    g, n = _plan(variant=variant)
    vr = validate(g)
    assert not vr.ok and vr.node == n["com"]


def test_misspelled_strategy_rejected():
    g, n = _plan(strategy="within_comunity")
    vr = validate(g)
    assert not vr.ok and vr.node == n["pbt"]


def test_partition_over_different_vertex_set_rejected():
    g, n = _plan(community_on_lcc=False)              # partition of the pre-LCC graph
    vr = validate(g)
    assert not vr.ok and vr.node == n["pbt"]


# ---- invariants ----------------------------------------------------------- #
def _triangle_plus_pendant():
    src = np.array([0, 1, 0, 2, 1, 2, 0, 3])
    dst = np.array([1, 0, 2, 0, 2, 1, 3, 0])
    return GraphData(n=4, src=src, dst=dst, directed=False)


def test_simple_symmetric_catches_dropped_arc_and_duplicate():
    g = _triangle_plus_pendant()
    assert V.simple_symmetric(g).ok
    one_way = GraphData(n=4, src=g.src[1:], dst=g.dst[1:], directed=False)
    assert not V.simple_symmetric(one_way).ok
    dup = GraphData(n=4, src=np.r_[g.src, 0, 1], dst=np.r_[g.dst, 1, 0], directed=False)
    assert V.degree_sum(dup).ok                       # the identity alone cannot see this
    assert not V.simple_symmetric(dup).ok


def test_partition_covers_checks_length_against_graph():
    assert not V.partition_covers(np.zeros(3, dtype=int), 4).ok


def test_two_sided_triangle_check_catches_undercounts():
    g = _triangle_plus_pendant()
    labels = np.zeros(4, dtype=int)
    assert V.per_block_matches_independent({0: 1}, g, labels).ok
    for wrong in ({0: 0}, {}, {0: 2}):
        assert not V.per_block_matches_independent(wrong, g, labels).ok
    assert V.per_block_le_global({0: 0}, g).ok        # the one-sided bound misses it


def test_two_sided_check_agrees_with_backend_on_real_run():
    be = SimulationBackend(n=600, blocks=4, p_in=0.06, p_out=0.001, seed=3)
    e, _ = be.load_edges("x"); g, _ = be.build_graph(e); s, _ = be.symmetrize(g)
    lcc, _ = be.largest_cc(s); part, _ = be.community(lcc, "louvain")
    per = be._count_triangles(lcc, restrict=part)
    assert V.per_block_matches_independent(per, lcc, part).ok
    assert V.simple_symmetric(lcc).ok
