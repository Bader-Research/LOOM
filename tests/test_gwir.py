"""Tests for GW-IR static validation (paper Algorithm 2)."""
from loom import GWIR, validate, ArtifactType


def test_valid_plan_accepts():
    g = GWIR()
    n_load = g.add("load", params={"src": "x.pq"})
    n_build = g.add("build_graph", inputs=[n_load])
    n_sym = g.add("symmetrize", inputs=[n_build])
    n_cc = g.add("largest_cc", inputs=[n_sym])
    n_com = g.add("community", inputs=[n_cc], params={"variant": "louvain"})
    g.add("per_block_triangles", inputs=[n_cc, n_com], params={"strategy": "global"})
    vr = validate(g)
    assert vr.ok, vr.message
    # postconditions propagated correctly
    assert g.node(n_sym).output.type == ArtifactType.GRAPH
    assert "sym" in g.node(n_sym).output.refinements
    assert "1cc" in g.node(n_cc).output.refinements


def test_digraph_into_symmetric_kernel_rejected():
    # community requires Graph[sym]; feeding a DiGraph must be rejected statically
    g = GWIR()
    n_load = g.add("load", params={"src": "x.pq"})
    n_build = g.add("build_graph", inputs=[n_load])   # DiGraph, no 'sym'
    g.add("community", inputs=[n_build], params={"variant": "louvain"})
    vr = validate(g)
    assert not vr.ok
    assert "community" in vr.message


def test_missing_largest_cc_still_typechecks_but_refinement_tracked():
    # community works on any symmetric graph; refinement set should lack '1cc'
    g = GWIR()
    n_load = g.add("load", params={"src": "x.pq"})
    n_build = g.add("build_graph", inputs=[n_load])
    n_sym = g.add("symmetrize", inputs=[n_build])
    n_com = g.add("community", inputs=[n_sym], params={"variant": "label_propagation"})
    vr = validate(g)
    assert vr.ok
    assert "1cc" not in g.node(n_sym).output.refinements
    assert g.node(n_com).output.type == ArtifactType.PARTITION


def test_wrong_arity_rejected():
    g = GWIR()
    n_load = g.add("load", params={"src": "x.pq"})
    # symmetrize takes exactly one input; give it none
    g.add("symmetrize", inputs=[])
    vr = validate(g)
    assert not vr.ok
