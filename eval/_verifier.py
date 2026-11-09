"""Run LOOM's own VerifierAgent on a given graph / partition / per-block result,
so the evaluation measures exactly the checks LOOM applies in its control loop."""
from loom.agents import VerifierAgent
from loom.blackboard import Blackboard


def verifier_failures(G, P, B):
    bb = Blackboard(); g = bb.gwir
    n_load = g.add("load", params={"src": "x"})
    n_build = g.add("build_graph", inputs=[n_load])
    n_sym = g.add("symmetrize", inputs=[n_build])
    n_cc = g.add("largest_cc", inputs=[n_sym])
    n_com = g.add("community", inputs=[n_cc], params={"variant": "louvain"})
    n_pbt = g.add("per_block_triangles", inputs=[n_cc, n_com], params={"strategy": "global"})
    for nid in g.nodes:
        g.nodes[nid].executed = True
    bb.results.update({n_cc: G, n_com: P, n_pbt: B})
    bb.last_graph_node = n_cc
    invs = VerifierAgent().check_results(bb)
    return [i.name for i in invs if not i.ok and not i.skipped]
