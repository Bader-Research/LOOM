"""E3 (usage: python -I eval/e3_community_validity.py [path/to/LOOM]): Does a passing Verifier imply a meaningful community result? Compare LOOM's community
output to the planted blocks the simulation backend generates (ground truth hidden from LOOM)."""
import os, sys, json
import numpy as np
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _verifier import verifier_failures
from loom.backend import SimulationBackend
from loom import verify as V

def nmi(a, b):
    a = np.unique(a, return_inverse=True)[1]; b = np.unique(b, return_inverse=True)[1]
    n = len(a); C = np.zeros((a.max()+1, b.max()+1)); np.add.at(C, (a, b), 1); P = C / n
    pa, pb = P.sum(1), P.sum(0); nz = P > 0
    I = (P[nz] * np.log(P[nz] / np.outer(pa, pb)[nz])).sum()
    H = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum()
    d = (H(pa) + H(pb)) / 2
    return float(I / d) if d > 0 else (1.0 if H(pa) == H(pb) == 0 else 0.0)

rows = []
for p_out in [0.0008, 0.002, 0.004, 0.008, 0.012]:
    for variant in ["louvain", "label_propagation"]:
        for seed in range(10):
            be = SimulationBackend(n=1500, p_out=p_out, seed=seed)
            e, _ = be.load_edges("x"); g, _ = be.build_graph(e); s, _ = be.symmetrize(g)
            G, _ = be.largest_cc(s); P, _ = be.community(G, variant)
            B = be._count_triangles(G, restrict=P)
            fails = verifier_failures(G, P, B)
            truth = be._planted_sub
            # mean internal vs external degree of the planted structure
            same = truth[G.src] == truth[G.dst]
            rows.append(dict(p_out=p_out, variant=variant, seed=seed, found=int(len(np.unique(P))),
                             nmi=nmi(P, truth), verifier_pass=not fails,
                             frac_internal_edges=float(same.mean())))
json.dump(rows, open("e3_results.json", "w"), indent=1)
print(f"{'p_out':>7} {'variant':>18} {'int.edges':>9} {'found(med)':>10} {'single-block':>12} {'NMI(mean)':>9} {'verifier pass':>13}")
for p_out in sorted({r['p_out'] for r in rows}):
    for v in ["louvain", "label_propagation"]:
        R = [r for r in rows if r['p_out'] == p_out and r['variant'] == v]
        print(f"{p_out:7.4f} {v:>18} {np.mean([r['frac_internal_edges'] for r in R]):9.2f} "
              f"{np.median([r['found'] for r in R]):10.0f} {sum(r['found']==1 for r in R):9d}/10 "
              f"{np.mean([r['nmi'] for r in R]):9.2f} {sum(r['verifier_pass'] for r in R):10d}/10")
