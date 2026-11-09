"""E2 (usage: python -I eval/e2_invariants.py [path/to/LOOM]): Fault injection into kernel outputs; does LOOM's Verifier detect it?
Pipeline per trial: load->build->symmetrize->largest_cc (G) -> community (P) -> per_block_triangles (B).
A fault corrupts one kernel's output; downstream kernels consume the corrupted value (as in a real run);
then the Verifier's checks are applied exactly as VerifierAgent.check_results does."""
import os, sys, json, collections, zlib
import numpy as np
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loom.backend import SimulationBackend, GraphData
from loom import verify as V
from _verifier import verifier_failures

def pipeline(seed, p_out):
    be = SimulationBackend(n=1500, p_out=p_out, seed=seed)
    e, _ = be.load_edges("x"); g, _ = be.build_graph(e); s, _ = be.symmetrize(g)
    G, _ = be.largest_cc(s)
    P, _ = be.community(G, "louvain")
    return be, s, G, P

def mkgraph(n, src, dst):
    return GraphData(n=n, src=src, dst=dst, directed=False)

def verifier(G, P, B):
    return verifier_failures(G, P, B)

def undirected_pairs(G):
    m = G.src < G.dst
    return G.src[m], G.dst[m]

def from_pairs(n, u, v):
    return mkgraph(n, np.concatenate([u, v]), np.concatenate([v, u]))

FAULTS = {}
def fault(name, stage):
    def deco(f): FAULTS[name] = (stage, f); return f
    return deco

# ---- graph-kernel faults (symmetrize / largest_cc output) ----
@fault("G1 drop 1% of edges (both arcs)", "graph")
def _(G, rng, **k):
    u, v = undirected_pairs(G); keep = rng.random(len(u)) > 0.01
    return from_pairs(G.n, u[keep], v[keep])
@fault("G2 drop 1% of arcs (one direction)", "graph")
def _(G, rng, **k):
    keep = rng.random(len(G.src)) > 0.01
    return mkgraph(G.n, G.src[keep], G.dst[keep])
@fault("G3 add 1% spurious edges (both arcs)", "graph")
def _(G, rng, **k):
    u, v = undirected_pairs(G); a = max(1, len(u) // 100)
    x = rng.integers(0, G.n, a); y = rng.integers(0, G.n, a); ok = x != y
    return from_pairs(G.n, np.concatenate([u, x[ok]]), np.concatenate([v, y[ok]]))
@fault("G4 duplicate 1% of edges", "graph")
def _(G, rng, **k):
    u, v = undirected_pairs(G); d = rng.random(len(u)) < 0.01
    return from_pairs(G.n, np.concatenate([u, u[d]]), np.concatenate([v, v[d]]))
# ---- partition-kernel faults (community output) ----
@fault("P1 1% of vertices unassigned (-1)", "part")
def _(P, rng, **k):
    P = P.copy(); P[rng.random(len(P)) < 0.01] = -1; return P
@fault("P2 10% of vertices moved to random blocks", "part")
def _(P, rng, **k):
    P = P.copy(); m = rng.random(len(P)) < 0.10; P[m] = rng.choice(np.unique(P), m.sum()); return P
@fault("P3 all vertices merged into one block", "part")
def _(P, rng, **k):
    return np.zeros_like(P)
# ---- triangle-kernel faults (per_block_triangles output) ----
for name, f in [("T1 overcount by 1%", lambda c: int(round(c*1.01))),
                ("T2 overcount by 10%", lambda c: int(round(c*1.10))),
                ("T3 undercount by 50%", lambda c: c//2),
                ("T4 all counts zero", lambda c: 0)]:
    FAULTS[name] = ("tri", (lambda f: (lambda B, rng, **k: {b: f(c) for b, c in B.items()}))(f))
FAULTS["T5 one block's count dropped"] = ("tri", lambda B, rng, **k: {b: c for i, (b, c) in enumerate(B.items()) if i})

REGIMES = {"paper default (p_out=0.0008)": 0.0008, "moderate (p_out=0.004)": 0.004, "weak (p_out=0.012)": 0.012}
SEEDS = range(10)
res = collections.defaultdict(lambda: collections.Counter())
ratio = collections.defaultdict(list)
for rname, p_out in REGIMES.items():
    for seed in SEEDS:
        be, pre_graph, G, P = pipeline(seed, p_out)
        B = be._count_triangles(G, restrict=P)
        ratio[rname].append(sum(B.values()) / max(1, V._independent_global_triangles(G)))
        assert verifier(G, P, B) == [], "clean run must pass"
        # partition computed on pre-LCC graph, for P4
        P_pre, _ = be.community(pre_graph, "louvain")
        for fname, (stage, f) in FAULTS.items():
            rng = np.random.default_rng(1000 * seed + zlib.crc32(fname.encode()) % 997)
            G2, P2, B2 = G, P, B
            try:
                if stage == "graph":
                    G2 = f(G, rng); P2, _ = be.community(G2, "louvain"); B2 = be._count_triangles(G2, restrict=P2)
                elif stage == "part":
                    P2 = f(P, rng, pre=P_pre); B2 = be._count_triangles(G, restrict=P2)
                else:
                    B2 = f(B, rng)
                fired = verifier(G2, P2, B2)
                res[(rname, fname)]["detected" if fired else "missed"] += 1
                for x in fired: res[(rname, fname)]["by:" + x] += 1
            except Exception as e:
                res[(rname, fname)]["crash:" + type(e).__name__] += 1
out = {f"{r} | {f}": dict(c) for (r, f), c in res.items()}
out["internal_over_global_ratio"] = {r: [float(np.mean(v)), float(np.min(v)), float(np.max(v))] for r, v in ratio.items()}
json.dump(out, open("e2_results.json", "w"), indent=1)
print("internal/global triangle ratio:", {r: round(np.mean(v), 3) for r, v in ratio.items()})
print(f"{'fault':44s}" + "".join(f"{r.split()[0]:>14s}" for r in REGIMES))
for fname in FAULTS:
    cells = []
    for r in REGIMES:
        c = res[(r, fname)]; n = sum(v for k, v in c.items() if not k.startswith("by:"))
        crash = sum(v for k, v in c.items() if k.startswith("crash"))
        cells.append(f"{c['detected']}/{n}" + (f" c{crash}" if crash else ""))
    print(f"{fname:44s}" + "".join(f"{x:>14s}" for x in cells))
