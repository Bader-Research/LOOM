"""
loom.backend
============

Execution backends behind the GW-IR. ``ArachneBackend`` is the interface the
agents call; it mirrors the operation library in ``gwir``. Two implementations
are provided:

* ``SimulationBackend`` -- pure NumPy, no cluster required. It builds a real
  synthetic graph, computes real results (so the Verifier's invariants are
  meaningful), and synthesizes plausible runtime telemetry so the Performance
  Engineer's reflection loop has something to react to.

* ``ArkoudaArachneBackend`` -- a thin adapter over the real Arkouda + Arachne
  stack. It is guarded so the package imports without those libraries present.
  The exact Arachne entry points should be checked against your installed
  build (see comments); they are named to match the public Arachne API.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .telemetry import GraphStats, Telemetry


# --------------------------------------------------------------------------- #
#  In-memory graph value used by the simulation backend                       #
# --------------------------------------------------------------------------- #
@dataclass
class GraphData:
    n: int
    src: np.ndarray
    dst: np.ndarray
    directed: bool = True
    _adj: Optional[List[set]] = field(default=None, repr=False)

    @property
    def m(self) -> int:
        # number of undirected edges when symmetric, else directed edges
        return int(len(self.src) // (2 if not self.directed else 1))

    def adjacency(self) -> List[set]:
        if self._adj is None:
            adj: List[set] = [set() for _ in range(self.n)]
            for u, v in zip(self.src.tolist(), self.dst.tolist()):
                adj[u].add(v)
            self._adj = adj
        return self._adj

    def degrees(self) -> np.ndarray:
        deg = np.zeros(self.n, dtype=np.int64)
        np.add.at(deg, self.src, 1)
        return deg

    def stats(self) -> GraphStats:
        deg = self.degrees()
        return GraphStats(n=self.n, m=self.m, max_degree=int(deg.max(initial=0)),
                          avg_degree=float(deg.mean() if self.n else 0.0))


class ArachneBackend(ABC):
    """Interface mirroring the GW-IR operation library."""

    @abstractmethod
    def load_edges(self, src: str): ...
    @abstractmethod
    def build_graph(self, edges): ...
    @abstractmethod
    def symmetrize(self, g): ...
    @abstractmethod
    def largest_cc(self, g): ...
    @abstractmethod
    def connected_components(self, g): ...
    @abstractmethod
    def community(self, g, variant: str): ...
    @abstractmethod
    def triangle_count(self, g): ...
    @abstractmethod
    def per_block_triangles(self, g, partition, strategy: str): ...

    # optional kernels with default "not implemented" telemetry
    def k_truss(self, g, k: int):
        raise NotImplementedError
    def bfs(self, g, source: int):
        raise NotImplementedError
    def betweenness(self, g):
        raise NotImplementedError


# --------------------------------------------------------------------------- #
#  Simulation backend                                                         #
# --------------------------------------------------------------------------- #
class SimulationBackend(ArachneBackend):
    """Runs the whole workflow locally with real computation + synthetic telemetry."""

    def __init__(self, n: int = 1500, blocks: int = 6, p_in: float = 0.04,
                 p_out: float = 0.0008, locales: int = 8, seed: int = 7):
        self.params = dict(n=n, blocks=blocks, p_in=p_in, p_out=p_out)
        self.locales = locales
        self.rng = np.random.default_rng(seed)
        self._planted: Optional[np.ndarray] = None  # ground-truth blocks (for reference only)

    # ---- ingest / construction ------------------------------------------- #
    def load_edges(self, src: str) -> Tuple[Dict, Telemetry]:
        n = self.params["n"]; b = self.params["blocks"]
        p_in = self.params["p_in"]; p_out = self.params["p_out"]
        block = self.rng.integers(0, b, size=n)
        self._planted = block
        # planted partition (stochastic block model), generated as a directed edge list
        srcs, dsts = [], []
        for u in range(n):
            # sample candidate partners cheaply: a Poisson-ish number of out-edges
            same = np.where(block == block[u])[0]
            diff = np.where(block != block[u])[0]
            k_in = self.rng.binomial(len(same), p_in)
            k_out = self.rng.binomial(len(diff), p_out)
            if k_in:
                srcs.extend([u] * k_in); dsts.extend(self.rng.choice(same, k_in))
            if k_out:
                srcs.extend([u] * k_out); dsts.extend(self.rng.choice(diff, k_out))
        src = np.asarray(srcs, dtype=np.int64)
        dst = np.asarray(dsts, dtype=np.int64)
        mask = src != dst
        edges = {"n": n, "src": src[mask], "dst": dst[mask]}
        tel = self._tel("load", n=n, m=len(edges["src"]), base=2.0e-7 * len(edges["src"]))
        return edges, tel

    def build_graph(self, edges: Dict) -> Tuple[GraphData, Telemetry]:
        g = GraphData(n=edges["n"], src=edges["src"], dst=edges["dst"], directed=True)
        return g, self._tel("build_graph", n=g.n, m=len(g.src), base=1.0e-7 * len(g.src))

    def symmetrize(self, g: GraphData) -> Tuple[GraphData, Telemetry]:
        src = np.concatenate([g.src, g.dst]); dst = np.concatenate([g.dst, g.src])
        # deduplicate undirected edges
        key = np.minimum(src, dst).astype(np.int64) * g.n + np.maximum(src, dst)
        _, idx = np.unique(key, return_index=True)
        s2, d2 = src[idx], dst[idx]
        # store both directions for adjacency
        sym = GraphData(n=g.n, src=np.concatenate([s2, d2]),
                        dst=np.concatenate([d2, s2]), directed=False)
        return sym, self._tel("symmetrize", n=sym.n, m=sym.m, base=1.5e-7 * len(src))

    def largest_cc(self, g: GraphData) -> Tuple[GraphData, Telemetry]:
        labels = self._union_find(g)
        vals, counts = np.unique(labels, return_counts=True)
        keep = vals[int(np.argmax(counts))]
        keep_mask = labels == keep
        remap = -np.ones(g.n, dtype=np.int64)
        kept = np.where(keep_mask)[0]
        remap[kept] = np.arange(len(kept))
        emask = keep_mask[g.src] & keep_mask[g.dst]
        sub = GraphData(n=len(kept), src=remap[g.src[emask]], dst=remap[g.dst[emask]],
                        directed=False)
        if self._planted is not None:
            self._planted_sub = self._planted[kept]
        return sub, self._tel("largest_cc", n=sub.n, m=sub.m, base=5.0e-8 * (g.n + g.m))

    # ---- analytics -------------------------------------------------------- #
    def connected_components(self, g: GraphData) -> Tuple[np.ndarray, Telemetry]:
        labels = self._union_find(g)
        return labels, self._tel("connected_components", n=g.n, m=g.m, base=5.0e-8 * (g.n + g.m))

    def community(self, g: GraphData, variant: str) -> Tuple[np.ndarray, Telemetry]:
        if variant == "louvain":
            labels = self._label_prop(g, rounds=8)   # stand-in: more passes -> better quality
            cost = 2.0e-7 * g.m
        else:
            labels = self._label_prop(g, rounds=3)
            cost = 6.0e-8 * g.m
        return labels, self._tel("community", n=g.n, m=g.m, base=cost, variant=variant)

    def triangle_count(self, g: GraphData) -> Tuple[int, Telemetry]:
        total = self._count_triangles(g, restrict=None)
        return total, self._tel("triangle_count", n=g.n, m=g.m, base=3.0e-7 * g.m)

    def per_block_triangles(self, g: GraphData, partition: np.ndarray,
                            strategy: str = "global") -> Tuple[Dict[int, int], Telemetry]:
        per_block = self._count_triangles(g, restrict=partition)
        # synthesize telemetry that differs by execution strategy:
        if strategy == "within_community":
            # touch only intra-community edges, batched -> low comm, balanced
            internal = self._internal_edge_count(g, partition)
            tel = self._tel("per_block_triangles", n=g.n, m=g.m,
                            base=2.0e-7 * internal, imbalance=0.18,
                            comm_mb=0.02 * internal / 1e3, variant=strategy)
        else:  # "global": scan all edges across locales -> heavy comm, imbalanced
            tel = self._tel("per_block_triangles", n=g.n, m=g.m,
                            base=4.0e-7 * g.m, imbalance=0.82,
                            comm_mb=0.9 * g.m / 1e3, variant=strategy)
        return per_block, tel

    # ---- helpers ---------------------------------------------------------- #
    def _union_find(self, g: GraphData) -> np.ndarray:
        parent = np.arange(g.n)

        def find(x):
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        for u, v in zip(g.src.tolist(), g.dst.tolist()):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
        return np.array([find(i) for i in range(g.n)], dtype=np.int64)

    def _label_prop(self, g: GraphData, rounds: int) -> np.ndarray:
        adj = g.adjacency()
        labels = np.arange(g.n, dtype=np.int64)
        order = np.arange(g.n)
        for _ in range(rounds):
            self.rng.shuffle(order)
            changed = 0
            for u in order:
                if not adj[u]:
                    continue
                nbr = list(adj[u])
                vals, counts = np.unique(labels[nbr], return_counts=True)
                best = vals[int(np.argmax(counts))]
                if best != labels[u]:
                    labels[u] = best
                    changed += 1
            if changed == 0:
                break
        # compact labels to a dense range
        _, inv = np.unique(labels, return_inverse=True)
        return inv.astype(np.int64)

    def _count_triangles(self, g: GraphData, restrict: Optional[np.ndarray]) -> "Dict[int,int] | int":
        adj = [frozenset(s) for s in g.adjacency()]
        if restrict is None:
            total = 0
            for u in range(g.n):
                for v in adj[u]:
                    if v > u:
                        total += len(adj[u] & adj[v])
            return total // 3
        # per block: count triangles whose three vertices share a block
        per: Dict[int, int] = {}
        for u in range(g.n):
            bu = int(restrict[u])
            for v in adj[u]:
                if v > u and restrict[v] == bu:
                    common = adj[u] & adj[v]
                    c = sum(1 for w in common if restrict[w] == bu)
                    if c:
                        per[bu] = per.get(bu, 0) + c
        return {b: c // 3 for b, c in per.items()}

    def _internal_edge_count(self, g: GraphData, partition: np.ndarray) -> int:
        return int(np.sum(partition[g.src] == partition[g.dst]) // 2)

    def _tel(self, kernel: str, n: int, m: int, base: float,
             imbalance: float = 0.22, comm_mb: Optional[float] = None,
             variant: Optional[str] = None) -> Telemetry:
        if comm_mb is None:
            comm_mb = 0.05 * m / 1e3
        return Telemetry(kernel=kernel, variant=variant,
                         time_s=max(base, 1e-4), n=n, m=m,
                         load_imbalance=imbalance, comm_volume_mb=comm_mb,
                         peak_mem_gb=max(1.0, (n + m) * 16 / 1e9), locales=self.locales)


# --------------------------------------------------------------------------- #
#  Real Arkouda + Arachne backend (guarded stub)                              #
# --------------------------------------------------------------------------- #
class ArkoudaArachneBackend(ArachneBackend):
    """Adapter over the real stack. Import-guarded so the package loads without
    Arkouda/Arachne installed. Entry-point names follow the public Arachne API;
    verify them against your installed build (github.com/Bears-R-Us/arkouda-njit)."""

    def __init__(self, server: str = "localhost", port: int = 5555, locales: int = 1):
        try:
            import arkouda as ak           # noqa: F401
            import arachne as ar           # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on cluster env
            raise RuntimeError(
                "Arkouda/Arachne not available. Install them and start a Chapel "
                "server, or use SimulationBackend.") from exc
        self.ak = ak
        self.ar = ar
        self.locales = locales
        ak.connect(server, port)

    def _t(self, kernel, n=0, m=0, t=0.0, variant=None):
        # Real Arkouda exposes server-side timing/comm counters; wire them here.
        return Telemetry(kernel=kernel, variant=variant, time_s=t, n=n, m=m,
                         locales=self.locales)

    def load_edges(self, src):  # pragma: no cover - requires cluster
        t0 = time.time()
        df = self.ak.read(src) if "*" in src or src.endswith(".pq") else self.ak.read_csv(src)
        edges = {"src": df["src"], "dst": df["dst"]}
        return edges, self._t("load", t=time.time() - t0)

    def build_graph(self, edges):  # pragma: no cover
        t0 = time.time()
        g = self.ar.DiGraph()
        g.add_edges_from(edges["src"], edges["dst"])
        return g, self._t("build_graph", n=g.size(), t=time.time() - t0)

    def symmetrize(self, g):  # pragma: no cover
        t0 = time.time()
        gu = self.ar.Graph()
        gu.add_edges_from(g.edges()[0], g.edges()[1])  # undirected ingest
        return gu, self._t("symmetrize", t=time.time() - t0)

    def largest_cc(self, g):  # pragma: no cover
        t0 = time.time()
        comp = self.ar.connected_components(g)
        # subgraph induced by the largest label is built client-side via Arkouda ops
        return self.ar.largest_component_subgraph(g, comp), self._t("largest_cc", t=time.time() - t0)

    def connected_components(self, g):  # pragma: no cover
        t0 = time.time()
        comp = self.ar.connected_components(g)
        return comp, self._t("connected_components", t=time.time() - t0)

    def community(self, g, variant):  # pragma: no cover
        t0 = time.time()
        if variant == "louvain":
            part = self.ar.community_detection(g, method="louvain")
        else:
            part = self.ar.community_detection(g, method="label_propagation")
        return part, self._t("community", t=time.time() - t0, variant=variant)

    def triangle_count(self, g):  # pragma: no cover
        t0 = time.time()
        tri = self.ar.triangles(g)
        return int(tri.sum()) // 3, self._t("triangle_count", t=time.time() - t0)

    def per_block_triangles(self, g, partition, strategy="global"):  # pragma: no cover
        t0 = time.time()
        result = self.ar.per_community_triangles(g, partition, strategy=strategy)
        return result, self._t("per_block_triangles", t=time.time() - t0, variant=strategy)

    def k_truss(self, g, k):  # pragma: no cover
        t0 = time.time()
        return self.ar.k_truss(g, kTrussValue=k), self._t("k_truss", t=time.time() - t0)
