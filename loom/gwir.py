"""
loom.gwir
=========

The Graph Workflow Intermediate Representation (GW-IR).

This module is the heart of LOOM. A GW-IR is a directed acyclic graph whose
nodes are *typed* Arachne operations and whose edges carry *typed* graph
artifacts. Every operation declares which input types and refinements it
requires (its precondition) and what it produces (its postcondition). A cheap
client-side static validator (``validate``) propagates types and refinements in
topological order and rejects incoherent plans before any cluster cycles are
spent. This realizes Algorithm 2 of the paper.

Nothing here touches a cluster: the GW-IR is pure metadata. Execution lives in
``loom.backend``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple


class ArtifactType(Enum):
    """The (small) type lattice over graph-analytics artifacts."""
    EDGE_SET = "EdgeSet"
    GRAPH = "Graph"
    DIGRAPH = "DiGraph"
    PROPERTY_GRAPH = "PropertyGraph"
    VERTEX_SET = "VertexSet"
    PARTITION = "Partition"          # a labeling of vertices (communities / components)
    SCALAR_FIELD = "ScalarField"     # a per-vertex or per-edge or per-block value
    SCALAR = "Scalar"                # a single number (e.g. global triangle count)
    SUMMARY = "Summary"              # the final report artifact


# Refinement attributes are lightweight string tags carried alongside a type.
# Conventions used by the operation library below:
#   "sym"    : graph is symmetric (undirected)
#   "1cc"    : graph is a single connected component
#   "weighted"
#   "truss"  : graph is a k-truss subgraph
#   "covers" : a Partition covers the vertex set exactly once
#   "vertex" / "edge" / "block" : the domain of a ScalarField
Refinements = FrozenSet[str]


@dataclass(frozen=True)
class Artifact:
    """A typed value flowing along a GW-IR edge."""
    type: ArtifactType
    refinements: Refinements = field(default_factory=frozenset)
    var: Optional[str] = None  # optional symbolic name on the blackboard

    def __str__(self) -> str:
        tags = (",".join(sorted(self.refinements))) if self.refinements else ""
        return f"{self.type.value}" + (f"[{tags}]" if tags else "")


@dataclass(frozen=True)
class Port:
    """A typed input slot of an operation (its precondition for that input)."""
    types: FrozenSet[ArtifactType]
    required: Refinements = field(default_factory=frozenset)
    name: str = ""

    def accepts(self, art: Artifact) -> Tuple[bool, str]:
        if art.type not in self.types:
            want = "|".join(sorted(t.value for t in self.types))
            return False, f"input '{self.name}' expects {want}, got {art.type.value}"
        missing = self.required - art.refinements
        if missing:
            return False, (f"input '{self.name}' missing refinement(s) "
                           f"{{{','.join(sorted(missing))}}} (have {{{','.join(sorted(art.refinements))}}})")
        return True, ""


def port(*types: ArtifactType, required: Tuple[str, ...] = (), name: str = "") -> Port:
    return Port(frozenset(types), frozenset(required), name)


# A postcondition: given resolved input artifacts and the node's scalar params,
# compute the produced output artifact (type + refinements).
Postcondition = Callable[[List[Artifact], dict], Artifact]


@dataclass(frozen=True)
class OpSpec:
    """Signature of an Arachne operation: precondition (inputs) + postcondition (produce)."""
    name: str
    inputs: Tuple[Port, ...]
    produce: Postcondition
    params: Tuple[str, ...] = ()
    executable: bool = True
    doc: str = ""


def _build_library() -> Dict[str, OpSpec]:
    G = ArtifactType.GRAPH
    DG = ArtifactType.DIGRAPH
    ES = ArtifactType.EDGE_SET
    PART = ArtifactType.PARTITION
    SF = ArtifactType.SCALAR_FIELD
    SC = ArtifactType.SCALAR
    SUM = ArtifactType.SUMMARY

    lib: Dict[str, OpSpec] = {}

    def add(spec: OpSpec):
        lib[spec.name] = spec

    add(OpSpec(
        "load", inputs=(), params=("src",),
        produce=lambda ins, p: Artifact(ES),
        doc="Read an edge list from a data source into an EdgeSet."))

    add(OpSpec(
        "build_graph", inputs=(port(ES, name="edges"),),
        produce=lambda ins, p: Artifact(DG),
        doc="Construct a directed graph from an EdgeSet."))

    add(OpSpec(
        "symmetrize", inputs=(port(G, DG, name="g"),),
        produce=lambda ins, p: Artifact(G, ins[0].refinements | {"sym"}),
        doc="Add reverse edges and deduplicate, producing an undirected Graph."))

    add(OpSpec(
        "largest_cc", inputs=(port(G, required=("sym",), name="g"),),
        produce=lambda ins, p: Artifact(G, ins[0].refinements | {"1cc"}),
        doc="Restrict to the largest connected component."))

    add(OpSpec(
        "connected_components", inputs=(port(G, required=("sym",), name="g"),),
        produce=lambda ins, p: Artifact(PART, {"covers"}),
        doc="Label each vertex with its connected component id."))

    add(OpSpec(
        "community", inputs=(port(G, required=("sym",), name="g"),), params=("variant",),
        produce=lambda ins, p: Artifact(PART, {"covers"}),
        doc="Detect communities. Variant is filled by the Algorithm Strategist."))

    add(OpSpec(
        "triangle_count", inputs=(port(G, required=("sym",), name="g"),),
        produce=lambda ins, p: Artifact(SC),
        doc="Global triangle count."))

    add(OpSpec(
        "per_block_triangles",
        inputs=(port(G, required=("sym",), name="g"), port(PART, required=("covers",), name="part")),
        params=("strategy",),
        produce=lambda ins, p: Artifact(SF, {"block"}),
        doc="Count triangles internal to each block of a partition."))

    add(OpSpec(
        "k_truss", inputs=(port(G, required=("sym",), name="g"),), params=("k",),
        produce=lambda ins, p: Artifact(G, (ins[0].refinements & {"sym"}) | {"truss"}),
        doc="Extract the k-truss subgraph."))

    add(OpSpec(
        "bfs", inputs=(port(G, DG, name="g"),), params=("source",),
        produce=lambda ins, p: Artifact(SF, {"vertex"}),
        doc="Breadth-first distances from a source vertex."))

    add(OpSpec(
        "betweenness", inputs=(port(G, DG, name="g"),),
        produce=lambda ins, p: Artifact(SF, {"vertex"}),
        doc="Betweenness centrality per vertex."))

    add(OpSpec(
        "report", inputs=(port(*list(ArtifactType), name="any"),),
        produce=lambda ins, p: Artifact(SUM),
        doc="Render a Summary from one or more upstream artifacts."))

    return lib


# The operation library is the single source of truth for typing.
OP_LIBRARY: Dict[str, OpSpec] = _build_library()


@dataclass
class GWNode:
    nid: int
    op: str
    params: dict = field(default_factory=dict)
    inputs: List[int] = field(default_factory=list)   # predecessor node ids
    output: Optional[Artifact] = None                 # set by validate() (postcondition)
    result_key: Optional[str] = None                  # where the executed value lives
    executed: bool = False
    note: str = ""                                    # which agent introduced / edited it


class GWIR:
    """A typed workflow DAG. Nodes are operations; edges are predecessor links."""

    def __init__(self) -> None:
        self.nodes: Dict[int, GWNode] = {}
        self._next = 0

    def add(self, op: str, inputs: Optional[List[int]] = None,
            params: Optional[dict] = None, note: str = "") -> int:
        if op not in OP_LIBRARY:
            raise KeyError(f"unknown operation '{op}'")
        nid = self._next
        self._next += 1
        self.nodes[nid] = GWNode(nid=nid, op=op, params=dict(params or {}),
                                 inputs=list(inputs or []), note=note)
        return nid

    def node(self, nid: int) -> GWNode:
        return self.nodes[nid]

    def remove(self, nids: List[int]) -> None:
        """Drop nodes (used to roll back a fragment rejected by static validation)."""
        for nid in nids:
            self.nodes.pop(nid, None)

    def topo(self) -> List[int]:
        """Kahn topological order. Raises if the graph contains a cycle."""
        indeg = {nid: 0 for nid in self.nodes}
        for n in self.nodes.values():
            for _ in n.inputs:
                indeg[n.nid] += 1
        ready = [nid for nid, d in indeg.items() if d == 0]
        order: List[int] = []
        while ready:
            nid = ready.pop(0)
            order.append(nid)
            for m in self.nodes.values():
                if nid in m.inputs:
                    indeg[m.nid] -= 1
                    if indeg[m.nid] == 0:
                        ready.append(m.nid)
        if len(order) != len(self.nodes):
            raise ValueError("GW-IR contains a cycle")
        return order

    def leaves(self) -> List[int]:
        used = {i for n in self.nodes.values() for i in n.inputs}
        return [nid for nid in self.nodes if nid not in used]

    def pretty(self) -> str:
        lines = []
        for nid in self.topo():
            n = self.nodes[nid]
            args = ", ".join(f"{k}={v}" for k, v in n.params.items()) if n.params else ""
            ins = ("<- " + ", ".join(f"n{i}" for i in n.inputs)) if n.inputs else ""
            out = f": {n.output}" if n.output is not None else ""
            lines.append(f"  n{nid}: {n.op}({args}) {ins}{out}".rstrip())
        return "GW-IR\n" + "\n".join(lines)


@dataclass
class ValidationResult:
    ok: bool
    node: Optional[int] = None
    message: str = ""

    def __bool__(self) -> bool:
        return self.ok


def validate(gwir: GWIR) -> ValidationResult:
    """Static GW-IR validation (paper Algorithm 2).

    Traverse the DAG in topological order, propagate output types/refinements
    via each op's postcondition, and check every node's precondition against the
    inferred types of its inputs. Returns the first violation, or Accept.
    """
    for nid in gwir.topo():
        n = gwir.nodes[nid]
        spec = OP_LIBRARY[n.op]

        # arity (the 'report' op is variadic and accepts >= 1 input)
        if n.op == "report":
            if len(n.inputs) < 1:
                return ValidationResult(False, nid, "report requires at least one input")
            ports = [spec.inputs[0]] * len(n.inputs)
        else:
            if len(n.inputs) != len(spec.inputs):
                return ValidationResult(
                    False, nid,
                    f"{n.op} expects {len(spec.inputs)} input(s), got {len(n.inputs)}")
            ports = list(spec.inputs)

        in_arts: List[Artifact] = []
        for in_nid, p in zip(n.inputs, ports):
            src = gwir.nodes[in_nid].output
            if src is None:
                return ValidationResult(False, nid, f"input n{in_nid} has no inferred type")
            ok, why = p.accepts(src)
            if not ok:
                return ValidationResult(False, nid, f"{n.op}: {why}")
            in_arts.append(src)

        # postcondition: annotate this node's output
        n.output = spec.produce(in_arts, n.params)

    return ValidationResult(True, None, "accepted")
