"""E1: Mutation study (usage: python -I eval/e1_gwir_mutation.py [path/to/LOOM])
 of LOOM's static GW-IR validator (Algorithm 2).
Each mutant is (a) statically validated, (b) labelled by an independent semantic
oracle written from kernel semantics, and (c) executed on the simulation backend."""
import os, sys, copy, json, collections
import numpy as np
LOOM = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), ".."); sys.path.insert(0, LOOM)
from loom import gwir as _GW
from loom.gwir import GWIR
validate = _GW.validate
from loom.backend import SimulationBackend
from loom.blackboard import Blackboard
from loom.agents import DataEngineerAgent

# ---------- base plan (paper Fig. 1 plus two extra analytics) ----------
# node spec: (op, [inputs], params)
BASE = {
 0: ("load", [], {"src": "edges.pq"}),
 1: ("build_graph", [0], {}),
 2: ("symmetrize", [1], {}),
 3: ("largest_cc", [2], {}),
 4: ("community", [3], {"variant": "louvain"}),
 5: ("per_block_triangles", [3, 4], {"strategy": "global"}),
 6: ("triangle_count", [3], {}),
 7: ("connected_components", [3], {}),
 8: ("report", [5, 6, 7], {}),
}

def build(spec):
    g = GWIR()
    ids = {}
    for k in sorted(spec):
        op, ins, p = spec[k]
        nid = g.add(op, params=p)
        ids[k] = nid
    for k in sorted(spec):
        g.nodes[ids[k]].inputs = [ids[i] for i in spec[k][1]]
    return g

# ---------- independent semantic oracle ----------
GRAPH_OPS = {"build_graph", "symmetrize", "largest_cc"}
PART_OPS = {"community", "connected_components"}
NEEDS_UNDIRECTED = {"largest_cc", "community", "connected_components", "triangle_count"}
VALID_PARAMS = {"community": {"variant": {"louvain", "label_propagation"}},
                "per_block_triangles": {"strategy": {"global", "within_community"}}}
ARITY = {"load": 0, "build_graph": 1, "symmetrize": 1, "largest_cc": 1, "community": 1,
         "per_block_triangles": 2, "triangle_count": 1, "connected_components": 1}

def oracle(spec):
    """Return (first_invalid_node or None, reason, redundant_flag). Topological order = key order
    for non-cyclic mutants; cycles flagged separately."""
    # cycle check
    order, seen = [], set()
    indeg = {k: len(v[1]) for k, v in spec.items()}
    ready = [k for k in spec if indeg[k] == 0]
    while ready:
        k = ready.pop(0); order.append(k)
        for j, v in spec.items():
            for i in v[1]:
                if i == k:
                    indeg[j] -= 1
                    if indeg[j] == 0: ready.append(j)
    if len(order) != len(spec):
        return ("cycle", "cyclic plan", False)
    kind, undirected, domain = {}, {}, {}
    redundant = False
    for k in order:
        op, ins, p = spec[k]
        if op != "report" and len(ins) != ARITY[op]:
            return (k, "arity", redundant)
        if op == "report" and not ins:
            return (k, "arity", redundant)
        for name, allowed in VALID_PARAMS.get(op, {}).items():
            if p.get(name) not in allowed:
                return (k, "invalid parameter", redundant)
        if op == "load":
            kind[k] = "edges"; domain[k] = "D0"
        elif op == "build_graph":
            if kind[ins[0]] != "edges": return (k, "type", redundant)
            kind[k] = "graph"; undirected[k] = False; domain[k] = domain[ins[0]]
        elif op == "symmetrize":
            if kind[ins[0]] != "graph": return (k, "type", redundant)
            if undirected[ins[0]]: redundant = True
            kind[k] = "graph"; undirected[k] = True; domain[k] = domain[ins[0]]
        elif op in NEEDS_UNDIRECTED:
            if kind[ins[0]] != "graph": return (k, "type", redundant)
            if not undirected[ins[0]]: return (k, "directed input to undirected kernel", redundant)
            if op == "largest_cc":
                if spec[ins[0]][0] == "largest_cc": redundant = True
                kind[k] = "graph"; undirected[k] = True; domain[k] = f"lcc({k})"
            elif op in PART_OPS:
                kind[k] = "partition"; domain[k] = domain[ins[0]]
            else:
                kind[k] = "scalar"
        elif op == "per_block_triangles":
            g, part = ins
            if kind[g] != "graph" or kind[part] != "partition": return (k, "type", redundant)
            if not undirected[g]: return (k, "directed input to undirected kernel", redundant)
            if domain[g] != domain[part]: return (k, "partition/graph vertex-domain mismatch", redundant)
            kind[k] = "field"
        elif op == "report":
            kind[k] = "summary"
    return (None, "valid", redundant)

# ---------- execution on the simulation backend ----------
def execute(spec, seed=7):
    g = build(spec)
    bb = Blackboard(); bb.gwir = g
    de = DataEngineerAgent(SimulationBackend(n=600, p_in=0.012, seed=seed))
    ran = 0
    try:
        order = g.topo()
    except ValueError:
        return ("crash", 0, "cycle")
    for nid in order:
        node = g.nodes[nid]
        if node.op == "report": continue
        try:
            val, tel = de.execute(node, bb)
        except Exception as e:
            return ("crash", ran, type(e).__name__)
        bb.results[nid] = val; node.executed = True; ran += 1
    return ("completed", ran, "")

# ---------- mutant generation ----------
def descendants(spec, k):
    out, stack = set(), [k]
    while stack:
        x = stack.pop()
        for j, v in spec.items():
            if x in v[1] and j not in out:
                out.add(j); stack.append(j)
    return out

def mutants():
    M = []
    # 1. single-input rewiring (excluding the variadic report node)
    for k, (op, ins, p) in BASE.items():
        if op == "report": continue
        for slot, cur in enumerate(ins):
            for j in BASE:
                if j in (cur, k) or BASE[j][0] == "report": continue
                s = copy.deepcopy(BASE); s[k][1][slot] = j
                M.append(("rewire", f"n{k}.in{slot}: n{cur}->n{j}", s))
    # 2. delete a unary node, bypassing its consumers to its input
    for k in (1, 2, 3):
        s = copy.deepcopy(BASE); src = s[k][1][0]; del s[k]
        for v in s.values():
            v[1][:] = [src if i == k else i for i in v[1]]
        M.append(("delete", f"delete n{k} ({BASE[k][0]})", s))
    # 3. insert a redundant operation
    for after, op in ((2, "symmetrize"), (3, "symmetrize"), (3, "largest_cc"), (2, "build_graph")):
        s = copy.deepcopy(BASE); new = max(s) + 1
        for j, v in s.items():
            if j != new: v[1][:] = [new if i == after else i for i in v[1]]
        s[new] = (op, [after], {})
        # keep keys topologically sortable by inserting in place: renumber via order
        M.append(("insert", f"insert {op} after n{after}", s))
    # 4. arity faults
    s = copy.deepcopy(BASE); s[5][1].pop(); M.append(("arity", "n5 drops partition input", s))
    s = copy.deepcopy(BASE); s[4][1].append(2); M.append(("arity", "n4 gets extra input", s))
    # 5. parameter faults
    s = copy.deepcopy(BASE); s[4][2]["variant"] = "spectral"; M.append(("param", "variant=spectral", s))
    s = copy.deepcopy(BASE); s[5][2]["strategy"] = "within_comunity"; M.append(("param", "strategy typo", s))
    s = copy.deepcopy(BASE); del s[4][2]["variant"]; M.append(("param", "variant missing", s))
    # 6. cycle
    s = copy.deepcopy(BASE); s[2][1][0] = 3; M.append(("cycle", "n2 <- n3", s))
    return M

def topo_keys(spec):
    # oracle and build() assume key order is topological; re-key mutants that aren't
    indeg = {k: len(set(v[1])) for k, v in spec.items()}
    order, ready = [], sorted(k for k in spec if indeg[k] == 0)
    while ready:
        k = ready.pop(0); order.append(k)
        for j in sorted(spec):
            if k in spec[j][1]:
                indeg[j] -= 1
                if indeg[j] == 0: ready.append(j)
    if len(order) != len(spec): return spec  # cycle; leave as is
    remap = {old: new for new, old in enumerate(order)}
    return {remap[k]: (v[0], [remap[i] for i in v[1]], v[2]) for k, v in spec.items()}

rows = []
for cls, desc, s in mutants():
    s = topo_keys(s)
    bad, reason, redundant = oracle(s)
    try:
        g = build(s); vr = validate(g)
        static = "reject" if not vr.ok else "accept"
        rej_node = vr.node
        # map validator node id back to spec key (build() assigns ids in key order)
        rej_key = sorted(s)[rej_node] if rej_node is not None else None
        msg = vr.message
    except ValueError as e:
        static, rej_key, msg = "exception", None, str(e)
    run, ran, err = execute(s)
    rows.append(dict(cls=cls, desc=desc, oracle=("invalid" if bad is not None else "valid"),
                     reason=reason, redundant=redundant, static=static,
                     localized=(bad is not None and rej_key == bad) if static == "reject" else None,
                     run=run, kernels_before_failure=ran, err=err, msg=msg))
json.dump(rows, open("e1_results.json", "w"), indent=1)

# ---------- summary ----------
inv = [r for r in rows if r["oracle"] == "invalid"]
val = [r for r in rows if r["oracle"] == "valid"]
print(f"mutants={len(rows)} invalid={len(inv)} valid={len(val)} (of which redundant={sum(r['redundant'] for r in val)})")
print("valid mutants rejected (false rejections):", sum(r["static"] != "accept" for r in val))
by = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
for r in inv:
    b = by[r["reason"]]; b[0] += 1
    b[1] += r["static"] != "accept"
    b[2] += bool(r["localized"])
    b[3] += (r["static"] == "accept" and r["run"] == "completed")
    b[4] += (r["static"] == "accept" and r["run"] == "crash")
print(f"{'reason':42s} n  caught localized  missed->silent  missed->crash")
for k, b in sorted(by.items(), key=lambda x: -x[1][0]):
    print(f"{k:42s} {b[0]:2d} {b[1]:4d} {b[2]:6d} {b[3]:10d} {b[4]:10d}")
tot = [sum(b[i] for b in by.values()) for i in range(5)]
print("TOTAL", tot)
# what would rejected mutants have done at run time?
rej = [r for r in inv if r["static"] != "accept"]
print("rejected-invalid at runtime: crash", sum(r["run"]=="crash" for r in rej),
      "completed(silent)", sum(r["run"]=="completed" for r in rej))
print("redundant-but-valid mutants accepted:", [r["desc"] for r in val if r["redundant"]])
for r in rows:
    if r["oracle"]=="invalid" and r["static"]=="accept":
        print("  MISSED:", r["desc"], "|", r["reason"], "|", r["run"], r["err"])
for r in rows:
    if r["static"]=="reject" and not r["localized"]:
        print("  MISLOCALIZED:", r["desc"], "| oracle:", r["reason"], "| validator:", r["msg"])
