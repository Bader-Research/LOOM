# Evaluation: what LOOM's self-verification catches

Fault-injection evaluation of LOOM's static GW-IR validator and invariant
Verifier, reported in A. Saxena and D. A. Bader, *What a Passing Check Certifies
in LOOM: A Fault-Injection Study of Agent Self-Verification*, Agent Evaluation
Science Symposium, 2026. All experiments use the simulation backend and fixed
seeds, and call LOOM's own `validate()` and `VerifierAgent`.

    python -I eval/e1_gwir_mutation.py        # E1: 61 single-fault plan mutants (~5 s)
    python -I eval/e2_invariants.py           # E2: 12 output-fault classes x 3 densities x 10 seeds (~1.5 min)
    python -I eval/e3_community_validity.py   # E3: verifier verdict vs. planted communities, 100 runs (~35 s)

Each script takes an optional path to a LOOM checkout (default: this repo), so
the same harness can be run against the original release for comparison:

    git worktree add /tmp/loom-05e17d4 05e17d4
    python -I eval/e1_gwir_mutation.py /tmp/loom-05e17d4

| | 05e17d4 (original) | this version |
|---|---|---|
| E1 invalid plans rejected | 49 / 54 | 54 / 54 |
| E1 valid plans falsely rejected | 0 / 7 | 0 / 7 |
| E2 injected output faults detected | 103 / 360 | 270 / 360 |
| E3 runs passing verification | 100 / 100 | 100 / 100 |

The E2 and E3 residue is expected. A consistently dropped edge or a coherent
but wrong partition (including collapse to one community) is internally
consistent, so no consistency invariant can detect it; it needs a
reference-based check such as a planted-structure probe. The E3 community
kernel is the simulation backend's label-propagation stand-in, not Arachne's
community detection.

## What changed in the verification layer

Validator (`loom/gwir.py`):
- `GWIR.topo()` counted duplicate inputs on the way in but not on the way out,
  so a node feeding two ports of one operation was reported as a cycle.
- `validate()` raised on a cycle instead of returning a rejection; it now
  returns `ValidationResult(ok=False, ...)` as Algorithm 2 specifies.
- Enumerated parameters (`community.variant`, `per_block_triangles.strategy`)
  are checked against `OpSpec.param_domains`; previously unknown values were
  accepted and the backend silently fell back to a default.
- Multi-input operations must combine artifacts over the same vertex set
  (`VERTEX_SET_CREATORS`); previously a partition of the pre-LCC graph could be
  applied to the LCC.

Verifier (`loom/verify.py`, `loom/agents.py`):
- New `simple_symmetric`: under the both-directions arc representation,
  `degree_sum` holds for any even-length arc list, so it cannot detect most
  edge corruption. `degree_sum` is kept.
- `partition_covers` is now called with the graph's vertex count, so labelings
  of the wrong length are caught.
- New `per_block_matches_independent`: a two-sided check against an
  independent recount (trace(A_b^3)/6 per block), matching the paper's
  "agree within tolerance" cross-check. `per_block_le_global` is kept but is
  one-sided and cannot detect undercounts.

Regression tests: `tests/test_verification_fixes.py`.
