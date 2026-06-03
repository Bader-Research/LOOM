# LOOM

Reference implementation of **LOOM**, a team-of-agents architecture for
high-performance graph analytics workflows on **Arkouda** and **Arachne**.

Accompanies: A. Saxena and D. A. Bader, *LOOM: A Team-of-Agents Architecture for
High-Performance Graph Analytics Workflows with Arkouda and Arachne.*

LOOM organizes specialized LLM-capable agents around a shared, typed workflow
plan. The architecture's correctness- and performance-critical machinery is
real deterministic code (typed validation, cost model, telemetry reflection,
invariant checking); the language model only handles natural-language
decomposition and reporting and is optional. The system therefore runs fully
offline with no API key and no cluster.

## Install and run

```bash
pip install -r requirements.txt          # just numpy
python -m examples.community_motif        # the paper's worked example, end to end
python -m pytest -q                       # 8 tests (needs pytest)
```

Minimal use:

```python
from loom import LOOM
result = LOOM().run("Find communities in the connectome and rank them by "
                    "internal triangle density, interactive")
print(result.report)
print(result.gwir.pretty())     # the typed, validated workflow
print(result.log())             # the agent contract-net / reflection trace
```

## What maps to what in the paper

| Paper | Code |
|-------|------|
| GW-IR: typed workflow DAG, property refinements | `loom/gwir.py` (`Artifact`, `Port`, `OpSpec`, `GWIR`) |
| Algorithm 2: static GW-IR validation | `loom/gwir.py` (`validate`) |
| Algorithm 1: contract-net control loop | `loom/engine.py` (`LOOM.run`) |
| Blackboard (shared GW-IR + telemetry + provenance) | `loom/blackboard.py` |
| Six specialist agents | `loom/agents.py` |
| Cost model for kernel/variant selection | `loom/telemetry.py` (`CostModel`) |
| Telemetry-grounded performance reflection | `loom/agents.py` (`PerformanceEngineerAgent.reflect`), `Telemetry.is_inefficient` |
| Invariant-based verification without ground truth | `loom/verify.py`, `VerifierAgent` |
| Arkouda/Arachne execution | `loom/backend.py` |

## Backends

* `SimulationBackend` (default): pure NumPy. Builds a planted-community graph,
  computes real connected components / communities / triangle counts, and
  synthesizes plausible per-locale telemetry so the reflection loop is exercised.
* `ArkoudaArachneBackend`: adapter over the real stack, import-guarded. Start a
  Chapel/Arkouda server, install Arachne
  (`github.com/Bears-R-Us/arkouda-njit`), and verify the kernel entry-point
  names against your build (see comments in `backend.py`).

```python
from loom import LOOM, ArkoudaArachneBackend
loom = LOOM(backend=ArkoudaArachneBackend(server="localhost", port=5555, locales=64))
```

## Language model (optional)

```python
from loom import LOOM, AnthropicLLM        # needs: pip install anthropic; ANTHROPIC_API_KEY
loom = LOOM(llm=AnthropicLLM(model="claude-sonnet-4-6"))
```

Without it, `RuleBasedLLM` provides a deterministic keyword planner and a
pass-through report, so behavior is reproducible and testable.

## Extending

Add an Arachne kernel by registering an `OpSpec` in `loom/gwir.py` (its input
ports define the precondition, its `produce` callback the postcondition),
implementing it in a backend, and, if it returns a checkable artifact, adding an
invariant in `loom/verify.py`.

## Status

This is an architecture reference implementation with a working simulation
path. Large-scale empirical evaluation on the real Arkouda/Arachne stack is the
subject of ongoing work; the simulation telemetry is illustrative and should
not be read as measured cluster performance.
