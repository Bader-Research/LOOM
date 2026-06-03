"""
loom.engine
===========

The LOOM driver. ``LOOM.run(goal)`` implements the control loop of the paper
(Algorithm 1): seed subgoals, then for each open subgoal run a contract-net
round, commit the strongest bid, statically validate the GW-IR, execute ready
nodes on the backend, let the Performance Engineer reflect on telemetry and
re-run if needed, and let the Verifier check invariants (routing failures back
to the Algorithm Strategist). Finally the Interpreter renders the report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .agents import (AlgorithmStrategistAgent, DataEngineerAgent, InterpreterAgent,
                     OrchestratorAgent, PerformanceEngineerAgent, VerifierAgent)
from .backend import ArachneBackend, SimulationBackend
from .blackboard import Blackboard
from .gwir import GWIR, validate
from .llm import LLMClient, RuleBasedLLM
from .telemetry import CostModel
from .verify import Invariant


@dataclass
class LoomResult:
    goal: str
    report: str
    gwir: GWIR
    blackboard: Blackboard
    invariants: List[Invariant] = field(default_factory=list)

    def log(self) -> str:
        return "\n".join(self.blackboard.notes)


class LOOM:
    def __init__(self, backend: Optional[ArachneBackend] = None,
                 llm: Optional[LLMClient] = None,
                 cost_model: Optional[CostModel] = None):
        self.backend = backend or SimulationBackend()
        self.llm = llm or RuleBasedLLM()
        self.cost = cost_model or CostModel()
        self.orch = OrchestratorAgent(self.llm)
        self.data_engineer = DataEngineerAgent(self.backend)
        self.strategist = AlgorithmStrategistAgent(self.cost)
        self.perf = PerformanceEngineerAgent()
        self.verifier = VerifierAgent()
        self.interpreter = InterpreterAgent(self.llm)

    # ---- execution + reflection ----------------------------------------- #
    def _stabilize(self, bb: Blackboard) -> None:
        changed, guard = True, 0
        while changed and guard < 200:
            changed, guard = False, guard + 1
            for nid in bb.gwir.topo():
                node = bb.gwir.nodes[nid]
                if node.executed or node.op == "report":
                    continue
                if not all(bb.gwir.nodes[i].executed for i in node.inputs):
                    continue
                value, tel = self.data_engineer.execute(node, bb)
                bb.results[nid] = value
                node.executed = True
                node.result_key = f"n{nid}_{node.op}"
                bb.record(nid, node.op, node.params, tel, event="execute")
                changed = True
                # telemetry-grounded reflection may request a re-run
                self.perf.reflect(node, tel, bb)  # sets node.executed False if re-running

    def _verify_and_repair(self, bb: Blackboard) -> List[Invariant]:
        for _ in range(3):
            invs = self.verifier.check_results(bb)
            bb.last_invariants = invs  # type: ignore[attr-defined]
            for inv in invs:
                bb.record(-1, "verify", {}, None, event="verify")
                break
            fails = [i for i in invs if (not i.ok and not i.skipped)]
            if not fails:
                return invs
            bb.log(f"Verifier found failing invariant(s): "
                   f"{[i.name for i in fails]}; routing to AlgorithmStrategist")
            repaired = False
            for nid, node in list(bb.gwir.nodes.items()):
                if node.op == "community" and node.executed and \
                        self.strategist.alternative_for(node, bb):
                    self._stabilize(bb)
                    repaired = True
                    break
            if not repaired:
                return invs
        return self.verifier.check_results(bb)

    # ---- main loop (Algorithm 1) ---------------------------------------- #
    def run(self, goal: str, budget: int = 25) -> LoomResult:
        bb = Blackboard()
        self.orch.seed(bb, goal)
        steps = 0
        while steps < budget:
            sg = bb.open_subgoal()
            if sg is None:
                break
            steps += 1

            bids = []
            for agent in (self.data_engineer, self.strategist):
                b = agent.bid(sg, bb)
                if b:
                    bids.append(b)
            best = self.orch.commit(bids)
            if best is None:
                bb.log(f"No specialist bid for subgoal '{sg.intent}'; skipping")
                sg.closed = True
                continue

            added = best.apply(bb)
            vr = validate(bb.gwir)
            if not vr.ok:
                bb.log(f"Verifier statically REJECTED fragment for '{sg.intent}': "
                       f"{vr.message}; rolling back")
                bb.gwir.remove(added)
                sg.closed = True
                continue

            bb.log(f"Committed '{best.label}' (by {best.bidder}, conf={best.confidence:.2f}, "
                   f"est_cost={best.est_cost:.3f}s) for subgoal '{sg.intent}'")
            self._stabilize(bb)
            self._verify_and_repair(bb)
            sg.closed = True

        report = self.interpreter.render(bb, goal)
        return LoomResult(goal=goal, report=report, gwir=bb.gwir, blackboard=bb,
                          invariants=getattr(bb, "last_invariants", []))


def run(goal: str, **kwargs) -> LoomResult:
    """Convenience entry point with the simulation backend and offline planner."""
    return LOOM().run(goal, **kwargs)
