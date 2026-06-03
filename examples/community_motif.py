"""
Worked example from the paper, end to end on the simulation backend.

Goal: "In the connectome, find the densely interconnected neuron communities and
tell me which ones are richest in recurring three-neuron motifs."

Run:  python -m examples.community_motif
"""
from loom import LOOM, SimulationBackend
from loom.verify import report as inv_report

GOAL = ("In the connectome, find the densely interconnected neuron communities "
        "and tell me which ones are richest in recurring three-neuron motifs. "
        "This is interactive, keep it under 30 seconds.")


def main():
    # A planted-community synthetic graph stands in for the connectome.
    backend = SimulationBackend(n=1500, blocks=6, p_in=0.05, p_out=0.0008, locales=8, seed=11)
    loom = LOOM(backend=backend)            # offline rule-based planner by default
    result = loom.run(GOAL, budget=25)

    print("=" * 70)
    print("AGENT LOG")
    print("=" * 70)
    print(result.log())

    print("\n" + "=" * 70)
    print("FINAL GW-IR (typed, statically validated)")
    print("=" * 70)
    print(result.gwir.pretty())

    print("\n" + "=" * 70)
    print("VERIFIER INVARIANTS")
    print("=" * 70)
    print(inv_report(result.invariants))

    print("\n" + "=" * 70)
    print("INTERPRETER REPORT")
    print("=" * 70)
    print(result.report)


if __name__ == "__main__":
    main()
