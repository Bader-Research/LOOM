"""
LOOM: a team-of-agents architecture for high-performance graph analytics
workflows on Arkouda and Arachne.

Reference implementation accompanying Saxena & Bader, "LOOM: A Team-of-Agents
Architecture for High-Performance Graph Analytics Workflows with Arkouda and
Arachne."

Quick start (no cluster required, runs on the simulation backend):

    from loom import LOOM
    result = LOOM().run("Find communities in the connectome and rank them by "
                        "internal triangle density")
    print(result.report)
    print(result.gwir.pretty())
"""
from .engine import LOOM, LoomResult, run
from .backend import ArachneBackend, SimulationBackend, ArkoudaArachneBackend, GraphData
from .gwir import GWIR, validate, OP_LIBRARY, Artifact, ArtifactType, ValidationResult
from .telemetry import CostModel, Telemetry, GraphStats
from .llm import LLMClient, RuleBasedLLM, AnthropicLLM
from . import verify

__all__ = [
    "LOOM", "LoomResult", "run",
    "ArachneBackend", "SimulationBackend", "ArkoudaArachneBackend", "GraphData",
    "GWIR", "validate", "OP_LIBRARY", "Artifact", "ArtifactType", "ValidationResult",
    "CostModel", "Telemetry", "GraphStats",
    "LLMClient", "RuleBasedLLM", "AnthropicLLM", "verify",
]
__version__ = "0.1.0"
