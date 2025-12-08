"""
ChronoMoE: Pressure-Driven MoE Routing Framework

Wires Chronovisor-style pressure into MoE routing to discover how
slow global pressures reshape decision trees over time.

Core philosophy:
    MoE router makes a local decision;
    Chronovisor adds a slow, global "pressure" term that bends those decisions over time.

Modules:
    moe: Simulated Mixture-of-Experts layer
    router: Router with pressure injection hook
    bridge: Translation layer between MoE stats and Chronovisor signals
    experiment: Driver for baseline vs pressure experiments
"""

from chronomoe.moe import Expert, MoE
from chronomoe.router import Router, RoutingLog
from chronomoe.bridge import ChronoMoEBridge
from chronomoe.experiment import SyntheticTask, Experiment, ExperimentMetrics

__all__ = [
    "Expert",
    "MoE",
    "Router",
    "RoutingLog",
    "ChronoMoEBridge",
    "SyntheticTask",
    "Experiment",
    "ExperimentMetrics",
]
