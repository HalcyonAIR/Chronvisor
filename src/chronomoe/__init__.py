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
    alignment: V7 structural alignment between Chronovisor and MoE experts
    experiment: V1 driver for baseline vs pressure experiments
    experiment_v7: V7 driver with structural alignment tracking
"""

from chronomoe.moe import Expert, MoE
from chronomoe.router import Router, RoutingLog
from chronomoe.bridge import ChronoMoEBridge
from chronomoe.alignment import AlignmentMatrix, StructuralAligner, AlignmentEvent
from chronomoe.knob import MetaKnob, KnobFactors, KnobState, KnobDecision, RuleBasedKnobController
from chronomoe.experiment import SyntheticTask, Experiment, ExperimentMetrics
from chronomoe.experiment_v7 import V7Experiment, V7Metrics, run_v7_experiment

__all__ = [
    # V1: Core MoE
    "Expert",
    "MoE",
    "Router",
    "RoutingLog",
    "ChronoMoEBridge",
    "SyntheticTask",
    "Experiment",
    "ExperimentMetrics",
    # V7: Structural Alignment
    "AlignmentMatrix",
    "StructuralAligner",
    "AlignmentEvent",
    "V7Experiment",
    "V7Metrics",
    "run_v7_experiment",
    # Meta-Knob
    "MetaKnob",
    "KnobFactors",
    "KnobState",
    "KnobDecision",
    "RuleBasedKnobController",
]
