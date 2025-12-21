"""
ChronoMoE Telemetry

This module does not decide when the system is done.
It only proposes pause points and exposes the evidence.

Non-invasive observability layer for ChronoMoE dynamics.
Provides pause-point detection, stillness signals, and calibration artifacts.

Design principles:
- Read-only observation (never modifies model state)
- Session-local (no persistence across runs)
- Human-auditable (every measurement has clear interpretation)
- Proxy-capable (can run without real ChronoMoE for testing)

Key rule:
No imports from telemetry into ChronoMoE.
Telemetry depends on ChronoMoE. Never the reverse.
"""

from .observer import ChronoMoEObserver, ProxyObserver
from .metrics import compute_pc_projection, compute_routing_entropy, compute_expert_switch_rate
from .stillness import StillnessThresholds, StillnessFlags, StillnessDetector
from .pause import PauseDetector
from .schema import TelemetrySnapshot, create_gloss_reference
from .recorder import TelemetryRecorder

__all__ = [
    'ChronoMoEObserver',
    'ProxyObserver',
    'compute_pc_projection',
    'compute_routing_entropy',
    'compute_expert_switch_rate',
    'StillnessThresholds',
    'StillnessFlags',
    'StillnessDetector',
    'PauseDetector',
    'TelemetrySnapshot',
    'create_gloss_reference',
    'TelemetryRecorder',
]
