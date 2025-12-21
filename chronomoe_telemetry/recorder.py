"""
Recorder: Emit JSON, resume hooks.

Main orchestrator for telemetry collection.
Ties together observer, metrics, stillness, pause, and schema.
"""

from typing import Optional, List, Tuple, Union
import numpy as np

from .observer import ChronoMoEObserver, ProxyObserver
from .metrics import (
    compute_pc_projection,
    compute_routing_entropy,
    compute_expert_switch_rate,
    compute_pc_velocity,
)
from .stillness import StillnessThresholds, StillnessFlags, StillnessDetector
from .pause import PauseDetector
from .schema import TelemetrySnapshot, create_gloss_reference


class TelemetryRecorder:
    """
    Main telemetry recorder.

    Orchestrates observation, measurement, stillness detection, and pause proposals.
    Can work with real ChronoMoE or proxy data.

    Usage:
        recorder = TelemetryRecorder(session_id="exp_001")

        # During inference loop:
        snapshot = recorder.capture_snapshot(chrono_state)
        should_pause, reason = recorder.should_pause()

        if should_pause:
            print(snapshot.to_json())
            # Pause for human review
    """

    def __init__(
        self,
        session_id: str,
        observer: Optional[Union[ChronoMoEObserver, ProxyObserver]] = None,
        stillness_thresholds: Optional[StillnessThresholds] = None,
        pause_detector: Optional[PauseDetector] = None,
        window_size: int = 10,
    ):
        """
        Args:
            session_id: Unique identifier for this session
            observer: Observer instance (ChronoMoEObserver or ProxyObserver)
            stillness_thresholds: Thresholds for stillness detection
            pause_detector: Pause detection logic
            window_size: History window for computing dynamics
        """
        self.session_id = session_id
        self.observer = observer or ChronoMoEObserver()
        self.stillness_detector = StillnessDetector(
            thresholds=stillness_thresholds or StillnessThresholds()
        )
        self.pause_detector = pause_detector or PauseDetector(
            require_fast=True,
            require_medium=True,
            require_slow=False,
        )

        self.window_size = window_size

        # History buffers (session-local)
        self.pc_history: List[Tuple[float, float]] = []
        self.routing_history: List[np.ndarray] = []

        # Pause counter
        self.pause_index = 0

        # Most recent snapshot (for resume tests)
        self.last_snapshot: Optional[TelemetrySnapshot] = None

    def capture_snapshot(
        self,
        chrono_state=None,
        pause_reason: str = "unknown",
        layer_idx: int = 0,
    ) -> TelemetrySnapshot:
        """
        Capture a telemetry snapshot at a pause point.

        Args:
            chrono_state: ChronovisorState from model (or None for proxy)
            pause_reason: Why this pause occurred
            layer_idx: Which layer to observe (default: 0)

        Returns:
            TelemetrySnapshot with all measurements
        """
        # Get routing weights from observer
        if isinstance(self.observer, ProxyObserver):
            routing_weights = self.observer.get_routing_weights(layer_idx)
        else:
            routing_weights = self.observer.get_routing_weights(chrono_state, layer_idx)

        if routing_weights is None:
            # Fallback to zeros if no data available
            routing_weights = np.zeros(64)

        # Compute current position in PC space
        pc1, pc2 = compute_pc_projection(routing_weights)
        self.pc_history.append((pc1, pc2))

        # Compute dynamics (velocity)
        if len(self.pc_history) >= 2:
            prev_pc1, prev_pc2 = self.pc_history[-2]
            d_pc1, d_pc2 = compute_pc_velocity(
                (pc1, pc2),
                (prev_pc1, prev_pc2)
            )
        else:
            d_pc1 = 0.0
            d_pc2 = 0.0

        # Compute routing stats
        routing_entropy = compute_routing_entropy(routing_weights)

        if len(self.routing_history) > 0:
            expert_switch_rate = compute_expert_switch_rate(
                routing_weights,
                self.routing_history[-1]
            )
        else:
            expert_switch_rate = 0.0

        self.routing_history.append(routing_weights.copy())

        # Update stillness detection
        stillness_flags = self.stillness_detector.update(d_pc1, d_pc2)

        # Create snapshot
        snapshot = TelemetrySnapshot(
            session_id=self.session_id,
            pause_index=self.pause_index,
            pause_reason=pause_reason,
            pc1=pc1,
            pc2=pc2,
            d_pc1=d_pc1,
            d_pc2=d_pc2,
            routing_entropy=routing_entropy,
            expert_switch_rate=expert_switch_rate,
            stillness_flags={
                'fast': stillness_flags.fast,
                'medium': stillness_flags.medium,
                'slow': stillness_flags.slow,
            },
            gloss=create_gloss_reference()
        )

        self.pause_index += 1
        self.last_snapshot = snapshot

        # Trim history
        if len(self.pc_history) > self.window_size:
            self.pc_history.pop(0)
        if len(self.routing_history) > self.window_size:
            self.routing_history.pop(0)

        return snapshot

    def should_pause(self) -> Tuple[bool, str]:
        """
        Check if system should pause based on current stillness state.

        Returns:
            (should_pause, reason)
        """
        if self.last_snapshot is None:
            return (False, "no_snapshot_yet")

        stillness_flags = StillnessFlags(
            fast=self.last_snapshot.stillness_flags['fast'],
            medium=self.last_snapshot.stillness_flags['medium'],
            slow=self.last_snapshot.stillness_flags['slow'],
        )

        return self.pause_detector.should_pause(stillness_flags)

    def reset(self):
        """Reset all history (between sessions)."""
        self.pc_history.clear()
        self.routing_history.clear()
        self.pause_index = 0
        self.last_snapshot = None
        self.stillness_detector.reset()

    def get_history_summary(self) -> dict:
        """Get summary of current history (for debugging)."""
        return {
            'pc_history_length': len(self.pc_history),
            'routing_history_length': len(self.routing_history),
            'pause_count': self.pause_index,
            'stillness_counters': self.stillness_detector.get_counters(),
        }
