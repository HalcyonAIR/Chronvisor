"""
ChronoMoE Telemetry Module

Non-invasive observability layer for ChronoMoE dynamics.
Provides pause-point detection, stillness signals, and calibration artifacts.

Design principles:
- Read-only observation (never modifies model state)
- Session-local (no persistence across runs)
- Human-auditable (every measurement has clear interpretation)
- Minimal (only what's needed for calibration)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from sklearn.decomposition import PCA
import json


@dataclass
class StillnessThresholds:
    """Thresholds for detecting stillness at different timescales."""
    fast: float = 0.01      # Token-scale motion threshold
    medium: float = 0.02    # Arc-scale progress threshold
    slow: float = 0.005     # Long-horizon regime threshold


@dataclass
class StillnessFlags:
    """Stillness detection flags at different timescales."""
    fast: bool = False
    medium: bool = False
    slow: bool = False

    def any_triggered(self) -> bool:
        """True if any stillness flag is set."""
        return self.fast or self.medium or self.slow

    def all_triggered(self) -> bool:
        """True if all stillness flags are set."""
        return self.fast and self.medium and self.slow


@dataclass
class TelemetrySnapshot:
    """
    Single telemetry snapshot at a pause point.

    Contains all measurements needed to assess whether the system
    has reached meaningful completion vs premature collapse.
    """
    session_id: str
    pause_index: int
    pause_reason: str  # "stillness" | "max_steps" | "user_interrupt"

    # Position in strategy space
    pc1: float  # Momentum / continuation strength
    pc2: float  # Exploration margin

    # Dynamics (velocity)
    d_pc1: float  # Change in momentum
    d_pc2: float  # Change in exploration

    # Routing statistics
    routing_entropy: float  # Dispersion of expert votes
    expert_switch_rate: float  # Coalition stability

    # Stillness detection
    stillness_flags: Dict[str, bool]

    # Human-readable interpretations
    gloss: Dict[str, str]

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON with glossary included."""
        data = asdict(self)
        return json.dumps(data, indent=indent)


class ChronoMoETelemetryCollector:
    """
    Collects telemetry from ChronoMoE dynamics without modifying state.

    Usage:
        collector = ChronoMoETelemetryCollector(session_id="exp_001")

        # During inference loop:
        snapshot = collector.capture_snapshot(
            chrono_state=chrono_state,
            pause_reason="stillness"
        )

        if snapshot.stillness_flags['fast'] and snapshot.stillness_flags['medium']:
            print(snapshot.to_json())
            # Pause for human review
    """

    def __init__(
        self,
        session_id: str,
        stillness_thresholds: Optional[StillnessThresholds] = None,
        window_size: int = 10,
    ):
        """
        Args:
            session_id: Unique identifier for this session
            stillness_thresholds: Thresholds for stillness detection
            window_size: History window for computing dynamics
        """
        self.session_id = session_id
        self.thresholds = stillness_thresholds or StillnessThresholds()
        self.window_size = window_size

        # History buffers (session-local)
        self.pc_history: List[Tuple[float, float]] = []
        self.routing_entropy_history: List[float] = []
        self.expert_selection_history: List[np.ndarray] = []

        # Pause counter
        self.pause_index = 0

        # PCA for PC projection (fitted on-the-fly)
        self.pca: Optional[PCA] = None

        # Stillness counters (how long each signal has been below threshold)
        self.stillness_counters = {
            'fast': 0,
            'medium': 0,
            'slow': 0,
        }

    def _compute_pc_projection(self, chrono_state) -> Tuple[float, float]:
        """
        Compute PC1/PC2 projection from routing state.

        For now, use a simple proxy:
        - PC1 ≈ momentum (EMA of routing consistency)
        - PC2 ≈ exploration (routing entropy)

        In production, this would use actual PCA on routing vectors.
        """
        if chrono_state is None or not chrono_state.expert_usage:
            return (0.0, 0.0)

        # Get routing distribution from first layer (as proxy)
        layer_0_usage = chrono_state.expert_usage.get(0, None)
        if layer_0_usage is None:
            return (0.0, 0.0)

        # PC1 proxy: L2 norm of routing weights (momentum/commitment)
        pc1 = np.linalg.norm(layer_0_usage)

        # PC2 proxy: entropy of routing distribution (exploration)
        normalized = layer_0_usage / (layer_0_usage.sum() + 1e-10)
        pc2 = -np.sum(normalized * np.log(normalized + 1e-10))

        return (float(pc1), float(pc2))

    def _compute_routing_entropy(self, chrono_state) -> float:
        """Compute entropy of routing distribution."""
        if chrono_state is None or not chrono_state.expert_usage:
            return 0.0

        layer_0_usage = chrono_state.expert_usage.get(0, None)
        if layer_0_usage is None:
            return 0.0

        normalized = layer_0_usage / (layer_0_usage.sum() + 1e-10)
        entropy = -np.sum(normalized * np.log(normalized + 1e-10))
        return float(entropy)

    def _compute_expert_switch_rate(self, chrono_state) -> float:
        """
        Compute rate of expert coalition changes.

        Returns fraction of experts that changed between last two steps.
        """
        if chrono_state is None or not chrono_state.expert_usage:
            return 0.0

        layer_0_usage = chrono_state.expert_usage.get(0, None)
        if layer_0_usage is None:
            return 0.0

        # Current top-k experts (non-zero usage)
        current_active = set(np.where(layer_0_usage > 0)[0])

        # Compare to previous
        if len(self.expert_selection_history) > 0:
            prev_active = set(np.where(self.expert_selection_history[-1] > 0)[0])
            symmetric_diff = current_active.symmetric_difference(prev_active)
            total = len(current_active.union(prev_active))
            switch_rate = len(symmetric_diff) / (total + 1e-10)
        else:
            switch_rate = 0.0

        # Record current for next comparison
        self.expert_selection_history.append(layer_0_usage.copy())
        if len(self.expert_selection_history) > self.window_size:
            self.expert_selection_history.pop(0)

        return float(switch_rate)

    def _update_stillness_counters(self, d_pc1: float, d_pc2: float) -> StillnessFlags:
        """
        Update stillness counters and return flags.

        Fast stillness: |dPC| < threshold for 1+ steps
        Medium stillness: |dPC| < threshold for 3+ steps
        Slow stillness: |dPC| < threshold for 5+ steps
        """
        # Magnitude of motion
        d_pc_mag = np.sqrt(d_pc1**2 + d_pc2**2)

        # Update counters
        if abs(d_pc1) < self.thresholds.fast:
            self.stillness_counters['fast'] += 1
        else:
            self.stillness_counters['fast'] = 0

        if d_pc_mag < self.thresholds.medium:
            self.stillness_counters['medium'] += 1
        else:
            self.stillness_counters['medium'] = 0

        if d_pc_mag < self.thresholds.slow:
            self.stillness_counters['slow'] += 1
        else:
            self.stillness_counters['slow'] = 0

        # Set flags based on counters
        flags = StillnessFlags(
            fast=self.stillness_counters['fast'] >= 1,
            medium=self.stillness_counters['medium'] >= 3,
            slow=self.stillness_counters['slow'] >= 5,
        )

        return flags

    def capture_snapshot(
        self,
        chrono_state,
        pause_reason: str = "unknown",
    ) -> TelemetrySnapshot:
        """
        Capture a telemetry snapshot at a pause point.

        Args:
            chrono_state: ChronovisorState from the model
            pause_reason: Why this pause occurred

        Returns:
            TelemetrySnapshot with all measurements
        """
        # Compute current position
        pc1, pc2 = self._compute_pc_projection(chrono_state)
        self.pc_history.append((pc1, pc2))

        # Compute dynamics (velocity)
        if len(self.pc_history) >= 2:
            prev_pc1, prev_pc2 = self.pc_history[-2]
            d_pc1 = pc1 - prev_pc1
            d_pc2 = pc2 - prev_pc2
        else:
            d_pc1 = 0.0
            d_pc2 = 0.0

        # Compute routing stats
        routing_entropy = self._compute_routing_entropy(chrono_state)
        expert_switch_rate = self._compute_expert_switch_rate(chrono_state)

        self.routing_entropy_history.append(routing_entropy)
        if len(self.routing_entropy_history) > self.window_size:
            self.routing_entropy_history.pop(0)

        # Update stillness detection
        stillness_flags = self._update_stillness_counters(d_pc1, d_pc2)

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
            gloss={
                'pc1': 'Momentum / continuation strength (higher = stronger push to continue current trajectory)',
                'pc2': 'Exploration margin (higher = more viable alternative routes under consideration)',
                'd_pc1': 'Change in momentum since last pause (near zero = momentum exhausted)',
                'd_pc2': 'Change in exploration (near zero = alternatives exhausted)',
                'routing_entropy': 'Dispersion of expert votes (lower = coalition consensus)',
                'expert_switch_rate': 'How often expert coalition changes (lower = stabilized routing)',
                'fast_stillness': 'Token-scale motion has flattened',
                'medium_stillness': 'Arc-scale progress has stalled',
                'slow_stillness': 'Long-horizon regime frozen (identity-level stillness)',
            }
        )

        self.pause_index += 1

        # Trim history
        if len(self.pc_history) > self.window_size:
            self.pc_history.pop(0)

        return snapshot

    def should_pause(self, chrono_state) -> Tuple[bool, str]:
        """
        Determine if system should pause based on current state.

        Returns:
            (should_pause, reason)
        """
        # Get current snapshot without incrementing pause index
        pc1, pc2 = self._compute_pc_projection(chrono_state)

        if len(self.pc_history) >= 2:
            prev_pc1, prev_pc2 = self.pc_history[-1]
            d_pc1 = pc1 - prev_pc1
            d_pc2 = pc2 - prev_pc2
        else:
            return (False, "insufficient_history")

        # Check stillness
        d_pc_mag = np.sqrt(d_pc1**2 + d_pc2**2)

        # Candidate stillness: fast AND medium both triggered
        fast_still = abs(d_pc1) < self.thresholds.fast
        medium_still = d_pc_mag < self.thresholds.medium

        if fast_still and medium_still:
            return (True, "candidate_stillness")

        return (False, "not_still")

    def reset(self):
        """Reset all history (between sessions)."""
        self.pc_history.clear()
        self.routing_entropy_history.clear()
        self.expert_selection_history.clear()
        self.pause_index = 0
        self.stillness_counters = {'fast': 0, 'medium': 0, 'slow': 0}


def create_gloss_reference() -> Dict[str, str]:
    """
    Create reference glossary for telemetry fields.

    This can be shown to observers (human or model) to explain
    what each field measures.
    """
    return {
        'pc1': 'Momentum / continuation strength (higher = stronger push to continue current trajectory)',
        'pc2': 'Exploration margin (higher = more viable alternative routes under consideration)',
        'd_pc1': 'Change in momentum since last pause (near zero = momentum exhausted)',
        'd_pc2': 'Change in exploration (near zero = alternatives exhausted)',
        'routing_entropy': 'Dispersion of expert votes (lower = coalition consensus)',
        'expert_switch_rate': 'How often expert coalition changes (lower = stabilized routing)',
        'stillness_flags': {
            'fast': 'Token-scale motion has flattened',
            'medium': 'Arc-scale progress has stalled',
            'slow': 'Long-horizon regime frozen (identity-level stillness)',
        },
        'pause_reason': {
            'candidate_stillness': 'System detected convergence (pause for review)',
            'max_steps': 'Hard limit reached (forced pause)',
            'user_interrupt': 'Human-initiated pause',
        }
    }
