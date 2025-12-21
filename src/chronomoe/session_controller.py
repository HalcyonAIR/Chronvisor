"""
Session Controller for Multistep Generation

Manages chunked generation with pressure-based pause/continue logic.

Architecture:
    User Input → SessionController → ClockGatedModel → Pressure System → Pause/Continue

Design principles:
- Non-agentic: No auto-continuation without user input
- Explicit control: User commands change mode
- Observable: All decisions logged with pressure values
- Deterministic: Same signals → same pause decision

See: docs/006-multistep-pressure-system.md
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import List, Optional, Tuple

import torch

from chronomoe.pressure import PressureOutput, PressureSignals, compute_pressure


# =============================================================================
# Session Mode
# =============================================================================


class SessionMode(Enum):
    """Generation mode for session."""

    SINGLE_TURN = "single_turn"  # Generate until max_length (default)
    MULTISTEP = "multistep"      # Pause after each chunk, wait for user


# =============================================================================
# User Commands
# =============================================================================


class UserCommand(Enum):
    """Commands that users can issue during multistep generation."""

    CONTINUE = "continue"      # Continue with next chunk
    MULTISTEP_ON = "multistep on"   # Enable multistep mode
    MULTISTEP_OFF = "multistep off"  # Disable multistep mode
    END_LOOP = "end loop"      # Stop multistep generation
    RESET = "reset"            # Reset residual intent


def parse_user_command(text: str) -> Optional[UserCommand]:
    """
    Parse user input into a command.

    Args:
        text: User input string

    Returns:
        UserCommand if recognized, None otherwise
    """
    text = text.strip().lower()

    if text == "continue":
        return UserCommand.CONTINUE
    elif text in ["multistep on", "multistep", "/multistep"]:
        return UserCommand.MULTISTEP_ON
    elif text in ["multistep off", "single", "/single"]:
        return UserCommand.MULTISTEP_OFF
    elif text in ["end loop", "end", "stop"]:
        return UserCommand.END_LOOP
    elif text == "reset":
        return UserCommand.RESET

    return None


# =============================================================================
# Telemetry
# =============================================================================


@dataclass
class ChunkTelemetry:
    """
    Telemetry for a single generation chunk.

    Non-anthropomorphic: numbers and decisions, not narratives.
    """

    chunk_index: int
    tokens_generated: int

    # Pressure system
    fast_pressure: float
    mid_pressure: float
    slow_pressure: float
    fast_weight: float
    mid_weight: float
    slow_weight: float
    net_pressure: float
    residual_intent: float

    # Pause decision
    should_pause: bool
    pause_reason: Optional[str]

    # Model signals (final token)
    router_entropy: float
    router_margin: float
    coherence_R: float
    delta_R: float
    margin: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class SessionTelemetry:
    """
    Telemetry for entire session.

    Aggregates chunk-level metrics for analysis.
    """

    mode: str
    total_chunks: int
    total_tokens: int
    chunks: List[ChunkTelemetry]

    # Aggregate statistics
    avg_chunk_length: float
    avg_net_pressure: float
    pause_reasons: dict  # reason -> count

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "mode": self.mode,
            "total_chunks": self.total_chunks,
            "total_tokens": self.total_tokens,
            "chunks": [c.to_dict() for c in self.chunks],
            "avg_chunk_length": self.avg_chunk_length,
            "avg_net_pressure": self.avg_net_pressure,
            "pause_reasons": self.pause_reasons,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# =============================================================================
# Session Controller
# =============================================================================


class SessionController:
    """
    Controls multistep generation with pressure-based pausing.

    Responsibilities:
        - Manage session mode (single_turn vs multistep)
        - Extract signals from model outputs
        - Compute pressure and pause decisions
        - Log telemetry
        - Handle user commands

    Does NOT:
        - Auto-continue (requires user input)
        - Modify clock state (read-only observer)
        - Run second forward pass
    """

    def __init__(
        self,
        mode: SessionMode = SessionMode.SINGLE_TURN,
        chunk_size: int = 50,
        max_chunks: int = 10,
        verbose: bool = False,
    ):
        """
        Initialize session controller.

        Args:
            mode: Initial generation mode
            chunk_size: Tokens per chunk in multistep mode
            max_chunks: Maximum chunks before forcing stop
            verbose: Print telemetry as generation proceeds
        """
        self.mode = mode
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self.verbose = verbose

        # State
        self.residual_intent = 0.0
        self.chunk_history: List[ChunkTelemetry] = []
        self.current_chunk_index = 0

    def set_mode(self, mode: SessionMode) -> None:
        """Change session mode."""
        self.mode = mode
        if self.verbose:
            print(f"Session mode: {mode.value}")

    def reset(self) -> None:
        """Reset session state."""
        self.residual_intent = 0.0
        self.chunk_history = []
        self.current_chunk_index = 0

    def extract_signals(
        self,
        final_hidden_state: torch.Tensor,
        router_stats: dict,
        chrono_state: dict,
        clock_state: dict,
        margin: float,
    ) -> PressureSignals:
        """
        Extract pressure signals from model outputs.

        NO second forward pass. All signals already computed.

        Args:
            final_hidden_state: Final hidden state from model [d_model]
            router_stats: Router statistics (entropy, margin)
            chrono_state: Chronovisor state (coherence_R, delta_R)
            clock_state: Clock state (proximities, confidences)
            margin: Token margin (stiffness metric)

        Returns:
            PressureSignals ready for pressure computation
        """
        return PressureSignals(
            # Router signals
            router_entropy=router_stats.get("entropy", 0.0),
            router_margin=router_stats.get("margin", 0.0),
            # Coherence signals
            coherence_R=chrono_state.get("coherence_R", 0.0),
            delta_R=chrono_state.get("delta_R", 0.0),
            # Token margin
            margin=margin,
            # Fast clock
            fast_confidence=clock_state.get("fast_confidence", 0.0),
            # Medium clock
            mid_proximity_meso=clock_state.get("mid_proximity_meso", 0.0),
            mid_transition_prob=clock_state.get("mid_transition_prob", 0.0),
            mid_residual_intent=self.residual_intent,
            # Slow clock
            slow_confidence=clock_state.get("slow_confidence", 0.0),
            slow_proximity_macro=clock_state.get("slow_proximity_macro", 0.0),
            slow_constraint_penalty=clock_state.get("slow_constraint_penalty", 0.0),
        )

    def decide_and_log(
        self,
        signals: PressureSignals,
        tokens_generated: int,
    ) -> Tuple[bool, Optional[str], ChunkTelemetry]:
        """
        Compute pressure, decide whether to pause, and log telemetry.

        Args:
            signals: Extracted pressure signals
            tokens_generated: Number of tokens in this chunk

        Returns:
            Tuple of (should_pause, pause_reason, telemetry)
        """
        # Compute pressure
        pressure_output = compute_pressure(signals, mode=self.mode.value)

        # Update residual intent
        self.residual_intent = pressure_output.new_residual_intent

        # Create telemetry
        telemetry = ChunkTelemetry(
            chunk_index=self.current_chunk_index,
            tokens_generated=tokens_generated,
            # Pressure
            fast_pressure=pressure_output.fast_pressure,
            mid_pressure=pressure_output.mid_pressure,
            slow_pressure=pressure_output.slow_pressure,
            fast_weight=pressure_output.fast_weight,
            mid_weight=pressure_output.mid_weight,
            slow_weight=pressure_output.slow_weight,
            net_pressure=pressure_output.net_pressure,
            residual_intent=self.residual_intent,
            # Decision
            should_pause=pressure_output.should_pause,
            pause_reason=pressure_output.pause_reason,
            # Signals
            router_entropy=signals.router_entropy,
            router_margin=signals.router_margin,
            coherence_R=signals.coherence_R,
            delta_R=signals.delta_R,
            margin=signals.margin,
        )

        # Log
        self.chunk_history.append(telemetry)
        self.current_chunk_index += 1

        if self.verbose:
            self._print_telemetry(telemetry)

        return (
            pressure_output.should_pause,
            pressure_output.pause_reason,
            telemetry,
        )

    def get_session_telemetry(self) -> SessionTelemetry:
        """
        Aggregate chunk-level telemetry into session summary.

        Returns:
            SessionTelemetry with aggregate statistics
        """
        if not self.chunk_history:
            return SessionTelemetry(
                mode=self.mode.value,
                total_chunks=0,
                total_tokens=0,
                chunks=[],
                avg_chunk_length=0.0,
                avg_net_pressure=0.0,
                pause_reasons={},
            )

        total_tokens = sum(c.tokens_generated for c in self.chunk_history)
        avg_chunk_length = total_tokens / len(self.chunk_history)
        avg_net_pressure = sum(c.net_pressure for c in self.chunk_history) / len(
            self.chunk_history
        )

        # Pause reason distribution
        pause_reasons = {}
        for chunk in self.chunk_history:
            if chunk.pause_reason:
                pause_reasons[chunk.pause_reason] = (
                    pause_reasons.get(chunk.pause_reason, 0) + 1
                )

        return SessionTelemetry(
            mode=self.mode.value,
            total_chunks=len(self.chunk_history),
            total_tokens=total_tokens,
            chunks=self.chunk_history,
            avg_chunk_length=avg_chunk_length,
            avg_net_pressure=avg_net_pressure,
            pause_reasons=pause_reasons,
        )

    def _print_telemetry(self, t: ChunkTelemetry) -> None:
        """Print chunk telemetry (non-anthropomorphic)."""
        print(f"\n{'=' * 60}")
        print(f"Chunk {t.chunk_index}: {t.tokens_generated} tokens")
        print(f"{'=' * 60}")
        print(f"  Pressures:")
        print(f"    Fast:  {t.fast_pressure:+.4f} (w={t.fast_weight:.2f})")
        print(f"    Mid:   {t.mid_pressure:+.4f} (w={t.mid_weight:.2f})")
        print(f"    Slow:  {t.slow_pressure:+.4f} (w={t.slow_weight:.2f})")
        print(f"    Net:   {t.net_pressure:+.4f}")
        print(f"  Residual intent: {t.residual_intent:.4f}")
        print(f"  Decision: {'PAUSE' if t.should_pause else 'CONTINUE'}")
        if t.pause_reason:
            print(f"  Reason: {t.pause_reason}")
        print()

    def handle_user_input(self, user_input: str) -> Optional[str]:
        """
        Handle user commands during generation.

        Args:
            user_input: User input string

        Returns:
            Status message or None
        """
        command = parse_user_command(user_input)

        if command == UserCommand.CONTINUE:
            return "Continuing..."

        elif command == UserCommand.MULTISTEP_ON:
            self.set_mode(SessionMode.MULTISTEP)
            return "Multistep mode enabled. Generation will pause after each chunk."

        elif command == UserCommand.MULTISTEP_OFF:
            self.set_mode(SessionMode.SINGLE_TURN)
            return "Single-turn mode enabled. Generation will continue until max_length."

        elif command == UserCommand.END_LOOP:
            return "Ending multistep loop."

        elif command == UserCommand.RESET:
            self.reset()
            return "Session reset. Residual intent cleared."

        return None


# =============================================================================
# Testing
# =============================================================================


if __name__ == "__main__":
    print("Testing Session Controller")
    print("=" * 60)
    print()

    # Test 1: Mode switching
    print("Test 1: Mode switching")
    controller = SessionController(verbose=True)
    print(f"  Initial mode: {controller.mode.value}")

    controller.handle_user_input("multistep on")
    assert controller.mode == SessionMode.MULTISTEP

    controller.handle_user_input("multistep off")
    assert controller.mode == SessionMode.SINGLE_TURN

    print("  ✓ Mode switching works")
    print()

    # Test 2: Signal extraction and pressure computation
    print("Test 2: Signal extraction and pressure computation")
    controller.set_mode(SessionMode.MULTISTEP)

    # Mock signals
    signals = controller.extract_signals(
        final_hidden_state=torch.randn(256),
        router_stats={"entropy": 0.5, "margin": 1.0},
        chrono_state={"coherence_R": 0.8, "delta_R": 0.1},
        clock_state={
            "fast_confidence": 0.9,
            "mid_proximity_meso": 0.2,
            "mid_transition_prob": 0.9,
            "slow_confidence": 0.7,
            "slow_proximity_macro": 0.1,
            "slow_constraint_penalty": 0.0,
        },
        margin=1.5,
    )

    should_pause, reason, telemetry = controller.decide_and_log(signals, tokens_generated=50)

    print(f"  Should pause: {should_pause}")
    print(f"  Reason: {reason}")
    print(f"  Net pressure: {telemetry.net_pressure:+.4f}")
    print(f"  Residual intent: {telemetry.residual_intent:.4f}")

    assert should_pause is True  # Multistep mode always pauses
    assert reason == "multistep_chunk_complete"

    print("  ✓ Pressure computation works")
    print()

    # Test 3: Session telemetry
    print("Test 3: Session telemetry aggregation")

    # Generate a few more chunks
    for i in range(3):
        signals.mid_proximity_meso += 0.1 * i  # Vary signals
        controller.decide_and_log(signals, tokens_generated=45 + i * 5)

    session_telemetry = controller.get_session_telemetry()

    print(f"  Total chunks: {session_telemetry.total_chunks}")
    print(f"  Total tokens: {session_telemetry.total_tokens}")
    print(f"  Avg chunk length: {session_telemetry.avg_chunk_length:.2f}")
    print(f"  Avg net pressure: {session_telemetry.avg_net_pressure:+.4f}")
    print(f"  Pause reasons: {session_telemetry.pause_reasons}")

    assert session_telemetry.total_chunks == 4
    assert session_telemetry.pause_reasons["multistep_chunk_complete"] == 4

    print("  ✓ Session telemetry works")
    print()

    # Test 4: JSON export
    print("Test 4: JSON export")
    json_output = session_telemetry.to_json(indent=2)
    print("  Sample JSON output:")
    print("  " + json_output[:200] + "...")
    print("  ✓ JSON serialization works")
    print()

    # Test 5: Reset
    print("Test 5: Reset")
    controller.reset()
    assert controller.residual_intent == 0.0
    assert len(controller.chunk_history) == 0
    assert controller.current_chunk_index == 0
    print("  ✓ Reset works")
    print()

    print("=" * 60)
    print("✓ All SessionController tests passed!")
    print()
    print("Key properties verified:")
    print("  - Mode switching works")
    print("  - Signal extraction from existing computation")
    print("  - Pressure computation and pause decisions")
    print("  - Telemetry aggregation")
    print("  - JSON serialization")
    print("  - Non-agentic by construction (no auto-continue)")
