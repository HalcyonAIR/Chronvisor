"""
Toy World Simulation for Chronovisor.

A sandbox for prototyping the control loop before touching
any real model or geometry. Everything here is fake but dynamic.

V2: Kuramoto coherence + emergent trace field lens + decay gating
"""

import math
import random
from dataclasses import dataclass, field
from chronovisor.expert_harness import ExpertHarness


# =============================================================================
# Lens as accumulated trace field
# =============================================================================

class TraceFieldLens:
    """
    Emergent lens surface from accumulated expert traces.

    The lens is not set directly — it emerges from expert imprints.
    Controller only controls decay rate γ.

    T(t+1) = γ(t) * T(t) + Σ_k v_k(t)
    L(t) = T(t) / (||T(t)|| + ε)
    """

    def __init__(self, dim: int = 2, epsilon: float = 1e-6) -> None:
        """
        Initialise the trace field.

        Args:
            dim: Dimensionality of trace field.
            epsilon: Small constant for normalisation stability.
        """
        self.dim = dim
        self.epsilon = epsilon
        self.trace = [0.0] * dim  # T(t) - raw accumulated traces
        self.gamma = 0.9  # Current decay factor, controlled by controller

    @property
    def vector(self) -> list:
        """
        Normalised lens state L(t) that experts read.

        L(t) = T(t) / (||T(t)|| + ε)
        """
        norm = math.sqrt(sum(x * x for x in self.trace)) + self.epsilon
        return [x / norm for x in self.trace]

    @property
    def magnitude(self) -> float:
        """Raw magnitude of trace field ||T(t)||."""
        return math.sqrt(sum(x * x for x in self.trace))

    def accumulate(self, traces: list) -> None:
        """
        Accumulate expert traces with current decay.

        T(t+1) = γ * T(t) + Σ traces

        Args:
            traces: List of 2D trace vectors from experts.
        """
        # Apply decay
        self.trace = [self.gamma * t for t in self.trace]

        # Add all traces
        for v in traces:
            for i in range(self.dim):
                self.trace[i] += v[i]

    def set_gamma(self, gamma: float) -> None:
        """Set decay factor (called by controller)."""
        self.gamma = max(0.0, min(1.0, gamma))

    def __repr__(self) -> str:
        L = self.vector
        return f"Lens(L=[{L[0]:+.3f},{L[1]:+.3f}] |T|={self.magnitude:.3f} γ={self.gamma:.3f})"


# =============================================================================
# Kuramoto-style expert with phase dynamics
# =============================================================================

class KuramotoExpert(ExpertHarness):
    """
    Expert with Kuramoto phase dynamics.

    Each expert has:
    - Phase φ ∈ [0, 2π) that evolves over time
    - Intrinsic frequency ω (how fast it naturally rotates)
    - Coupling strength that determines trace amplitude

    Emits trace vector: v = a * [cos(φ), sin(φ)]
    where a = gain * stability
    """

    def __init__(
        self,
        name: str = None,
        omega: float = 0.1,
        noise_scale: float = 0.05,
        coupling: float = 1.0,
    ) -> None:
        """
        Initialise the Kuramoto expert.

        Args:
            name: Optional name for this expert.
            omega: Intrinsic frequency (radians per tick).
            noise_scale: Scale of phase noise.
            coupling: Strength of trace emission.
        """
        super().__init__(name=name)
        self.omega = omega
        self.noise_scale = noise_scale
        self.coupling = coupling

        # Phase state
        self.phase = random.uniform(0, 2 * math.pi)

    def tick_phase(self) -> None:
        """
        Evolve phase by one step.

        φ(t+1) = φ(t) + ω + η(t)
        """
        noise = random.gauss(0, self.noise_scale)
        self.phase = (self.phase + self.omega + noise) % (2 * math.pi)

    def sense(self, lens_state) -> dict:
        """
        Generate sensor readings based on current phase and lens.

        Returns dict with gain, tilt, stability, out_of_tolerance, plus:
        - phase: current phase
        - trace: emitted trace vector
        """
        if lens_state is None:
            return super().sense(lens_state)

        L = lens_state.vector

        # Project phase onto lens direction
        cos_phi = math.cos(self.phase)
        sin_phi = math.sin(self.phase)

        # Alignment with lens (dot product with unit phase vector)
        alignment = L[0] * cos_phi + L[1] * sin_phi

        # Gain increases when aligned with lens
        gain = 1.0 + 0.2 * alignment + random.gauss(0, 0.02)

        # Tilt is the perpendicular component
        tilt = L[0] * (-sin_phi) + L[1] * cos_phi + random.gauss(0, 0.02)

        # Stability is higher when well-aligned
        stability = 0.5 + 0.5 * abs(alignment) + random.gauss(0, 0.02)
        stability = max(0.0, min(1.0, stability))

        # Out of tolerance if stability too low
        out_of_tolerance = stability < 0.3

        # Trace emission: v = a * [cos(φ), sin(φ)]
        amplitude = gain * stability * self.coupling
        trace = [amplitude * cos_phi, amplitude * sin_phi]

        return {
            "gain": gain,
            "tilt": tilt,
            "stability": stability,
            "out_of_tolerance": out_of_tolerance,
            "phase": self.phase,
            "trace": trace,
        }


# =============================================================================
# Kuramoto order parameter (coherence)
# =============================================================================

def compute_kuramoto_R(phases: list) -> tuple:
    """
    Compute Kuramoto order parameter R and mean phase ψ.

    R * exp(iψ) = (1/N) Σ_k exp(i φ_k)

    Args:
        phases: List of phase values in radians.

    Returns:
        (R, psi) where R ∈ [0,1] is coherence and ψ is mean phase.
    """
    if not phases:
        return 0.0, 0.0

    N = len(phases)
    sum_cos = sum(math.cos(p) for p in phases)
    sum_sin = sum(math.sin(p) for p in phases)

    # Complex order parameter
    real = sum_cos / N
    imag = sum_sin / N

    R = math.sqrt(real * real + imag * imag)
    psi = math.atan2(imag, real)

    return R, psi


def sigmoid(x: float) -> float:
    """Logistic sigmoid function."""
    if x < -500:
        return 0.0
    if x > 500:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def compute_ensemble_coherence(expert_signals: list) -> float:
    """
    V1 coherence: variance-based score from expert signals.

    Uses variance of stability values as a proxy for coherence.
    Low variance = experts agree = high coherence.
    Returns negative variance so higher = better.

    Args:
        expert_signals: List of dicts from expert.sense().

    Returns:
        Coherence score (higher is more coherent).
    """
    if not expert_signals:
        return 0.0

    stabilities = [s["stability"] for s in expert_signals]
    gains = [s["gain"] for s in expert_signals]

    n = len(stabilities)
    if n < 2:
        return 0.0

    # Mean and variance of stability
    mean_stab = sum(stabilities) / n
    var_stab = sum((x - mean_stab) ** 2 for x in stabilities) / n

    # Mean and variance of gain
    mean_gain = sum(gains) / n
    var_gain = sum((x - mean_gain) ** 2 for x in gains) / n

    # Coherence: negative combined variance (higher = more agreement)
    coherence = -10 * (var_stab + 0.5 * var_gain)

    return coherence


# =============================================================================
# Controller with decay gating
# =============================================================================

@dataclass
class ControllerState:
    """Mutable state for the gating controller."""

    # Clocks
    fast_clock: int = 0
    micro_clock: int = 0
    macro_clock: int = 0

    # Coherence tracking
    R_history: list = field(default_factory=list)  # Recent R values
    R_micro_avg: float = 0.5  # Smoothed R over micro window

    # Gating parameters (can be adapted at macro clock)
    gamma_min: float = 0.7
    gamma_max: float = 0.99
    R0: float = 0.5  # Reference coherence level
    beta: float = 10.0  # Sigmoid sharpness

    # For macro adaptation
    R_micro_history: list = field(default_factory=list)


class GatingController:
    """
    Controller that gates lens decay based on Kuramoto coherence.

    The controller does NOT paint geometry. It only controls how quickly
    the trace field decays:

    γ = γ_min + (γ_max - γ_min) * σ(β * (R̄_micro - R0))

    High coherence → slow decay → lens holds structure
    Low coherence → fast decay → lens washes out
    """

    def __init__(
        self,
        micro_period: int = 5,
        macro_period: int = 20,
        gamma_min: float = 0.7,
        gamma_max: float = 0.99,
        R0: float = 0.5,
        beta: float = 10.0,
    ) -> None:
        """
        Initialise the gating controller.

        Args:
            micro_period: Fast ticks per micro tick.
            macro_period: Fast ticks per macro tick.
            gamma_min: Minimum decay (fast decay, short memory).
            gamma_max: Maximum decay (slow decay, long memory).
            R0: Reference coherence level for gating.
            beta: Sigmoid sharpness.
        """
        self.micro_period = micro_period
        self.macro_period = macro_period

        self.state = ControllerState(
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            R0=R0,
            beta=beta,
        )

    def compute_gamma(self, R_avg: float) -> float:
        """
        Compute decay factor from average coherence.

        γ = γ_min + (γ_max - γ_min) * σ(β * (R̄ - R0))
        """
        s = self.state
        x = s.beta * (R_avg - s.R0)
        return s.gamma_min + (s.gamma_max - s.gamma_min) * sigmoid(x)

    def tick(self, experts: list, lens: TraceFieldLens) -> dict:
        """
        Advance one fast tick.

        1. Update expert phases
        2. Collect signals and compute R
        3. Accumulate traces on lens
        4. At micro boundary: update γ
        5. At macro boundary: adapt R0

        Returns dict with tick info.
        """
        s = self.state

        # === Fast clock: every tick ===
        s.fast_clock += 1

        # Update expert phases
        for expert in experts:
            expert.tick_phase()

        # Collect signals
        signals = [expert.sense(lens) for expert in experts]

        # Extract phases and compute R
        phases = [sig["phase"] for sig in signals]
        R, psi = compute_kuramoto_R(phases)
        s.R_history.append(R)

        # Keep only recent R values (one micro window worth)
        if len(s.R_history) > self.micro_period:
            s.R_history = s.R_history[-self.micro_period:]

        # Collect traces and accumulate on lens
        traces = [sig["trace"] for sig in signals]
        lens.accumulate(traces)

        # Track clock boundaries
        is_micro = s.fast_clock % self.micro_period == 0
        is_macro = s.fast_clock % self.macro_period == 0

        # === Micro clock: every K fast ticks ===
        if is_micro:
            s.micro_clock += 1

            # Compute smoothed R over micro window
            s.R_micro_avg = sum(s.R_history) / len(s.R_history)
            s.R_micro_history.append(s.R_micro_avg)

            # Update gamma based on coherence
            gamma = self.compute_gamma(s.R_micro_avg)
            lens.set_gamma(gamma)

        # === Macro clock: every M fast ticks ===
        if is_macro:
            s.macro_clock += 1

            # Adapt R0 toward long-term mean R
            # This makes the gate self-tune to the system's natural coherence
            if len(s.R_micro_history) >= 2:
                long_term_R = sum(s.R_micro_history) / len(s.R_micro_history)
                # Nudge R0 toward observed mean (slow adaptation)
                s.R0 = s.R0 + 0.1 * (long_term_R - s.R0)

            # Keep history bounded
            if len(s.R_micro_history) > 10:
                s.R_micro_history = s.R_micro_history[-10:]

        return {
            "fast": s.fast_clock,
            "micro": s.micro_clock,
            "macro": s.macro_clock,
            "R": R,
            "R_micro": s.R_micro_avg,
            "gamma": lens.gamma,
            "R0": s.R0,
            "is_micro": is_micro,
            "is_macro": is_macro,
            "signals": signals,
        }


# =============================================================================
# Simulation runner
# =============================================================================

def run_kuramoto_simulation(
    num_experts: int = 5,
    num_ticks: int = 200,
    micro_period: int = 5,
    macro_period: int = 25,
    seed: int = None,
) -> None:
    """
    Run the Kuramoto-based Chronovisor simulation.

    Experts have phase dynamics. Lens emerges from accumulated traces.
    Controller gates decay based on Kuramoto coherence R.
    """
    if seed is not None:
        random.seed(seed)

    # Create experts with varied frequencies
    # Spread of ω creates natural decoherence that must be overcome
    expert_configs = [
        ("Alpha", 0.08, 0.04),   # (name, omega, noise)
        ("Beta", 0.10, 0.04),
        ("Gamma", 0.12, 0.04),
        ("Delta", 0.09, 0.05),
        ("Epsilon", 0.11, 0.05),
    ]

    experts = [
        KuramotoExpert(name=name, omega=omega, noise_scale=noise)
        for name, omega, noise in expert_configs[:num_experts]
    ]

    # Create lens and controller
    lens = TraceFieldLens(dim=2)
    controller = GatingController(
        micro_period=micro_period,
        macro_period=macro_period,
        gamma_min=0.7,
        gamma_max=0.99,
        R0=0.5,
        beta=10.0,
    )

    # Header
    print("=" * 78)
    print("CHRONOVISOR KURAMOTO SIMULATION")
    print("=" * 78)
    print(f"Experts: {[e.name for e in experts]}")
    print(f"Frequencies (ω): {[e.omega for e in experts]}")
    print(f"Periods: micro={micro_period}, macro={macro_period}")
    print("=" * 78)
    print()
    print("Legend: R=Kuramoto coherence, γ=decay, R0=reference (adapts at MACRO)")
    print()

    # Run simulation
    for tick in range(1, num_ticks + 1):
        info = controller.tick(experts, lens)

        # Determine what to log
        is_micro = info["is_micro"]
        is_macro = info["is_macro"]

        # Log at boundaries
        if is_micro or is_macro:
            marker = " [MACRO]" if is_macro else " [micro]"

            # Compute average stability
            avg_stab = sum(s["stability"] for s in info["signals"]) / len(info["signals"])

            print(
                f"t={tick:3d} | "
                f"R={info['R']:.3f} R̄={info['R_micro']:.3f} | "
                f"γ={info['gamma']:.3f} R0={info['R0']:.3f} | "
                f"{lens} | "
                f"stab={avg_stab:.3f}"
                f"{marker}"
            )

    # Summary
    print()
    print("=" * 78)
    print("SIMULATION COMPLETE")
    print("=" * 78)
    print(f"Final lens: {lens}")
    print(f"Final clocks: fast={info['fast']}, micro={info['micro']}, macro={info['macro']}")
    print(f"Final R0 (adapted): {controller.state.R0:.3f}")
    print()
    print("Final expert states:")
    for expert in experts:
        sig = expert.sense(lens)
        print(
            f"  {expert.name}: φ={sig['phase']:.2f} rad, "
            f"gain={sig['gain']:.3f}, stab={sig['stability']:.3f}"
        )


# Keep old simulation for backwards compatibility
def run_simulation(*args, **kwargs):
    """Run the original simulation (V1). For V2, use run_kuramoto_simulation()."""
    # Import old implementation inline to avoid circular issues
    print("Note: run_simulation() is V1. For Kuramoto dynamics, use run_kuramoto_simulation()")
    print()
    _run_simulation_v1(*args, **kwargs)


def _run_simulation_v1(
    num_experts: int = 4,
    num_ticks: int = 100,
    micro_period: int = 5,
    macro_period: int = 20,
    drift_interval: int = 7,
    drift_magnitude: float = 0.1,
    seed: int = None,
) -> None:
    """Original V1 simulation (preserved for comparison)."""
    from chronovisor.controller import Controller

    if seed is not None:
        random.seed(seed)

    controller = Controller(micro_period=micro_period, macro_period=macro_period)
    controller._prev_coherence = 0.0

    def tracking_coherence(expert_states: list) -> float:
        coherence = compute_ensemble_coherence(expert_states)
        delta = coherence - controller._prev_coherence
        controller._prev_coherence = coherence
        return delta

    controller.compute_delta_coherence = tracking_coherence

    # V1 ToyExpert (inline to avoid breaking changes)
    class ToyExpertV1(ExpertHarness):
        def __init__(self, name=None, sensitivity=None):
            super().__init__(name=name)
            self.sensitivity = sensitivity if sensitivity else [1.0, 1.0]

        def sense(self, lens_state):
            if lens_state is None:
                return super().sense(lens_state)
            vec = lens_state.vector
            s = self.sensitivity
            noise = lambda: random.gauss(0, 0.02)
            gain = 1.0 + 0.1 * s[0] * vec[0] + noise()
            tilt = 0.0 + 0.15 * s[1] * vec[1] + noise()
            stretch = abs(s[0] * vec[0]) + abs(s[1] * vec[1])
            stability = max(0.0, 1.0 - 0.2 * stretch + noise())
            return {
                "gain": gain, "tilt": tilt, "stability": stability,
                "out_of_tolerance": stability < 0.5,
            }

    # V1 LensState (inline)
    class LensStateV1:
        def __init__(self, initial=None):
            self.vector = list(initial) if initial else [0.0, 0.0]
        def update(self, delta):
            self.vector[0] += delta[0]
            self.vector[1] += delta[1]
        def __repr__(self):
            return f"Lens([{self.vector[0]:+.3f}, {self.vector[1]:+.3f}])"

    personalities = [
        ([1.0, 0.5], "Alpha"), ([0.5, 1.0], "Beta"),
        ([0.8, 0.8], "Gamma"), ([1.2, 0.3], "Delta"), ([0.3, 1.2], "Epsilon"),
    ]
    experts = [ToyExpertV1(name=n, sensitivity=s) for s, n in personalities[:num_experts]]
    lens = LensStateV1()

    print("=" * 70)
    print("CHRONOVISOR TOY SIMULATION (V1)")
    print("=" * 70)

    for tick in range(1, num_ticks + 1):
        if tick % drift_interval == 0:
            lens.update([random.gauss(0, drift_magnitude), random.gauss(0, drift_magnitude)])

        controller.fast_clock += 1
        if controller.fast_clock % micro_period == 0:
            controller.micro_clock += 1
        if controller.fast_clock % macro_period == 0:
            controller.macro_clock += 1

        signals = [e.sense(lens) for e in experts]
        delta = controller.compute_delta_coherence(signals)

        is_micro = controller.fast_clock % micro_period == 0
        is_macro = controller.fast_clock % macro_period == 0

        if is_micro or is_macro:
            avg_stab = sum(s["stability"] for s in signals) / len(signals)
            marker = " [MACRO]" if is_macro else " [micro]"
            print(f"t={tick:3d} | {lens} | Δ={delta:+.4f} | stab={avg_stab:.3f}{marker}")

    print("=" * 70)
    print("SIMULATION COMPLETE (V1)")


if __name__ == "__main__":
    run_kuramoto_simulation(seed=42)
