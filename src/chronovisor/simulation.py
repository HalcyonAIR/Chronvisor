"""
Toy World Simulation for Chronovisor.

A sandbox for prototyping the control loop before touching
any real model or geometry. Everything here is fake but dynamic.
"""

import random
from chronovisor.controller import Controller
from chronovisor.expert_harness import ExpertHarness


class LensState:
    """
    Minimal placeholder lens surface.

    Internal state is a 2D vector that experts pretend to read through.
    No constraints, no bounds, no real geometry.
    """

    def __init__(self, initial: list = None) -> None:
        """
        Initialise the lens state.

        Args:
            initial: Starting 2D vector. Defaults to [0.0, 0.0].
        """
        if initial is None:
            initial = [0.0, 0.0]
        self.vector = list(initial)

    def update(self, delta: list) -> None:
        """
        Add delta to the internal vector.

        Args:
            delta: 2D vector to add.
        """
        self.vector[0] += delta[0]
        self.vector[1] += delta[1]

    def __repr__(self) -> str:
        return f"Lens([{self.vector[0]:+.3f}, {self.vector[1]:+.3f}])"


class ToyExpert(ExpertHarness):
    """
    A toy expert that generates noisy readings based on lens state.

    Each expert has a "personality" — a sensitivity vector that determines
    how it responds to the lens. This creates diversity in the ensemble.
    """

    def __init__(self, name: str = None, sensitivity: list = None) -> None:
        """
        Initialise the toy expert.

        Args:
            name: Optional name for this expert.
            sensitivity: 2D vector controlling response to lens. Defaults to [1.0, 1.0].
        """
        super().__init__(name=name)
        if sensitivity is None:
            sensitivity = [1.0, 1.0]
        self.sensitivity = list(sensitivity)

    def sense(self, lens_state) -> dict:
        """
        Generate noisy sensor readings based on the lens.

        The expert's personality (sensitivity) shapes how it interprets
        the lens geometry. Small noise keeps things dynamic.

        Args:
            lens_state: LensState object (or None for testing).

        Returns:
            Dictionary with gain, tilt, stability, out_of_tolerance.
        """
        if lens_state is None:
            return super().sense(lens_state)

        vec = lens_state.vector
        s = self.sensitivity

        # Small noise term
        noise = lambda: random.gauss(0, 0.02)

        # Gain responds to first lens dimension
        gain = 1.0 + 0.1 * s[0] * vec[0] + noise()

        # Tilt responds to second lens dimension
        tilt = 0.0 + 0.15 * s[1] * vec[1] + noise()

        # Stability is inverse of how "stretched" the expert feels
        stretch = abs(s[0] * vec[0]) + abs(s[1] * vec[1])
        stability = max(0.0, 1.0 - 0.2 * stretch + noise())

        # Out of tolerance if stability drops too low
        out_of_tolerance = stability < 0.5

        return {
            "gain": gain,
            "tilt": tilt,
            "stability": stability,
            "out_of_tolerance": out_of_tolerance,
        }


def compute_ensemble_coherence(expert_signals: list) -> float:
    """
    Compute a coherence score from expert signals.

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
    # Scale so typical values are in a readable range
    coherence = -10 * (var_stab + 0.5 * var_gain)

    return coherence


def run_simulation(
    num_experts: int = 4,
    num_ticks: int = 100,
    micro_period: int = 5,
    macro_period: int = 20,
    drift_interval: int = 7,
    drift_magnitude: float = 0.1,
    seed: int = None,
) -> None:
    """
    Run a toy simulation of the Chronovisor control loop.

    Creates experts with varied personalities, a drifting lens,
    and logs the system's behaviour over time.

    Args:
        num_experts: Number of ToyExperts to create.
        num_ticks: How many ticks to run.
        micro_period: Controller micro clock period.
        macro_period: Controller macro clock period.
        drift_interval: Apply lens drift every N ticks.
        drift_magnitude: Size of random lens drift.
        seed: Random seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)

    # Create controller with coherence computation
    controller = Controller(micro_period=micro_period, macro_period=macro_period)

    # Override the placeholder coherence method
    controller._prev_coherence = 0.0

    def tracking_coherence(expert_states: list) -> float:
        coherence = compute_ensemble_coherence(expert_states)
        delta = coherence - controller._prev_coherence
        controller._prev_coherence = coherence
        return delta

    controller.compute_delta_coherence = tracking_coherence

    # Create experts with varied personalities
    personalities = [
        ([1.0, 0.5], "Alpha"),
        ([0.5, 1.0], "Beta"),
        ([0.8, 0.8], "Gamma"),
        ([1.2, 0.3], "Delta"),
        ([0.3, 1.2], "Epsilon"),
    ]
    experts = [
        ToyExpert(name=name, sensitivity=sens)
        for sens, name in personalities[:num_experts]
    ]

    # Create lens
    lens = LensState()

    # Header
    print("=" * 70)
    print("CHRONOVISOR TOY SIMULATION")
    print("=" * 70)
    print(f"Experts: {[e.name for e in experts]}")
    print(f"Periods: micro={micro_period}, macro={macro_period}")
    print(f"Drift: every {drift_interval} ticks, magnitude={drift_magnitude}")
    print("=" * 70)
    print()

    # Silence the controller's default logging
    original_tick = controller.tick

    def quiet_tick(experts, lens):
        # Store clocks before tick
        fast_before = controller.fast_clock

        # Increment fast clock
        controller.fast_clock += 1

        # Update micro clock
        if controller.fast_clock % controller.micro_period == 0:
            controller.micro_clock += 1

        # Update macro clock
        if controller.fast_clock % controller.macro_period == 0:
            controller.macro_clock += 1

        # Collect expert signals
        expert_signals = [expert.sense(lens) for expert in experts]

        # Compute delta coherence
        delta = controller.compute_delta_coherence(expert_signals)

        return expert_signals, delta

    # Run simulation
    for tick in range(1, num_ticks + 1):
        # Apply lens drift periodically
        drifted = False
        if tick % drift_interval == 0:
            drift = [
                random.gauss(0, drift_magnitude),
                random.gauss(0, drift_magnitude),
            ]
            lens.update(drift)
            drifted = True

        # Run one tick
        signals, delta = quiet_tick(experts, lens)

        # Compute summary stats
        avg_stability = sum(s["stability"] for s in signals) / len(signals)
        avg_gain = sum(s["gain"] for s in signals) / len(signals)
        any_oot = any(s["out_of_tolerance"] for s in signals)

        # Log at interesting moments
        is_micro = controller.fast_clock % micro_period == 0
        is_macro = controller.fast_clock % macro_period == 0

        marker = ""
        if is_macro:
            marker = " [MACRO]"
        elif is_micro:
            marker = " [micro]"

        if drifted or is_micro or is_macro or any_oot:
            oot_flag = " OOT!" if any_oot else ""
            drift_flag = " ~drift~" if drifted else ""
            print(
                f"t={tick:3d} | {lens} | "
                f"Δ={delta:+.4f} | "
                f"stab={avg_stability:.3f} gain={avg_gain:.3f}"
                f"{oot_flag}{drift_flag}{marker}"
            )

    # Summary
    print()
    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"Final lens state: {lens}")
    print(f"Final clocks: fast={controller.fast_clock}, "
          f"micro={controller.micro_clock}, macro={controller.macro_clock}")

    # Final expert states
    print()
    print("Final expert readings:")
    for expert in experts:
        reading = expert.sense(lens)
        print(f"  {expert.name}: gain={reading['gain']:.3f}, "
              f"tilt={reading['tilt']:.3f}, "
              f"stability={reading['stability']:.3f}, "
              f"oot={reading['out_of_tolerance']}")


if __name__ == "__main__":
    run_simulation(seed=42)
