"""
Chronovisor V3: Per-Expert Temperament Simulation.

Memory lives in per-expert inertia (velocity) and damping (lambda).
No global lens, no geometry. Each expert learns how quickly it is allowed to change.

Key insight: bias the expert's acceleration, not its position or the world it sees.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional


def compute_kuramoto_R_and_psi(phases: List[float]) -> tuple:
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
    re = sum(math.cos(phi) for phi in phases) / N
    im = sum(math.sin(phi) for phi in phases) / N

    R = math.sqrt(re * re + im * im)
    psi = math.atan2(im, re)

    return R, psi


def sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def alignment(phi_k: float, psi: float) -> float:
    """
    Per-expert alignment with ensemble mean phase.

    Returns value in [-1, 1]:
    - +1: perfectly in phase with group
    - -1: perfectly anti-phase
    - 0: orthogonal

    Args:
        phi_k: Expert's phase.
        psi: Ensemble mean phase.

    Returns:
        Alignment score.
    """
    return math.cos(phi_k - psi)


@dataclass
class AdaptiveExpert:
    """
    Expert with per-expert temperament dynamics.

    Each expert carries:
    - phi: Kuramoto phase in [0, 2π)
    - omega: intrinsic frequency
    - theta: scalar "tilt" or preference (position)
    - v: tilt velocity
    - lambd: damping/responsiveness (how much alignment affects acceleration)
    - s: slow reliability score

    Memory is in velocity and damping, not in a shared lens.
    """

    name: str
    phi: float  # Kuramoto phase
    omega: float  # Intrinsic frequency

    # Trajectory state
    theta: float = 0.0  # Tilt (position)
    v: float = 0.0  # Velocity

    # Temperament
    lambd: float = 0.05  # Damping/responsiveness

    # Reliability (slow-moving trust score)
    s: float = 0.0

    # Rolling stats for micro/macro updates
    micro_align_sum: float = 0.0
    micro_align_count: int = 0
    macro_align_sum: float = 0.0
    macro_align_count: int = 0

    def tick_fast(
        self,
        psi: float,
        dv: float = 0.05,
        noise_phi_std: float = 0.01,
        noise_v_std: float = 0.01,
    ) -> Dict[str, float]:
        """
        Fast clock update: phase, velocity, tilt.

        Args:
            psi: Ensemble mean phase from Kuramoto.
            dv: Friction on velocity.
            noise_phi_std: Phase noise standard deviation.
            noise_v_std: Velocity noise standard deviation.

        Returns:
            Dict with observable signals.
        """
        # Phase update
        self.phi = (
            self.phi + self.omega + random.gauss(0.0, noise_phi_std)
        ) % (2.0 * math.pi)

        # Alignment with ensemble
        a_k = alignment(self.phi, psi)

        # Track stats for micro/macro updates
        self.micro_align_sum += a_k
        self.micro_align_count += 1
        self.macro_align_sum += a_k
        self.macro_align_count += 1

        # Velocity update: damping * alignment - friction + noise
        # This is where temperament (lambd) biases acceleration
        self.v = (
            self.v
            + self.lambd * a_k  # Alignment accelerates toward group
            - dv * self.v  # Friction dampens velocity
            + random.gauss(0.0, noise_v_std)
        )

        # Tilt update
        self.theta = self.theta + self.v

        # Observable signals
        gain = 1.0 + 0.2 * a_k
        stability = max(0.0, 1.0 - 0.5 * abs(self.v))

        return {
            "name": self.name,
            "phi": self.phi,
            "align": a_k,
            "tilt": self.theta,
            "velocity": self.v,
            "lambd": self.lambd,
            "gain": gain,
            "stability": stability,
        }

    def do_micro_update(
        self,
        lambda_min: float = 0.01,
        lambda_max: float = 0.2,
        eta_lambda: float = 0.1,
    ) -> float:
        """
        Micro clock update: adjust damping based on recent alignment.

        Experts that have been in phase get more responsive (higher lambda).
        Out of phase experts drift toward lambda_min and become sluggish.

        Args:
            lambda_min: Minimum damping (sluggish).
            lambda_max: Maximum damping (nimble).
            eta_lambda: Learning rate for damping update.

        Returns:
            New lambda value.
        """
        if self.micro_align_count == 0:
            return self.lambd

        a_avg = self.micro_align_sum / self.micro_align_count
        self.micro_align_sum = 0.0
        self.micro_align_count = 0

        # Only reward in-phase behaviour
        x = max(0.0, a_avg)
        target = lambda_min + (lambda_max - lambda_min) * x

        # Smooth update toward target
        self.lambd = (1.0 - eta_lambda) * self.lambd + eta_lambda * target

        return self.lambd

    def do_macro_update(
        self,
        eta_s: float = 0.05,
        beta_s: float = 3.0,
    ) -> float:
        """
        Macro clock update: adjust reliability score and compute trust weight.

        Args:
            eta_s: Learning rate for reliability score.
            beta_s: Sigmoid sharpness for weight computation.

        Returns:
            Trust weight w_k.
        """
        if self.macro_align_count == 0:
            return sigmoid(beta_s * self.s)

        a_avg = self.macro_align_sum / self.macro_align_count
        self.macro_align_sum = 0.0
        self.macro_align_count = 0

        # Update slow reliability score
        self.s = (1.0 - eta_s) * self.s + eta_s * a_avg

        # Compute trust weight
        w = sigmoid(beta_s * self.s)

        return w


@dataclass
class TemperamentController:
    """
    Controller for per-expert temperament dynamics.

    Does not paint geometry. Only maintains clocks, computes Kuramoto R,
    and triggers micro/macro updates on each expert.

    The three clocks:
    - Fast: update phases, velocities, tilts
    - Micro: update damping (lambda) based on recent alignment
    - Macro: update reliability (s) and trust weights (w)
    """

    experts: List[AdaptiveExpert]

    # Clocks
    fast_clock: int = 0
    micro_clock: int = 0
    macro_clock: int = 0

    # Periods
    micro_period: int = 5
    macro_period: int = 4  # In micro ticks, so 4 * 5 = 20 fast ticks

    # Global stats
    micro_R_sum: float = 0.0
    micro_R_count: int = 0
    macro_R_sum: float = 0.0
    macro_R_count: int = 0

    # Hyperparameters
    dv: float = 0.05
    noise_phi_std: float = 0.01
    noise_v_std: float = 0.01
    lambda_min: float = 0.01
    lambda_max: float = 0.2
    eta_lambda: float = 0.1
    eta_s: float = 0.05
    beta_s: float = 3.0

    def tick(self) -> Dict[str, object]:
        """
        Advance one fast tick.

        Returns dict with tick info for logging.
        """
        self.fast_clock += 1

        # Compute Kuramoto order parameter
        phases = [e.phi for e in self.experts]
        R, psi = compute_kuramoto_R_and_psi(phases)

        # Track global R
        self.micro_R_sum += R
        self.micro_R_count += 1
        self.macro_R_sum += R
        self.macro_R_count += 1

        # Fast update for each expert
        expert_signals = [
            e.tick_fast(
                psi=psi,
                dv=self.dv,
                noise_phi_std=self.noise_phi_std,
                noise_v_std=self.noise_v_std,
            )
            for e in self.experts
        ]

        micro_event = False
        macro_event = False
        R_micro_avg = None
        R_macro_avg = None
        weights = None

        # Micro clock check
        if self.fast_clock % self.micro_period == 0:
            self.micro_clock += 1
            micro_event = True

            R_micro_avg = self.micro_R_sum / max(1, self.micro_R_count)
            self.micro_R_sum = 0.0
            self.micro_R_count = 0

            # Update damping for each expert
            for e in self.experts:
                e.do_micro_update(
                    self.lambda_min,
                    self.lambda_max,
                    self.eta_lambda,
                )

        # Macro clock check
        if self.micro_clock > 0 and self.micro_clock % self.macro_period == 0:
            # Only trigger if micro just incremented
            if micro_event:
                self.macro_clock += 1
                macro_event = True

                R_macro_avg = self.macro_R_sum / max(1, self.macro_R_count)
                self.macro_R_sum = 0.0
                self.macro_R_count = 0

                # Update reliability and weights for each expert
                weights = {}
                for e in self.experts:
                    w = e.do_macro_update(self.eta_s, self.beta_s)
                    weights[e.name] = w

        return {
            "fast_clock": self.fast_clock,
            "micro_clock": self.micro_clock,
            "macro_clock": self.macro_clock,
            "R": R,
            "psi": psi,
            "R_micro_avg": R_micro_avg,
            "R_macro_avg": R_macro_avg,
            "expert_signals": expert_signals,
            "micro_event": micro_event,
            "macro_event": macro_event,
            "weights": weights,
        }


def run_simulation_v3(
    seed: int = 42,
    num_experts: int = 5,
    num_ticks: int = 200,
    micro_period: int = 5,
    macro_period: int = 4,
) -> None:
    """
    Run the V3 per-expert temperament simulation.

    No global lens. Memory is in per-expert inertia and damping.
    """
    random.seed(seed)

    # Expert configurations
    names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"][:num_experts]
    omegas = [0.08, 0.10, 0.12, 0.09, 0.11][:num_experts]

    experts = [
        AdaptiveExpert(
            name=n,
            phi=random.uniform(0.0, 2.0 * math.pi),
            omega=w,
            theta=0.0,
            v=0.0,
            lambd=0.05,
        )
        for n, w in zip(names, omegas)
    ]

    controller = TemperamentController(
        experts=experts,
        micro_period=micro_period,
        macro_period=macro_period,
    )

    # Header
    print("=" * 80)
    print("CHRONOVISOR V3: PER-EXPERT TEMPERAMENT SIMULATION")
    print("=" * 80)
    print(f"Experts: {names}")
    print(f"Frequencies (ω): {omegas}")
    print(f"Periods: micro={micro_period} fast ticks, macro={macro_period} micro ticks")
    print("=" * 80)
    print()
    print("Legend:")
    print("  R = global Kuramoto coherence")
    print("  λ = per-expert damping (higher = more responsive to alignment)")
    print("  s = reliability score (slow-moving)")
    print("  w = trust weight (sigmoid of s)")
    print()

    # Run simulation
    for t in range(1, num_ticks + 1):
        info = controller.tick()
        R = info["R"]

        if info["micro_event"] or info["macro_event"]:
            label = "[micro]" if info["micro_event"] else ""
            if info["macro_event"]:
                label = "[MACRO]"

            # Compute average alignment and velocity
            avg_align = sum(s["align"] for s in info["expert_signals"]) / len(experts)
            avg_v = sum(abs(s["velocity"]) for s in info["expert_signals"]) / len(experts)

            print(
                f"t={t:3d} | R={R:.3f} ā={avg_align:+.3f} |v̄|={avg_v:.3f} | "
                f"clocks: {info['fast_clock']}/{info['micro_clock']}/{info['macro_clock']} "
                f"{label}"
            )

            # Show per-expert damping
            lambda_str = " ".join(f"{e.name}:λ={e.lambd:.3f}" for e in experts)
            print(f"       | {lambda_str}")

            # Show weights on macro events
            if info["weights"] is not None:
                w_str = " ".join(
                    f"{name}:w={w:.3f}" for name, w in info["weights"].items()
                )
                print(f"       | {w_str}")
            print()

    # Summary
    print("=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    print()
    print("Final expert states:")
    print("-" * 80)
    print(f"{'Name':<10} {'φ':>6} {'θ':>8} {'v':>8} {'λ':>6} {'s':>6} {'w':>6}")
    print("-" * 80)
    for e in experts:
        w = sigmoid(controller.beta_s * e.s)
        print(
            f"{e.name:<10} {e.phi:>6.2f} {e.theta:>8.3f} {e.v:>8.4f} "
            f"{e.lambd:>6.3f} {e.s:>6.3f} {w:>6.3f}"
        )
    print("-" * 80)


if __name__ == "__main__":
    run_simulation_v3(seed=42)
