"""
Chronovisor V4: Expert Bifurcation and Population Dynamics.

Building on V3's per-expert temperament, this version adds:
- Bifurcation: experts that drift far from home spawn offspring
- Culling: consistently anti-phase experts are pruned
- Population dynamics: the ensemble can grow and differentiate

Key insight: when an expert wanders too far, it doesn't get pulled back.
It becomes a new specialist, and a fresh expert spawns at the abandoned origin.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import copy


def compute_kuramoto_R_and_psi(phases: List[float]) -> Tuple[float, float]:
    """Compute Kuramoto order parameter R and mean phase ψ."""
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
    """Per-expert alignment with ensemble mean phase."""
    return math.cos(phi_k - psi)


# Global counter for unique expert IDs
_expert_id_counter = 0


def _next_expert_id() -> int:
    global _expert_id_counter
    _expert_id_counter += 1
    return _expert_id_counter


def reset_expert_id_counter():
    """Reset the counter (useful for tests)."""
    global _expert_id_counter
    _expert_id_counter = 0


@dataclass
class BifurcatingExpert:
    """
    Expert with bifurcation capability.

    Extends V3's AdaptiveExpert with:
    - theta_home: birth position (where this expert "belongs")
    - generation: how many bifurcations in lineage
    - parent_id: who spawned this expert (None for originals)
    - birth_tick: when this expert was created
    """

    name: str
    expert_id: int
    phi: float  # Kuramoto phase
    omega: float  # Intrinsic frequency

    # Trajectory state
    theta: float = 0.0  # Current tilt
    theta_home: float = 0.0  # Birth position
    v: float = 0.0  # Velocity

    # Temperament
    lambd: float = 0.05  # Damping

    # Reliability
    s: float = 0.0  # Slow reliability score

    # Lineage
    generation: int = 0
    parent_id: Optional[int] = None
    birth_tick: int = 0

    # Rolling stats
    micro_align_sum: float = 0.0
    micro_align_count: int = 0
    macro_align_sum: float = 0.0
    macro_align_count: int = 0

    # For tracking consecutive low reliability
    low_reliability_ticks: int = 0

    def drift_distance(self) -> float:
        """How far has this expert drifted from home?"""
        return abs(self.theta - self.theta_home)

    def tick_fast(
        self,
        psi: float,
        dv: float = 0.05,
        noise_phi_std: float = 0.01,
        noise_v_std: float = 0.01,
    ) -> Dict[str, float]:
        """Fast clock update."""
        # Phase update
        self.phi = (
            self.phi + self.omega + random.gauss(0.0, noise_phi_std)
        ) % (2.0 * math.pi)

        a_k = alignment(self.phi, psi)

        # Track stats
        self.micro_align_sum += a_k
        self.micro_align_count += 1
        self.macro_align_sum += a_k
        self.macro_align_count += 1

        # Velocity and tilt update
        self.v = (
            self.v
            + self.lambd * a_k
            - dv * self.v
            + random.gauss(0.0, noise_v_std)
        )
        self.theta = self.theta + self.v

        # Observables
        gain = 1.0 + 0.2 * a_k
        stability = max(0.0, 1.0 - 0.5 * abs(self.v))

        return {
            "name": self.name,
            "id": self.expert_id,
            "phi": self.phi,
            "align": a_k,
            "tilt": self.theta,
            "drift": self.drift_distance(),
            "velocity": self.v,
            "lambd": self.lambd,
            "gain": gain,
            "stability": stability,
            "generation": self.generation,
        }

    def do_micro_update(
        self,
        lambda_min: float = 0.01,
        lambda_max: float = 0.2,
        eta_lambda: float = 0.1,
    ) -> float:
        """Micro clock: adjust damping."""
        if self.micro_align_count == 0:
            return self.lambd

        a_avg = self.micro_align_sum / self.micro_align_count
        self.micro_align_sum = 0.0
        self.micro_align_count = 0

        x = max(0.0, a_avg)
        target = lambda_min + (lambda_max - lambda_min) * x
        self.lambd = (1.0 - eta_lambda) * self.lambd + eta_lambda * target

        return self.lambd

    def do_macro_update(
        self,
        eta_s: float = 0.05,
        beta_s: float = 3.0,
    ) -> float:
        """Macro clock: adjust reliability."""
        if self.macro_align_count == 0:
            return sigmoid(beta_s * self.s)

        a_avg = self.macro_align_sum / self.macro_align_count
        self.macro_align_sum = 0.0
        self.macro_align_count = 0

        self.s = (1.0 - eta_s) * self.s + eta_s * a_avg

        # Track low reliability
        if self.s < -0.2:
            self.low_reliability_ticks += 1
        else:
            self.low_reliability_ticks = 0

        return sigmoid(beta_s * self.s)

    def spawn_offspring(self, current_tick: int) -> "BifurcatingExpert":
        """
        Create offspring at this expert's old home position.

        The offspring inherits omega (with mutation) but starts fresh otherwise.
        """
        # Mutate omega slightly
        omega_mutation = random.gauss(0, 0.01)
        new_omega = max(0.01, self.omega + omega_mutation)

        offspring = BifurcatingExpert(
            name=f"{self.name}.{self.generation + 1}",
            expert_id=_next_expert_id(),
            phi=random.uniform(0, 2 * math.pi),
            omega=new_omega,
            theta=self.theta_home,  # Spawn at parent's old home
            theta_home=self.theta_home,
            v=0.0,
            lambd=0.05,  # Fresh temperament
            s=0.0,  # Fresh reliability
            generation=self.generation + 1,
            parent_id=self.expert_id,
            birth_tick=current_tick,
        )

        return offspring

    def settle_at_current_position(self):
        """This expert has founded a new cluster at its current position."""
        self.theta_home = self.theta
        # Slight name update to indicate settlement
        if "→" not in self.name:
            self.name = f"{self.name}→"


@dataclass
class EvolutionaryController:
    """
    Controller with bifurcation and culling.

    At macro clock:
    - Check each expert's drift from home
    - If drift > D_max and expert is coherent: BIFURCATE
    - If reliability < s_cull for too long: CULL (if population allows)
    """

    experts: List[BifurcatingExpert]

    # Clocks
    fast_clock: int = 0
    micro_clock: int = 0
    macro_clock: int = 0

    # Periods
    micro_period: int = 5
    macro_period: int = 4

    # Dynamics hyperparameters
    dv: float = 0.05
    noise_phi_std: float = 0.01
    noise_v_std: float = 0.01
    lambda_min: float = 0.01
    lambda_max: float = 0.2
    eta_lambda: float = 0.1
    eta_s: float = 0.05
    beta_s: float = 3.0

    # Bifurcation parameters
    D_max: float = 50.0  # Drift threshold for bifurcation
    s_bifurcate: float = 0.1  # Must be this reliable to bifurcate
    max_population: int = 20  # Cap on total experts

    # Culling parameters
    s_cull: float = -0.3  # Reliability below which expert may be culled
    cull_grace_ticks: int = 3  # Macro ticks of low reliability before culling
    min_population: int = 3  # Never cull below this

    # Global stats
    micro_R_sum: float = 0.0
    micro_R_count: int = 0

    # Event log
    events: List[Dict] = field(default_factory=list)

    def tick(self) -> Dict:
        """Advance one fast tick."""
        self.fast_clock += 1

        # Compute Kuramoto
        phases = [e.phi for e in self.experts]
        R, psi = compute_kuramoto_R_and_psi(phases)

        self.micro_R_sum += R
        self.micro_R_count += 1

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
        bifurcations = []
        culled = []

        # Micro clock
        if self.fast_clock % self.micro_period == 0:
            self.micro_clock += 1
            micro_event = True

            for e in self.experts:
                e.do_micro_update(self.lambda_min, self.lambda_max, self.eta_lambda)

        # Macro clock
        if self.micro_clock > 0 and self.micro_clock % self.macro_period == 0:
            if micro_event:
                self.macro_clock += 1
                macro_event = True

                # Update reliability for each expert
                weights = {}
                for e in self.experts:
                    w = e.do_macro_update(self.eta_s, self.beta_s)
                    weights[e.name] = w

                # Check for bifurcation
                bifurcations = self._check_bifurcation()

                # Check for culling
                culled = self._check_culling()

        return {
            "fast_clock": self.fast_clock,
            "micro_clock": self.micro_clock,
            "macro_clock": self.macro_clock,
            "R": R,
            "psi": psi,
            "population": len(self.experts),
            "expert_signals": expert_signals,
            "micro_event": micro_event,
            "macro_event": macro_event,
            "bifurcations": bifurcations,
            "culled": culled,
        }

    def _check_bifurcation(self) -> List[Dict]:
        """Check each expert for bifurcation eligibility."""
        bifurcations = []

        if len(self.experts) >= self.max_population:
            return bifurcations  # At capacity

        # Iterate over copy since we may modify the list
        for expert in list(self.experts):
            drift = expert.drift_distance()

            # Eligible: drifted far AND is reliable AND not already at capacity
            if (
                drift > self.D_max
                and expert.s > self.s_bifurcate
                and len(self.experts) < self.max_population
            ):
                # Spawn offspring at old home
                offspring = expert.spawn_offspring(self.fast_clock)
                self.experts.append(offspring)

                # Parent settles at new location
                old_home = expert.theta_home
                expert.settle_at_current_position()

                event = {
                    "type": "bifurcation",
                    "tick": self.fast_clock,
                    "parent": expert.name,
                    "parent_id": expert.expert_id,
                    "offspring": offspring.name,
                    "offspring_id": offspring.expert_id,
                    "old_home": old_home,
                    "new_home": expert.theta_home,
                    "drift": drift,
                }
                bifurcations.append(event)
                self.events.append(event)

        return bifurcations

    def _check_culling(self) -> List[Dict]:
        """Cull experts with consistently low reliability."""
        culled = []

        if len(self.experts) <= self.min_population:
            return culled  # Can't cull below minimum

        experts_to_remove = []

        for expert in self.experts:
            if (
                expert.s < self.s_cull
                and expert.low_reliability_ticks >= self.cull_grace_ticks
                and len(self.experts) - len(experts_to_remove) > self.min_population
            ):
                experts_to_remove.append(expert)

                event = {
                    "type": "culling",
                    "tick": self.fast_clock,
                    "name": expert.name,
                    "id": expert.expert_id,
                    "reliability": expert.s,
                    "generation": expert.generation,
                }
                culled.append(event)
                self.events.append(event)

        for expert in experts_to_remove:
            self.experts.remove(expert)

        return culled


def run_simulation_v4(
    seed: int = 42,
    num_experts: int = 5,
    num_ticks: int = 500,
    micro_period: int = 5,
    macro_period: int = 4,
    D_max: float = 50.0,
) -> None:
    """
    Run the V4 simulation with bifurcation and culling.
    """
    random.seed(seed)
    reset_expert_id_counter()

    # Initial experts
    names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"][:num_experts]
    omegas = [0.08, 0.10, 0.12, 0.09, 0.11][:num_experts]

    experts = [
        BifurcatingExpert(
            name=n,
            expert_id=_next_expert_id(),
            phi=random.uniform(0, 2 * math.pi),
            omega=w,
            theta=0.0,
            theta_home=0.0,
            v=0.0,
            lambd=0.05,
            generation=0,
            birth_tick=0,
        )
        for n, w in zip(names, omegas)
    ]

    controller = EvolutionaryController(
        experts=experts,
        micro_period=micro_period,
        macro_period=macro_period,
        D_max=D_max,
    )

    # Header
    print("=" * 85)
    print("CHRONOVISOR V4: EXPERT BIFURCATION & POPULATION DYNAMICS")
    print("=" * 85)
    print(f"Initial experts: {names}")
    print(f"D_max (drift threshold): {D_max}")
    print(f"Periods: micro={micro_period}, macro={macro_period * micro_period}")
    print("=" * 85)
    print()

    # Run simulation
    for t in range(1, num_ticks + 1):
        info = controller.tick()

        # Log bifurcation events immediately
        for bif in info["bifurcations"]:
            print(f"  🌱 BIFURCATION: {bif['parent']} → spawned {bif['offspring']} "
                  f"(drift={bif['drift']:.1f})")

        # Log culling events
        for cull in info["culled"]:
            print(f"  💀 CULLED: {cull['name']} (s={cull['reliability']:.3f})")

        # Regular logging at macro events
        if info["macro_event"]:
            pop = info["population"]
            R = info["R"]

            # Summary stats
            avg_drift = sum(e.drift_distance() for e in controller.experts) / pop
            avg_s = sum(e.s for e in controller.experts) / pop
            generations = [e.generation for e in controller.experts]
            max_gen = max(generations) if generations else 0

            print(
                f"t={t:3d} [MACRO] | R={R:.3f} | pop={pop:2d} | "
                f"avg_drift={avg_drift:6.1f} | avg_s={avg_s:+.3f} | max_gen={max_gen}"
            )

    # Final summary
    print()
    print("=" * 85)
    print("SIMULATION COMPLETE")
    print("=" * 85)
    print(f"Final population: {len(controller.experts)}")
    print(f"Total bifurcations: {sum(1 for e in controller.events if e['type'] == 'bifurcation')}")
    print(f"Total cullings: {sum(1 for e in controller.events if e['type'] == 'culling')}")
    print()

    # Expert census
    print("Expert Census:")
    print("-" * 85)
    print(f"{'Name':<15} {'Gen':>3} {'θ':>8} {'θ_home':>8} {'Drift':>7} {'λ':>6} {'s':>6} {'w':>6}")
    print("-" * 85)

    for e in sorted(controller.experts, key=lambda x: -x.s):
        w = sigmoid(controller.beta_s * e.s)
        drift = e.drift_distance()
        print(
            f"{e.name:<15} {e.generation:>3} {e.theta:>8.1f} {e.theta_home:>8.1f} "
            f"{drift:>7.1f} {e.lambd:>6.3f} {e.s:>6.3f} {w:>6.3f}"
        )

    print("-" * 85)

    # Lineage summary
    print()
    print("Lineage Tree:")
    originals = [e for e in controller.experts if e.parent_id is None]
    descendants = [e for e in controller.experts if e.parent_id is not None]

    for orig in originals:
        print(f"  {orig.name} (gen 0)")
        # Find descendants
        for desc in sorted(descendants, key=lambda x: x.generation):
            if desc.name.startswith(orig.name.split("→")[0]):
                indent = "    " * desc.generation
                print(f"  {indent}└─ {desc.name} (gen {desc.generation})")


if __name__ == "__main__":
    run_simulation_v4(seed=42, num_ticks=500)
