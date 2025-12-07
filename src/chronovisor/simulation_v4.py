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
class Governor:
    """
    Meta-controller that gates bifurcation and culling.

    The Governor observes ecosystem-level signals and decides:
    - gate_spawn: whether bifurcation is allowed
    - gate_cull: whether culling is allowed
    - mode: "normal" or "outside_box"

    Outside-box mode triggers when the system is stagnant:
    near capacity but no progress. It relaxes thresholds to
    encourage exploration.
    """

    # Population bounds
    max_population: int = 20
    min_population: int = 3

    # Coherence thresholds
    R_high: float = 0.8  # Above this, population is "in agreement"
    R_low: float = 0.3   # Below this, population is "chaotic"

    # Reliability thresholds
    s_spawn_threshold: float = 0.1  # Must be this reliable to spawn
    s_cull_threshold: float = -0.3  # Below this for too long → cull

    # Stagnation detection
    stagnation_window: int = 5  # Macro ticks to look back
    progress_threshold: float = 0.02  # R must improve by this much

    # History buffers
    R_history: List[float] = field(default_factory=list)
    s_avg_history: List[float] = field(default_factory=list)

    # Current state
    mode: str = "normal"
    mode_ticks_remaining: int = 0  # How long to stay in current mode

    # Outside-box parameters
    outside_box_duration: int = 10  # Macro ticks in outside_box mode
    relaxed_D_max_factor: float = 0.7  # Multiply D_max by this in outside_box
    relaxed_s_spawn_factor: float = 0.5  # Multiply s_spawn threshold
    noise_boost_factor: float = 2.0  # Multiply noise in outside_box

    def update(
        self,
        R: float,
        N: int,
        s_values: List[float],
        d_values: List[float],
    ) -> Tuple[bool, bool, str]:
        """
        Update governor state and return gates.

        Args:
            R: Current Kuramoto coherence
            N: Current population size
            s_values: List of expert reliability scores
            d_values: List of expert drift distances

        Returns:
            (gate_spawn, gate_cull, mode)
        """
        # Update history
        self.R_history.append(R)
        if len(self.R_history) > self.stagnation_window:
            self.R_history.pop(0)

        s_avg = sum(s_values) / len(s_values) if s_values else 0.0
        self.s_avg_history.append(s_avg)
        if len(self.s_avg_history) > self.stagnation_window:
            self.s_avg_history.pop(0)

        # Check for mode transition
        if self.mode == "outside_box":
            self.mode_ticks_remaining -= 1
            if self.mode_ticks_remaining <= 0:
                self.mode = "normal"
        else:
            # Check for stagnation
            if self._is_stagnant(N):
                self.mode = "outside_box"
                self.mode_ticks_remaining = self.outside_box_duration

        # Compute gates
        gate_spawn = self._compute_gate_spawn(R, N, s_avg)
        gate_cull = self._compute_gate_cull(R, N)

        return gate_spawn, gate_cull, self.mode

    def _is_stagnant(self, N: int) -> bool:
        """
        Detect stagnation: near capacity but no coherence progress.
        """
        # Need enough history
        if len(self.R_history) < self.stagnation_window:
            return False

        # Must be near capacity
        capacity_ratio = N / self.max_population
        if capacity_ratio < 0.8:
            return False

        # Check for progress in R
        R_start = self.R_history[0]
        R_end = self.R_history[-1]
        progress = R_end - R_start

        # Stagnant if no improvement (or decline) in coherence
        return progress < self.progress_threshold

    def _compute_gate_spawn(self, R: float, N: int, s_avg: float) -> bool:
        """
        Should bifurcation be allowed?

        Open when:
        - Below capacity
        - Either: R is low (need diversity) OR s_avg is positive (healthy)
        - In outside_box mode: always open if below capacity
        """
        if N >= self.max_population:
            return False

        if self.mode == "outside_box":
            return True  # Encourage exploration

        # Normal mode: spawn when R is low (need diversity) or when healthy
        return R < self.R_high or s_avg > 0.0

    def _compute_gate_cull(self, R: float, N: int) -> bool:
        """
        Should culling be allowed?

        Open when:
        - Above minimum population
        - R is reasonable (not in chaos)
        - In outside_box mode: more conservative about culling
        """
        if N <= self.min_population:
            return False

        if self.mode == "outside_box":
            # More conservative: only cull if well above minimum
            return N > self.min_population + 2

        # Normal mode: cull when not in chaos
        return R > self.R_low

    def get_effective_params(
        self,
        base_D_max: float,
        base_s_bifurcate: float,
        base_noise_phi_std: float,
        base_noise_v_std: float,
    ) -> Dict[str, float]:
        """
        Get effective parameters, potentially relaxed in outside_box mode.
        """
        if self.mode == "outside_box":
            return {
                "D_max": base_D_max * self.relaxed_D_max_factor,
                "s_bifurcate": base_s_bifurcate * self.relaxed_s_spawn_factor,
                "noise_phi_std": base_noise_phi_std * self.noise_boost_factor,
                "noise_v_std": base_noise_v_std * self.noise_boost_factor,
            }
        else:
            return {
                "D_max": base_D_max,
                "s_bifurcate": base_s_bifurcate,
                "noise_phi_std": base_noise_phi_std,
                "noise_v_std": base_noise_v_std,
            }


@dataclass
class EvolutionaryController:
    """
    Controller with bifurcation and culling, gated by a Governor.

    At macro clock:
    - Governor decides gate_spawn, gate_cull, and mode
    - If gate_spawn: check each expert's drift from home
    - If drift > D_max and expert is coherent: BIFURCATE
    - If gate_cull and reliability < s_cull for too long: CULL
    """

    experts: List[BifurcatingExpert]

    # Governor (meta-controller)
    governor: Governor = field(default_factory=Governor)

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

    # Bifurcation parameters (base values, may be modified by Governor)
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

    # Current governor state (for external access)
    current_gate_spawn: bool = True
    current_gate_cull: bool = True
    current_mode: str = "normal"

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

        # Get effective parameters from Governor (may be modified in outside_box mode)
        effective = self.governor.get_effective_params(
            self.D_max,
            self.s_bifurcate,
            self.noise_phi_std,
            self.noise_v_std,
        )

        # Fast update for each expert
        expert_signals = [
            e.tick_fast(
                psi=psi,
                dv=self.dv,
                noise_phi_std=effective["noise_phi_std"],
                noise_v_std=effective["noise_v_std"],
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

                # Update Governor and get gates
                s_values = [e.s for e in self.experts]
                d_values = [e.drift_distance() for e in self.experts]
                gate_spawn, gate_cull, mode = self.governor.update(
                    R=R,
                    N=len(self.experts),
                    s_values=s_values,
                    d_values=d_values,
                )

                # Store current state
                self.current_gate_spawn = gate_spawn
                self.current_gate_cull = gate_cull
                self.current_mode = mode

                # Check for bifurcation (if gate is open)
                bifurcations = self._check_bifurcation(
                    gate_spawn, effective["D_max"], effective["s_bifurcate"]
                )

                # Check for culling (if gate is open)
                culled = self._check_culling(gate_cull)

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
            "gate_spawn": self.current_gate_spawn,
            "gate_cull": self.current_gate_cull,
            "mode": self.current_mode,
        }

    def _check_bifurcation(
        self,
        gate_spawn: bool,
        effective_D_max: float,
        effective_s_bifurcate: float,
    ) -> List[Dict]:
        """Check each expert for bifurcation eligibility."""
        bifurcations = []

        # Gate must be open
        if not gate_spawn:
            return bifurcations

        if len(self.experts) >= self.max_population:
            return bifurcations  # At capacity

        # Iterate over copy since we may modify the list
        for expert in list(self.experts):
            drift = expert.drift_distance()

            # Eligible: drifted far AND is reliable AND not already at capacity
            if (
                drift > effective_D_max
                and expert.s > effective_s_bifurcate
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
                    "mode": self.current_mode,
                }
                bifurcations.append(event)
                self.events.append(event)

        return bifurcations

    def _check_culling(self, gate_cull: bool) -> List[Dict]:
        """Cull experts with consistently low reliability."""
        culled = []

        # Gate must be open
        if not gate_cull:
            return culled

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
                    "mode": self.current_mode,
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
    Run the V4 simulation with bifurcation, culling, and Governor.
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

    # Create Governor with matching parameters
    governor = Governor(
        max_population=20,
        min_population=3,
    )

    controller = EvolutionaryController(
        experts=experts,
        governor=governor,
        micro_period=micro_period,
        macro_period=macro_period,
        D_max=D_max,
    )

    # Header
    print("=" * 90)
    print("CHRONOVISOR V4: EXPERT BIFURCATION & POPULATION DYNAMICS (with Governor)")
    print("=" * 90)
    print(f"Initial experts: {names}")
    print(f"D_max (drift threshold): {D_max}")
    print(f"Periods: micro={micro_period}, macro={macro_period * micro_period}")
    print("Governor: gates spawn/cull, detects stagnation, triggers outside_box mode")
    print("=" * 90)
    print()

    outside_box_events = 0

    # Run simulation
    for t in range(1, num_ticks + 1):
        info = controller.tick()

        # Log bifurcation events immediately
        for bif in info["bifurcations"]:
            mode_tag = " [OOB]" if bif.get("mode") == "outside_box" else ""
            print(f"  🌱 BIFURCATION{mode_tag}: {bif['parent']} → spawned {bif['offspring']} "
                  f"(drift={bif['drift']:.1f})")

        # Log culling events
        for cull in info["culled"]:
            mode_tag = " [OOB]" if cull.get("mode") == "outside_box" else ""
            print(f"  💀 CULLED{mode_tag}: {cull['name']} (s={cull['reliability']:.3f})")

        # Regular logging at macro events
        if info["macro_event"]:
            pop = info["population"]
            R = info["R"]

            # Summary stats
            avg_drift = sum(e.drift_distance() for e in controller.experts) / pop
            avg_s = sum(e.s for e in controller.experts) / pop
            generations = [e.generation for e in controller.experts]
            max_gen = max(generations) if generations else 0

            # Governor state
            mode = info.get("mode", "normal")
            gate_spawn = "✓" if info.get("gate_spawn", True) else "✗"
            gate_cull = "✓" if info.get("gate_cull", True) else "✗"

            if mode == "outside_box":
                outside_box_events += 1
                mode_display = "OUTSIDE_BOX"
            else:
                mode_display = "normal"

            print(
                f"t={t:3d} [MACRO] | R={R:.3f} | pop={pop:2d} | "
                f"avg_drift={avg_drift:6.1f} | avg_s={avg_s:+.3f} | max_gen={max_gen} | "
                f"spawn={gate_spawn} cull={gate_cull} mode={mode_display}"
            )

    # Final summary
    print()
    print("=" * 90)
    print("SIMULATION COMPLETE")
    print("=" * 90)
    print(f"Final population: {len(controller.experts)}")
    total_bifs = sum(1 for e in controller.events if e['type'] == 'bifurcation')
    oob_bifs = sum(1 for e in controller.events if e['type'] == 'bifurcation' and e.get('mode') == 'outside_box')
    total_culls = sum(1 for e in controller.events if e['type'] == 'culling')
    oob_culls = sum(1 for e in controller.events if e['type'] == 'culling' and e.get('mode') == 'outside_box')
    print(f"Total bifurcations: {total_bifs} ({oob_bifs} in outside_box mode)")
    print(f"Total cullings: {total_culls} ({oob_culls} in outside_box mode)")
    print(f"Outside-box macro events: {outside_box_events}")
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
