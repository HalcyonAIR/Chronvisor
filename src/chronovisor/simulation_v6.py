"""
Chronovisor V6: Cultural Transmission Between Experts.

Building on V4/V5's ecology and governance, this version adds:
- Motifs: shared strategies that emerge from clusters of successful experts
- Cultural clock: slower than macro, identifies and updates motifs
- Cultural transmission: motifs nudge experts' temperament and behavior
- Cultural capital: experts aligned with strong motifs gain reliability

Key insight: genetic/ecological change (V4) determines WHO exists,
temperament/physiology (V3) determines HOW they behave moment-to-moment,
and cultural change (V6) determines WHAT strategies they share and learn.

The cultural layer is the slowest and weakest—it biases but doesn't overwrite.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set


def compute_kuramoto_R_and_psi(phases: List[float]) -> Tuple[float, float]:
    """Compute Kuramoto order parameter R and mean phase psi."""
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


# Global counter for unique IDs
_expert_id_counter = 0
_motif_id_counter = 0


def _next_expert_id() -> int:
    global _expert_id_counter
    _expert_id_counter += 1
    return _expert_id_counter


def _next_motif_id() -> int:
    global _motif_id_counter
    _motif_id_counter += 1
    return _motif_id_counter


def reset_id_counters():
    """Reset counters (useful for tests)."""
    global _expert_id_counter, _motif_id_counter
    _expert_id_counter = 0
    _motif_id_counter = 0


@dataclass
class Lens:
    """
    Global lens that modulates expert drift based on agreement with flow.

    The lens tracks a smoothed orientation L representing "where the global
    flow is going" in theta-space. Experts whose drift agrees with L get
    amplified; those opposing get damped. The effect is gated by coherence R.

    Key principle: the lens does NOT choose outcomes—it rescales F_drift
    per expert based on agreement with global flow. Soft, bounded, reversible.
    """

    # Lens orientation (EMA of mean expert drift)
    L: float = 0.0

    # Smoothing factor for L updates
    eta_L: float = 0.1

    # Lens strength (how much gain can deviate from 1.0)
    gamma_lens: float = 0.3

    # Gain bounds (keeps system stable)
    g_min: float = 0.5
    g_max: float = 1.5

    def update(self, drifts: List[float]) -> None:
        """
        Update lens orientation from expert drift signals.

        L_t = (1 - eta_L) * L_{t-1} + eta_L * mean(drifts)
        """
        if not drifts:
            return
        mean_drift = sum(drifts) / len(drifts)
        self.L = (1.0 - self.eta_L) * self.L + self.eta_L * mean_drift

    def compute_gain(self, raw_drift: float, R: float) -> Tuple[float, float]:
        """
        Compute lens gain for an expert given its raw drift and global coherence.

        Returns (gain, alpha) where:
        - gain: multiplicative factor for F_drift
        - alpha: agreement score in [-1, 1]

        Formula:
        - alpha = sign agreement between raw_drift and L (normalized)
        - g = 1 + gamma_lens * R * alpha
        - g clamped to [g_min, g_max]
        """
        epsilon = 1e-8

        # Compute agreement with lens orientation
        if abs(self.L) < epsilon or abs(raw_drift) < epsilon:
            alpha = 0.0
        else:
            # Normalized product gives sign agreement in [-1, 1]
            alpha = (raw_drift * self.L) / (abs(raw_drift) * abs(self.L))

        # Compute gain, gated by coherence R
        g = 1.0 + self.gamma_lens * R * alpha

        # Clamp to bounds
        g = max(self.g_min, min(self.g_max, g))

        return g, alpha

    def get_params(self) -> Tuple[float, float, float, float]:
        """Return parameters needed by experts: (L, gamma_lens, g_min, g_max)."""
        return self.L, self.gamma_lens, self.g_min, self.g_max


@dataclass
class CulturalExpert:
    """
    Expert with cultural capabilities and pressure-based dynamics.

    Extends BifurcatingExpert with:
    - motif_ids: which motifs this expert currently aligns with
    - motif_affinity: strength of connection to each motif
    - cultural_capital: bonus from strong motif membership

    Dynamics governed by pressure fields:
    - F_drift: expert's own belief (lambda * alignment)
    - F_home: rubber band to theta_home
    - F_culture: pull toward affiliated motifs
    - F_safety: (stub) avoidance of unsafe regions
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

    # Culling tracking
    low_reliability_ticks: int = 0

    # Cultural layer (V6)
    motif_ids: Set[int] = field(default_factory=set)
    motif_affinity: Dict[int, float] = field(default_factory=dict)  # motif_id -> affinity
    cultural_capital: float = 0.0  # Bonus from motif membership

    # Pressure constants (can be overridden per-expert if needed)
    k_home: float = 0.01  # Home pressure stiffness
    k_safety: float = 0.1  # Safety pressure stiffness (stub)

    # === V6.1: Prediction Success Axis ===
    # Δs tracking (prediction success)
    s_prev: float = 0.0           # Previous s value
    delta_s: float = 0.0          # Recent Δs
    delta_s_ema: float = 0.0      # Smoothed Δs for gating

    # Absorption statistics (lens feedback)
    g_lens_ema: float = 1.0       # EMA of lens gain
    R_ema: float = 0.0            # EMA of coherence
    alpha_lens_ema: float = 0.0   # EMA of lens alignment

    # Absorption rates
    eta_theta_home: float = 0.02  # Slow identity shift
    eta_k_home: float = 0.01      # Very slow adventurousness change
    eta_absorb_stats: float = 0.1 # EMA rate for absorption stats
    eta_delta_s: float = 0.2      # EMA rate for Δs

    # Diversity bonus
    diversity_bonus: float = 0.0  # Reward for correct outliers
    diversity_bonus_rate: float = 0.05

    def drift_distance(self) -> float:
        """How far has this expert drifted from home?"""
        return abs(self.theta - self.theta_home)

    def effective_reliability(self) -> float:
        """Reliability including cultural capital and diversity bonus."""
        return self.s + self.cultural_capital + self.diversity_bonus

    # === V6.1: Prediction Success Methods ===

    def update_delta_s(self):
        """Track change in reliability (called after s updates)."""
        self.delta_s = self.s - self.s_prev
        self.delta_s_ema = (1 - self.eta_delta_s) * self.delta_s_ema + self.eta_delta_s * self.delta_s
        self.s_prev = self.s

    def update_absorption_stats(self, g_lens: float, R: float, alpha_lens: float):
        """Update running statistics for absorption decisions."""
        eta = self.eta_absorb_stats
        self.g_lens_ema = (1 - eta) * self.g_lens_ema + eta * g_lens
        self.R_ema = (1 - eta) * self.R_ema + eta * R
        self.alpha_lens_ema = (1 - eta) * self.alpha_lens_ema + eta * alpha_lens

    def maybe_absorb(self) -> Dict[str, float]:
        """
        Absorb lens pressure into deep parameters IF conditions met.

        Three conditions required:
        1. System is coherent (R high)
        2. Lens is giving clear signal (g ≠ 1)
        3. This expert is improving (Δs > 0) ← THE KEY GATE

        Returns diagnostics dict.
        """
        absorb_info = {
            "absorbed": False,
            "delta_theta_home": 0.0,
            "delta_k_home": 0.0,
            "reason": "none",
        }

        # Gating conditions
        R_threshold = 0.5
        g_threshold = 0.05
        delta_s_threshold = 0.0

        if self.R_ema < R_threshold:
            absorb_info["reason"] = "R_too_low"
            return absorb_info

        if abs(self.g_lens_ema - 1.0) < g_threshold:
            absorb_info["reason"] = "g_near_1"
            return absorb_info

        if self.delta_s_ema <= delta_s_threshold:
            absorb_info["reason"] = "delta_s_not_positive"
            return absorb_info

        # All conditions met - absorb!
        absorb_info["absorbed"] = True
        absorb_info["reason"] = "success"

        # Success weight: how much we're improving
        success_weight = max(0, self.delta_s_ema)

        # θ_home absorption: drift becomes identity
        drift = self.theta - self.theta_home
        delta_home = (
            self.eta_theta_home
            * drift
            * (self.g_lens_ema - 1.0)
            * self.R_ema
            * success_weight
        )
        # Rate limit
        delta_home = max(-0.5, min(0.5, delta_home))
        self.theta_home += delta_home
        absorb_info["delta_theta_home"] = delta_home

        # k_home adaptation: success while far → more adventurous
        drift_dist = abs(drift)
        delta_k = (
            -self.eta_k_home
            * drift_dist
            * (self.g_lens_ema - 1.0)
            * self.R_ema
            * success_weight
        )
        # Bound k_home
        new_k = self.k_home + delta_k
        new_k = max(0.001, min(0.1, new_k))
        absorb_info["delta_k_home"] = new_k - self.k_home
        self.k_home = new_k

        return absorb_info

    def compute_diversity_bonus(self) -> float:
        """
        Reward experts who were damped but turned out correct.

        This is the "feet someday" mechanism:
        - If lens damped you (g < 1), you're an outlier
        - If you're still improving (Δs > 0), you're a CORRECT outlier
        - You get a diversity bonus that feeds into effective_reliability
        """
        # Was I damped? (lens didn't like my direction)
        damping_threshold = 0.95
        was_damped = self.g_lens_ema < damping_threshold

        # Was I right anyway?
        success_threshold = 0.02
        was_correct = self.delta_s_ema > success_threshold

        if was_damped and was_correct:
            # I resisted consensus and was vindicated
            outlier_strength = 1.0 - self.g_lens_ema  # How much I was damped
            success_strength = self.delta_s_ema       # How right I was
            self.diversity_bonus = self.diversity_bonus_rate * outlier_strength * success_strength
        else:
            # Decay diversity bonus when not actively being a correct outlier
            self.diversity_bonus *= 0.9

        return self.diversity_bonus

    def tick_fast(
        self,
        psi: float,
        dv: float = 0.05,
        noise_phi_std: float = 0.01,
        noise_v_std: float = 0.01,
        motifs: Optional[Dict[int, "Motif"]] = None,
        # Lens parameters
        L: float = 0.0,
        R: float = 0.0,
        gamma_lens: float = 0.3,
        g_min: float = 0.5,
        g_max: float = 1.5,
    ) -> Dict[str, float]:
        """
        Fast clock update using pressure-field dynamics with lens modulation.

        Pressures:
        - F_drift: expert's own intent (lambda * alignment), modulated by lens
        - F_home: rubber band to birth position
        - F_culture: pull toward affiliated motifs
        - F_safety: avoidance of unsafe regions (stub)

        Lens:
        - Computes raw_drift = lambda * alignment
        - Computes agreement alpha with lens orientation L
        - Applies gain g = 1 + gamma_lens * R * alpha (clamped)
        - F_drift = g * raw_drift
        """
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

        # === PRESSURE FRAMEWORK WITH LENS ===

        # 1. Raw drift (expert's proposal before lens)
        raw_drift = self.lambd * a_k

        # 2. Lens gain calculation
        epsilon = 1e-8
        if abs(L) < epsilon or abs(raw_drift) < epsilon:
            alpha_lens = 0.0
        else:
            # Normalized product gives sign agreement in [-1, 1]
            alpha_lens = (raw_drift * L) / (abs(raw_drift) * abs(L))

        # Gain is gated by coherence R
        g_lens = 1.0 + gamma_lens * R * alpha_lens
        g_lens = max(g_min, min(g_max, g_lens))

        # 3. Lensed drift pressure
        F_drift = g_lens * raw_drift

        # 4. Home pressure (rubber band to theta_home)
        F_home = -self.k_home * (self.theta - self.theta_home)

        # 5. Cultural pressure (pull toward affiliated motifs)
        F_culture = 0.0
        if motifs and self.motif_ids:
            for motif_id in self.motif_ids:
                if motif_id in motifs:
                    m = motifs[motif_id]
                    w = self.motif_affinity.get(motif_id, 0.0)
                    # Pull toward motif center, weighted by affinity and motif's alpha
                    F_culture += -w * m.alpha_theta * (self.theta - m.theta_center)

        # 6. Safety pressure (stub - can be expanded later)
        F_safety = 0.0

        # Total pressure
        F_total = F_drift + F_home + F_culture + F_safety

        # Apply dynamics: v += F_total - damping + noise
        self.v = self.v + F_total - dv * self.v + random.gauss(0.0, noise_v_std)
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
            "cultural_capital": self.cultural_capital,
            "num_motifs": len(self.motif_ids),
            # Pressure diagnostics
            "raw_drift": raw_drift,
            "F_drift": F_drift,
            "F_home": F_home,
            "F_culture": F_culture,
            "F_total": F_total,
            # Lens diagnostics
            "g_lens": g_lens,
            "alpha_lens": alpha_lens,
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

    def spawn_offspring(self, current_tick: int) -> "CulturalExpert":
        """Create offspring at this expert's old home position."""
        omega_mutation = random.gauss(0, 0.01)
        new_omega = max(0.01, self.omega + omega_mutation)

        offspring = CulturalExpert(
            name=f"{self.name}.{self.generation + 1}",
            expert_id=_next_expert_id(),
            phi=random.uniform(0, 2 * math.pi),
            omega=new_omega,
            theta=self.theta_home,
            theta_home=self.theta_home,
            v=0.0,
            lambd=0.05,
            s=0.0,
            generation=self.generation + 1,
            parent_id=self.expert_id,
            birth_tick=current_tick,
            # Start fresh culturally
            motif_ids=set(),
            motif_affinity={},
            cultural_capital=0.0,
            # Inherit pressure constants from parent
            k_home=self.k_home,
            k_safety=self.k_safety,
        )

        return offspring

    def settle_at_current_position(self):
        """This expert has founded a new cluster at its current position."""
        self.theta_home = self.theta
        if ">" not in self.name:
            self.name = f"{self.name}>"


@dataclass
class Motif:
    """
    A shared strategy that emerges from clusters of successful experts.

    Motifs are slow, smoothed summaries of successful local cultures.
    They act as "teachers" that exert cultural pressure on nearby experts,
    pulling them toward proven behaviors in theta-space.
    """

    motif_id: int
    name: str

    # Location in temperament space
    theta_center: float  # Mean theta of supporting experts

    # Canonical reliability
    S: float  # Mean s of supporting experts (how good this motif is)

    # Canonical temperament sketch
    mean_lambda: float  # Typical damping of this culture
    mean_abs_v: float  # Typical velocity magnitude
    var_theta: float  # How spread out the cluster is

    # Pressure parameters
    alpha_theta: float = 0.01  # Cultural pressure stiffness (how strongly this motif pulls)

    # Support set
    support_ids: Set[int] = field(default_factory=set)

    # Lifecycle
    age: int = 0  # Cultural ticks since formation
    birth_tick: int = 0

    # Stability tracking
    s_history: List[float] = field(default_factory=list)
    stability_window: int = 5

    def support_size(self) -> int:
        """Number of experts currently aligned with this motif."""
        return len(self.support_ids)

    def is_stable(self) -> bool:
        """Is this motif reliably successful over time?"""
        if len(self.s_history) < self.stability_window:
            return False
        return all(s > 0.0 for s in self.s_history[-self.stability_window:])

    def update_from_cluster(
        self,
        theta_values: List[float],
        s_values: List[float],
        lambda_values: List[float],
        v_values: List[float],
        expert_ids: List[int],
        eta_motif: float = 0.1,
    ):
        """Update motif statistics from its supporting cluster (slow EMA)."""
        if not theta_values:
            return

        # Compute cluster stats
        new_theta = sum(theta_values) / len(theta_values)
        new_S = sum(s_values) / len(s_values)
        new_lambda = sum(lambda_values) / len(lambda_values)
        new_abs_v = sum(abs(v) for v in v_values) / len(v_values)
        new_var = sum((t - new_theta) ** 2 for t in theta_values) / len(theta_values)

        # Slow EMA update
        self.theta_center = (1 - eta_motif) * self.theta_center + eta_motif * new_theta
        self.S = (1 - eta_motif) * self.S + eta_motif * new_S
        self.mean_lambda = (1 - eta_motif) * self.mean_lambda + eta_motif * new_lambda
        self.mean_abs_v = (1 - eta_motif) * self.mean_abs_v + eta_motif * new_abs_v
        self.var_theta = (1 - eta_motif) * self.var_theta + eta_motif * new_var

        # Update support set
        self.support_ids = set(expert_ids)

        # Track S history
        self.s_history.append(self.S)
        if len(self.s_history) > self.stability_window * 2:
            self.s_history.pop(0)

        self.age += 1


@dataclass
class CulturalController:
    """
    Controller for cultural dynamics.

    Runs on the cultural clock (slower than macro).
    - Clusters experts in theta-space
    - Forms/updates motifs from dense, reliable clusters
    - Applies motif->expert teaching (slow nudges)
    - Prunes dead motifs
    """

    motifs: List[Motif] = field(default_factory=list)

    # Clustering parameters
    theta_cluster_radius: float = 30.0  # How close experts must be to cluster
    min_cluster_size: int = 2  # Minimum experts to form a motif
    min_cluster_reliability: float = 0.05  # Minimum mean s to form a motif

    # Teaching parameters
    D_culture: float = 50.0  # Max distance for a motif to influence an expert
    eta_cultural_lambda: float = 0.02  # How fast experts learn lambda from motifs
    eta_cultural_theta: float = 0.005  # How fast experts are pulled toward motifs
    cultural_capital_rate: float = 0.01  # How fast cultural capital accrues

    # Outside-box overrides
    cross_cluster_enabled: bool = False  # Allow merging distant clusters
    boosted_learning_for_low_s: bool = False  # Faster learning for struggling experts

    # Event log
    events: List[Dict] = field(default_factory=list)

    def cultural_tick(
        self,
        experts: List[CulturalExpert],
        current_tick: int,
        mode: str = "normal",
    ) -> Dict:
        """
        Run one cultural tick.

        1. Cluster experts in theta-space
        2. Form/update motifs from reliable clusters
        3. Apply motif->expert teaching
        4. Prune dead motifs
        """
        # Configure for mode
        if mode == "outside_box":
            self.cross_cluster_enabled = True
            self.boosted_learning_for_low_s = True
        else:
            self.cross_cluster_enabled = False
            self.boosted_learning_for_low_s = False

        # Step 1: Cluster experts
        clusters = self._cluster_experts(experts)

        # Step 2: Form/update motifs
        new_motifs = []
        updated_motifs = []
        for cluster in clusters:
            if len(cluster) < self.min_cluster_size:
                continue

            # Compute cluster stats
            theta_values = [e.theta for e in cluster]
            s_values = [e.s for e in cluster]
            lambda_values = [e.lambd for e in cluster]
            v_values = [e.v for e in cluster]
            expert_ids = [e.expert_id for e in cluster]

            mean_s = sum(s_values) / len(s_values)
            if mean_s < self.min_cluster_reliability:
                continue

            # Find nearby existing motif
            mean_theta = sum(theta_values) / len(theta_values)
            nearby_motif = self._find_nearby_motif(mean_theta)

            if nearby_motif is None:
                # Spawn new motif
                motif = self._spawn_motif(
                    theta_values, s_values, lambda_values, v_values,
                    expert_ids, current_tick
                )
                self.motifs.append(motif)
                new_motifs.append(motif.name)

                self.events.append({
                    "type": "motif_spawn",
                    "tick": current_tick,
                    "motif_name": motif.name,
                    "motif_id": motif.motif_id,
                    "theta_center": motif.theta_center,
                    "S": motif.S,
                    "support_size": motif.support_size(),
                    "mode": mode,
                })
            else:
                # Update existing motif
                nearby_motif.update_from_cluster(
                    theta_values, s_values, lambda_values, v_values, expert_ids
                )
                updated_motifs.append(nearby_motif.name)

        # Step 3: Apply motif->expert teaching
        taught = self._apply_teaching(experts)

        # Step 4: Prune dead motifs
        pruned = self._prune_dead_motifs(experts, current_tick, mode)

        return {
            "num_motifs": len(self.motifs),
            "new_motifs": new_motifs,
            "updated_motifs": updated_motifs,
            "pruned_motifs": pruned,
            "experts_taught": taught,
            "mode": mode,
        }

    def _cluster_experts(
        self,
        experts: List[CulturalExpert],
    ) -> List[List[CulturalExpert]]:
        """
        Simple greedy clustering in theta-space.

        Returns list of clusters, where each cluster is a list of experts.
        """
        if not experts:
            return []

        # Sort by theta
        sorted_experts = sorted(experts, key=lambda e: e.theta)
        clusters = []
        current_cluster = [sorted_experts[0]]

        for expert in sorted_experts[1:]:
            # Check if expert is close to cluster center
            cluster_theta = sum(e.theta for e in current_cluster) / len(current_cluster)

            if abs(expert.theta - cluster_theta) <= self.theta_cluster_radius:
                current_cluster.append(expert)
            else:
                # Start new cluster
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = [expert]

        if current_cluster:
            clusters.append(current_cluster)

        # In outside_box mode, try merging distant but similar clusters
        if self.cross_cluster_enabled and len(clusters) > 1:
            clusters = self._try_cross_cluster_merge(clusters)

        return clusters

    def _try_cross_cluster_merge(
        self,
        clusters: List[List[CulturalExpert]],
    ) -> List[List[CulturalExpert]]:
        """
        In outside_box mode, merge clusters that have similar temperament
        profiles even if they're distant in theta-space.
        """
        if len(clusters) < 2:
            return clusters

        def cluster_profile(cluster):
            """Compute (mean_lambda, mean_s) for a cluster."""
            mean_lambda = sum(e.lambd for e in cluster) / len(cluster)
            mean_s = sum(e.s for e in cluster) / len(cluster)
            return mean_lambda, mean_s

        merged = []
        used = set()

        for i, c1 in enumerate(clusters):
            if i in used:
                continue

            merged_cluster = list(c1)

            for j, c2 in enumerate(clusters[i + 1:], i + 1):
                if j in used:
                    continue

                # Check if profiles are similar
                p1 = cluster_profile(c1)
                p2 = cluster_profile(c2)

                lambda_diff = abs(p1[0] - p2[0])
                s_diff = abs(p1[1] - p2[1])

                # Similar temperament profile → merge
                if lambda_diff < 0.05 and s_diff < 0.1:
                    merged_cluster.extend(c2)
                    used.add(j)

            merged.append(merged_cluster)
            used.add(i)

        return merged

    def _find_nearby_motif(self, theta: float) -> Optional[Motif]:
        """Find a motif close to the given theta."""
        for motif in self.motifs:
            if abs(motif.theta_center - theta) < self.theta_cluster_radius:
                return motif
        return None

    def _spawn_motif(
        self,
        theta_values: List[float],
        s_values: List[float],
        lambda_values: List[float],
        v_values: List[float],
        expert_ids: List[int],
        current_tick: int,
    ) -> Motif:
        """Create a new motif from a cluster."""
        mean_theta = sum(theta_values) / len(theta_values)
        mean_s = sum(s_values) / len(s_values)
        mean_lambda = sum(lambda_values) / len(lambda_values)
        mean_abs_v = sum(abs(v) for v in v_values) / len(v_values)
        var_theta = sum((t - mean_theta) ** 2 for t in theta_values) / len(theta_values)

        motif_id = _next_motif_id()
        motif = Motif(
            motif_id=motif_id,
            name=f"M{motif_id}",
            theta_center=mean_theta,
            S=mean_s,
            mean_lambda=mean_lambda,
            mean_abs_v=mean_abs_v,
            var_theta=var_theta,
            support_ids=set(expert_ids),
            birth_tick=current_tick,
        )

        return motif

    def _apply_teaching(self, experts: List[CulturalExpert]) -> int:
        """
        Apply motif->expert teaching.

        For each expert, find nearby motifs and:
        1. Record affiliation (motif_ids)
        2. Set motif_affinity (weight for pressure-based dynamics)
        3. Nudge lambda toward motif's style
        4. Accrue cultural capital

        Returns number of experts taught.
        """
        taught_count = 0

        for expert in experts:
            # Clear old motif affiliations and affinities
            expert.motif_ids.clear()
            expert.motif_affinity.clear()
            expert.cultural_capital = 0.0

            nearby_motifs = []
            for motif in self.motifs:
                distance = abs(expert.theta - motif.theta_center)
                if distance < self.D_culture:
                    nearby_motifs.append((motif, distance))

            if not nearby_motifs:
                continue

            taught_count += 1

            # Sort by reliability * proximity
            nearby_motifs.sort(key=lambda m: -m[0].S / (1 + m[1] / self.D_culture))

            for motif, distance in nearby_motifs:
                # Record affiliation
                expert.motif_ids.add(motif.motif_id)

                # Compute affinity weight (used by pressure-based dynamics)
                proximity = 1.0 - distance / self.D_culture
                reliability_weight = max(0.0, motif.S)
                weight = proximity * reliability_weight

                # Store affinity for use in tick_fast() pressure calculations
                expert.motif_affinity[motif.motif_id] = weight

                # Determine learning rate
                eta_lambda = self.eta_cultural_lambda

                # In outside_box mode, low-s experts learn faster
                if self.boosted_learning_for_low_s and expert.s < 0.0:
                    eta_lambda *= 2.0

                # Nudge lambda toward motif's style (direct teaching)
                expert.lambd = (
                    (1 - eta_lambda * weight) * expert.lambd
                    + eta_lambda * weight * motif.mean_lambda
                )

                # Note: theta pull now happens via F_culture in pressure dynamics
                # during every fast tick, not just during cultural tick

                # Cultural capital accrues from strong motif membership
                expert.cultural_capital += (
                    self.cultural_capital_rate * weight * motif.S
                )

        return taught_count

    def _prune_dead_motifs(
        self,
        experts: List[CulturalExpert],
        current_tick: int,
        mode: str,
    ) -> List[str]:
        """Remove motifs with no supporting experts or consistently low S."""
        expert_ids = {e.expert_id for e in experts}
        pruned_names = []

        surviving = []
        for motif in self.motifs:
            # Check if motif has living supporters
            alive_supporters = motif.support_ids & expert_ids
            motif.support_ids = alive_supporters

            # Prune if no supporters or consistently negative S
            should_prune = False

            if len(alive_supporters) == 0 and motif.age > 3:
                should_prune = True

            if motif.S < -0.2 and motif.age > 5:
                should_prune = True

            if should_prune:
                pruned_names.append(motif.name)
                self.events.append({
                    "type": "motif_prune",
                    "tick": current_tick,
                    "motif_name": motif.name,
                    "motif_id": motif.motif_id,
                    "reason": "no_supporters" if len(alive_supporters) == 0 else "low_S",
                    "mode": mode,
                })
            else:
                surviving.append(motif)

        self.motifs = surviving
        return pruned_names


@dataclass
class Governor:
    """
    Meta-controller that gates bifurcation, culling, and cultural modes.
    (Same as V5, extended for V6 cultural awareness.)
    """

    max_population: int = 20
    min_population: int = 3

    R_high: float = 0.8
    R_low: float = 0.3

    s_spawn_threshold: float = 0.1
    s_cull_threshold: float = -0.3

    stagnation_window: int = 5
    progress_threshold: float = 0.02

    R_history: List[float] = field(default_factory=list)
    s_avg_history: List[float] = field(default_factory=list)

    mode: str = "normal"
    mode_ticks_remaining: int = 0

    outside_box_duration: int = 10
    relaxed_D_max_factor: float = 0.7
    relaxed_s_spawn_factor: float = 0.5
    noise_boost_factor: float = 2.0

    def update(
        self,
        R: float,
        N: int,
        s_values: List[float],
        d_values: List[float],
    ) -> Tuple[bool, bool, str]:
        """Update governor state and return gates."""
        self.R_history.append(R)
        if len(self.R_history) > self.stagnation_window:
            self.R_history.pop(0)

        s_avg = sum(s_values) / len(s_values) if s_values else 0.0
        self.s_avg_history.append(s_avg)
        if len(self.s_avg_history) > self.stagnation_window:
            self.s_avg_history.pop(0)

        if self.mode == "outside_box":
            self.mode_ticks_remaining -= 1
            if self.mode_ticks_remaining <= 0:
                self.mode = "normal"
        else:
            if self._is_stagnant(N):
                self.mode = "outside_box"
                self.mode_ticks_remaining = self.outside_box_duration

        gate_spawn = self._compute_gate_spawn(R, N, s_avg)
        gate_cull = self._compute_gate_cull(R, N)

        return gate_spawn, gate_cull, self.mode

    def _is_stagnant(self, N: int) -> bool:
        if len(self.R_history) < self.stagnation_window:
            return False
        capacity_ratio = N / self.max_population
        if capacity_ratio < 0.8:
            return False
        R_start = self.R_history[0]
        R_end = self.R_history[-1]
        progress = R_end - R_start
        return progress < self.progress_threshold

    def _compute_gate_spawn(self, R: float, N: int, s_avg: float) -> bool:
        if N >= self.max_population:
            return False
        if self.mode == "outside_box":
            return True
        return R < self.R_high or s_avg > 0.0

    def _compute_gate_cull(self, R: float, N: int) -> bool:
        if N <= self.min_population:
            return False
        if self.mode == "outside_box":
            return N > self.min_population + 2
        return R > self.R_low

    def get_effective_params(
        self,
        base_D_max: float,
        base_s_bifurcate: float,
        base_noise_phi_std: float,
        base_noise_v_std: float,
    ) -> Dict[str, float]:
        if self.mode == "outside_box":
            return {
                "D_max": base_D_max * self.relaxed_D_max_factor,
                "s_bifurcate": base_s_bifurcate * self.relaxed_s_spawn_factor,
                "noise_phi_std": base_noise_phi_std * self.noise_boost_factor,
                "noise_v_std": base_noise_v_std * self.noise_boost_factor,
            }
        return {
            "D_max": base_D_max,
            "s_bifurcate": base_s_bifurcate,
            "noise_phi_std": base_noise_phi_std,
            "noise_v_std": base_noise_v_std,
        }


@dataclass
class CulturalEvolutionaryController:
    """
    Controller with four clocks: fast, micro, macro, cultural.

    Extends V5's EvolutionaryController with:
    - Cultural clock (slower than macro)
    - CulturalController for motif dynamics
    - Cultural transmission between experts
    """

    experts: List[CulturalExpert]

    # Governor (meta-controller)
    governor: Governor = field(default_factory=Governor)

    # Cultural controller
    cultural: CulturalController = field(default_factory=CulturalController)

    # Global lens for drift modulation
    lens: Lens = field(default_factory=Lens)

    # Clocks
    fast_clock: int = 0
    micro_clock: int = 0
    macro_clock: int = 0
    cultural_clock: int = 0

    # Periods
    micro_period: int = 5
    macro_period: int = 4
    cultural_period: int = 3  # Cultural ticks every N macro ticks

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
    D_max: float = 50.0
    s_bifurcate: float = 0.1
    max_population: int = 20

    # Culling parameters
    s_cull: float = -0.3
    cull_grace_ticks: int = 3
    min_population: int = 3

    # Global stats
    micro_R_sum: float = 0.0
    micro_R_count: int = 0

    # Current state
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

        # Get effective parameters from Governor
        effective = self.governor.get_effective_params(
            self.D_max,
            self.s_bifurcate,
            self.noise_phi_std,
            self.noise_v_std,
        )

        # Build motifs dict for pressure-based dynamics
        motifs_dict = {m.motif_id: m for m in self.cultural.motifs}

        # Get lens parameters for this tick
        L, gamma_lens, g_min, g_max = self.lens.get_params()

        # Fast update for each expert (with pressure-based dynamics and lens)
        expert_signals = [
            e.tick_fast(
                psi=psi,
                dv=self.dv,
                noise_phi_std=effective["noise_phi_std"],
                noise_v_std=effective["noise_v_std"],
                motifs=motifs_dict,
                # Lens parameters
                L=L,
                R=R,
                gamma_lens=gamma_lens,
                g_min=g_min,
                g_max=g_max,
            )
            for e in self.experts
        ]

        # Update lens with raw drifts from this tick
        drifts = [sig["raw_drift"] for sig in expert_signals]
        self.lens.update(drifts)

        # V6.1: Update absorption statistics for each expert
        for e, sig in zip(self.experts, expert_signals):
            e.update_absorption_stats(sig["g_lens"], R, sig["alpha_lens"])

        micro_event = False
        macro_event = False
        cultural_event = False
        bifurcations = []
        culled = []
        cultural_info = {}

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

                # Update reliability
                for e in self.experts:
                    e.do_macro_update(self.eta_s, self.beta_s)

                # V6.1: Prediction success axis
                for e in self.experts:
                    e.update_delta_s()           # Track Δs
                    e.maybe_absorb()             # Absorb if conditions met
                    e.compute_diversity_bonus()  # Reward correct outliers

                # Update Governor
                s_values = [e.s for e in self.experts]
                d_values = [e.drift_distance() for e in self.experts]
                gate_spawn, gate_cull, mode = self.governor.update(
                    R=R,
                    N=len(self.experts),
                    s_values=s_values,
                    d_values=d_values,
                )

                self.current_gate_spawn = gate_spawn
                self.current_gate_cull = gate_cull
                self.current_mode = mode

                # Bifurcation and culling
                bifurcations = self._check_bifurcation(
                    gate_spawn, effective["D_max"], effective["s_bifurcate"]
                )
                culled = self._check_culling(gate_cull)

                # Cultural clock (every N macro ticks)
                if self.macro_clock % self.cultural_period == 0:
                    self.cultural_clock += 1
                    cultural_event = True

                    cultural_info = self.cultural.cultural_tick(
                        self.experts,
                        self.fast_clock,
                        mode=self.current_mode,
                    )

        # Compute lens diagnostics
        avg_g_lens = sum(sig["g_lens"] for sig in expert_signals) / len(expert_signals) if expert_signals else 1.0

        # V6.1: Compute prediction success diagnostics
        avg_delta_s = sum(e.delta_s_ema for e in self.experts) / len(self.experts) if self.experts else 0.0
        avg_diversity_bonus = sum(e.diversity_bonus for e in self.experts) / len(self.experts) if self.experts else 0.0

        return {
            "fast_clock": self.fast_clock,
            "micro_clock": self.micro_clock,
            "macro_clock": self.macro_clock,
            "cultural_clock": self.cultural_clock,
            "R": R,
            "psi": psi,
            "population": len(self.experts),
            "expert_signals": expert_signals,
            "micro_event": micro_event,
            "macro_event": macro_event,
            "cultural_event": cultural_event,
            "bifurcations": bifurcations,
            "culled": culled,
            "gate_spawn": self.current_gate_spawn,
            "gate_cull": self.current_gate_cull,
            "mode": self.current_mode,
            "num_motifs": len(self.cultural.motifs),
            "cultural_info": cultural_info,
            # Lens diagnostics
            "lens_L": self.lens.L,
            "avg_g_lens": avg_g_lens,
            # V6.1: Prediction success diagnostics
            "avg_delta_s": avg_delta_s,
            "avg_diversity_bonus": avg_diversity_bonus,
        }

    def _check_bifurcation(
        self,
        gate_spawn: bool,
        effective_D_max: float,
        effective_s_bifurcate: float,
    ) -> List[Dict]:
        """Check each expert for bifurcation eligibility."""
        bifurcations = []

        if not gate_spawn:
            return bifurcations

        if len(self.experts) >= self.max_population:
            return bifurcations

        for expert in list(self.experts):
            drift = expert.drift_distance()

            # Use effective reliability (includes cultural capital)
            eff_s = expert.effective_reliability()

            if (
                drift > effective_D_max
                and eff_s > effective_s_bifurcate
                and len(self.experts) < self.max_population
            ):
                offspring = expert.spawn_offspring(self.fast_clock)
                self.experts.append(offspring)

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
                    "cultural_capital": expert.cultural_capital,
                }
                bifurcations.append(event)
                self.events.append(event)

        return bifurcations

    def _check_culling(self, gate_cull: bool) -> List[Dict]:
        """Cull experts with consistently low reliability."""
        culled = []

        if not gate_cull:
            return culled

        if len(self.experts) <= self.min_population:
            return culled

        experts_to_remove = []

        for expert in self.experts:
            # Use effective reliability for culling decisions
            eff_s = expert.effective_reliability()

            if (
                eff_s < self.s_cull
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
                    "effective_reliability": eff_s,
                    "generation": expert.generation,
                    "mode": self.current_mode,
                }
                culled.append(event)
                self.events.append(event)

        for expert in experts_to_remove:
            self.experts.remove(expert)

        return culled


def run_simulation_v6(
    seed: int = 42,
    num_experts: int = 5,
    num_ticks: int = 600,
    micro_period: int = 5,
    macro_period: int = 4,
    cultural_period: int = 3,
    D_max: float = 50.0,
    # Lens parameters
    eta_L: float = 0.1,
    gamma_lens: float = 0.3,
    g_min: float = 0.5,
    g_max: float = 1.5,
) -> None:
    """
    Run the V6 simulation with cultural transmission.
    """
    random.seed(seed)
    reset_id_counters()

    # Initial experts
    names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"][:num_experts]
    omegas = [0.08, 0.10, 0.12, 0.09, 0.11][:num_experts]

    experts = [
        CulturalExpert(
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

    governor = Governor(max_population=20, min_population=3)
    cultural = CulturalController()
    lens = Lens(eta_L=eta_L, gamma_lens=gamma_lens, g_min=g_min, g_max=g_max)

    controller = CulturalEvolutionaryController(
        experts=experts,
        governor=governor,
        cultural=cultural,
        lens=lens,
        micro_period=micro_period,
        macro_period=macro_period,
        cultural_period=cultural_period,
        D_max=D_max,
    )

    # Header
    print("=" * 100)
    print("CHRONOVISOR V6: CULTURAL TRANSMISSION BETWEEN EXPERTS")
    print("=" * 100)
    print(f"Initial experts: {names}")
    print(f"D_max (drift threshold): {D_max}")
    print(f"Lens: gamma={gamma_lens}, eta_L={eta_L}, bounds=[{g_min}, {g_max}]")
    print(f"Periods: micro={micro_period}, macro={macro_period * micro_period}, "
          f"cultural={cultural_period * macro_period * micro_period}")
    print("Four clocks: fast (token) < micro (lambda) < macro (s, bifurcate) < cultural (motifs)")
    print("=" * 100)
    print()

    outside_box_events = 0

    for t in range(1, num_ticks + 1):
        info = controller.tick()

        # Log bifurcation events
        for bif in info["bifurcations"]:
            mode_tag = " [OOB]" if bif.get("mode") == "outside_box" else ""
            cc_tag = f" (CC={bif.get('cultural_capital', 0):.2f})" if bif.get("cultural_capital", 0) > 0 else ""
            print(f"  [BIFURCATE{mode_tag}] {bif['parent']} -> {bif['offspring']} "
                  f"(drift={bif['drift']:.1f}){cc_tag}")

        # Log culling events
        for cull in info["culled"]:
            mode_tag = " [OOB]" if cull.get("mode") == "outside_box" else ""
            print(f"  [CULL{mode_tag}] {cull['name']} (s={cull['reliability']:.3f})")

        # Log cultural events
        if info["cultural_event"] and info.get("cultural_info"):
            ci = info["cultural_info"]
            if ci.get("new_motifs"):
                for m in ci["new_motifs"]:
                    print(f"  [CULTURE] New motif: {m}")
            if ci.get("pruned_motifs"):
                for m in ci["pruned_motifs"]:
                    print(f"  [CULTURE] Pruned motif: {m}")

        # Regular logging at macro events
        if info["macro_event"]:
            pop = info["population"]
            R = info["R"]
            lens_L = info.get("lens_L", 0.0)
            avg_g = info.get("avg_g_lens", 1.0)

            avg_drift = sum(e.drift_distance() for e in controller.experts) / pop
            avg_s = sum(e.s for e in controller.experts) / pop
            avg_cc = sum(e.cultural_capital for e in controller.experts) / pop
            generations = [e.generation for e in controller.experts]
            max_gen = max(generations) if generations else 0

            mode = info.get("mode", "normal")
            gate_spawn = "Y" if info.get("gate_spawn", True) else "N"
            gate_cull = "Y" if info.get("gate_cull", True) else "N"
            num_motifs = info.get("num_motifs", 0)

            if mode == "outside_box":
                outside_box_events += 1
                mode_display = "OOB"
            else:
                mode_display = "norm"

            cultural_tag = f"M={num_motifs}" if info["cultural_event"] else ""

            print(
                f"t={t:3d} | R={R:.3f} | N={pop:2d} | "
                f"drift={avg_drift:6.1f} | s={avg_s:+.3f} | L={lens_L:+.4f} | g={avg_g:.3f} | "
                f"gen={max_gen} | {gate_spawn}/{gate_cull} {mode_display} {cultural_tag}"
            )

    # Final summary
    print()
    print("=" * 100)
    print("SIMULATION COMPLETE")
    print("=" * 100)
    print(f"Final population: {len(controller.experts)}")

    total_bifs = sum(1 for e in controller.events if e['type'] == 'bifurcation')
    oob_bifs = sum(1 for e in controller.events if e['type'] == 'bifurcation' and e.get('mode') == 'outside_box')
    total_culls = sum(1 for e in controller.events if e['type'] == 'culling')
    oob_culls = sum(1 for e in controller.events if e['type'] == 'culling' and e.get('mode') == 'outside_box')

    print(f"Total bifurcations: {total_bifs} ({oob_bifs} in outside_box mode)")
    print(f"Total cullings: {total_culls} ({oob_culls} in outside_box mode)")
    print(f"Outside-box macro events: {outside_box_events}")
    print()

    # Motif census
    print(f"Active Motifs: {len(controller.cultural.motifs)}")
    print("-" * 80)
    if controller.cultural.motifs:
        print(f"{'Name':<10} {'theta':>8} {'S':>6} {'lambda':>6} {'|v|':>6} {'support':>8} {'age':>4}")
        print("-" * 80)
        for m in sorted(controller.cultural.motifs, key=lambda x: -x.S):
            print(
                f"{m.name:<10} {m.theta_center:>8.1f} {m.S:>6.3f} "
                f"{m.mean_lambda:>6.3f} {m.mean_abs_v:>6.3f} "
                f"{m.support_size():>8} {m.age:>4}"
            )
    print()

    # Expert census
    print("Expert Census:")
    print("-" * 100)
    print(f"{'Name':<15} {'Gen':>3} {'theta':>8} {'home':>8} {'Drift':>7} "
          f"{'lambda':>6} {'s':>6} {'CC':>6} {'Motifs':>8}")
    print("-" * 100)

    for e in sorted(controller.experts, key=lambda x: -x.effective_reliability()):
        motif_str = ",".join(str(mid) for mid in sorted(e.motif_ids)) if e.motif_ids else "-"
        print(
            f"{e.name:<15} {e.generation:>3} {e.theta:>8.1f} {e.theta_home:>8.1f} "
            f"{e.drift_distance():>7.1f} {e.lambd:>6.3f} {e.s:>6.3f} "
            f"{e.cultural_capital:>6.3f} {motif_str:>8}"
        )

    print("-" * 100)

    # Cultural event log
    cultural_events = [e for e in controller.cultural.events]
    if cultural_events:
        print()
        print("Cultural Event Log:")
        for ev in cultural_events[-10:]:  # Last 10 events
            if ev["type"] == "motif_spawn":
                print(f"  t={ev['tick']:3d} SPAWN {ev['motif_name']} at theta={ev['theta_center']:.1f} "
                      f"(S={ev['S']:.3f}, support={ev['support_size']})")
            elif ev["type"] == "motif_prune":
                print(f"  t={ev['tick']:3d} PRUNE {ev['motif_name']} ({ev['reason']})")


if __name__ == "__main__":
    run_simulation_v6(seed=42, num_ticks=600)
