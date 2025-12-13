"""
Bridge between MoE routing and Chronovisor dynamics.

Translates:
- MoE routing stats -> Chronovisor observables (alignment signals)
- Chronovisor state -> pressure biases for router

Core principle: MoE router still makes the decision, but Chronovisor
gently tilts the logits in favor of globally trusted experts.
Pressure is a wind, not a hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from chronovisor.simulation_v6 import (
    CulturalExpert,
    CulturalEvolutionaryController,
    CulturalController,
    Governor,
    Lens,
    Motif,
    compute_kuramoto_R_and_psi,
    sigmoid,
    reset_id_counters,
    _next_expert_id,
)


def _sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


@dataclass
class PressureBias:
    """
    Pressure bias terms for router injection.

    b_k = alpha_T * T_k + alpha_P * P_k + alpha_C * C_k

    Where:
    - T_k = trust score (from reliability s_k)
    - P_k = lens pressure (from g_lens deviation from 1.0)
    - C_k = cultural bonus (from motif membership)
    """

    trust: np.ndarray  # T_k for each expert
    pressure: np.ndarray  # P_k for each expert
    cultural: np.ndarray  # C_k for each expert
    combined: np.ndarray  # Final b_k

    alpha_T: float  # Trust weight
    alpha_P: float  # Pressure weight
    alpha_C: float  # Cultural weight


@dataclass
class TemperatureField:
    """
    Per-expert temperature field for routing topology.

    Fast temperature:
        T_fast_k = base_T × coherence_factor × drift_factor_k × reliability_factor_k

    Effective temperature (includes structural/geological):
        T_effective_k = T_fast_k × T̄_structural_k

    The structural temperature evolves slowly via EMA, creating "geology":
    - Regions with consistently high T become ridges (permanently hot)
    - Regions with consistently low T become valleys (permanently cold)

    High temperature = diffuse, exploratory, muddy terrain
    Low temperature = sharp, exploitative, solid bedrock
    """

    # Fast-time temperature (instantaneous)
    fast_temperatures: np.ndarray  # T_fast_k for each expert

    # Structural temperature (slow geological evolution)
    structural_temperatures: np.ndarray  # T̄_k for each expert

    # Effective temperature (what routing actually uses)
    effective_temperatures: np.ndarray  # T_effective_k = T_fast × T̄_structural

    base_temperature: float
    coherence_R: float  # Current system coherence

    # Components for debugging
    coherence_factor: float  # Global factor from R
    drift_factors: np.ndarray  # Per-expert drift contribution
    reliability_factors: np.ndarray  # Per-expert reliability contribution

    # Backwards compatibility alias
    @property
    def temperatures(self) -> np.ndarray:
        """Alias for effective_temperatures (backwards compatibility)."""
        return self.effective_temperatures


@dataclass
class RoutingStats:
    """
    Statistics from MoE routing for one batch.

    Collected from router and fed to Chronovisor.
    """

    expert_usage: np.ndarray  # Count of times each expert was chosen
    mean_gate_weights: np.ndarray  # Mean gate weight per expert
    batch_loss: float  # Overall batch loss
    per_expert_contribution: Optional[np.ndarray] = None  # Optional: loss contribution per expert


@dataclass
class ChronoMoEBridge:
    """
    Bridge connecting MoE routing to Chronovisor dynamics.

    Maintains a Chronovisor controller with one expert per MoE expert.
    Translates routing stats into alignment signals, runs Chronovisor ticks,
    and produces pressure biases for the router.
    """

    n_experts: int
    controller: CulturalEvolutionaryController

    # Pressure coefficients (small-signal regime)
    alpha_T: float = 0.3  # Trust weight
    alpha_P: float = 0.2  # Lens pressure weight
    alpha_C: float = 0.1  # Cultural weight

    # Reliability conversion
    beta_s: float = 3.0  # Steepness of trust sigmoid

    # Temperature field coefficients
    base_temperature: float = 1.0  # Baseline temperature
    beta_R: float = 0.5  # Coherence → temperature (low R → high T)
    beta_drift: float = 0.3  # Drift → temperature (high drift → high T)
    beta_reliability: float = 0.2  # Reliability → temperature (low s → high T)
    T_min: float = 0.3  # Minimum temperature (sharp routing)
    T_max: float = 3.0  # Maximum temperature (diffuse routing)

    # Structural temperature (slow geological evolution)
    eta_structural_T: float = 0.01  # Very slow EMA rate (geological timescale)
    structural_T: np.ndarray = field(default=None, repr=False)  # Per-expert structural temp

    # Stats tracking
    total_ticks: int = 0
    total_macro_ticks: int = 0

    # History for analysis
    pressure_history: list = field(default_factory=list)
    temperature_history: list = field(default_factory=list)
    structural_T_history: list = field(default_factory=list)
    R_history: list = field(default_factory=list)

    @classmethod
    def create(
        cls,
        n_experts: int,
        alpha_T: float = 0.3,
        alpha_P: float = 0.2,
        alpha_C: float = 0.1,
        # Temperature field parameters
        base_temperature: float = 1.0,
        beta_R: float = 0.5,
        beta_drift: float = 0.3,
        beta_reliability: float = 0.2,
        T_min: float = 0.3,
        T_max: float = 3.0,
        # Structural temperature (geological) parameters
        eta_structural_T: float = 0.01,
        # Controller parameters
        seed: int = 42,
        micro_period: int = 5,
        macro_period: int = 4,
        cultural_period: int = 3,
    ) -> ChronoMoEBridge:
        """
        Factory method to create a bridge with fresh Chronovisor controller.

        Args:
            n_experts: Number of MoE experts (1:1 mapping).
            alpha_T: Trust weight for pressure bias.
            alpha_P: Lens pressure weight.
            alpha_C: Cultural weight.
            base_temperature: Baseline temperature for routing.
            beta_R: Coherence factor for temperature (low R → high T).
            beta_drift: Drift factor for temperature (high drift → high T).
            beta_reliability: Reliability factor (low s → high T).
            T_min: Minimum temperature bound.
            T_max: Maximum temperature bound.
            eta_structural_T: EMA rate for structural temperature (slow, ~0.01).
            seed: Random seed for reproducibility.
            micro_period: Fast ticks between micro updates.
            macro_period: Micro ticks between macro updates.
            cultural_period: Macro ticks between cultural updates.
        """
        import random

        random.seed(seed)
        reset_id_counters()

        # Create Chronovisor experts (one per MoE expert)
        experts = []
        for i in range(n_experts):
            expert = CulturalExpert(
                name=f"E{i}",
                expert_id=_next_expert_id(),
                phi=random.uniform(0, 2 * math.pi),
                omega=0.1 + random.uniform(-0.02, 0.02),
                theta=0.0,
                theta_home=0.0,
                v=0.0,
                lambd=0.05,
                generation=0,
                birth_tick=0,
            )
            experts.append(expert)

        # Create controller components
        governor = Governor(max_population=n_experts + 5, min_population=n_experts)
        cultural = CulturalController()
        lens = Lens(eta_L=0.1, gamma_lens=0.3, g_min=0.5, g_max=1.5)

        controller = CulturalEvolutionaryController(
            experts=experts,
            governor=governor,
            cultural=cultural,
            lens=lens,
            micro_period=micro_period,
            macro_period=macro_period,
            cultural_period=cultural_period,
            max_population=n_experts + 5,  # Allow some growth
            min_population=n_experts,  # Don't cull below initial count
        )

        return cls(
            n_experts=n_experts,
            controller=controller,
            alpha_T=alpha_T,
            alpha_P=alpha_P,
            alpha_C=alpha_C,
            base_temperature=base_temperature,
            beta_R=beta_R,
            beta_drift=beta_drift,
            beta_reliability=beta_reliability,
            T_min=T_min,
            T_max=T_max,
            eta_structural_T=eta_structural_T,
            structural_T=np.ones(n_experts),  # Initialize to 1.0 (neutral terrain)
        )

    def translate_routing_to_alignment(
        self,
        stats: RoutingStats,
        cluster_assignments: Optional[np.ndarray] = None,
        ground_truth_experts: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Translate MoE routing stats into alignment signals for Chronovisor.

        The alignment signal tells Chronovisor which experts are "agreeing"
        with the global direction. High usage + good loss = positive alignment.

        Args:
            stats: Routing statistics from MoE.
            cluster_assignments: Optional cluster IDs for tokens in batch.
            ground_truth_experts: Optional ground-truth expert for each cluster.

        Returns:
            Dictionary with alignment values per expert.
        """
        # Normalize usage to get usage rate
        total_usage = stats.expert_usage.sum()
        if total_usage > 0:
            usage_rate = stats.expert_usage / total_usage
        else:
            usage_rate = np.ones(self.n_experts) / self.n_experts

        # Compute alignment signal
        # Higher usage and lower loss -> higher alignment
        # We want: Δs_k ∝ (usage_k * −loss)
        # Since we don't have per-expert loss yet, use overall loss

        # Simple heuristic: alignment = usage_rate - uniform_rate
        # Experts used more than average get positive alignment
        uniform_rate = 1.0 / self.n_experts
        alignment_signal = usage_rate - uniform_rate

        # If we have ground-truth matching (synthetic task), use it
        if cluster_assignments is not None and ground_truth_experts is not None:
            # Compute how well each expert matched its "correct" assignments
            correct_matches = np.zeros(self.n_experts)
            total_per_cluster = np.zeros(self.n_experts)

            for cluster_id, gt_expert in enumerate(ground_truth_experts):
                mask = cluster_assignments == cluster_id
                if mask.any():
                    # Count how often this expert was chosen for its cluster
                    # This requires per-token data which we'd get from RoutingLog
                    pass

            # For now, use simpler proxy based on loss
            if stats.batch_loss < 0.5:  # Good loss
                alignment_signal = alignment_signal * (1.0 - stats.batch_loss)
            else:  # Bad loss - invert signal
                alignment_signal = -alignment_signal * stats.batch_loss

        return {
            "alignment": alignment_signal,
            "usage_rate": usage_rate,
            "batch_loss": stats.batch_loss,
        }

    def feed_routing_stats(
        self,
        stats: RoutingStats,
        num_chronovisor_ticks: int = 20,
    ) -> dict:
        """
        Feed MoE routing stats to Chronovisor and run ticks.

        This is the main integration point. Called once per batch:
        1. Translate routing stats to alignment signals
        2. Inject alignment into Chronovisor experts
        3. Run Chronovisor ticks
        4. Return updated state

        Args:
            stats: Routing statistics from MoE.
            num_chronovisor_ticks: Number of fast ticks to run.

        Returns:
            Dictionary with Chronovisor state after ticks.
        """
        # Translate routing to alignment
        alignment_info = self.translate_routing_to_alignment(stats)
        alignment = alignment_info["alignment"]

        # Inject alignment into Chronovisor experts
        # We modify their phase to reflect the alignment signal
        for i, expert in enumerate(self.controller.experts[:self.n_experts]):
            # Higher alignment -> push phase toward mean (more aligned)
            # Lower alignment -> push phase away (less aligned)
            # This is a soft influence, not a hard override
            alignment_nudge = alignment[i] * 0.1  # Small nudge
            expert.phi = (expert.phi + alignment_nudge) % (2 * math.pi)

        # Run Chronovisor ticks
        tick_results = []
        for _ in range(num_chronovisor_ticks):
            result = self.controller.tick()
            tick_results.append(result)
            self.total_ticks += 1
            if result.get("macro_event"):
                self.total_macro_ticks += 1

        # Get final state
        final_result = tick_results[-1] if tick_results else {}

        # Track history
        if final_result.get("R") is not None:
            self.R_history.append(final_result["R"])

        return {
            "alignment_info": alignment_info,
            "final_R": final_result.get("R", 0.0),
            "final_psi": final_result.get("psi", 0.0),
            "lens_L": final_result.get("lens_L", 0.0),
            "avg_g_lens": final_result.get("avg_g_lens", 1.0),
            "num_ticks": num_chronovisor_ticks,
            "macro_events": sum(1 for r in tick_results if r.get("macro_event")),
            "cultural_events": sum(1 for r in tick_results if r.get("cultural_event")),
        }

    def get_pressure_bias(self) -> PressureBias:
        """
        Extract pressure bias from current Chronovisor state.

        b_k = alpha_T * T_k + alpha_P * P_k + alpha_C * C_k

        Where:
        - T_k = log(w_k) where w_k = sigmoid(beta_s * s_k)
        - P_k = g_lens_ema - 1.0 (zero-centered)
        - C_k = cultural_capital (from motif membership)

        Returns:
            PressureBias with all components and combined bias.
        """
        n = min(self.n_experts, len(self.controller.experts))

        trust = np.zeros(self.n_experts)
        pressure = np.zeros(self.n_experts)
        cultural = np.zeros(self.n_experts)

        for i in range(n):
            expert = self.controller.experts[i]

            # Trust: from reliability (symmetrized via logit so it can push up or down)
            w_k = _sigmoid(self.beta_s * expert.s)
            w_k = min(max(w_k, 1e-6), 1.0 - 1e-6)
            trust[i] = math.log(w_k) - math.log(1.0 - w_k)

            # Pressure: lens gain deviation from 1.0
            pressure[i] = expert.g_lens_ema - 1.0

            # Cultural: cultural capital
            cultural[i] = expert.cultural_capital

        # Combine with weights
        combined = (
            self.alpha_T * trust
            + self.alpha_P * pressure
            + self.alpha_C * cultural
        )

        # Store in history
        self.pressure_history.append(combined.copy())

        return PressureBias(
            trust=trust,
            pressure=pressure,
            cultural=cultural,
            combined=combined,
            alpha_T=self.alpha_T,
            alpha_P=self.alpha_P,
            alpha_C=self.alpha_C,
        )

    def get_temperature_field(self, update_structural: bool = True) -> TemperatureField:
        """
        Compute per-expert temperature field from Chronovisor state.

        Fast temperature (instantaneous):
            T_fast_k = base_T × (1 + β_R(1-R)) × (1 + β_drift·d_k) × (1 + β_s(1-σ(s_k)))

        Structural temperature (slow geological evolution):
            T̄_k(t+1) = (1 - η_T) × T̄_k(t) + η_T × T_fast_k(t)

        Effective temperature (what routing uses):
            T_effective_k = T_fast_k × T̄_k

        The structural temperature creates "geology":
        - Consistently hot regions become ridges (permanently high T)
        - Consistently cold regions become valleys (permanently low T)

        Args:
            update_structural: Whether to update structural temperatures via EMA.

        Returns:
            TemperatureField with fast, structural, and effective temperatures.
        """
        n = min(self.n_experts, len(self.controller.experts))

        # Initialize structural_T if needed
        if self.structural_T is None:
            self.structural_T = np.ones(self.n_experts)

        # Get current coherence R
        if self.R_history:
            R = self.R_history[-1]
        else:
            R = 0.5  # Default moderate coherence

        # Coherence factor: low R → high temperature (explore when uncertain)
        coherence_factor = 1.0 + self.beta_R * (1.0 - R)

        # Per-expert factors
        drift_factors = np.ones(self.n_experts)
        reliability_factors = np.ones(self.n_experts)

        # Compute max drift for normalization
        drifts = []
        for i in range(n):
            drifts.append(self.controller.experts[i].drift_distance())
        max_drift = max(drifts) if drifts and max(drifts) > 0 else 1.0

        for i in range(n):
            expert = self.controller.experts[i]

            # Drift factor: high drift → high temperature (less trusted)
            normalized_drift = expert.drift_distance() / max_drift
            drift_factors[i] = 1.0 + self.beta_drift * normalized_drift

            # Reliability factor: low reliability → high temperature
            # sigmoid maps s to [0, 1], we want (1 - sigmoid(s)) for inverse
            reliability = _sigmoid(self.beta_s * expert.s)
            reliability_factors[i] = 1.0 + self.beta_reliability * (1.0 - reliability)

        # Compute fast temperatures (instantaneous)
        fast_temperatures = (
            self.base_temperature
            * coherence_factor
            * drift_factors
            * reliability_factors
        )

        # Clamp fast temperatures
        fast_temperatures = np.clip(fast_temperatures, self.T_min, self.T_max)

        # Update structural temperature via EMA (geological erosion)
        if update_structural:
            self.structural_T = (
                (1 - self.eta_structural_T) * self.structural_T
                + self.eta_structural_T * fast_temperatures
            )

        # Compute effective temperature = fast × structural
        # This is what routing actually uses
        effective_temperatures = fast_temperatures * self.structural_T

        # Clamp effective temperatures
        effective_temperatures = np.clip(effective_temperatures, self.T_min, self.T_max)

        # Store in history
        self.temperature_history.append(fast_temperatures.copy())
        self.structural_T_history.append(self.structural_T.copy())

        return TemperatureField(
            fast_temperatures=fast_temperatures,
            structural_temperatures=self.structural_T.copy(),
            effective_temperatures=effective_temperatures,
            base_temperature=self.base_temperature,
            coherence_R=R,
            coherence_factor=coherence_factor,
            drift_factors=drift_factors,
            reliability_factors=reliability_factors,
        )

    def get_expert_states(self) -> list[dict]:
        """
        Get current state of all Chronovisor experts.

        Returns:
            List of dicts with expert state for analysis.
        """
        states = []
        for i, expert in enumerate(self.controller.experts[:self.n_experts]):
            states.append({
                "id": i,
                "name": expert.name,
                "s": expert.s,
                "lambd": expert.lambd,
                "theta": expert.theta,
                "theta_home": expert.theta_home,
                "drift": expert.drift_distance(),
                "g_lens_ema": expert.g_lens_ema,
                "delta_s_ema": expert.delta_s_ema,
                "cultural_capital": expert.cultural_capital,
                "effective_reliability": expert.effective_reliability(),
            })
        return states

    def get_diagnostics(self) -> dict:
        """
        Get diagnostic information about bridge state.

        Returns:
            Dictionary with diagnostic metrics.
        """
        experts = self.controller.experts[:self.n_experts]

        # Structural temperature stats
        if self.structural_T is not None:
            structural_T_mean = float(np.mean(self.structural_T))
            structural_T_std = float(np.std(self.structural_T))
            structural_T_min = float(np.min(self.structural_T))
            structural_T_max = float(np.max(self.structural_T))
        else:
            structural_T_mean = 1.0
            structural_T_std = 0.0
            structural_T_min = 1.0
            structural_T_max = 1.0

        return {
            "total_ticks": self.total_ticks,
            "total_macro_ticks": self.total_macro_ticks,
            "current_R": self.R_history[-1] if self.R_history else 0.0,
            "R_history_len": len(self.R_history),
            "lens_L": self.controller.lens.L,
            "num_motifs": len(self.controller.cultural.motifs),
            "avg_s": sum(e.s for e in experts) / len(experts) if experts else 0.0,
            "avg_lambd": sum(e.lambd for e in experts) / len(experts) if experts else 0.0,
            "avg_drift": sum(e.drift_distance() for e in experts) / len(experts) if experts else 0.0,
            "mode": self.controller.current_mode,
            # Structural temperature diagnostics
            "structural_T_mean": structural_T_mean,
            "structural_T_std": structural_T_std,
            "structural_T_min": structural_T_min,
            "structural_T_max": structural_T_max,
        }

    def get_structural_temperature_diagnostics(self) -> dict:
        """
        Get detailed structural temperature diagnostics for visualization.

        Returns:
            Dictionary with structural temperature analysis.
        """
        if self.structural_T is None:
            return {
                "structural_T": np.ones(self.n_experts),
                "variance": 0.0,
                "entropy": 0.0,
                "landscape_formed": False,
                "valleys": [],
                "ridges": [],
                "evolution_steps": 0,
            }

        structural_T = self.structural_T

        # Basic stats
        variance = float(np.var(structural_T))
        mean = float(np.mean(structural_T))

        # Normalized entropy of structural temperature distribution
        # Low entropy = specialized landscape (clear valleys/ridges)
        # High entropy = flat landscape
        normalized = structural_T / structural_T.sum()
        entropy = float(-np.sum(normalized * np.log(normalized + 1e-10)))
        max_entropy = np.log(len(structural_T))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Identify valleys (low T, stable regions) and ridges (high T, unstable)
        threshold_low = mean - 0.5 * np.std(structural_T)
        threshold_high = mean + 0.5 * np.std(structural_T)

        valleys = [i for i, t in enumerate(structural_T) if t < threshold_low]
        ridges = [i for i, t in enumerate(structural_T) if t > threshold_high]

        # Landscape is "formed" when variance exceeds threshold
        landscape_formed = bool(variance > 0.01)

        return {
            "structural_T": structural_T.copy(),
            "variance": variance,
            "mean": mean,
            "std": float(np.std(structural_T)),
            "entropy": normalized_entropy,
            "landscape_formed": landscape_formed,
            "valleys": valleys,  # Experts that became stable (low T)
            "ridges": ridges,  # Experts that became unstable (high T)
            "evolution_steps": len(self.structural_T_history),
        }

    def get_valley_health_diagnostics(self) -> dict:
        """
        Monitor valley health to detect "bad valleys" (low T̄ + low reliability).

        This diagnostic helps verify that the self-correction mechanism is working:
        - Bad experts should have high T_fast (behavioral avoidance)
        - Over time, their T̄ should rise (structural correction)
        - Persistent bad valleys indicate the mechanism is failing

        The design relies on reliability → T_fast → T̄ feedback, not asymmetric
        erosion. This diagnostic lets us monitor if that assumption holds.

        Returns:
            Dictionary with valley health analysis.
        """
        if self.structural_T is None:
            return {
                "valleys": [],
                "valley_health": {},
                "healthy_valleys": [],
                "unhealthy_valleys": [],
                "at_risk_experts": [],
                "mean_valley_health": 1.0,
                "self_correction_working": True,
            }

        n = min(self.n_experts, len(self.controller.experts))
        structural_T = self.structural_T

        # Identify valleys
        mean_T = float(np.mean(structural_T))
        std_T = float(np.std(structural_T))
        threshold_valley = mean_T - 0.5 * std_T

        valleys = [i for i in range(n) if structural_T[i] < threshold_valley]

        # Compute reliability for each expert
        reliabilities = {}
        for i in range(n):
            expert = self.controller.experts[i]
            reliabilities[i] = _sigmoid(self.beta_s * expert.s)

        # Valley health = reliability (for valleys only)
        # High reliability + valley = good (healthy valley)
        # Low reliability + valley = bad (unhealthy valley, should self-correct)
        valley_health = {}
        healthy_valleys = []
        unhealthy_valleys = []

        reliability_threshold = 0.5  # Below this = unhealthy

        for v in valleys:
            health = reliabilities[v]
            valley_health[v] = health
            if health >= reliability_threshold:
                healthy_valleys.append(v)
            else:
                unhealthy_valleys.append(v)

        # Identify "at risk" experts: high T_fast but still low T̄
        # These are experts trending toward bad valley if T_fast stays high
        at_risk_experts = []
        if self.temperature_history:
            latest_T_fast = self.temperature_history[-1]
            for i in range(n):
                is_hot = latest_T_fast[i] > mean_T + 0.5 * std_T
                is_still_valley = structural_T[i] < threshold_valley
                if is_hot and is_still_valley:
                    # High T_fast (behavioral avoidance) but still a valley
                    # This means structural correction is lagging
                    at_risk_experts.append(i)

        # Mean valley health
        if valley_health:
            mean_health = sum(valley_health.values()) / len(valley_health)
        else:
            mean_health = 1.0  # No valleys = healthy

        # Self-correction is working if:
        # 1. No persistent unhealthy valleys (they should fill in)
        # 2. At-risk experts list is small or empty
        self_correction_working = (
            len(unhealthy_valleys) == 0 or
            len(at_risk_experts) < len(unhealthy_valleys)  # Correction in progress
        )

        return {
            "valleys": valleys,
            "valley_health": valley_health,
            "healthy_valleys": healthy_valleys,
            "unhealthy_valleys": unhealthy_valleys,
            "at_risk_experts": at_risk_experts,
            "mean_valley_health": mean_health,
            "reliabilities": reliabilities,
            "self_correction_working": self_correction_working,
        }

    def reset(self, seed: int = 42) -> None:
        """
        Reset the bridge to initial state.

        Args:
            seed: Random seed for reproducibility.
        """
        import random

        random.seed(seed)
        reset_id_counters()

        # Recreate experts
        for i, expert in enumerate(self.controller.experts[:self.n_experts]):
            expert.phi = random.uniform(0, 2 * math.pi)
            expert.theta = 0.0
            expert.theta_home = 0.0
            expert.v = 0.0
            expert.lambd = 0.05
            expert.s = 0.0
            expert.s_prev = 0.0
            expert.delta_s = 0.0
            expert.delta_s_ema = 0.0
            expert.g_lens_ema = 1.0
            expert.cultural_capital = 0.0
            expert.motif_ids.clear()
            expert.motif_affinity.clear()

        # Reset controller state
        self.controller.fast_clock = 0
        self.controller.micro_clock = 0
        self.controller.macro_clock = 0
        self.controller.cultural_clock = 0
        self.controller.lens.L = 0.0
        self.controller.cultural.motifs.clear()

        # Reset history
        self.total_ticks = 0
        self.total_macro_ticks = 0
        self.pressure_history.clear()
        self.temperature_history.clear()
        self.structural_T_history.clear()
        self.R_history.clear()

        # Reset structural temperature to neutral (flat terrain)
        self.structural_T = np.ones(self.n_experts)
