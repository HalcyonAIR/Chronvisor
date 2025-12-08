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

    # Stats tracking
    total_ticks: int = 0
    total_macro_ticks: int = 0

    # History for analysis
    pressure_history: list = field(default_factory=list)
    R_history: list = field(default_factory=list)

    @classmethod
    def create(
        cls,
        n_experts: int,
        alpha_T: float = 0.3,
        alpha_P: float = 0.2,
        alpha_C: float = 0.1,
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

            # Trust: from reliability
            w_k = _sigmoid(self.beta_s * expert.s)
            # Log transform for additive bias (avoid log(0))
            trust[i] = math.log(max(w_k, 1e-6))

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
        self.R_history.clear()
