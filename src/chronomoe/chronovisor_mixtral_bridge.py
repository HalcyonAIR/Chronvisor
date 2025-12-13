"""
Chronovisor ↔ Mixtral Integration Bridge

Connects Chronovisor's geometric control layer (lens, controller) with
the internal Mixtral MoE implementation.

Flow:
    1. Mixtral layers forward pass → routing statistics
    2. Bridge translates routing stats → Chronovisor expert signals
    3. Controller computes Kuramoto R (coherence) → lens updates
    4. Lens state → P×T fields → Mixtral router

This creates a closed loop where:
    - Mixtral's routing behavior informs the geometric lens
    - The lens reshapes future routing decisions via P×T geometry
    - Expert specialization emerges from geometric stability
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from chronomoe.mixtral_core import (
    MixtralConfig,
    MixtralDecoderLayer,
    MixtralSparseMoELayer,
)
from chronomoe.knob import MetaKnob, KnobState, KnobDecision, KnobFactors


def compute_kuramoto_R_and_psi(phases: np.ndarray) -> Tuple[float, float]:
    """
    Compute Kuramoto order parameter R and mean phase ψ.

    R measures synchronization: R=1 (perfect sync), R=0 (desynchronized).

    Args:
        phases: Array of phase angles (n_experts,) in radians

    Returns:
        Tuple of (R, psi) where R ∈ [0, 1] and psi ∈ [0, 2π)
    """
    if len(phases) == 0:
        return 0.0, 0.0

    # Compute complex order parameter: Z = (1/N) Σ exp(i*φ_k)
    Z = np.mean(np.exp(1j * phases))

    # R is the magnitude, ψ is the angle
    R = np.abs(Z)
    psi = np.angle(Z) % (2 * np.pi)

    return float(R), float(psi)


@dataclass
class ChronovisorMixtralState:
    """
    State of the Chronovisor-Mixtral system at a given timestep.

    Tracks:
        - Lens parameters (geometric surface)
        - Expert coherence (how aligned experts are)
        - Pressure biases (how to reshape routing)
        - Controller clock state
    """
    # Lens state (per layer)
    lens_vectors: Dict[int, np.ndarray] = field(default_factory=dict)  # {layer_idx: vector}
    lens_magnitude: Dict[int, float] = field(default_factory=dict)

    # Expert signals (per layer)
    expert_gains: Dict[int, np.ndarray] = field(default_factory=dict)  # {layer_idx: (n_experts,)}
    expert_stability: Dict[int, np.ndarray] = field(default_factory=dict)

    # Routing statistics
    expert_usage: Dict[int, np.ndarray] = field(default_factory=dict)  # {layer_idx: (n_experts,)}
    routing_entropy: Dict[int, float] = field(default_factory=dict)

    # Coherence tracking
    coherence: float = 0.0
    delta_coherence: float = 0.0
    coherence_history: List[float] = field(default_factory=list)

    # Geological temperature tracking
    T_bar: Optional[np.ndarray] = None  # Global geological temperature (T̄_global)
    T_bar_local: Dict[int, np.ndarray] = field(default_factory=dict)  # Per-layer T̄_local
    T_bar_hierarchical: Dict[int, np.ndarray] = field(default_factory=dict)  # Per-layer T̄_eff
    T_effective: Dict[int, np.ndarray] = field(default_factory=dict)  # Per-layer T_final

    # Controller clock state
    fast_clock: int = 0
    micro_clock: int = 0
    macro_clock: int = 0


class MixtralLens:
    """
    Geometric lens for a single Mixtral MoE layer with P×T geometry.

    The lens is a deformable surface in expert space that reshapes
    how the router sees semantic regions.

    State:
        - vector: Current position in expert geometry space (n_experts,)
        - gamma: Adaptation rate
        - pressure: Current pressure bias to apply to routing (P field)
        - temperature: Current temperature field (T field)
        - structural_T: Geological memory of temperature (EMA)
    """

    def __init__(
        self,
        n_experts: int = 8,
        gamma: float = 0.05,
        eta_structural_T: float = 0.01,
        base_temperature: float = 1.0,
    ):
        """
        Initialize lens for one layer.

        Args:
            n_experts: Number of experts in this layer
            gamma: Adaptation rate (how quickly lens moves)
            eta_structural_T: EMA rate for structural temperature
            base_temperature: Baseline temperature
        """
        self.n_experts = n_experts
        self.gamma = gamma
        self.eta_structural_T = eta_structural_T
        self.base_temperature = base_temperature

        # Lens state
        self.vector = np.zeros(n_experts)
        self.magnitude = 0.0

        # P×T fields
        self.pressure = np.zeros(n_experts)  # Pressure field (P)
        self.temperature_fast = np.ones(n_experts) * base_temperature  # Fast temperature
        self.structural_T = np.ones(n_experts) * base_temperature  # Local geological memory (T̄_local)
        self.structural_T_hierarchical = np.ones(n_experts) * base_temperature  # T̄_global × T̄_local
        self.temperature_effective = np.ones(n_experts) * base_temperature  # T_fast × T̄_hierarchical

    def update(self, drift: np.ndarray, coherence_gate: float = 1.0) -> None:
        """
        Update lens position based on expert drift signals.

        Args:
            drift: Direction to move in expert space (n_experts,)
            coherence_gate: Multiplier based on system coherence (0-1)
        """
        # Accumulate drift with gating
        self.vector += self.gamma * coherence_gate * drift

        # Update magnitude
        self.magnitude = np.linalg.norm(self.vector)

    def compute_pressure(self, expert_usage: np.ndarray) -> np.ndarray:
        """
        Compute pressure bias based on expert usage distribution.

        Pressure encourages underutilized experts and discourages overused ones.

        Args:
            expert_usage: Frequency of expert usage (n_experts,)

        Returns:
            Pressure bias vector (n_experts,)
        """
        # Normalize usage to distribution
        total = expert_usage.sum()
        if total == 0:
            return np.zeros(self.n_experts)

        actual_dist = expert_usage / total
        ideal_dist = np.ones(self.n_experts) / self.n_experts

        # Pressure = deviation from ideal
        # Negative pressure = underutilized (should be encouraged)
        pressure = (ideal_dist - actual_dist) * 2.0

        self.pressure = np.clip(pressure, -1.0, 1.0)
        return self.pressure

    def compute_temperature(
        self,
        coherence_R: float,
        expert_drifts: Optional[np.ndarray] = None,
        expert_reliabilities: Optional[np.ndarray] = None,
        beta_R: float = 0.5,
        beta_drift: float = 0.3,
        beta_reliability: float = 0.2,
        T_min: float = 0.3,
        T_max: float = 3.0,
    ) -> np.ndarray:
        """
        Compute temperature field from Kuramoto coherence and expert states.

        T_k = base_T * (1 + β_R*(1-R)) * (1 + β_drift*d_k) * (1 + β_rel*(1-s_k))

        Where:
            - R = Kuramoto order parameter (global coherence)
            - d_k = normalized drift for expert k
            - s_k = reliability score for expert k

        High temperature = diffuse routing (exploratory)
        Low temperature = sharp routing (exploitative)

        Args:
            coherence_R: Kuramoto order parameter (0-1)
            expert_drifts: Optional per-expert drift distances
            expert_reliabilities: Optional per-expert reliability scores
            beta_R: Coherence factor weight
            beta_drift: Drift factor weight
            beta_reliability: Reliability factor weight
            T_min: Minimum temperature bound
            T_max: Maximum temperature bound

        Returns:
            Temperature field (n_experts,)
        """
        # Coherence factor: low R → high temperature (explore when uncertain)
        coherence_factor = 1.0 + beta_R * (1.0 - coherence_R)

        # Start with base temperature modulated by coherence
        temperatures = np.ones(self.n_experts) * self.base_temperature * coherence_factor

        # Apply per-expert drift factor if provided
        if expert_drifts is not None:
            max_drift = expert_drifts.max() if expert_drifts.max() > 0 else 1.0
            normalized_drifts = expert_drifts / max_drift
            drift_factors = 1.0 + beta_drift * normalized_drifts
            temperatures *= drift_factors

        # Apply per-expert reliability factor if provided
        if expert_reliabilities is not None:
            # Low reliability → high temperature
            reliability_factors = 1.0 + beta_reliability * (1.0 - expert_reliabilities)
            temperatures *= reliability_factors

        # Clamp to bounds
        temperatures = np.clip(temperatures, T_min, T_max)

        # Update fast temperature
        self.temperature_fast = temperatures

        # Update local structural temperature (geological memory via EMA)
        # This is T̄_local - layer-specific geological pattern
        self.structural_T = (
            (1.0 - self.eta_structural_T) * self.structural_T
            + self.eta_structural_T * self.temperature_fast
        )

        # Update the local effective temperature so callers can use it immediately
        self.temperature_effective = self.temperature_fast * self.structural_T

        # Note: hierarchical structural T (T̄_global × T̄_local) is computed
        # by the controller, which has access to T̄_global

        return self.temperature_effective.copy()

    def get_state(self) -> dict:
        """Get current lens state with hierarchical structural temperature."""
        return {
            'vector': self.vector.copy(),
            'magnitude': self.magnitude,
            'pressure': self.pressure.copy(),
            'temperature_fast': self.temperature_fast.copy(),
            'structural_T_local': self.structural_T.copy(),
            'structural_T_hierarchical': self.structural_T_hierarchical.copy(),
            'temperature_effective': self.temperature_effective.copy(),
        }


class ChronovisorMixtralController:
    """
    Controller for the Chronovisor-Mixtral system with P×T geometry.

    Manages:
        - Multi-scale clocks (fast, micro, macro)
        - Per-layer lenses with P×T fields
        - Hierarchical structural temperature (global + local)
        - Kuramoto coherence computation (R)
        - Pressure and temperature updates
    """

    def __init__(
        self,
        config: MixtralConfig,
        micro_period: int = 5,
        macro_period: int = 20,
        pressure_scale: float = 0.1,
        temperature_scale: float = 1.0,
        eta_structural_T_global: float = 0.005,  # Even slower than local
        eta_structural_T_local: float = 0.01,
        enable_meta_knob: bool = True,
        usage_decay: float = 0.95,
    ):
        """
        Initialize controller with hierarchical structural temperature.

        Args:
            config: Mixtral model configuration
            micro_period: Ticks between micro-scale updates
            macro_period: Ticks between macro-scale updates
            pressure_scale: How strongly to apply pressure to routing
            temperature_scale: Global temperature scale factor
            eta_structural_T_global: EMA rate for global structural T (very slow)
            eta_structural_T_local: EMA rate for local structural T (slow)
            enable_meta_knob: Whether to enable meta-knob modulation
        """
        self.config = config
        self.micro_period = micro_period
        self.macro_period = macro_period
        self.pressure_scale = pressure_scale
        self.temperature_scale = temperature_scale
        self.eta_structural_T_global = eta_structural_T_global
        self.eta_structural_T_local = eta_structural_T_local
        self.enable_meta_knob = enable_meta_knob
        self.usage_decay = usage_decay

        # Create lenses for each layer
        self.lenses: Dict[int, MixtralLens] = {
            i: MixtralLens(
                n_experts=config.num_experts,
                eta_structural_T=eta_structural_T_local,
            )
            for i in range(config.num_layers)
        }

        # Global structural temperature (shared across all layers)
        # T̄_global captures system-wide geological patterns
        self.structural_T_global = np.ones(config.num_experts)

        # Clock state
        self.fast_clock = 0
        self.micro_clock = 0
        self.macro_clock = 0

        # Kuramoto coherence tracking
        self.coherence_R = 0.0  # Kuramoto order parameter
        self.mean_phase_psi = 0.0
        self.delta_coherence = 0.0
        self.coherence_history: List[float] = []

        # Expert states (per layer)
        self.expert_usage: Dict[int, np.ndarray] = {
            i: np.zeros(config.num_experts)
            for i in range(config.num_layers)
        }
        self.expert_phases: Dict[int, np.ndarray] = {
            i: np.random.uniform(0, 2 * np.pi, config.num_experts)
            for i in range(config.num_layers)
        }

        # Meta-knob for LLM-controlled modulation
        # κ ∈ [-1, +1] modulates pressure and temperature scales
        self.meta_knob = MetaKnob(use_smoothing=True) if enable_meta_knob else None

        # Loss tracking for knob state computation
        self.loss_history: List[float] = []
        self.current_loss: float = 0.0

    def tick(self, routing_stats: Dict[int, dict]) -> ChronovisorMixtralState:
        """
        Advance the controller by one tick.

        Args:
            routing_stats: Routing statistics from all Mixtral layers
                {layer_idx: {'selected_experts': tensor, 'routing_weights': tensor}}

        Returns:
            Current system state
        """
        # Increment clocks
        self.fast_clock += 1

        if self.fast_clock % self.micro_period == 0:
            self.micro_clock += 1

        if self.fast_clock % self.macro_period == 0:
            self.macro_clock += 1

        # Update expert usage from routing stats
        for layer_idx, stats in routing_stats.items():
            selected = stats['selected_experts'].detach().cpu().numpy()
            current_usage = np.zeros(self.config.num_experts)
            for expert_id in range(self.config.num_experts):
                current_usage[expert_id] = np.sum(selected == expert_id)

            self.expert_usage[layer_idx] = (
                self.usage_decay * self.expert_usage[layer_idx]
                + (1.0 - self.usage_decay) * current_usage
            )

        # Update expert phases (simple phase evolution model)
        for layer_idx in routing_stats.keys():
            # Phase evolves based on routing activity
            selected = routing_stats[layer_idx]['selected_experts'].detach().cpu().numpy()
            weights = routing_stats[layer_idx]['routing_weights'].detach().cpu().numpy()

            # Compute average contribution per expert
            phase_nudge = np.zeros(self.config.num_experts)
            for expert_id in range(self.config.num_experts):
                mask = (selected == expert_id)
                if mask.any():
                    # Average weight when this expert is selected
                    phase_nudge[expert_id] = weights[mask].mean()

            # Apply nudge (small perturbation based on usage)
            self.expert_phases[layer_idx] = (
                self.expert_phases[layer_idx] + phase_nudge * 0.1
            ) % (2 * np.pi)

        # Compute Kuramoto coherence (R)
        prev_coherence = self.coherence_R
        self.coherence_R, self.mean_phase_psi = self._compute_kuramoto_coherence()
        self.delta_coherence = self.coherence_R - prev_coherence
        self.coherence_history.append(self.coherence_R)

        # On micro boundaries, update lens pressure and temperature
        if self.fast_clock % self.micro_period == 0:
            # Update global structural temperature first
            # Average fast temperature across all layers
            avg_fast_T = np.zeros(self.config.num_experts)
            for layer_idx, lens in self.lenses.items():
                avg_fast_T += lens.temperature_fast
            avg_fast_T /= len(self.lenses)

            # Update global structural T (very slow EMA)
            self.structural_T_global = (
                (1.0 - self.eta_structural_T_global) * self.structural_T_global
                + self.eta_structural_T_global * avg_fast_T
            )

            # Update per-layer pressure and temperature
            for layer_idx, lens in self.lenses.items():
                lens.compute_pressure(self.expert_usage[layer_idx])

                # Compute per-expert drift from usage imbalance
                usage = self.expert_usage[layer_idx]
                total_usage = usage.sum()
                if total_usage > 0:
                    usage_dist = usage / total_usage
                    ideal_dist = 1.0 / self.config.num_experts
                    # Drift = deviation from uniform (same as pressure but unsigned)
                    expert_drifts = np.abs(usage_dist - ideal_dist)
                else:
                    expert_drifts = np.zeros(self.config.num_experts)

                # Compute per-expert reliability from routing weights consistency
                # Higher weight = more reliable (router trusts this expert)
                # For now, use normalized usage as proxy for reliability
                if total_usage > 0:
                    expert_reliabilities = usage / usage.max()  # Normalize to [0, 1]
                else:
                    expert_reliabilities = np.ones(self.config.num_experts)

                # NOW pass per-expert signals to temperature computation!
                lens.compute_temperature(
                    coherence_R=self.coherence_R,
                    expert_drifts=expert_drifts,
                    expert_reliabilities=expert_reliabilities,
                )

                # Apply hierarchical structural temperature
                # T̄_eff = T̄_global × T̄_local
                lens.structural_T_hierarchical = self.structural_T_global * lens.structural_T
                lens.temperature_effective = lens.temperature_fast * lens.structural_T_hierarchical

        # On macro boundaries, update lens positions
        if self.fast_clock % self.macro_period == 0:
            coherence_gate = self._compute_coherence_gate()
            for layer_idx, lens in self.lenses.items():
                # Drift is proportional to pressure
                drift = lens.pressure
                lens.update(drift, coherence_gate)

        # Build state snapshot
        state = ChronovisorMixtralState(
            lens_vectors={i: lens.vector.copy() for i, lens in self.lenses.items()},
            lens_magnitude={i: lens.magnitude for i, lens in self.lenses.items()},
            expert_gains={},  # TODO: compute from routing stats
            expert_stability={},  # TODO: compute from routing variance
            expert_usage={i: usage.copy() for i, usage in self.expert_usage.items()},
            routing_entropy={},  # TODO: compute from routing stats
            coherence=self.coherence_R,  # Kuramoto R
            delta_coherence=self.delta_coherence,
            coherence_history=self.coherence_history.copy(),
            # Geological temperature tracking
            T_bar=self.structural_T_global.copy(),  # Global T̄
            T_bar_local={i: lens.structural_T.copy() for i, lens in self.lenses.items()},
            T_bar_hierarchical={i: lens.structural_T_hierarchical.copy() for i, lens in self.lenses.items()},
            T_effective={i: lens.temperature_effective.copy() for i, lens in self.lenses.items()},
            fast_clock=self.fast_clock,
            micro_clock=self.micro_clock,
            macro_clock=self.macro_clock,
        )

        return state

    def _compute_kuramoto_coherence(self) -> Tuple[float, float]:
        """
        Compute Kuramoto order parameter R across all layers.

        R measures phase synchronization of experts:
            R = 1: Perfect synchronization (high coherence)
            R = 0: Complete desynchronization (low coherence)

        This is the patent-compliant coherence metric.

        Returns:
            Tuple of (R, psi) where R is coherence and psi is mean phase
        """
        all_phases = []
        for layer_idx in range(self.config.num_layers):
            if layer_idx in self.expert_phases:
                all_phases.extend(self.expert_phases[layer_idx])

        if not all_phases:
            return 0.0, 0.0

        phases_array = np.array(all_phases)
        R, psi = compute_kuramoto_R_and_psi(phases_array)

        return R, psi

    def _compute_coherence_gate(self) -> float:
        """
        Compute gating factor based on coherence trajectory.

        If coherence is improving (Δcoherence > 0), gate opens (closer to 1).
        If coherence is degrading, gate closes (closer to 0).

        Returns:
            Gate value (0-1)
        """
        # Simple sigmoid gating
        return 1.0 / (1.0 + np.exp(-5.0 * self.delta_coherence))

    def get_pressure_for_layer(self, layer_idx: int) -> torch.Tensor:
        """
        Get current pressure bias for a specific layer.

        Applies meta-knob modulation if enabled:
            P_effective = P_raw × pressure_scale × κ_pressure_scale

        Args:
            layer_idx: Layer index

        Returns:
            Pressure tensor (n_experts,) ready to add to router logits
        """
        if layer_idx not in self.lenses:
            return torch.zeros(self.config.num_experts)

        pressure = self.lenses[layer_idx].pressure * self.pressure_scale

        # Apply meta-knob modulation
        if self.meta_knob is not None:
            factors = self.meta_knob.get_factors()
            pressure = pressure * factors.pressure_scale

        return torch.from_numpy(pressure).float()

    def get_temperature_for_layer(self, layer_idx: int) -> torch.Tensor:
        """
        Get current temperature field for a specific layer.

        Applies meta-knob modulation if enabled:
            T_effective = T_raw × temperature_scale × κ_temp_scale

        Args:
            layer_idx: Layer index

        Returns:
            Temperature tensor (n_experts,) ready to divide router logits
        """
        if layer_idx not in self.lenses:
            return torch.ones(self.config.num_experts)

        temperature = self.lenses[layer_idx].temperature_effective * self.temperature_scale

        # Apply meta-knob modulation
        if self.meta_knob is not None:
            factors = self.meta_knob.get_factors()
            temperature = temperature * factors.temp_scale

        return torch.from_numpy(temperature).float()

    # =========================================================================
    # Meta-Knob Interface
    # =========================================================================

    def set_knob(self, kappa: float, intent: str = "") -> KnobFactors:
        """
        Set the meta-knob value.

        The knob κ ∈ [-1, +1] modulates:
            κ > 0: More pressure, more exploration, higher temperature
            κ < 0: Less pressure, more exploitation, lower temperature
            κ = 0: Baseline behavior

        Args:
            kappa: Meta-knob value in [-1, +1]
            intent: Optional intent description

        Returns:
            KnobFactors with the computed multiplicative factors
        """
        if self.meta_knob is None:
            return KnobFactors(
                kappa=0.0,
                pressure_scale=1.0,
                explore_bias=1.0,
                alignment_lr_mul=1.0,
                temp_scale=1.0,
            )

        return self.meta_knob.set_kappa(kappa, intent)

    def apply_knob_decision(self, decision: KnobDecision) -> KnobFactors:
        """
        Apply a knob decision from an LLM controller.

        Args:
            decision: KnobDecision with κ and intent

        Returns:
            KnobFactors with the computed multiplicative factors
        """
        if self.meta_knob is None:
            return KnobFactors(
                kappa=0.0,
                pressure_scale=1.0,
                explore_bias=1.0,
                alignment_lr_mul=1.0,
                temp_scale=1.0,
            )

        return self.meta_knob.apply_decision(decision)

    def get_knob_state(self, loss: Optional[float] = None) -> KnobState:
        """
        Get current system state for knob decision-making.

        This provides a compact summary of the system state that can be
        fed to an LLM controller to decide on κ.

        Args:
            loss: Optional current loss value

        Returns:
            KnobState for LLM consumption
        """
        # Update loss tracking
        if loss is not None:
            self.current_loss = loss
            self.loss_history.append(loss)

        # Compute loss trend
        if len(self.loss_history) >= 2:
            loss_trend = self.loss_history[-1] - self.loss_history[-2]
        else:
            loss_trend = 0.0

        # Compute routing entropy (average across layers)
        routing_entropy = 0.0
        for layer_idx, usage in self.expert_usage.items():
            total = usage.sum()
            if total > 0:
                p = usage / total
                p = p[p > 0]  # Avoid log(0)
                layer_entropy = -np.sum(p * np.log(p + 1e-10))
                routing_entropy += layer_entropy
        routing_entropy /= len(self.expert_usage) if self.expert_usage else 1

        # Alignment entropy (using structural T variance as proxy)
        structural_T_var = np.var(self.structural_T_global)
        alignment_entropy = np.log(1 + structural_T_var * 10)  # Scaled

        # Drift correlation (not directly available, use coherence as proxy)
        drift_correlation = self.coherence_R

        return KnobState(
            loss=self.current_loss,
            loss_trend=loss_trend,
            routing_entropy=routing_entropy,
            alignment_entropy=alignment_entropy,
            drift_correlation=drift_correlation,
            coherence_R=self.coherence_R,
            population=self.config.num_experts * self.config.num_layers,
            avg_pressure_magnitude=np.mean([
                np.mean(np.abs(lens.pressure)) for lens in self.lenses.values()
            ]) if self.lenses else 0.0,
        )

    def get_knob_factors(self) -> KnobFactors:
        """
        Get current meta-knob factors.

        Returns:
            Current KnobFactors or neutral factors if knob is disabled
        """
        if self.meta_knob is None:
            return KnobFactors(
                kappa=0.0,
                pressure_scale=1.0,
                explore_bias=1.0,
                alignment_lr_mul=1.0,
                temp_scale=1.0,
            )

        return self.meta_knob.get_factors()

    def get_structural_temperature_diagnostics(self, layer_idx: Optional[int] = None) -> dict:
        """
        Get detailed structural temperature diagnostics for visualization.

        Analyzes the geological landscape formed by structural temperatures:
        - Variance indicates landscape differentiation
        - Low T̄ regions = valleys (stable experts)
        - High T̄ regions = ridges (unstable experts)

        Args:
            layer_idx: Specific layer to analyze, or None for global T̄

        Returns:
            Dictionary with structural temperature analysis.
        """
        if layer_idx is not None and layer_idx in self.lenses:
            structural_T = self.lenses[layer_idx].structural_T
            label = f"layer_{layer_idx}"
        else:
            structural_T = self.structural_T_global
            label = "global"

        n_experts = len(structural_T)

        # Basic stats
        variance = float(np.var(structural_T))
        mean = float(np.mean(structural_T))
        std = float(np.std(structural_T))

        # Normalized entropy of structural temperature distribution
        # Low entropy = specialized landscape (clear valleys/ridges)
        # High entropy = flat landscape
        normalized = structural_T / structural_T.sum()
        entropy = float(-np.sum(normalized * np.log(normalized + 1e-10)))
        max_entropy = np.log(n_experts)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Identify valleys (low T, stable regions) and ridges (high T, unstable)
        threshold_low = mean - 0.5 * std
        threshold_high = mean + 0.5 * std

        valleys = [i for i, t in enumerate(structural_T) if t < threshold_low]
        ridges = [i for i, t in enumerate(structural_T) if t > threshold_high]

        # Landscape is "formed" when variance exceeds threshold
        landscape_formed = bool(variance > 0.01)

        return {
            "label": label,
            "structural_T": structural_T.copy(),
            "variance": variance,
            "mean": mean,
            "std": std,
            "entropy": normalized_entropy,
            "landscape_formed": landscape_formed,
            "valleys": valleys,  # Experts that became stable (low T)
            "ridges": ridges,  # Experts that became unstable (high T)
            "evolution_steps": self.fast_clock,
        }

    def get_valley_health_diagnostics(self, layer_idx: Optional[int] = None) -> dict:
        """
        Monitor valley health to detect "bad valleys" (low T̄ + low reliability).

        This diagnostic helps verify that the self-correction mechanism is working:
        - Bad experts should have high T_fast (behavioral avoidance)
        - Over time, their T̄ should rise (structural correction)
        - Persistent bad valleys indicate the mechanism is failing

        The design relies on reliability → T_fast → T̄ feedback, not asymmetric
        erosion. This diagnostic lets us monitor if that assumption holds.

        For Mixtral, we use expert usage as a proxy for reliability:
        - Frequently used experts = reliable (low T desired)
        - Rarely used experts = unreliable (high T expected)

        Args:
            layer_idx: Specific layer to analyze, or None for global analysis

        Returns:
            Dictionary with valley health analysis.
        """
        n_experts = self.config.num_experts

        # Get structural temperature
        if layer_idx is not None and layer_idx in self.lenses:
            structural_T = self.lenses[layer_idx].structural_T
            usage = self.expert_usage.get(layer_idx, np.zeros(n_experts))
            label = f"layer_{layer_idx}"
        else:
            structural_T = self.structural_T_global
            # Aggregate usage across all layers
            usage = np.zeros(n_experts)
            for u in self.expert_usage.values():
                usage += u
            label = "global"

        # Identify valleys based on structural temperature
        mean_T = float(np.mean(structural_T))
        std_T = float(np.std(structural_T))
        threshold_valley = mean_T - 0.5 * std_T

        valleys = [i for i in range(n_experts) if structural_T[i] < threshold_valley]

        # Compute reliability proxy from usage
        # More usage = more reliable
        total_usage = usage.sum()
        if total_usage > 0:
            reliabilities = usage / total_usage * n_experts  # Normalize so average = 1.0
            reliabilities = np.clip(reliabilities, 0.0, 2.0) / 2.0  # Scale to [0, 1]
        else:
            reliabilities = np.ones(n_experts) * 0.5  # Neutral if no usage data

        # Valley health = reliability (for valleys only)
        # High reliability + valley = good (healthy valley)
        # Low reliability + valley = bad (unhealthy valley, should self-correct)
        valley_health = {}
        healthy_valleys = []
        unhealthy_valleys = []

        reliability_threshold = 0.3  # Below this = unhealthy

        for v in valleys:
            health = float(reliabilities[v])
            valley_health[v] = health
            if health >= reliability_threshold:
                healthy_valleys.append(v)
            else:
                unhealthy_valleys.append(v)

        # Identify "at risk" experts: low usage but still low T̄
        # These are experts trending toward bad valley if T_fast stays high
        at_risk_experts = []
        if layer_idx is not None and layer_idx in self.lenses:
            fast_T = self.lenses[layer_idx].temperature_fast
        else:
            # Average fast T across layers
            fast_T = np.zeros(n_experts)
            for lens in self.lenses.values():
                fast_T += lens.temperature_fast
            fast_T /= len(self.lenses) if self.lenses else 1

        for i in range(n_experts):
            is_hot = fast_T[i] > mean_T + 0.5 * std_T
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
            "label": label,
            "valleys": valleys,
            "valley_health": valley_health,
            "healthy_valleys": healthy_valleys,
            "unhealthy_valleys": unhealthy_valleys,
            "at_risk_experts": at_risk_experts,
            "mean_valley_health": mean_health,
            "reliabilities": {i: float(r) for i, r in enumerate(reliabilities)},
            "self_correction_working": self_correction_working,
        }

    def get_diagnostics(self) -> dict:
        """
        Get comprehensive diagnostic information about controller state.

        Returns:
            Dictionary with all diagnostic metrics.
        """
        # Basic state
        diagnostics = {
            "fast_clock": self.fast_clock,
            "micro_clock": self.micro_clock,
            "macro_clock": self.macro_clock,
            "kuramoto_R": self.coherence_R,
            "delta_R": self.delta_coherence,
            "mean_phase_psi": self.mean_phase_psi,
            "coherence_history_len": len(self.coherence_history),
        }

        # Per-layer lens states
        diagnostics["lens_magnitudes"] = {
            i: lens.magnitude for i, lens in self.lenses.items()
        }

        # Global structural temperature stats
        diagnostics["structural_T_global_mean"] = float(np.mean(self.structural_T_global))
        diagnostics["structural_T_global_std"] = float(np.std(self.structural_T_global))
        diagnostics["structural_T_global_min"] = float(np.min(self.structural_T_global))
        diagnostics["structural_T_global_max"] = float(np.max(self.structural_T_global))

        # Structural temperature diagnostics
        diagnostics["structural_T_diagnostics"] = self.get_structural_temperature_diagnostics()

        # Valley health
        diagnostics["valley_health"] = self.get_valley_health_diagnostics()

        # Meta-knob state
        if self.meta_knob is not None:
            diagnostics["meta_knob"] = self.meta_knob.get_diagnostics()
        else:
            diagnostics["meta_knob"] = {"enabled": False}

        return diagnostics

    def reset(self) -> None:
        """Reset all controller state."""
        for lens in self.lenses.values():
            lens.vector = np.zeros(lens.n_experts)
            lens.magnitude = 0.0
            lens.pressure = np.zeros(lens.n_experts)
            lens.temperature_fast = np.ones(lens.n_experts) * lens.base_temperature
            lens.structural_T = np.ones(lens.n_experts) * lens.base_temperature
            lens.structural_T_hierarchical = np.ones(lens.n_experts) * lens.base_temperature
            lens.temperature_effective = np.ones(lens.n_experts) * lens.base_temperature

        # Reset global structural T
        self.structural_T_global = np.ones(self.config.num_experts)

        self.fast_clock = 0
        self.micro_clock = 0
        self.macro_clock = 0
        self.coherence_R = 0.0
        self.delta_coherence = 0.0
        self.coherence_history = []

        for layer_idx in self.expert_usage:
            self.expert_usage[layer_idx] = np.zeros(self.config.num_experts)
        for layer_idx in self.expert_phases:
            self.expert_phases[layer_idx] = np.random.uniform(0, 2 * np.pi, self.config.num_experts)

        # Reset meta-knob
        if self.meta_knob is not None:
            self.meta_knob.reset()

        # Reset loss tracking
        self.loss_history = []
        self.current_loss = 0.0


class ChronovisorMixtralModel(nn.Module):
    """
    Mixtral model with Chronovisor geometric control layer.

    This wraps a stack of Mixtral layers and integrates them with
    the Chronovisor controller for adaptive routing.
    """

    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.config = config

        # Create Mixtral layers
        self.layers = nn.ModuleList([
            MixtralDecoderLayer(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # Chronovisor controller
        self.controller = ChronovisorMixtralController(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_chronovisor: bool = True,
    ) -> Tuple[torch.Tensor, ChronovisorMixtralState]:
        """
        Forward pass through the model with Chronovisor updates.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            attention_mask: Optional attention mask
            update_chronovisor: Whether to update controller state

        Returns:
            Tuple of (final_hidden_states, chronovisor_state)
        """
        all_routing_stats = {}

        # Apply P×T fields from controller to each layer
        for layer_idx, layer in enumerate(self.layers):
            if update_chronovisor and self.config.enable_chronovisor:
                device = hidden_states.device
                pressure = self.controller.get_pressure_for_layer(layer_idx).to(device, non_blocking=True)
                temperature = self.controller.get_temperature_for_layer(layer_idx).to(device, non_blocking=True)
                layer.moe.pressure_bias = pressure
                layer.moe.temperature_field = temperature

            # Forward through layer
            hidden_states, routing_stats = layer(hidden_states, attention_mask)
            all_routing_stats[layer_idx] = routing_stats

        # Update controller
        chronovisor_state = None
        if update_chronovisor and self.config.enable_chronovisor:
            chronovisor_state = self.controller.tick(all_routing_stats)

        return hidden_states, chronovisor_state


class ChronovisorMixtralForCausalLM(nn.Module):
    """
    Mixtral language model with Chronovisor P×T geometric control.

    Complete causal LM with:
        - Token embeddings
        - Chronovisor-controlled Mixtral decoder stack
        - Language modeling head

    Usage:
        model = ChronovisorMixtralForCausalLM(config)
        logits, chrono_state = model(input_ids)
    """

    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Chronovisor Mixtral decoder
        self.model = ChronovisorMixtralModel(config)

        # Language modeling head (often tied with embed_tokens.weight)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Tie weights (common practice in LMs)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_chronovisor: bool = True,
    ) -> Tuple[torch.Tensor, Optional[ChronovisorMixtralState]]:
        """
        Forward pass through the language model.

        Args:
            input_ids: Token indices (batch, seq_len)
            attention_mask: Optional attention mask
            update_chronovisor: Whether to update controller state

        Returns:
            Tuple of (logits, chronovisor_state)
                - logits: (batch, seq_len, vocab_size)
                - chronovisor_state: Current Chronovisor state or None
        """
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Forward through Chronovisor Mixtral
        hidden_states, chronovisor_state = self.model(
            hidden_states,
            attention_mask=attention_mask,
            update_chronovisor=update_chronovisor,
        )

        # Project to vocabulary
        logits = self.lm_head(hidden_states)

        return logits, chronovisor_state

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Tuple[torch.Tensor, List[ChronovisorMixtralState]]:
        """
        Simple greedy/sampling generation.

        Args:
            input_ids: Initial token indices (batch, seq_len)
            max_length: Maximum total sequence length
            temperature: Sampling temperature
            top_k: Number of top tokens to sample from

        Returns:
            Tuple of (generated_ids, chronovisor_states)
        """
        batch_size = input_ids.shape[0]
        current_ids = input_ids.clone()
        chronovisor_states = []

        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            logits, chrono_state = self.forward(current_ids)
            chronovisor_states.append(chrono_state)

            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('inf')

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            current_ids = torch.cat([current_ids, next_token], dim=-1)

            # Break if EOS token (if applicable)
            # if (next_token == eos_token_id).all():
            #     break

        return current_ids, chronovisor_states


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == '__main__':
    # Create small config for testing
    config = MixtralConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=512,
        num_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=4,
        num_experts_per_token=2,
        max_seq_length=512,
        enable_chronovisor=True,
    )

    print("=" * 60)
    print("Testing ChronovisorMixtralModel (hidden states)")
    print("=" * 60)

    # Create model
    model = ChronovisorMixtralModel(config)

    # Test forward pass
    batch_size = 2
    seq_len = 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)

    print("Running forward pass with Chronovisor...")
    output, chrono_state = model(hidden_states)

    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Kuramoto R: {chrono_state.coherence:.4f}")
    print(f"✓ Δ R: {chrono_state.delta_coherence:.4f}")
    print(f"✓ Fast clock: {chrono_state.fast_clock}")

    # Run a few more ticks to see coherence evolve
    print("\nRunning 10 more forward passes...")
    for i in range(10):
        output, chrono_state = model(hidden_states)
        print(f"Tick {i+2}: R={chrono_state.coherence:.4f}, Δ={chrono_state.delta_coherence:.4f}")

    print("\n" + "=" * 60)
    print("Testing ChronovisorMixtralForCausalLM (language model)")
    print("=" * 60)

    # Create language model
    lm = ChronovisorMixtralForCausalLM(config)

    # Test with token IDs
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    print("Running forward pass through language model...")
    logits, chrono_state = lm(input_ids)

    print(f"✓ Logits shape: {logits.shape}")
    print(f"✓ Kuramoto R: {chrono_state.coherence:.4f}")
    print(f"✓ Δ R: {chrono_state.delta_coherence:.4f}")
    print(f"✓ Vocab size: {config.vocab_size}")

    print("\n" + "=" * 60)
    print("Testing Diagnostics and Meta-Knob")
    print("=" * 60)

    controller = model.controller

    # Test structural temperature diagnostics
    print("\nStructural Temperature Diagnostics (Global):")
    st_diag = controller.get_structural_temperature_diagnostics()
    print(f"  Variance: {st_diag['variance']:.6f}")
    print(f"  Landscape formed: {st_diag['landscape_formed']}")
    print(f"  Valleys: {st_diag['valleys']}")
    print(f"  Ridges: {st_diag['ridges']}")

    # Test valley health diagnostics
    print("\nValley Health Diagnostics:")
    vh_diag = controller.get_valley_health_diagnostics()
    print(f"  Healthy valleys: {vh_diag['healthy_valleys']}")
    print(f"  Unhealthy valleys: {vh_diag['unhealthy_valleys']}")
    print(f"  Self-correction working: {vh_diag['self_correction_working']}")

    # Test meta-knob
    print("\nMeta-Knob Test:")
    print(f"  Initial κ: {controller.get_knob_factors().kappa}")

    # Set knob to explore mode
    factors = controller.set_knob(0.5, intent="explore")
    print(f"  After κ=0.5 (explore):")
    print(f"    pressure_scale: {factors.pressure_scale:.3f}")
    print(f"    temp_scale: {factors.temp_scale:.3f}")

    # Set knob to exploit mode
    factors = controller.set_knob(-0.5, intent="exploit")
    print(f"  After κ=-0.5 (exploit):")
    print(f"    pressure_scale: {factors.pressure_scale:.3f}")
    print(f"    temp_scale: {factors.temp_scale:.3f}")

    # Get knob state for LLM
    knob_state = controller.get_knob_state(loss=0.5)
    print(f"\nKnob State for LLM:")
    print(knob_state.to_prompt())

    # Full diagnostics
    print("\nFull Diagnostics:")
    diag = controller.get_diagnostics()
    print(f"  Kuramoto R: {diag['kuramoto_R']:.4f}")
    print(f"  Meta-knob κ: {diag['meta_knob'].get('kappa', 'N/A')}")

    print("\n✓ P×T Mixtral integration complete!")
    print("  - Pressure field (P): Biases routing toward/away from experts")
    print("  - Temperature field (T): Controls routing sharpness/diffuseness")
    print("  - Structural T̄: Geological memory via EMA (global + local)")
    print("  - Kuramoto R: Patent-compliant coherence metric")
    print("  - Valley health diagnostics: Monitors self-correction")
    print("  - Meta-knob κ: LLM-controlled pressure/temperature modulation")
    print("  - Language modeling: Embeddings + LM head ready for training")
