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
        self.structural_T = np.ones(n_experts) * base_temperature  # Geological memory (T̄)
        self.temperature_effective = np.ones(n_experts) * base_temperature  # T_eff = T_fast × T̄

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

        # Update structural temperature (geological memory via EMA)
        self.structural_T = (
            (1.0 - self.eta_structural_T) * self.structural_T
            + self.eta_structural_T * self.temperature_fast
        )

        # Effective temperature is element-wise product
        self.temperature_effective = self.temperature_fast * self.structural_T

        return self.temperature_effective

    def get_state(self) -> dict:
        """Get current lens state."""
        return {
            'vector': self.vector.copy(),
            'magnitude': self.magnitude,
            'pressure': self.pressure.copy(),
            'temperature_fast': self.temperature_fast.copy(),
            'structural_T': self.structural_T.copy(),
            'temperature_effective': self.temperature_effective.copy(),
        }


class ChronovisorMixtralController:
    """
    Controller for the Chronovisor-Mixtral system with P×T geometry.

    Manages:
        - Multi-scale clocks (fast, micro, macro)
        - Per-layer lenses with P×T fields
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
    ):
        """
        Initialize controller.

        Args:
            config: Mixtral model configuration
            micro_period: Ticks between micro-scale updates
            macro_period: Ticks between macro-scale updates
            pressure_scale: How strongly to apply pressure to routing
            temperature_scale: Global temperature scale factor
        """
        self.config = config
        self.micro_period = micro_period
        self.macro_period = macro_period
        self.pressure_scale = pressure_scale
        self.temperature_scale = temperature_scale

        # Create lenses for each layer
        self.lenses: Dict[int, MixtralLens] = {
            i: MixtralLens(n_experts=config.num_experts)
            for i in range(config.num_layers)
        }

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
            for expert_id in range(self.config.num_experts):
                count = np.sum(selected == expert_id)
                self.expert_usage[layer_idx][expert_id] += count

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
            for layer_idx, lens in self.lenses.items():
                lens.compute_pressure(self.expert_usage[layer_idx])
                lens.compute_temperature(coherence_R=self.coherence_R)

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
            expert_usage=self.expert_usage.copy(),
            routing_entropy={},  # TODO: compute from routing stats
            coherence=self.coherence_R,  # Kuramoto R
            delta_coherence=self.delta_coherence,
            coherence_history=self.coherence_history.copy(),
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

        Args:
            layer_idx: Layer index

        Returns:
            Pressure tensor (n_experts,) ready to add to router logits
        """
        if layer_idx not in self.lenses:
            return torch.zeros(self.config.num_experts)

        pressure = self.lenses[layer_idx].pressure * self.pressure_scale
        return torch.from_numpy(pressure).float()

    def get_temperature_for_layer(self, layer_idx: int) -> torch.Tensor:
        """
        Get current temperature field for a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            Temperature tensor (n_experts,) ready to divide router logits
        """
        if layer_idx not in self.lenses:
            return torch.ones(self.config.num_experts)

        temperature = self.lenses[layer_idx].temperature_effective * self.temperature_scale
        return torch.from_numpy(temperature).float()

    def reset(self) -> None:
        """Reset all controller state."""
        for lens in self.lenses.values():
            lens.vector = np.zeros(lens.n_experts)
            lens.magnitude = 0.0
            lens.pressure = np.zeros(lens.n_experts)

        self.fast_clock = 0
        self.micro_clock = 0
        self.macro_clock = 0
        self.coherence = 0.0
        self.delta_coherence = 0.0
        self.coherence_history = []

        for layer_idx in self.expert_usage:
            self.expert_usage[layer_idx] = np.zeros(self.config.num_experts)


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
                pressure = self.controller.get_pressure_for_layer(layer_idx)
                temperature = self.controller.get_temperature_for_layer(layer_idx)
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

    print("\n✓ P×T Mixtral integration complete!")
    print("  - Pressure field (P): Biases routing toward/away from experts")
    print("  - Temperature field (T): Controls routing sharpness/diffuseness")
    print("  - Structural T: Geological memory via EMA")
    print("  - Kuramoto R: Patent-compliant coherence metric")
    print("  - Language modeling: Embeddings + LM head ready for training")
