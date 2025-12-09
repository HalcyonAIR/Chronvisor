"""
Chronovisor ↔ Mixtral Integration Bridge

Connects Chronovisor's geometric control layer (lens, controller) with
the internal Mixtral MoE implementation.

Flow:
    1. Mixtral layers forward pass → routing statistics
    2. Bridge translates routing stats → Chronovisor expert signals
    3. Controller computes Δcoherence → lens updates
    4. Lens state → pressure bias → Mixtral router

This creates a closed loop where:
    - Mixtral's routing behavior informs the geometric lens
    - The lens reshapes future routing decisions
    - Expert specialization emerges from geometric stability
"""

from __future__ import annotations

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
    Geometric lens for a single Mixtral MoE layer.

    The lens is a deformable surface in expert space that reshapes
    how the router sees semantic regions.

    State:
        - vector: Current position in expert geometry space (n_experts,)
        - gamma: Adaptation rate
        - pressure: Current pressure bias to apply to routing
    """

    def __init__(self, n_experts: int = 8, gamma: float = 0.05):
        """
        Initialize lens for one layer.

        Args:
            n_experts: Number of experts in this layer
            gamma: Adaptation rate (how quickly lens moves)
        """
        self.n_experts = n_experts
        self.gamma = gamma

        # Lens state
        self.vector = np.zeros(n_experts)
        self.magnitude = 0.0

        # Pressure bias (set by controller)
        self.pressure = np.zeros(n_experts)

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

    def get_state(self) -> dict:
        """Get current lens state."""
        return {
            'vector': self.vector.copy(),
            'magnitude': self.magnitude,
            'pressure': self.pressure.copy(),
        }


class ChronovisorMixtralController:
    """
    Controller for the Chronovisor-Mixtral system.

    Manages:
        - Multi-scale clocks (fast, micro, macro)
        - Per-layer lenses
        - Coherence computation
        - Pressure updates
    """

    def __init__(
        self,
        config: MixtralConfig,
        micro_period: int = 5,
        macro_period: int = 20,
        pressure_scale: float = 0.1,
    ):
        """
        Initialize controller.

        Args:
            config: Mixtral model configuration
            micro_period: Ticks between micro-scale updates
            macro_period: Ticks between macro-scale updates
            pressure_scale: How strongly to apply pressure to routing
        """
        self.config = config
        self.micro_period = micro_period
        self.macro_period = macro_period
        self.pressure_scale = pressure_scale

        # Create lenses for each layer
        self.lenses: Dict[int, MixtralLens] = {
            i: MixtralLens(n_experts=config.num_experts)
            for i in range(config.num_layers)
        }

        # Clock state
        self.fast_clock = 0
        self.micro_clock = 0
        self.macro_clock = 0

        # Coherence tracking
        self.coherence = 0.0
        self.delta_coherence = 0.0
        self.coherence_history: List[float] = []

        # Expert usage tracking (per layer)
        self.expert_usage: Dict[int, np.ndarray] = {
            i: np.zeros(config.num_experts)
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

        # Compute coherence
        prev_coherence = self.coherence
        self.coherence = self._compute_coherence(routing_stats)
        self.delta_coherence = self.coherence - prev_coherence
        self.coherence_history.append(self.coherence)

        # On micro boundaries, update lens pressure
        if self.fast_clock % self.micro_period == 0:
            for layer_idx, lens in self.lenses.items():
                lens.compute_pressure(self.expert_usage[layer_idx])

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
            coherence=self.coherence,
            delta_coherence=self.delta_coherence,
            coherence_history=self.coherence_history.copy(),
            fast_clock=self.fast_clock,
            micro_clock=self.micro_clock,
            macro_clock=self.macro_clock,
        )

        return state

    def _compute_coherence(self, routing_stats: Dict[int, dict]) -> float:
        """
        Compute system coherence from routing statistics.

        Coherence measures how "aligned" expert selection is across layers.
        Higher coherence = experts are settling into stable patterns.

        Args:
            routing_stats: Routing statistics from all layers

        Returns:
            Coherence value (0-1)
        """
        if not routing_stats:
            return 0.0

        # Simple coherence: entropy of routing distributions
        # Low entropy = high coherence (concentrated on few experts)
        # High entropy = low coherence (spread across many experts)

        total_entropy = 0.0
        for layer_idx, stats in routing_stats.items():
            selected = stats['selected_experts'].detach().cpu().numpy()
            weights = stats['routing_weights'].detach().cpu().numpy()

            # Flatten to get overall distribution
            all_selections = selected.flatten()
            all_weights = weights.flatten()

            # Compute distribution
            dist = np.zeros(self.config.num_experts)
            for expert_id in range(self.config.num_experts):
                mask = (all_selections == expert_id)
                dist[expert_id] = all_weights[mask].sum()

            # Normalize
            if dist.sum() > 0:
                dist = dist / dist.sum()

            # Entropy
            entropy = -np.sum(dist * np.log(dist + 1e-10))
            total_entropy += entropy

        avg_entropy = total_entropy / len(routing_stats)
        max_entropy = np.log(self.config.num_experts)

        # Coherence = 1 - normalized_entropy
        coherence = 1.0 - (avg_entropy / max_entropy)

        return coherence

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

        # Apply pressure from controller to each layer
        for layer_idx, layer in enumerate(self.layers):
            if update_chronovisor and self.config.enable_chronovisor:
                pressure = self.controller.get_pressure_for_layer(layer_idx)
                layer.moe.pressure_bias = pressure

            # Forward through layer
            hidden_states, routing_stats = layer(hidden_states, attention_mask)
            all_routing_stats[layer_idx] = routing_stats

        # Update controller
        chronovisor_state = None
        if update_chronovisor and self.config.enable_chronovisor:
            chronovisor_state = self.controller.tick(all_routing_stats)

        return hidden_states, chronovisor_state


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

    # Create model
    model = ChronovisorMixtralModel(config)

    # Test forward pass
    batch_size = 2
    seq_len = 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)

    print("Running forward pass with Chronovisor...")
    output, chrono_state = model(hidden_states)

    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Coherence: {chrono_state.coherence:.4f}")
    print(f"✓ Δ Coherence: {chrono_state.delta_coherence:.4f}")
    print(f"✓ Fast clock: {chrono_state.fast_clock}")

    # Run a few more ticks to see coherence evolve
    print("\nRunning 10 more forward passes...")
    for i in range(10):
        output, chrono_state = model(hidden_states)
        print(f"Tick {i+2}: coherence={chrono_state.coherence:.4f}, Δ={chrono_state.delta_coherence:.4f}")
