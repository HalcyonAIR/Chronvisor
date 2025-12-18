"""
Chronovisor ↔ DeepSeek Integration Bridge

Connects Chronovisor's P×T coupling with DeepSeek-MoE's shared+routed architecture.

Key challenge: DeepSeek has TWO types of experts:
1. Shared experts (always activated, no routing)
2. Routed experts (sparse top-k routing)

Chronovisor P×T coupling applies to ROUTED experts only.
Shared experts provide stable base - they don't need geometric control.

This tests whether P×T coupling works on radically different architecture:
- Fine-grained experts (64 vs Mixtral's 8)
- Hybrid shared+routed design
- Different load balancing mechanism
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from chronomoe.deepseek_core import (
    DeepSeekConfig,
    DeepSeekDecoderLayer,
    DeepSeekSparseMoELayer,
)


@dataclass
class ChronovisorDeepSeekState:
    """
    State of Chronovisor-DeepSeek system.

    Tracks P×T coupling for ROUTED experts only.
    Shared experts are always active and don't need geometric control.
    """

    # Pressure (fast bias for routed experts)
    pressure: Dict[int, np.ndarray] = field(default_factory=dict)  # {layer_idx: [num_routed_experts]}

    # Temperature (slow structural variable)
    T_bar_local: Dict[int, np.ndarray] = field(default_factory=dict)  # Per-layer geological T
    T_effective: Dict[int, np.ndarray] = field(default_factory=dict)  # Effective temperature

    # Routing statistics (routed experts only)
    expert_usage: Dict[int, np.ndarray] = field(default_factory=dict)  # {layer_idx: [num_routed_experts]}
    routing_entropy: Dict[int, float] = field(default_factory=dict)

    # Shared expert statistics (for monitoring only, no P×T coupling)
    shared_expert_activity: Dict[int, float] = field(default_factory=dict)  # Average activation

    # Coherence tracking
    coherence: float = 0.0
    delta_coherence: float = 0.0

    # Clock state
    fast_clock: int = 0
    micro_clock: int = 0
    macro_clock: int = 0


class MixtralLens:
    """
    Geometric lens for P×T coupling.

    (Reusing from Mixtral - same math, different scale)

    Computes:
    - Pressure bias from routing utility
    - Temperature updates from routing statistics
    """

    def __init__(
        self,
        num_experts: int,
        eta_structural_T: float = 0.015,
        pressure_scale: float = 0.5
    ):
        self.num_experts = num_experts
        self.eta_structural_T = eta_structural_T
        self.pressure_scale = pressure_scale

        # Geological temperature (slow variable)
        self.T_bar = np.ones(num_experts)

        # Expert usage tracking (EMA)
        self.expert_usage_ema = np.ones(num_experts) / num_experts
        self.ema_beta = 0.99

    def compute_pressure(self, routing_stats: Dict) -> np.ndarray:
        """
        Compute pressure bias from routing utility.

        Pressure = α * ∇utility
        Where utility encourages specialization and load balance.
        """
        expert_usage = routing_stats.get('expert_usage', np.ones(self.num_experts) / self.num_experts)

        # Pressure encourages underused experts
        avg_usage = np.mean(expert_usage)
        pressure = self.pressure_scale * (avg_usage - expert_usage)

        return pressure

    def update_temperature(self, routing_stats: Dict):
        """
        Update geological temperature based on routing patterns.

        T increases when routing is certain (low entropy).
        T decreases when routing is uncertain (high entropy).
        """
        expert_usage = routing_stats.get('expert_usage', np.ones(self.num_experts) / self.num_experts)

        # Update EMA of expert usage
        self.expert_usage_ema = (
            self.ema_beta * self.expert_usage_ema +
            (1 - self.ema_beta) * expert_usage
        )

        # Compute routing certainty (how peaked is the usage distribution)
        usage_variance = np.var(self.expert_usage_ema)

        # T̄ increases with certainty, decreases with uncertainty
        # High usage variance → certain routing → increase T
        # Low usage variance → uncertain routing → decrease T
        delta_T = self.eta_structural_T * (usage_variance - 0.1)  # 0.1 is baseline

        self.T_bar = self.T_bar + delta_T

        # Keep T̄ in reasonable range
        self.T_bar = np.clip(self.T_bar, 0.1, 10.0)


class ChronovisorDeepSeekController:
    """
    Multi-timescale controller for DeepSeek-MoE.

    Manages P×T coupling across:
    - Fast: Per-token pressure updates
    - Micro: Per-turn temperature updates
    - Macro: Global structure refinement

    Key difference from Mixtral: Only controls ROUTED experts.
    Shared experts provide stable base without geometric control.
    """

    def __init__(self, config: DeepSeekConfig):
        self.config = config
        self.num_layers = config.num_layers
        self.num_routed_experts = config.num_routed_experts

        # Create lens for each layer (one per layer of routed experts)
        self.lenses: Dict[int, MixtralLens] = {}
        for layer_idx in range(config.num_layers):
            self.lenses[layer_idx] = MixtralLens(
                num_experts=config.num_routed_experts,
                eta_structural_T=0.015,  # Geological learning rate
                pressure_scale=0.5,
            )

        # Learning rates
        self.eta_structural_T_local = 0.015
        self.eta_structural_T_global = 0.0075
        self.pressure_scale = 0.5

        # Clock periods
        self.fast_period = 1  # Every step
        self.micro_period = 5  # Every 5 steps
        self.macro_period = 20  # Every 20 steps

        # Clock state
        self.fast_clock = 0
        self.micro_clock = 0
        self.macro_clock = 0

    def tick(self) -> Dict[str, bool]:
        """Advance clocks and return which ticked."""
        self.fast_clock += 1
        ticked = {'fast': True, 'micro': False, 'macro': False}

        if self.fast_clock % self.micro_period == 0:
            self.micro_clock += 1
            ticked['micro'] = True

        if self.fast_clock % self.macro_period == 0:
            self.macro_clock += 1
            ticked['macro'] = True

        return ticked

    def compute_pressure_biases(
        self,
        routing_stats: Dict[int, Dict]
    ) -> Dict[int, np.ndarray]:
        """
        Compute pressure biases for routed experts.

        Returns:
            {layer_idx: pressure_bias[num_routed_experts]}
        """
        pressure_biases = {}

        for layer_idx, stats in routing_stats.items():
            if layer_idx in self.lenses:
                pressure = self.lenses[layer_idx].compute_pressure(stats)
                pressure_biases[layer_idx] = pressure

        return pressure_biases

    def update_temperatures(
        self,
        routing_stats: Dict[int, Dict],
        ticked: Dict[str, bool]
    ):
        """Update geological temperatures based on routing patterns."""
        # Only update on micro clock ticks
        if not ticked['micro']:
            return

        for layer_idx, stats in routing_stats.items():
            if layer_idx in self.lenses:
                self.lenses[layer_idx].update_temperature(stats)

    def get_state(self, routing_stats: Dict[int, Dict] = None) -> ChronovisorDeepSeekState:
        """Get current system state."""
        state = ChronovisorDeepSeekState()

        if routing_stats is None:
            routing_stats = {}

        for layer_idx, lens in self.lenses.items():
            # Pressure
            layer_stats = routing_stats.get(layer_idx, {})
            state.pressure[layer_idx] = lens.compute_pressure(layer_stats)

            # Temperature
            state.T_bar_local[layer_idx] = lens.T_bar.copy()
            state.T_effective[layer_idx] = lens.T_bar.copy()

            # Routing statistics (if available)
            if 'expert_usage' in layer_stats:
                state.expert_usage[layer_idx] = layer_stats['expert_usage']
            if 'routing_entropy' in layer_stats:
                state.routing_entropy[layer_idx] = layer_stats['routing_entropy']

        # Clocks
        state.fast_clock = self.fast_clock
        state.micro_clock = self.micro_clock
        state.macro_clock = self.macro_clock

        return state


class ChronovisorDeepSeekModel(nn.Module):
    """
    DeepSeek-MoE model with Chronovisor P×T coupling.

    Tests whether P×T coupling works on radically different architecture:
    - Shared + routed experts (vs Mixtral's all-routed)
    - Fine-grained routing (64 experts vs 8)
    - Different load balancing

    If this works, it proves P×T coupling is general, not Mixtral-specific.
    """

    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config

        # DeepSeek layers
        self.layers = nn.ModuleList([
            DeepSeekDecoderLayer(config) for _ in range(config.num_layers)
        ])

        # Chronovisor controller
        self.controller = ChronovisorDeepSeekController(config) if config.enable_chronovisor else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_chronovisor: bool = True
    ) -> Tuple[torch.Tensor, Optional[ChronovisorDeepSeekState], torch.Tensor]:
        """
        Forward pass through DeepSeek with Chronovisor control.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            attention_mask: Optional attention mask
            update_chronovisor: Whether to update P×T coupling

        Returns:
            hidden_states: Final layer output
            chrono_state: Chronovisor state (if enabled)
            total_aux_loss: Sum of all MoE auxiliary losses
        """
        total_aux_loss = 0.0
        routing_stats = {}

        # Get pressure biases if Chronovisor enabled
        pressure_biases = {}
        if self.controller is not None and update_chronovisor:
            # Tick clocks
            ticked = self.controller.tick()

            # Compute pressure biases for this forward pass
            # (Would need routing stats from previous pass - simplified here)
            pressure_biases = self.controller.compute_pressure_biases(routing_stats)

        # Forward through layers
        for layer_idx, layer in enumerate(self.layers):
            # Get pressure bias for this layer
            pressure_bias = None
            if layer_idx in pressure_biases:
                pressure_bias = torch.from_numpy(pressure_biases[layer_idx]).float().to(hidden_states.device)

            # Layer forward
            hidden_states, aux_loss, layer_routing_stats = layer(
                hidden_states,
                attention_mask=attention_mask,
                pressure_bias=pressure_bias
            )

            total_aux_loss = total_aux_loss + aux_loss

            # Collect routing statistics
            routing_stats[layer_idx] = layer_routing_stats

        # Update temperatures if Chronovisor enabled
        if self.controller is not None and update_chronovisor:
            self.controller.update_temperatures(routing_stats, ticked)

        # Get current state (with routing stats)
        chrono_state = self.controller.get_state(routing_stats) if self.controller is not None else None

        return hidden_states, chrono_state, total_aux_loss


class ChronovisorDeepSeekForCausalLM(nn.Module):
    """
    DeepSeek language model with Chronovisor P×T coupling.

    Complete LM with embeddings + DeepSeek decoder + LM head.
    """

    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)

        # DeepSeek decoder with Chronovisor
        self.model = ChronovisorDeepSeekModel(config)

        # LM head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_chronovisor: bool = True
    ) -> Tuple[torch.Tensor, Optional[ChronovisorDeepSeekState], torch.Tensor]:
        """
        Forward pass through LM.

        Returns:
            logits: [batch, seq_len, vocab_size]
            chrono_state: Chronovisor state
            aux_loss: MoE auxiliary loss
        """
        # Embed
        hidden_states = self.embed_tokens(input_ids)

        # Decode
        hidden_states, chrono_state, aux_loss = self.model(
            hidden_states,
            attention_mask=attention_mask,
            update_chronovisor=update_chronovisor
        )

        # Project to vocab
        logits = self.lm_head(hidden_states)

        return logits, chrono_state, aux_loss

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute language modeling loss + MoE auxiliary loss.

        Returns:
            total_loss: LM loss + aux loss
            aux_loss: Auxiliary loss (for monitoring)
        """
        # Forward
        logits, _, aux_loss = self.forward(input_ids, attention_mask)

        # LM loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        lm_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )

        # Total loss
        total_loss = lm_loss + aux_loss

        return total_loss, aux_loss


# Export
__all__ = [
    'ChronovisorDeepSeekState',
    'ChronovisorDeepSeekController',
    'ChronovisorDeepSeekModel',
    'ChronovisorDeepSeekForCausalLM',
]
