"""
Switch Transformer Core Implementation with P×T Coupling

Implements Google's Switch Transformer architecture:
- Top-1 routing (each token routed to exactly 1 expert)
- Load balancing auxiliary loss
- Simpler routing than Mixtral (no top-k selection)

Key difference from Mixtral:
  Mixtral: top-2 routing (each token → 2 experts)
  Switch:  top-1 routing (each token → 1 expert)

This tests whether P×T coupling generalizes across routing strategies.

Reference: Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models"
           https://arxiv.org/abs/2101.03961
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Reuse components from Mixtral
from chronomoe.mixtral_core import (
    RMSNorm,
    RotaryPositionalEmbedding,
    GroupedQueryAttention,
    SwiGLUFeedForward,
)


@dataclass
class SwitchConfig:
    """Configuration for Switch Transformer architecture."""
    # Model dimensions
    vocab_size: int = 32000
    hidden_dim: int = 4096
    intermediate_dim: int = 14336
    num_layers: int = 32
    max_seq_length: int = 32768

    # Attention
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA
    head_dim: int = 128

    # MoE (Switch uses top-1 routing)
    num_experts: int = 128  # Switch typically uses many experts
    num_experts_per_token: int = 1  # KEY DIFFERENCE: top-1 routing

    # Training
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1e6

    # Load balancing
    router_z_loss_coef: float = 0.001  # Router z-loss coefficient
    router_aux_loss_coef: float = 0.01  # Load balancing loss coefficient

    # Chronovisor integration
    enable_chronovisor: bool = True
    chronovisor_pressure_scale: float = 0.1


class SwitchExpert(nn.Module):
    """
    A single expert in the Switch MoE layer.

    Identical to Mixtral expert - SwiGLU feedforward network.
    """

    def __init__(self, config: SwitchConfig, expert_id: int):
        super().__init__()
        self.expert_id = expert_id
        self.ffn = SwiGLUFeedForward(
            hidden_dim=config.hidden_dim,
            intermediate_dim=config.intermediate_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class SwitchRouter(nn.Module):
    """
    Top-1 router for Switch Transformer with P×T geometry.

    Key differences from Mixtral router:
    - Selects top-1 expert (not top-k)
    - Includes load balancing loss
    - Tracks routing probabilities for auxiliary loss

    P×T Integration:
        logits'_k = (logits_k + pressure_k) / temperature_k

    Load Balancing Loss:
        Encourages uniform expert usage across batch
        L_aux = num_experts * Σ_i (f_i * P_i)
        where:
            f_i = fraction of tokens routed to expert i
            P_i = mean routing probability for expert i
    """

    def __init__(self, config: SwitchConfig):
        super().__init__()
        self.num_experts = config.num_experts

        # Router weights: hidden_dim → num_experts logits
        self.gate = nn.Linear(config.hidden_dim, config.num_experts, bias=False)

        # Load balancing coefficients
        self.router_z_loss_coef = config.router_z_loss_coef
        self.router_aux_loss_coef = config.router_aux_loss_coef

    def forward(
        self,
        hidden_states: torch.Tensor,
        pressure_bias: Optional[torch.Tensor] = None,
        temperature_field: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute top-1 routing with P×T geometry and load balancing loss.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            pressure_bias: Optional Chronovisor pressure (num_experts,)
            temperature_field: Optional per-expert temperatures (num_experts,)

        Returns:
            Tuple of:
                - routing_weights: (batch, seq_len, 1) - Weight for selected expert
                - selected_experts: (batch, seq_len, 1) - Index of selected expert
                - router_probs: (batch, seq_len, num_experts) - Full probability distribution
                - aux_loss: scalar - Load balancing auxiliary loss
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute router logits
        router_logits = self.gate(hidden_states)  # (batch, seq_len, num_experts)

        # Apply Chronovisor pressure if provided
        if pressure_bias is not None:
            router_logits = router_logits + pressure_bias.unsqueeze(0).unsqueeze(0)

        # Apply temperature field if provided
        if temperature_field is not None:
            # Temperature warping: divide logits by per-expert temperature
            temp_safe = torch.clamp(temperature_field, min=0.1, max=10.0)
            router_logits = router_logits / temp_safe.unsqueeze(0).unsqueeze(0)

        # Compute router probabilities (for load balancing)
        router_probs = F.softmax(router_logits, dim=-1)  # (batch, seq_len, num_experts)

        # Top-1 selection
        routing_weights, selected_experts = torch.max(
            router_probs, dim=-1, keepdim=True
        )  # Both: (batch, seq_len, 1)

        # Compute load balancing auxiliary loss
        aux_loss = self._compute_load_balancing_loss(
            router_probs,
            selected_experts.squeeze(-1),
            router_logits
        )

        return routing_weights, selected_experts, router_probs, aux_loss

    def _compute_load_balancing_loss(
        self,
        router_probs: torch.Tensor,
        selected_experts: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Switch Transformer load balancing loss.

        Two components:
        1. Auxiliary loss: Encourages uniform expert usage
        2. Router z-loss: Encourages router logits to stay small

        Args:
            router_probs: (batch, seq_len, num_experts)
            selected_experts: (batch, seq_len)
            router_logits: (batch, seq_len, num_experts)

        Returns:
            Combined auxiliary loss (scalar)
        """
        batch_size, seq_len, num_experts = router_probs.shape

        # 1. Load balancing auxiliary loss
        # f_i = fraction of tokens routed to expert i
        f_i = torch.bincount(
            selected_experts.flatten(),
            minlength=num_experts
        ).float() / (batch_size * seq_len)

        # P_i = mean routing probability for expert i
        P_i = router_probs.mean(dim=(0, 1))  # (num_experts,)

        # L_aux = num_experts * Σ(f_i * P_i)
        # This encourages f_i and P_i to be uniform (ideally 1/num_experts each)
        load_balance_loss = num_experts * torch.sum(f_i * P_i)

        # 2. Router z-loss (encourages small logits for stability)
        # L_z = log²(Σ exp(logits))
        router_z_loss = torch.logsumexp(router_logits, dim=-1) ** 2
        router_z_loss = router_z_loss.mean()

        # Combine losses
        aux_loss = (
            self.router_aux_loss_coef * load_balance_loss +
            self.router_z_loss_coef * router_z_loss
        )

        return aux_loss


class SwitchSparseMoELayer(nn.Module):
    """
    Sparse Mixture-of-Experts layer with top-1 routing and P×T integration.

    Architecture:
        - N experts (typically 64-256 in Switch Transformer)
        - Top-1 routing (each token → 1 expert)
        - Load balancing via auxiliary loss

    Chronovisor P×T Integration:
        - Pressure field (P): Force toward/away from experts
        - Temperature field (T): Per-expert routing permeability
        - Combined: logits' = (logits + P) / T
    """

    def __init__(self, config: SwitchConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_experts = config.num_experts

        # Create experts
        self.experts = nn.ModuleList([
            SwitchExpert(config, expert_id=i)
            for i in range(self.num_experts)
        ])

        # Router
        self.router = SwitchRouter(config)

        # Chronovisor state (set externally)
        if config.enable_chronovisor:
            self.register_buffer('pressure_bias', torch.zeros(self.num_experts))
            self.register_buffer('temperature_field', torch.ones(self.num_experts))
        else:
            self.register_buffer('pressure_bias', None)
            self.register_buffer('temperature_field', None)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through Switch MoE layer with P×T geometry.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            Tuple of:
                - output: (batch, seq_len, hidden_dim)
                - routing_stats: Dict with routing info and auxiliary loss
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Compute routing with P×T fields
        routing_weights, selected_experts, router_probs, aux_loss = self.router(
            hidden_states,
            pressure_bias=self.pressure_bias if self.config.enable_chronovisor else None,
            temperature_field=self.temperature_field if self.config.enable_chronovisor else None,
        )

        # Initialize output
        output = torch.zeros_like(hidden_states)

        # Reshape for expert processing
        # Since top-1, we can batch all tokens by expert
        flat_hidden = hidden_states.view(-1, hidden_dim)  # (batch*seq, hidden_dim)
        flat_experts = selected_experts.view(-1)  # (batch*seq,)
        flat_weights = routing_weights.view(-1, 1)  # (batch*seq, 1)

        # Process each expert
        for expert_id in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (flat_experts == expert_id)

            if not expert_mask.any():
                continue

            # Get tokens for this expert
            expert_input = flat_hidden[expert_mask]

            # Process through expert
            expert_output = self.experts[expert_id](expert_input)

            # Weight and scatter back
            expert_output = expert_output * flat_weights[expert_mask]

            # Scatter to output
            output.view(-1, hidden_dim)[expert_mask] = expert_output

        # Collect routing statistics
        routing_stats = {
            'layer_idx': self.layer_idx,
            'selected_experts': selected_experts.squeeze(-1),  # (batch, seq)
            'routing_weights': routing_weights.squeeze(-1),
            'router_probs': router_probs,
            'aux_loss': aux_loss,
        }

        return output, routing_stats


class SwitchDecoderLayer(nn.Module):
    """
    A single Switch Transformer decoder layer.

    Structure (same as Mixtral):
        1. RMSNorm
        2. Grouped-Query Attention
        3. Residual connection
        4. RMSNorm
        5. Switch Sparse MoE (top-1)
        6. Residual connection
    """

    def __init__(self, config: SwitchConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Attention
        self.input_layernorm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)

        # MoE (Switch instead of Mixtral)
        self.post_attention_layernorm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.moe = SwitchSparseMoELayer(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            attention_mask: Optional attention mask

        Returns:
            Tuple of (output_states, routing_stats)
        """
        # Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # MoE block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, routing_stats = self.moe(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, routing_stats


# =============================================================================
# Compatibility wrappers to work with existing Chronovisor bridge
# =============================================================================

# The existing ChronovisorMixtralController can work with Switch Transformer
# because it only needs:
#   - config.num_layers
#   - config.num_experts
#   - routing_stats['selected_experts'] and routing_stats['routing_weights']
#
# These are compatible between Mixtral and Switch.

# Just need to add aux_loss tracking for Switch
def get_total_aux_loss(routing_stats_dict):
    """
    Sum auxiliary losses across all layers.

    Args:
        routing_stats_dict: {layer_idx: routing_stats}

    Returns:
        Total auxiliary loss (scalar tensor)
    """
    total_aux_loss = 0.0
    for stats in routing_stats_dict.values():
        if 'aux_loss' in stats:
            total_aux_loss = total_aux_loss + stats['aux_loss']
    return total_aux_loss


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == '__main__':
    # Create small config for testing
    config = SwitchConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=512,
        num_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=8,  # Small for testing (Switch typically uses 64-256)
        num_experts_per_token=1,  # Top-1 routing
        max_seq_length=512,
        enable_chronovisor=True,
    )

    print("=" * 60)
    print("Testing Switch Transformer with P×T Coupling")
    print("=" * 60)

    # Create a single layer
    layer = SwitchDecoderLayer(config, layer_idx=0)

    # Test forward pass
    batch_size = 2
    seq_len = 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)

    print("\nForward pass...")
    output, routing_stats = layer(hidden_states)

    print(f"✓ Input shape: {hidden_states.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Selected experts shape: {routing_stats['selected_experts'].shape}")
    print(f"✓ Routing weights shape: {routing_stats['routing_weights'].shape}")
    print(f"✓ Auxiliary loss: {routing_stats['aux_loss'].item():.6f}")

    # Check top-1 routing
    print(f"\nTop-1 routing verification:")
    print(f"  Each token routed to: {routing_stats['selected_experts'].shape[-1]} expert(s)")
    assert routing_stats['selected_experts'].shape[-1] == seq_len, "Should be seq_len for top-1"

    # Apply P×T fields
    print("\nTesting P×T field application...")
    layer.moe.pressure_bias = torch.randn(config.num_experts) * 0.1
    layer.moe.temperature_field = torch.ones(config.num_experts) * 1.5

    output2, routing_stats2 = layer(hidden_states)
    print(f"✓ With P×T: Auxiliary loss: {routing_stats2['aux_loss'].item():.6f}")

    # Check that expert usage changes with P×T
    expert_usage_1 = torch.bincount(routing_stats['selected_experts'].flatten(), minlength=config.num_experts)
    expert_usage_2 = torch.bincount(routing_stats2['selected_experts'].flatten(), minlength=config.num_experts)

    print(f"\nExpert usage (no P×T): {expert_usage_1.tolist()}")
    print(f"Expert usage (with P×T): {expert_usage_2.tolist()}")

    if not torch.equal(expert_usage_1, expert_usage_2):
        print("✓ P×T fields affect routing!")
    else:
        print("⚠ P×T fields had no effect (may be too small)")

    print("\n" + "=" * 60)
    print("Switch Transformer P×T integration test complete!")
    print("=" * 60)
