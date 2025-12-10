"""
Mixtral MoE Core Implementation

Internal implementation of Mixtral's architecture components:
- Multi-head grouped-query attention (GQA) with RoPE
- Sparse Mixture-of-Experts with top-k routing
- SwiGLU feedforward networks

This is built from scratch to integrate deeply with Chronovisor's
geometric control layer, rather than wrapping external models.

Architecture References:
    - Mixtral 8x7B: 32 layers, 8 experts per layer, top-2 routing
    - Hidden dim: 4096, Intermediate (FFN): 14336
    - 32 attention heads, 8 key-value heads (GQA)
    - Vocab: 32000, Max seq length: 32768
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class MixtralConfig:
    """Configuration for Mixtral architecture."""
    # Model dimensions
    vocab_size: int = 32000
    hidden_dim: int = 4096
    intermediate_dim: int = 14336
    num_layers: int = 32
    max_seq_length: int = 32768

    # Attention
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA: fewer KV heads than Q heads
    head_dim: int = 128  # hidden_dim / num_attention_heads

    # MoE
    num_experts: int = 8
    num_experts_per_token: int = 2  # top-k routing

    # Training
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1e6  # RoPE base frequency

    # Chronovisor integration
    enable_chronovisor: bool = True
    chronovisor_pressure_scale: float = 0.1


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Used in Mixtral instead of LayerNorm for better training stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            Normalized tensor (batch, seq_len, dim)
        """
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Instead of adding positional info, RoPE rotates the query/key vectors
    in a way that encodes relative positions.
    """

    def __init__(self, dim: int, max_seq_length: int = 32768, theta: float = 1e6):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.theta = theta

        # Precompute frequencies
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_length).float()
        freqs = torch.outer(t, freqs)

        # Create rotation matrices (cos, sin)
        self.register_buffer('cos', freqs.cos())
        self.register_buffer('sin', freqs.sin())

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Apply rotary embeddings to query or key.

        Args:
            x: (batch, seq_len, num_heads, head_dim)
            seq_len: Sequence length

        Returns:
            Rotated tensor (batch, seq_len, num_heads, head_dim)
        """
        # Split into even/odd dimensions
        x1, x2 = x[..., ::2], x[..., 1::2]

        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, dim/2)
        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(2)

        # Apply rotation
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ], dim=-1)

        return rotated


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).

    Mixtral uses GQA: multiple query heads share the same key/value heads.
    This reduces memory and computation compared to full multi-head attention.

    Example: 32 query heads, 8 KV heads → each KV head serves 4 query heads.
    """

    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.config = config

        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim

        # Number of query heads per KV head
        self.num_q_per_kv = self.num_q_heads // self.num_kv_heads

        # Linear projections
        self.q_proj = nn.Linear(self.hidden_dim, self.num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_q_heads * self.head_dim, self.hidden_dim, bias=False)

        # RoPE
        self.rope = RotaryPositionalEmbedding(
            dim=self.head_dim,
            max_seq_length=config.max_seq_length,
            theta=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            attention_mask: Optional (batch, seq_len, seq_len)

        Returns:
            Output tensor (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states)  # (batch, seq_len, num_q_heads * head_dim)
        k = self.k_proj(hidden_states)  # (batch, seq_len, num_kv_heads * head_dim)
        v = self.v_proj(hidden_states)  # (batch, seq_len, num_kv_heads * head_dim)

        # Reshape to separate heads
        q = q.view(batch_size, seq_len, self.num_q_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE to Q and K
        q = self.rope(q, seq_len)
        k = self.rope(k, seq_len)

        # Expand KV heads to match Q heads (GQA)
        # Each KV head is repeated for multiple Q heads
        k = k.repeat_interleave(self.num_q_per_kv, dim=2)  # (batch, seq_len, num_q_heads, head_dim)
        v = v.repeat_interleave(self.num_q_per_kv, dim=2)

        # Transpose for attention: (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # Apply mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)

        # Final projection
        output = self.o_proj(attn_output)

        return output


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU activation feedforward network.

    SwiGLU: Swish-Gated Linear Unit
        FFN(x) = (Swish(x @ W1) ⊙ (x @ V)) @ W2

    Where ⊙ is element-wise multiplication and Swish(x) = x * sigmoid(x).

    This is used in each Mixtral expert.
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, intermediate_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)

        Returns:
            Output (batch, seq_len, hidden_dim)
        """
        # SwiGLU(x) = (Swish(x @ W1) ⊙ (x @ V)) @ W2
        swish_gate = F.silu(self.w1(x))  # SiLU = Swish
        gated = swish_gate * self.v(x)
        output = self.w2(gated)
        return output


class MixtralExpert(nn.Module):
    """
    A single expert in the Mixtral MoE layer.

    Each expert is a SwiGLU feedforward network.
    """

    def __init__(self, config: MixtralConfig, expert_id: int):
        super().__init__()
        self.expert_id = expert_id
        self.ffn = SwiGLUFeedForward(
            hidden_dim=config.hidden_dim,
            intermediate_dim=config.intermediate_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)

        Returns:
            Expert output (batch, seq_len, hidden_dim)
        """
        return self.ffn(x)


class MixtralRouter(nn.Module):
    """
    Top-k sparse router for MoE with P×T (Pressure × Temperature) geometry.

    For each token, computes routing probabilities over experts
    and selects the top-k experts to process that token.

    P×T Integration:
        logits'_k = (logits_k + pressure_k) / temperature_k

    Where:
        - pressure_k: Force field pushing toward/away from experts
        - temperature_k: Permeability controlling routing sharpness
    """

    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token

        # Router weights: hidden_dim → num_experts logits
        self.gate = nn.Linear(config.hidden_dim, config.num_experts, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pressure_bias: Optional[torch.Tensor] = None,
        temperature_field: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute routing weights and selected experts with P×T geometry.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            pressure_bias: Optional Chronovisor pressure (num_experts,)
            temperature_field: Optional per-expert temperatures (num_experts,)

        Returns:
            Tuple of:
                - routing_weights: (batch, seq_len, top_k) - Normalized weights for selected experts
                - selected_experts: (batch, seq_len, top_k) - Indices of selected experts
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
            # High temp = diffuse (exploratory), Low temp = sharp (exploitative)
            temp_safe = torch.clamp(temperature_field, min=0.1, max=10.0)
            router_logits = router_logits / temp_safe.unsqueeze(0).unsqueeze(0)

        # Select top-k experts per token
        routing_weights, selected_experts = torch.topk(
            router_logits, self.top_k, dim=-1
        )  # Both: (batch, seq_len, top_k)

        # Normalize weights with softmax over the top-k
        routing_weights = F.softmax(routing_weights, dim=-1)

        return routing_weights, selected_experts


class MixtralSparseMoELayer(nn.Module):
    """
    Sparse Mixture-of-Experts layer with Chronovisor P×T integration.

    Architecture:
        - N experts (default 8)
        - Top-k routing (default k=2)
        - Each token is processed by k experts
        - Output is a weighted combination

    Chronovisor P×T Integration:
        - Pressure field (P): Force toward/away from experts
        - Temperature field (T): Per-expert routing permeability
        - Combined: logits' = (logits + P) / T
        - Tracks expert usage statistics for feedback
    """

    def __init__(self, config: MixtralConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_experts = config.num_experts

        # Create experts
        self.experts = nn.ModuleList([
            MixtralExpert(config, expert_id=i)
            for i in range(self.num_experts)
        ])

        # Router
        self.router = MixtralRouter(config)

        # Chronovisor state (set externally)
        if config.enable_chronovisor:
            self.register_buffer('pressure_bias', torch.zeros(self.num_experts))
            self.register_buffer('temperature_field', torch.ones(self.num_experts))
        else:
            self.register_buffer('pressure_bias', None)
            self.register_buffer('temperature_field', None)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass through MoE layer with P×T geometry.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            Tuple of:
                - output: (batch, seq_len, hidden_dim)
                - routing_stats: Dict with expert usage info
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Compute routing with P×T fields
        routing_weights, selected_experts = self.router(
            hidden_states,
            pressure_bias=self.pressure_bias if self.config.enable_chronovisor else None,
            temperature_field=self.temperature_field if self.config.enable_chronovisor else None,
        )

        # Initialize output
        output = torch.zeros_like(hidden_states)

        # Process each expert
        for expert_id in range(self.num_experts):
            # Find which tokens route to this expert
            expert_mask = (selected_experts == expert_id)  # (batch, seq_len, top_k)

            if not expert_mask.any():
                continue

            # Get expert output for all tokens
            expert_output = self.experts[expert_id](hidden_states)  # (batch, seq_len, hidden_dim)

            # Weight expert output by routing weights where this expert is selected
            # This is tricky: we need to accumulate weighted outputs
            for k_idx in range(self.config.num_experts_per_token):
                mask = expert_mask[:, :, k_idx]  # (batch, seq_len)
                weights = routing_weights[:, :, k_idx]  # (batch, seq_len)

                # Add weighted contribution
                weighted_output = expert_output * weights.unsqueeze(-1)
                output = output + weighted_output * mask.unsqueeze(-1).float()

        # Collect routing statistics for Chronovisor
        routing_stats = {
            'layer_idx': self.layer_idx,
            'selected_experts': selected_experts,
            'routing_weights': routing_weights,
        }

        return output, routing_stats


class MixtralDecoderLayer(nn.Module):
    """
    A single Mixtral decoder layer.

    Structure:
        1. RMSNorm
        2. Grouped-Query Attention
        3. Residual connection
        4. RMSNorm
        5. Sparse MoE
        6. Residual connection
    """

    def __init__(self, config: MixtralConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Attention
        self.input_layernorm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)

        # MoE
        self.post_attention_layernorm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.moe = MixtralSparseMoELayer(config, layer_idx)

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
# Usage Example (for testing)
# =============================================================================

if __name__ == '__main__':
    # Create a small config for testing
    config = MixtralConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=512,
        num_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=4,
        num_experts_per_token=2,
        max_seq_length=512,
    )

    # Create a single layer
    layer = MixtralDecoderLayer(config, layer_idx=0)

    # Test forward pass
    batch_size = 2
    seq_len = 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)

    output, routing_stats = layer(hidden_states)

    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Selected experts shape: {routing_stats['selected_experts'].shape}")
    print(f"Routing weights shape: {routing_stats['routing_weights'].shape}")
