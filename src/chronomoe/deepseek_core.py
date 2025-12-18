"""
DeepSeek-MoE Core Implementation

Reference: DeepSeek-MoE: Towards Ultimate Expert Specialization
https://arxiv.org/abs/2401.06066

Key innovations:
1. Shared experts (always activated) + Routed experts (sparse top-k)
2. Fine-grained experts (64-160 vs typical 8-16)
3. Device-limited routing (experts assigned to devices, routing respects limits)
4. Load balancing via auxiliary loss

Architecture:
    y = Σ(shared_experts) + Σ(top_k(routed_experts))

The shared experts provide a stable base, while routed experts specialize.
This is radically different from Mixtral (no shared experts) and Switch (top-1).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DeepSeekConfig:
    """Configuration for DeepSeek-MoE architecture."""

    # Model dimensions
    vocab_size: int = 32000
    hidden_dim: int = 4096
    intermediate_dim: int = 14336  # FFN hidden size
    num_layers: int = 32
    max_seq_length: int = 4096

    # Attention
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA
    head_dim: int = 128

    # MoE configuration (DeepSeek-specific)
    num_shared_experts: int = 2  # Always activated
    num_routed_experts: int = 64  # Sparse top-k selection
    num_experts_per_token: int = 6  # Top-k for routed experts

    # Training
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1e6

    # Load balancing
    router_aux_loss_coef: float = 0.001  # Balance coefficient
    router_z_loss_coef: float = 0.001  # Router z-loss (encourages sparsity)

    # Chronovisor integration
    enable_chronovisor: bool = True
    chronovisor_pressure_scale: float = 0.1


class DeepSeekRMSNorm(nn.Module):
    """RMSNorm as used in DeepSeek."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class DeepSeekExpert(nn.Module):
    """
    Single expert (FFN with SwiGLU activation).

    Same as Mixtral expert, but DeepSeek uses many more of them.
    """

    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.w2 = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)
        self.w3 = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU: FFN_SwiGLU(x) = (Swish(xW1) ⊙ xW3)W2"""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DeepSeekRouter(nn.Module):
    """
    DeepSeek router with load balancing.

    Computes routing probabilities for routed experts.
    Shared experts are always activated (no routing needed).
    """

    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.num_routed_experts = config.num_routed_experts
        self.num_experts_per_token = config.num_experts_per_token

        # Router projection
        self.gate = nn.Linear(config.hidden_dim, config.num_routed_experts, bias=False)

        # Load balancing
        self.aux_loss_coef = config.router_aux_loss_coef
        self.z_loss_coef = config.router_z_loss_coef

    def forward(
        self,
        hidden_states: torch.Tensor,
        pressure_bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to top-k routed experts.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            pressure_bias: Optional [num_routed_experts] Chronovisor pressure

        Returns:
            routing_weights: [batch, seq_len, num_experts_per_token] (normalized)
            selected_experts: [batch, seq_len, num_experts_per_token] (indices)
            router_probs: [batch, seq_len, num_routed_experts] (full softmax)
            aux_loss: Scalar auxiliary loss for load balancing
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Compute router logits
        router_logits = self.gate(hidden_states)  # [batch, seq_len, num_routed_experts]

        # Apply Chronovisor pressure bias if provided
        if pressure_bias is not None:
            router_logits = router_logits + pressure_bias.unsqueeze(0).unsqueeze(0)

        # Softmax over experts
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k selection
        routing_weights, selected_experts = torch.topk(
            router_probs,
            self.num_experts_per_token,
            dim=-1
        )

        # Renormalize selected expert weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # Compute auxiliary losses
        aux_loss = self._compute_aux_loss(router_logits, router_probs)

        return routing_weights, selected_experts, router_probs, aux_loss

    def _compute_aux_loss(
        self,
        router_logits: torch.Tensor,
        router_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary loss for load balancing.

        Combines:
        1. Load balancing loss (encourage uniform expert usage)
        2. Router z-loss (encourage sparse activations)
        """
        # Load balancing loss
        # Want: Each expert gets equal fraction of tokens
        # f_i = fraction of tokens routed to expert i
        # Ideally: f_i = 1/num_experts for all i

        # Count expert assignments (approximate, pre-softmax)
        expert_mask = torch.zeros_like(router_probs)
        expert_mask.scatter_(-1, router_probs.topk(self.num_experts_per_token, dim=-1)[1], 1.0)

        # Fraction assigned to each expert
        tokens_per_expert = expert_mask.sum(dim=(0, 1))  # [num_routed_experts]
        fraction_per_expert = tokens_per_expert / (expert_mask.sum() + 1e-9)

        # Average router probability per expert
        avg_prob_per_expert = router_probs.mean(dim=(0, 1))  # [num_routed_experts]

        # Load balance loss: f_i * P_i (want both to be 1/N)
        load_balance_loss = (fraction_per_expert * avg_prob_per_expert).sum() * self.num_routed_experts

        # Router z-loss: log^2(sum(exp(logits))) - encourages confident routing
        z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()

        # Combined auxiliary loss
        aux_loss = (
            self.aux_loss_coef * load_balance_loss +
            self.z_loss_coef * z_loss
        )

        return aux_loss


class DeepSeekSparseMoELayer(nn.Module):
    """
    DeepSeek Sparse MoE Layer.

    Key innovation: Shared experts + Routed experts

    Output = SharedExperts(x) + RoutedExperts(x)

    Shared experts provide stable base representation.
    Routed experts specialize based on input.
    """

    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.num_shared_experts = config.num_shared_experts
        self.num_routed_experts = config.num_routed_experts
        self.num_experts_per_token = config.num_experts_per_token

        # Shared experts (always activated)
        self.shared_experts = nn.ModuleList([
            DeepSeekExpert(config) for _ in range(config.num_shared_experts)
        ])

        # Routed experts (sparse top-k)
        self.routed_experts = nn.ModuleList([
            DeepSeekExpert(config) for _ in range(config.num_routed_experts)
        ])

        # Router for routed experts
        self.router = DeepSeekRouter(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pressure_bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass through MoE layer.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            pressure_bias: Optional Chronovisor pressure for routed experts

        Returns:
            output: [batch, seq_len, hidden_dim]
            aux_loss: Auxiliary loss for load balancing
            routing_stats: Dict with routing statistics
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 1. Shared experts (always activated, averaged)
        shared_output = torch.zeros_like(hidden_states)
        for expert in self.shared_experts:
            shared_output = shared_output + expert(hidden_states)
        shared_output = shared_output / self.num_shared_experts

        # 2. Route to top-k routed experts
        routing_weights, selected_experts, router_probs, aux_loss = self.router(
            hidden_states,
            pressure_bias
        )

        # 3. Compute routed expert outputs
        # Reshape for expert computation
        hidden_states_flat = hidden_states.view(-1, hidden_dim)  # [batch*seq_len, hidden_dim]
        selected_experts_flat = selected_experts.view(-1, self.num_experts_per_token)  # [batch*seq_len, k]
        routing_weights_flat = routing_weights.view(-1, self.num_experts_per_token)  # [batch*seq_len, k]

        # Compute outputs for selected experts
        routed_output = torch.zeros_like(hidden_states_flat)

        for token_idx in range(hidden_states_flat.size(0)):
            token_hidden = hidden_states_flat[token_idx]  # [hidden_dim]

            for k_idx in range(self.num_experts_per_token):
                expert_idx = selected_experts_flat[token_idx, k_idx].item()
                weight = routing_weights_flat[token_idx, k_idx]

                expert_output = self.routed_experts[expert_idx](token_hidden.unsqueeze(0))
                routed_output[token_idx] = routed_output[token_idx] + weight * expert_output.squeeze(0)

        routed_output = routed_output.view(batch_size, seq_len, hidden_dim)

        # 4. Combine shared + routed
        final_output = shared_output + routed_output

        # 5. Collect routing statistics
        with torch.no_grad():
            # Expert usage (fraction of tokens routed to each expert)
            expert_usage = torch.zeros(self.num_routed_experts)
            for i in range(self.num_routed_experts):
                expert_usage[i] = (selected_experts == i).float().sum() / selected_experts.numel()

            # Routing entropy (average over batch and sequence)
            eps = 1e-10
            entropy = -torch.sum(router_probs * torch.log(router_probs + eps), dim=-1)
            routing_entropy = entropy.mean().item()

        routing_stats = {
            'expert_usage': expert_usage.cpu().numpy(),
            'routing_entropy': routing_entropy,
            'router_probs': router_probs.detach(),  # Keep for Takens analysis
        }

        return final_output, aux_loss, routing_stats


class DeepSeekDecoderLayer(nn.Module):
    """
    DeepSeek decoder layer: Attention + DeepSeek MoE.

    (Simplified - full implementation would include GQA attention)
    """

    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        # Pre-attention norm
        self.input_layernorm = DeepSeekRMSNorm(config.hidden_dim, config.rms_norm_eps)

        # Attention (placeholder - would be full GQA)
        self.self_attn = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_attention_heads,
            batch_first=True
        )

        # Post-attention norm
        self.post_attention_layernorm = DeepSeekRMSNorm(config.hidden_dim, config.rms_norm_eps)

        # DeepSeek MoE
        self.moe = DeepSeekSparseMoELayer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pressure_bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass through decoder layer.

        Returns:
            hidden_states: [batch, seq_len, hidden_dim]
            aux_loss: MoE auxiliary loss
            routing_stats: Routing statistics from MoE
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, hidden_states, hidden_states,
                                         attn_mask=attention_mask)
        hidden_states = residual + hidden_states

        # MoE with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, aux_loss, routing_stats = self.moe(hidden_states, pressure_bias)
        hidden_states = residual + hidden_states

        return hidden_states, aux_loss, routing_stats


# Export public API
__all__ = [
    'DeepSeekConfig',
    'DeepSeekSparseMoELayer',
    'DeepSeekRouter',
    'DeepSeekExpert',
    'DeepSeekDecoderLayer',
]
