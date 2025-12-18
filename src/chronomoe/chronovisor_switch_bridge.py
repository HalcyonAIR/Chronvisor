"""
Chronovisor ↔ Switch Transformer Integration Bridge

Adapts the existing Chronovisor controller to work with Switch Transformer's
top-1 routing mechanism.

Key differences from Mixtral bridge:
- Handles top-1 routing (1 expert per token instead of 2)
- Tracks auxiliary load balancing loss
- Otherwise identical control flow

The Chronovisor controller is routing-agnostic and works with both:
- Mixtral (top-k routing)
- Switch (top-1 routing)

This demonstrates P×T coupling generalization across routing strategies.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from chronomoe.switch_core import (
    SwitchConfig,
    SwitchDecoderLayer,
    get_total_aux_loss,
)

# Reuse the Chronovisor controller from Mixtral bridge
# It's routing-agnostic and works with any MoE architecture
from chronomoe.chronovisor_mixtral_bridge import (
    ChronovisorMixtralController,
    ChronovisorMixtralState,
)


class ChronovisorSwitchModel(nn.Module):
    """
    Switch Transformer model with Chronovisor geometric control layer.

    This wraps a stack of Switch layers and integrates them with
    the Chronovisor controller for adaptive routing.

    Key difference from ChronovisorMixtralModel:
    - Uses SwitchDecoderLayer (top-1 routing)
    - Tracks auxiliary load balancing loss
    - Otherwise identical control flow
    """

    def __init__(self, config: SwitchConfig):
        super().__init__()
        self.config = config

        # Create Switch layers
        self.layers = nn.ModuleList([
            SwitchDecoderLayer(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # Chronovisor controller (reuse from Mixtral)
        # The controller is routing-agnostic
        self.controller = ChronovisorMixtralController(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_chronovisor: bool = True,
    ) -> Tuple[torch.Tensor, Optional[ChronovisorMixtralState], torch.Tensor]:
        """
        Forward pass through the model with Chronovisor updates.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            attention_mask: Optional attention mask
            update_chronovisor: Whether to update controller state

        Returns:
            Tuple of (final_hidden_states, chronovisor_state, aux_loss)
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

        # Compute total auxiliary loss (Switch-specific)
        total_aux_loss = get_total_aux_loss(all_routing_stats)

        # Update controller
        chronovisor_state = None
        if update_chronovisor and self.config.enable_chronovisor:
            chronovisor_state = self.controller.tick(all_routing_stats)

        return hidden_states, chronovisor_state, total_aux_loss


class ChronovisorSwitchForCausalLM(nn.Module):
    """
    Switch Transformer language model with Chronovisor P×T geometric control.

    Complete causal LM with:
        - Token embeddings
        - Chronovisor-controlled Switch decoder stack
        - Language modeling head

    Includes auxiliary load balancing loss for Switch routing.
    """

    def __init__(self, config: SwitchConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Chronovisor Switch decoder
        self.model = ChronovisorSwitchModel(config)

        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_chronovisor: bool = True,
    ) -> Tuple[torch.Tensor, Optional[ChronovisorMixtralState], torch.Tensor]:
        """
        Forward pass through the language model.

        Args:
            input_ids: Token indices (batch, seq_len)
            attention_mask: Optional attention mask
            update_chronovisor: Whether to update controller state

        Returns:
            Tuple of (logits, chronovisor_state, aux_loss)
                - logits: (batch, seq_len, vocab_size)
                - chronovisor_state: Current Chronovisor state or None
                - aux_loss: Load balancing auxiliary loss (scalar)
        """
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Forward through Chronovisor Switch
        hidden_states, chronovisor_state, aux_loss = self.model(
            hidden_states,
            attention_mask=attention_mask,
            update_chronovisor=update_chronovisor,
        )

        # Project to vocabulary
        logits = self.lm_head(hidden_states)

        return logits, chronovisor_state, aux_loss

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute total loss including cross-entropy and auxiliary loss.

        Args:
            input_ids: (batch, seq_len)
            labels: (batch, seq_len)
            attention_mask: Optional mask

        Returns:
            Tuple of (total_loss, aux_loss)
        """
        # Forward pass
        logits, chrono_state, aux_loss = self.forward(
            input_ids,
            attention_mask=attention_mask,
            update_chronovisor=True
        )

        # Cross-entropy loss
        ce_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, self.vocab_size),
            labels.view(-1),
            reduction='mean'
        )

        # Total loss = cross-entropy + auxiliary loss
        total_loss = ce_loss + aux_loss

        return total_loss, aux_loss


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == '__main__':
    # Create small config for testing
    config = SwitchConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=512,
        num_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=8,
        num_experts_per_token=1,  # Top-1 routing
        max_seq_length=512,
        enable_chronovisor=True,
        router_aux_loss_coef=0.01,
    )

    print("=" * 60)
    print("Testing ChronovisorSwitchModel")
    print("=" * 60)

    # Create model
    model = ChronovisorSwitchModel(config)
    print(f"✓ Model created with {config.num_layers} layers, {config.num_experts} experts")
    print(f"✓ Routing: top-{config.num_experts_per_token} (Switch Transformer)")

    # Test forward pass
    batch_size = 2
    seq_len = 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)

    print("\n1. Initial forward pass...")
    output, state, aux_loss = model(hidden_states)
    print(f"   ✓ Output shape: {output.shape}")
    print(f"   ✓ Coherence (R): {state.coherence:.4f}")
    print(f"   ✓ Auxiliary loss: {aux_loss.item():.6f}")

    # Check T̄ export
    print("\n2. Checking T̄ export...")
    assert state.T_bar is not None, "T_bar should not be None!"
    print(f"   ✓ T_bar shape: {state.T_bar.shape}")
    print(f"   ✓ T_bar mean: {state.T_bar.mean():.4f}")

    # Run several ticks
    print("\n3. Running 10 ticks to observe geological evolution...")
    T_bar_history = [state.T_bar.copy()]

    for i in range(10):
        output, state, aux_loss = model(hidden_states)
        T_bar_history.append(state.T_bar.copy())

    import numpy as np
    T_bar_variance = np.var([np.mean(t) for t in T_bar_history])
    print(f"   ✓ Variance of mean T̄: {T_bar_variance:.6f}")
    print(f"   ✓ Geology is {'ALIVE' if T_bar_variance > 1e-10 else 'FROZEN'}!")

    print("\n" + "=" * 60)
    print("Testing ChronovisorSwitchForCausalLM")
    print("=" * 60)

    # Create language model
    lm = ChronovisorSwitchForCausalLM(config)

    # Test with token IDs
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    print("\nForward pass through language model...")
    logits, chrono_state, aux_loss = lm(input_ids)
    print(f"✓ Logits shape: {logits.shape}")
    print(f"✓ Auxiliary loss: {aux_loss.item():.6f}")

    print("\nComputing total loss (CE + aux)...")
    total_loss, aux_loss_check = lm.compute_loss(input_ids, labels)
    print(f"✓ Total loss: {total_loss.item():.6f}")
    print(f"✓ Aux loss component: {aux_loss_check.item():.6f}")

    print("\n" + "=" * 60)
    print("✓ Switch Transformer P×T integration complete!")
    print("=" * 60)
    print("\nKey differences from Mixtral:")
    print("  - Top-1 routing (vs top-2)")
    print("  - Load balancing auxiliary loss")
    print("  - Same P×T coupling mechanism")
    print("  - Same Chronovisor controller")
    print("\n✓ Ready for comparative evaluation!")
