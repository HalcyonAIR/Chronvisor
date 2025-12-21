#!/usr/bin/env python3
"""
ChronoMoE Adapter for Switch Transformers

Wraps the pretrained Switch router to inject geological temperature (T̄) bias.

Architecture:
    Original Switch Router → Logits
    + T̄ Pressure → Biased Logits
    → Softmax → Routing Probs
    → Update T̄ based on usage

This is a non-invasive wrapper that preserves the pretrained model.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class GeologicalLens:
    """
    Geological temperature tracking per expert.

    T̄ adapts slowly based on expert usage, creating a persistence field.
    """

    def __init__(
        self,
        num_experts: int,
        eta_structural_T: float = 0.015,  # Geological learning rate
    ):
        self.num_experts = num_experts
        self.eta_structural_T = eta_structural_T

        # Geological temperature (starts uniform)
        self.T_bar = np.ones(num_experts, dtype=np.float32)

        # EMA of expert usage
        self.expert_usage_ema = np.ones(num_experts, dtype=np.float32) / num_experts

    def compute_pressure(self) -> np.ndarray:
        """
        Compute pressure bias from T̄.

        Pressure = log(T̄) - mean(log(T̄))
        This creates a zero-sum bias that favors "warmer" experts.
        """
        log_T = np.log(self.T_bar + 1e-10)
        pressure = log_T - np.mean(log_T)
        return pressure

    def update(self, routing_probs: np.ndarray):
        """
        Update T̄ based on routing probabilities.

        routing_probs: [batch * seq_len, num_experts]

        T̄ increases for used experts, decreases for unused experts.
        """
        # Average routing over batch/sequence
        usage = routing_probs.mean(axis=0)  # [num_experts]

        # Update EMA
        alpha = 0.1  # EMA coefficient
        self.expert_usage_ema = (
            alpha * usage + (1 - alpha) * self.expert_usage_ema
        )

        # Update T̄: warm experts that are used, cool unused ones
        # T̄ <- T̄ + η * (usage - 1/N)
        target = usage - (1.0 / self.num_experts)
        self.T_bar += self.eta_structural_T * target

        # Clamp to prevent runaway
        self.T_bar = np.clip(self.T_bar, 0.5, 2.0)

    def reset(self):
        """Reset T̄ to uniform."""
        self.T_bar = np.ones(self.num_experts, dtype=np.float32)
        self.expert_usage_ema = np.ones(self.num_experts, dtype=np.float32) / self.num_experts


class ChronoMoESwitchAdapter:
    """
    Adapter that wraps a Switch Transformer router to inject T̄ bias.

    Usage:
        adapter = ChronoMoESwitchAdapter(num_experts=8)

        # Wrap the router's forward pass
        original_forward = router.forward
        router.forward = lambda x: adapter.forward_with_bias(original_forward, x)

        # Enable/disable adaptation
        adapter.enable_adaptation()
        model(inputs)  # T̄ updates during forward pass
        adapter.disable_adaptation()
    """

    def __init__(
        self,
        num_experts: int,
        eta_structural_T: float = 0.015,
    ):
        self.num_experts = num_experts
        self.lens = GeologicalLens(num_experts, eta_structural_T)

        # Control flags
        self.adaptation_enabled = False
        self.inject_bias = False

    def enable_adaptation(self):
        """Enable T̄ updates during forward pass."""
        self.adaptation_enabled = True
        self.inject_bias = True

    def disable_adaptation(self):
        """Disable T̄ updates (but keep bias injection if set)."""
        self.adaptation_enabled = False

    def enable_bias_only(self):
        """Inject T̄ bias without updating T̄."""
        self.adaptation_enabled = False
        self.inject_bias = True

    def disable_all(self):
        """Disable both adaptation and bias (vanilla mode)."""
        self.adaptation_enabled = False
        self.inject_bias = False

    def reset(self):
        """Reset T̄ to uniform."""
        self.lens.reset()

    def forward_with_bias(
        self,
        original_forward,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Wrapper around router's forward pass.

        Args:
            original_forward: The router's original forward method
            hidden_states: [batch, seq_len, hidden_dim]

        Returns:
            router_mask: [batch, seq_len, num_experts] (expert selection mask)
            router_probs: [batch, seq_len, num_experts]
            router_logits: [batch, seq_len, num_experts]
        """
        # Get original router output
        # Switch router returns (router_mask, router_probs, router_logits)
        router_mask, router_probs, router_logits = original_forward(hidden_states)

        # Store original for later
        original_logits = router_logits

        # Inject T̄ bias if enabled
        if self.inject_bias:
            pressure = self.lens.compute_pressure()
            pressure_tensor = torch.tensor(
                pressure, dtype=router_logits.dtype, device=router_logits.device
            )

            # Add pressure to logits
            biased_logits = router_logits + pressure_tensor.unsqueeze(0).unsqueeze(0)

            # Recompute full probability distribution
            full_probs = torch.softmax(biased_logits, dim=-1)

            # Recompute mask (top-1 selection for Switch)
            selected_experts = torch.argmax(full_probs, dim=-1)
            router_mask = torch.nn.functional.one_hot(
                selected_experts, num_classes=self.num_experts
            ).to(router_logits.dtype)

            # router_probs is the confidence for the selected expert
            # Shape: [batch, seq, 1]
            router_probs = full_probs.gather(
                dim=-1,
                index=selected_experts.unsqueeze(-1)
            )

            # Use biased logits
            router_logits = biased_logits

        # Update T̄ if adaptation enabled
        if self.adaptation_enabled:
            # Compute full probability distribution for T̄ update
            full_probs = torch.softmax(router_logits, dim=-1)

            # Convert to numpy for lens update
            routing_np = full_probs.detach().cpu().numpy()

            # Flatten batch and sequence dimensions
            batch_size, seq_len, _ = routing_np.shape
            routing_flat = routing_np.reshape(-1, self.num_experts)

            self.lens.update(routing_flat)

        return router_mask, router_probs, router_logits

    def get_state(self):
        """Get current T̄ state for inspection."""
        return {
            'T_bar': self.lens.T_bar.copy(),
            'expert_usage_ema': self.lens.expert_usage_ema.copy(),
            'pressure': self.lens.compute_pressure(),
            'adaptation_enabled': self.adaptation_enabled,
            'inject_bias': self.inject_bias,
        }


def wrap_switch_model_with_chronovisor(
    model,
    layer_indices: list = [1, 2, 3],  # Which layers to wrap (skip layer 0 - dense)
    eta_structural_T: float = 0.015,
) -> dict:
    """
    Wrap a Switch Transformer model with ChronoMoE adapters.

    Args:
        model: The Switch Transformer model
        layer_indices: Which encoder layers to wrap (default: layers 1-3)
        eta_structural_T: Geological learning rate

    Returns:
        adapters: Dict mapping layer_idx -> adapter
    """
    adapters = {}

    for layer_idx in layer_indices:
        if layer_idx == 0:
            print(f"⚠ Skipping layer 0 (dense, no router)")
            continue

        block = model.encoder.block[layer_idx]

        # Check if this block has MoE
        if not hasattr(block.layer[-1].mlp, 'router'):
            print(f"⚠ Layer {layer_idx} has no router, skipping")
            continue

        router = block.layer[-1].mlp.router

        # Get number of experts from router
        num_experts = router.classifier.out_features

        # Create adapter
        adapter = ChronoMoESwitchAdapter(num_experts, eta_structural_T)

        # Store original forward
        original_forward = router.forward

        # Wrap forward with adapter
        # Need to use closure to capture adapter and original_forward
        def make_wrapped_forward(adapter_ref, orig_forward):
            def wrapped_forward(hidden_states):
                return adapter_ref.forward_with_bias(orig_forward, hidden_states)
            return wrapped_forward

        router.forward = make_wrapped_forward(adapter, original_forward)

        adapters[layer_idx] = adapter

        print(f"✓ Wrapped layer {layer_idx} router with ChronoMoE adapter")

    return adapters


if __name__ == '__main__':
    """Test the adapter with a simple example."""

    print("Testing ChronoMoE Switch Adapter")
    print("="*70)
    print()

    # Create a mock router for testing
    class MockRouter:
        def __init__(self, num_experts=8):
            self.classifier = nn.Linear(768, num_experts, bias=False)

        def forward(self, hidden_states):
            """Mock Switch router forward."""
            router_logits = self.classifier(hidden_states)
            router_probs = torch.softmax(router_logits, dim=-1)
            selected_experts = torch.argmax(router_probs, dim=-1)
            return router_probs, selected_experts, router_logits

    # Test
    router = MockRouter(num_experts=8)
    adapter = ChronoMoESwitchAdapter(num_experts=8, eta_structural_T=0.015)

    # Create dummy input
    hidden_states = torch.randn(1, 10, 768)  # [batch, seq, hidden]

    print("Test 1: Vanilla (no bias)")
    adapter.disable_all()
    original_output = router.forward(hidden_states)
    wrapped_output = adapter.forward_with_bias(router.forward, hidden_states)

    print(f"  Original probs: {original_output[0][0, 0, :3]}")
    print(f"  Wrapped probs:  {wrapped_output[0][0, 0, :3]}")
    print(f"  Match: {torch.allclose(original_output[0], wrapped_output[0])}")
    print()

    print("Test 2: Enable adaptation")
    adapter.enable_adaptation()

    initial_T_bar = adapter.lens.T_bar.copy()

    # Run multiple passes
    for i in range(10):
        _ = adapter.forward_with_bias(router.forward, hidden_states)

    final_T_bar = adapter.lens.T_bar
    T_bar_drift = np.mean(np.abs(final_T_bar - initial_T_bar))

    print(f"  Initial T̄: {initial_T_bar[:4]}")
    print(f"  Final T̄:   {final_T_bar[:4]}")
    print(f"  T̄ drift:   {T_bar_drift:.6f}")
    print()

    print("Test 3: Bias injection (no adaptation)")
    adapter.reset()
    adapter.enable_bias_only()

    vanilla_output = router.forward(hidden_states)
    biased_output = adapter.forward_with_bias(router.forward, hidden_states)

    print(f"  Vanilla probs: {vanilla_output[0][0, 0, :4]}")
    print(f"  Biased probs:  {biased_output[0][0, 0, :4]}")
    print(f"  Different: {not torch.allclose(vanilla_output[0], biased_output[0])}")
    print()

    print("✓ Adapter tests complete")
