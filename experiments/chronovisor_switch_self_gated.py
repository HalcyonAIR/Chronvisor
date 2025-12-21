#!/usr/bin/env python3
"""
Self-Gated ChronoMoE Adapter for Switch Transformers

Phase 2: Test if margin-conditioned state can create path-dependent routing
where unconditional pressure alone could not.

Key mechanism:
    gate = f(margin)  # Low margin → high gate (listen to history)
                      # High margin → low gate (trust current evidence)

    biased_logits = logits + gate * pressure(T̄)

This creates a self-regulating system where T̄ influence is strong exactly where
routing is uncertain, but weak where it's confident.

Hypothesis:
    State with momentum can create path-dependent crossings where pressure
    alone cannot deform the manifold.

Success criteria:
    - 7→4→2 cliff shifts with approach direction (hysteresis)
    - WITHOUT causing permanent Expert 2 lock-in (pathological collapse)
    - In-distribution routing remains stable (no corruption)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class SelfGatedGeologicalLens:
    """
    Geological temperature with margin-based self-gating.

    T̄ influence is modulated by routing confidence:
    - Low confidence (low margin) → high gate → listen to history
    - High confidence (high margin) → low gate → trust current evidence
    """

    def __init__(
        self,
        num_experts: int,
        eta_structural_T: float = 0.015,
        gate_scale: float = 2.0,  # Controls gate sensitivity to margin
        gate_offset: float = 0.5,  # Margin threshold for 50% gating
    ):
        self.num_experts = num_experts
        self.eta_structural_T = eta_structural_T
        self.gate_scale = gate_scale
        self.gate_offset = gate_offset

        # Geological temperature (starts uniform)
        self.T_bar = np.ones(num_experts, dtype=np.float32)

        # EMA of expert usage
        self.expert_usage_ema = np.ones(num_experts, dtype=np.float32) / num_experts

        # Track gating statistics
        self.gate_history = []
        self.margin_history = []

    def compute_pressure(self) -> np.ndarray:
        """
        Compute pressure bias from T̄.

        Pressure = log(T̄) - mean(log(T̄))
        Zero-sum bias that favors warmer experts.
        """
        log_T = np.log(self.T_bar + 1e-10)
        pressure = log_T - np.mean(log_T)
        return pressure

    def compute_margin_gate(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute margin-based gate.

        Args:
            router_logits: [batch, seq, num_experts]

        Returns:
            gate: [batch, seq] where:
                - gate ≈ 1 when margin is low (uncertain → listen to history)
                - gate ≈ 0 when margin is high (confident → ignore history)

        Gate function:
            gate = 1 / (1 + exp(gate_scale * (margin - gate_offset)))

        This is a sigmoid centered at gate_offset:
            - margin < gate_offset → gate > 0.5
            - margin > gate_offset → gate < 0.5
        """
        # Compute margin: top1_logit - top2_logit
        sorted_logits, _ = torch.sort(router_logits, dim=-1, descending=True)
        margin = sorted_logits[..., 0] - sorted_logits[..., 1]  # [batch, seq]

        # Compute gate (inverted sigmoid)
        # When margin is low (uncertain), gate → 1 (listen to history)
        # When margin is high (confident), gate → 0 (ignore history)
        gate = 1.0 / (1.0 + torch.exp(self.gate_scale * (margin - self.gate_offset)))

        # Track statistics
        self.margin_history.append(margin.detach().cpu().numpy().mean())
        self.gate_history.append(gate.detach().cpu().numpy().mean())

        return gate

    def update(self, routing_probs: np.ndarray):
        """
        Update T̄ based on routing probabilities.

        routing_probs: [batch * seq_len, num_experts]
        """
        # Average routing over batch/sequence
        usage = routing_probs.mean(axis=0)  # [num_experts]

        # Update EMA
        alpha = 0.1
        self.expert_usage_ema = alpha * usage + (1 - alpha) * self.expert_usage_ema

        # Update T̄: warm used experts, cool unused ones
        target = usage - (1.0 / self.num_experts)
        self.T_bar += self.eta_structural_T * target

        # Clamp to prevent runaway
        self.T_bar = np.clip(self.T_bar, 0.5, 2.0)

    def reset(self):
        """Reset T̄ to uniform."""
        self.T_bar = np.ones(self.num_experts, dtype=np.float32)
        self.expert_usage_ema = np.ones(self.num_experts, dtype=np.float32) / self.num_experts
        self.gate_history = []
        self.margin_history = []

    def get_gating_stats(self):
        """Get statistics on gating behavior."""
        if not self.gate_history:
            return None

        return {
            'mean_gate': np.mean(self.gate_history),
            'mean_margin': np.mean(self.margin_history),
            'gate_history': self.gate_history.copy(),
            'margin_history': self.margin_history.copy(),
        }


class SelfGatedChronoMoEAdapter:
    """
    Self-gated ChronoMoE adapter with margin-conditioned influence.

    Key difference from base adapter:
        Base:       biased_logits = logits + pressure
        Self-gated: biased_logits = logits + gate(margin) * pressure

    Where gate is strong when routing is uncertain, weak when confident.
    """

    def __init__(
        self,
        num_experts: int,
        eta_structural_T: float = 0.015,
        gate_scale: float = 2.0,
        gate_offset: float = 0.5,
    ):
        self.num_experts = num_experts
        self.lens = SelfGatedGeologicalLens(
            num_experts,
            eta_structural_T,
            gate_scale,
            gate_offset,
        )

        # Control flags
        self.adaptation_enabled = False
        self.inject_bias = False

    def enable_adaptation(self):
        """Enable T̄ updates and bias injection."""
        self.adaptation_enabled = True
        self.inject_bias = True

    def disable_adaptation(self):
        """Disable T̄ updates (keep bias if set)."""
        self.adaptation_enabled = False

    def enable_bias_only(self):
        """Inject bias without updating T̄."""
        self.adaptation_enabled = False
        self.inject_bias = True

    def disable_all(self):
        """Disable everything (vanilla mode)."""
        self.adaptation_enabled = False
        self.inject_bias = False

    def reset(self):
        """Reset T̄ and gating history."""
        self.lens.reset()

    def forward_with_bias(
        self,
        original_forward,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Wrapper with self-gated bias injection.

        Args:
            original_forward: Router's original forward method
            hidden_states: [batch, seq_len, hidden_dim]

        Returns:
            router_mask: [batch, seq_len, num_experts]
            router_probs: [batch, seq_len, 1] (confidence for selected expert)
            router_logits: [batch, seq_len, num_experts]
        """
        # Get original router output
        router_mask, router_probs, router_logits = original_forward(hidden_states)

        # Inject self-gated bias if enabled
        if self.inject_bias:
            # Compute margin-based gate
            gate = self.lens.compute_margin_gate(router_logits)  # [batch, seq]

            # Compute pressure from T̄
            pressure = self.lens.compute_pressure()  # [num_experts]
            pressure_tensor = torch.tensor(
                pressure,
                dtype=router_logits.dtype,
                device=router_logits.device,
            )

            # Apply gated pressure
            # gate: [batch, seq]
            # pressure: [num_experts]
            # Result: [batch, seq, num_experts]
            gated_pressure = gate.unsqueeze(-1) * pressure_tensor.unsqueeze(0).unsqueeze(0)
            biased_logits = router_logits + gated_pressure

            # Recompute routing from biased logits
            full_probs = torch.softmax(biased_logits, dim=-1)

            # Top-1 selection
            selected_experts = torch.argmax(full_probs, dim=-1)
            router_mask = torch.nn.functional.one_hot(
                selected_experts,
                num_classes=self.num_experts,
            ).to(router_logits.dtype)

            # Confidence for selected expert [batch, seq, 1]
            router_probs = full_probs.gather(
                dim=-1,
                index=selected_experts.unsqueeze(-1),
            )

            # Use biased logits
            router_logits = biased_logits

        # Update T̄ if adaptation enabled
        if self.adaptation_enabled:
            # Compute full distribution for T̄ update
            full_probs = torch.softmax(router_logits, dim=-1)

            # Convert to numpy
            routing_np = full_probs.detach().cpu().numpy()
            batch_size, seq_len, _ = routing_np.shape
            routing_flat = routing_np.reshape(-1, self.num_experts)

            self.lens.update(routing_flat)

        return router_mask, router_probs, router_logits

    def get_state(self):
        """Get current state for inspection."""
        state = {
            'T_bar': self.lens.T_bar.copy(),
            'expert_usage_ema': self.lens.expert_usage_ema.copy(),
            'pressure': self.lens.compute_pressure(),
            'adaptation_enabled': self.adaptation_enabled,
            'inject_bias': self.inject_bias,
        }

        gating_stats = self.lens.get_gating_stats()
        if gating_stats:
            state.update(gating_stats)

        return state


def wrap_switch_model_with_self_gated_chronovisor(
    model,
    layer_indices: list = [1],
    eta_structural_T: float = 0.015,
    gate_scale: float = 2.0,
    gate_offset: float = 0.5,
) -> dict:
    """
    Wrap Switch Transformer with self-gated ChronoMoE adapters.

    Args:
        model: Switch Transformer model
        layer_indices: Which layers to wrap
        eta_structural_T: Geological learning rate
        gate_scale: Gate sensitivity to margin
        gate_offset: Margin threshold for 50% gating

    Returns:
        adapters: Dict mapping layer_idx -> adapter
    """
    adapters = {}

    for layer_idx in layer_indices:
        if layer_idx == 0:
            print(f"⚠ Skipping layer 0 (dense, no router)")
            continue

        block = model.encoder.block[layer_idx]

        if not hasattr(block.layer[-1].mlp, 'router'):
            print(f"⚠ Layer {layer_idx} has no router, skipping")
            continue

        router = block.layer[-1].mlp.router
        num_experts = router.classifier.out_features

        # Create self-gated adapter
        adapter = SelfGatedChronoMoEAdapter(
            num_experts,
            eta_structural_T,
            gate_scale,
            gate_offset,
        )

        # Store original forward
        original_forward = router.forward

        # Wrap with self-gated adapter
        def make_wrapped_forward(adapter_ref, orig_forward):
            def wrapped_forward(hidden_states):
                return adapter_ref.forward_with_bias(orig_forward, hidden_states)
            return wrapped_forward

        router.forward = make_wrapped_forward(adapter, original_forward)

        adapters[layer_idx] = adapter

        print(f"✓ Wrapped layer {layer_idx} router with self-gated ChronoMoE")
        print(f"  gate_scale={gate_scale:.2f}, gate_offset={gate_offset:.2f}")

    return adapters


if __name__ == '__main__':
    """Test self-gated adapter."""

    print("Testing Self-Gated ChronoMoE Adapter")
    print("="*70)
    print()

    # Create mock router
    class MockRouter:
        def __init__(self, num_experts=8):
            self.classifier = nn.Linear(768, num_experts, bias=False)

        def forward(self, hidden_states):
            router_logits = self.classifier(hidden_states)
            full_probs = torch.softmax(router_logits, dim=-1)
            selected_experts = torch.argmax(full_probs, dim=-1)

            # Return Switch-style output
            router_mask = torch.nn.functional.one_hot(
                selected_experts, num_classes=8
            ).to(router_logits.dtype)

            router_probs = full_probs.gather(
                dim=-1,
                index=selected_experts.unsqueeze(-1),
            )

            return router_mask, router_probs, router_logits

    router = MockRouter(num_experts=8)
    adapter = SelfGatedChronoMoEAdapter(
        num_experts=8,
        eta_structural_T=0.015,
        gate_scale=2.0,
        gate_offset=0.5,
    )

    # Test gate computation
    print("Test 1: Margin-based gating")
    print()

    # Create inputs with different margin levels
    hidden_high_margin = torch.randn(1, 10, 768)  # Will have random margins

    adapter.enable_adaptation()

    # Run a few passes to see gate behavior
    for i in range(5):
        _ = adapter.forward_with_bias(router.forward, hidden_high_margin)

    stats = adapter.get_state()

    print(f"  Mean gate: {stats['mean_gate']:.4f}")
    print(f"  Mean margin: {stats['mean_margin']:.4f}")
    print(f"  T̄ range: [{stats['T_bar'].min():.3f}, {stats['T_bar'].max():.3f}]")
    print()

    print("Test 2: Gate modulates pressure")
    print()

    # Manually set T̄ to create strong pressure
    adapter.lens.T_bar = np.array([2.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5])
    pressure = adapter.lens.compute_pressure()

    print(f"  Pressure range: [{pressure.min():.3f}, {pressure.max():.3f}]")

    # Run with bias enabled
    adapter.enable_bias_only()
    _, _, logits_biased = adapter.forward_with_bias(router.forward, hidden_high_margin)

    # Run vanilla
    adapter.disable_all()
    _, _, logits_vanilla = router.forward(hidden_high_margin)

    bias_applied = (logits_biased - logits_vanilla).abs().mean().item()

    print(f"  Bias applied: {bias_applied:.6f}")
    print()

    print("Test 3: Self-regulation")
    print("  High margin (confident) should reduce gate")
    print("  Low margin (uncertain) should increase gate")
    print()

    # Create high-confidence input (manually set one expert much higher)
    confident_logits = torch.tensor([[
        [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Expert 0 dominant
        [0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Expert 1 dominant
    ]])

    margin_confident = confident_logits[0, 0, 0] - confident_logits[0, 0, 1]
    gate_confident = adapter.lens.compute_margin_gate(confident_logits)

    # Create low-confidence input (all experts similar)
    uncertain_logits = torch.tensor([[
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.4, 0.3],
    ]])

    margin_uncertain = uncertain_logits[0, 0, 0] - uncertain_logits[0, 0, 1]
    gate_uncertain = adapter.lens.compute_margin_gate(uncertain_logits)

    print(f"  Confident: margin={margin_confident:.2f} → gate={gate_confident.mean():.4f}")
    print(f"  Uncertain: margin={margin_uncertain:.2f} → gate={gate_uncertain.mean():.4f}")
    print(f"  Gate ratio (uncertain/confident): {gate_uncertain.mean()/gate_confident.mean():.2f}x")
    print()

    if gate_uncertain.mean() > gate_confident.mean():
        print("✓ Self-gating works: uncertain regions have higher gate")
    else:
        print("⚠ Self-gating issue: gate not increasing with uncertainty")

    print()
    print("✓ Self-gated adapter tests complete")
