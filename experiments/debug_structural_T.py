"""
Debug script to validate structural temperature evolution.

This directly exercises the controller to verify structural T updates.
"""

import numpy as np
import torch
from chronomoe.chronovisor_mixtral_bridge import (
    ChronovisorMixtralForCausalLM,
)
from chronomoe.mixtral_core import MixtralConfig


def test_structural_T_evolution():
    """Test that structural T actually evolves over ticks."""

    print("🔬 Testing Structural Temperature Evolution")
    print("=" * 60)

    # Create small model
    config = MixtralConfig(
        vocab_size=32000,
        hidden_dim=256,
        intermediate_dim=1024,
        num_layers=2,  # Even smaller for this test
        num_experts=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        enable_chronovisor=True,
    )

    model = ChronovisorMixtralForCausalLM(config)
    controller = model.model.controller

    # Set fast geology for quick evolution
    controller.eta_structural_T_global = 0.1  # Very fast
    for lens in controller.lenses.values():
        lens.eta_structural_T = 0.2  # Very fast

    print(f"\nInitial state:")
    print(f"  η_global: {controller.eta_structural_T_global}")
    print(f"  η_local: {controller.lenses[0].eta_structural_T}")
    print(f"  T̄_global: {controller.structural_T_global}")
    print(f"  T̄_local[0]: {controller.lenses[0].structural_T}")
    print(f"  Variance: {np.var(controller.structural_T_global)}")

    # Run forward passes with different routing patterns
    for step in range(100):
        batch_size = 2
        seq_len = 8

        # Vary input to create different routing patterns
        if step % 2 == 0:
            # Pattern A: bias toward lower expert IDs
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        else:
            # Pattern B: bias toward higher expert IDs
            input_ids = torch.randint(1000, 2000, (batch_size, seq_len))

        # Forward pass
        with torch.no_grad():
            logits, chrono_state = model(input_ids, update_chronovisor=True)

        # Print every 10 steps
        if step % 10 == 0:
            var_global = np.var(controller.structural_T_global)
            var_local = np.var(controller.lenses[0].structural_T)
            print(f"\nStep {step}:")
            print(f"  Kuramoto R: {chrono_state.coherence:.4f}")
            print(f"  T̄_global mean: {controller.structural_T_global.mean():.4f}")
            print(f"  T̄_global variance: {var_global:.6f}")
            print(f"  T̄_local[0] mean: {controller.lenses[0].structural_T.mean():.4f}")
            print(f"  T̄_local[0] variance: {var_local:.6f}")
            print(f"  T̄_hierarchical[0] mean: {controller.lenses[0].structural_T_hierarchical.mean():.4f}")
            print(f"  Expert usage[0]: {controller.expert_usage[0]}")

    print("\n" + "=" * 60)
    print("Final state:")
    print(f"  T̄_global: {controller.structural_T_global}")
    print(f"  T̄_global variance: {np.var(controller.structural_T_global):.6f}")
    print(f"  T̄_local[0]: {controller.lenses[0].structural_T}")
    print(f"  T̄_local[0] variance: {np.var(controller.lenses[0].structural_T):.6f}")

    if np.var(controller.structural_T_global) > 0.001:
        print("\n✅ SUCCESS: Structural temperature evolved!")
    else:
        print("\n❌ FAILURE: Structural temperature did not evolve")
        print("   This suggests structural T is not being updated during forward passes")


if __name__ == "__main__":
    test_structural_T_evolution()
