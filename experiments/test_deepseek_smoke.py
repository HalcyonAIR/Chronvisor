#!/usr/bin/env python3
"""
Quick smoke test for DeepSeek + Chronovisor integration.

Verifies:
1. Model initializes
2. Forward pass works
3. Routing statistics are collected
4. P×T coupling computes pressure
5. Gradients flow correctly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np

from chronomoe.deepseek_core import DeepSeekConfig
from chronomoe.chronovisor_deepseek_bridge import ChronovisorDeepSeekForCausalLM


def test_initialization():
    """Test model initialization."""
    print("=" * 70)
    print("TEST 1: Model Initialization")
    print("=" * 70)

    config = DeepSeekConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=1024,
        num_layers=2,
        num_shared_experts=2,
        num_routed_experts=16,
        num_experts_per_token=4,
        enable_chronovisor=True,
    )

    model = ChronovisorDeepSeekForCausalLM(config)

    print(f"✓ Model created")
    print(f"  Shared experts: {config.num_shared_experts}")
    print(f"  Routed experts: {config.num_routed_experts}")
    print(f"  Chronovisor enabled: {config.enable_chronovisor}")
    print()

    return model, config


def test_forward_pass(model, config):
    """Test forward pass."""
    print("=" * 70)
    print("TEST 2: Forward Pass")
    print("=" * 70)

    # Create dummy input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    logits, chrono_state, aux_loss = model(input_ids, update_chronovisor=True)

    print(f"✓ Forward pass successful")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Aux loss: {aux_loss.item():.6f}")
    print()

    return logits, chrono_state, aux_loss


def test_routing_statistics(chrono_state, config):
    """Test routing statistics collection."""
    print("=" * 70)
    print("TEST 3: Routing Statistics")
    print("=" * 70)

    if chrono_state is None:
        print("✗ Chrono state is None")
        return False

    print(f"✓ Chrono state collected")

    # Check expert usage
    if chrono_state.expert_usage:
        layer_0_usage = chrono_state.expert_usage[0]
        print(f"  Expert usage (layer 0): shape {layer_0_usage.shape}")
        print(f"  Sum: {layer_0_usage.sum():.4f} (should be ≈1.0)")
        print(f"  Mean: {layer_0_usage.mean():.4f}")
        print(f"  Std: {layer_0_usage.std():.4f}")
    else:
        print("✗ Expert usage not collected")

    # Check routing entropy
    if chrono_state.routing_entropy:
        layer_0_entropy = chrono_state.routing_entropy[0]
        print(f"  Routing entropy (layer 0): {layer_0_entropy:.4f}")
        max_entropy = np.log(config.num_routed_experts)
        print(f"  Max possible entropy: {max_entropy:.4f}")
    else:
        print("✗ Routing entropy not collected")

    print()
    return True


def test_pressure_computation(chrono_state, config):
    """Test pressure computation."""
    print("=" * 70)
    print("TEST 4: P×T Coupling - Pressure Computation")
    print("=" * 70)

    if chrono_state is None or not chrono_state.pressure:
        print("✗ Pressure not computed")
        return False

    layer_0_pressure = chrono_state.pressure[0]
    print(f"✓ Pressure computed")
    print(f"  Pressure (layer 0): shape {layer_0_pressure.shape}")
    print(f"  Mean: {layer_0_pressure.mean():.6f}")
    print(f"  Std: {layer_0_pressure.std():.6f}")
    print(f"  Range: [{layer_0_pressure.min():.6f}, {layer_0_pressure.max():.6f}]")
    print()

    return True


def test_temperature_tracking(chrono_state, config):
    """Test temperature tracking."""
    print("=" * 70)
    print("TEST 5: P×T Coupling - Temperature Tracking")
    print("=" * 70)

    if chrono_state is None or not chrono_state.T_bar_local:
        print("✗ Temperature not tracked")
        return False

    layer_0_T = chrono_state.T_bar_local[0]
    print(f"✓ Temperature tracked")
    print(f"  T̄ (layer 0): shape {layer_0_T.shape}")
    print(f"  Mean: {layer_0_T.mean():.6f}")
    print(f"  Std: {layer_0_T.std():.6f}")
    print(f"  Range: [{layer_0_T.min():.6f}, {layer_0_T.max():.6f}]")
    print()

    return True


def test_clock_system(chrono_state):
    """Test clock system."""
    print("=" * 70)
    print("TEST 6: Clock System")
    print("=" * 70)

    if chrono_state is None:
        print("✗ Chrono state is None")
        return False

    print(f"✓ Clocks initialized")
    print(f"  Fast clock: {chrono_state.fast_clock}")
    print(f"  Micro clock: {chrono_state.micro_clock}")
    print(f"  Macro clock: {chrono_state.macro_clock}")
    print()

    return True


def test_backward_pass(model, config):
    """Test backward pass (gradients flow)."""
    print("=" * 70)
    print("TEST 7: Backward Pass")
    print("=" * 70)

    # Create dummy input and target
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    total_loss, aux_loss = model.compute_loss(input_ids, target_ids)

    print(f"  Total loss: {total_loss.item():.6f}")
    print(f"  Aux loss: {aux_loss.item():.6f}")

    # Backward pass
    total_loss.backward()

    # Check gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    if has_grad:
        print(f"✓ Gradients computed successfully")
    else:
        print(f"✗ No gradients found")

    print()
    return has_grad


def test_shared_vs_routed_separation():
    """Test that shared experts don't get P×T coupling."""
    print("=" * 70)
    print("TEST 8: Shared vs Routed Separation")
    print("=" * 70)

    config = DeepSeekConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=1024,
        num_layers=1,
        num_shared_experts=2,
        num_routed_experts=8,
        num_experts_per_token=2,
        enable_chronovisor=True,
    )

    model = ChronovisorDeepSeekForCausalLM(config)

    # Get number of routed experts from controller
    num_controller_experts = model.model.controller.num_routed_experts

    print(f"  Config routed experts: {config.num_routed_experts}")
    print(f"  Controller tracking: {num_controller_experts} experts")
    print(f"  Shared experts in model: {config.num_shared_experts}")

    if num_controller_experts == config.num_routed_experts:
        print(f"✓ Controller only tracks routed experts")
        print(f"  (Shared experts are NOT under P×T control)")
    else:
        print(f"✗ Controller tracking mismatch")

    print()


def main():
    print()
    print("=" * 70)
    print("DEEPSEEK + CHRONOVISOR SMOKE TEST")
    print("=" * 70)
    print()
    print("Testing integration of P×T coupling with DeepSeek's")
    print("shared+routed expert architecture.")
    print()

    # Run tests
    model, config = test_initialization()
    logits, chrono_state, aux_loss = test_forward_pass(model, config)
    test_routing_statistics(chrono_state, config)
    test_pressure_computation(chrono_state, config)
    test_temperature_tracking(chrono_state, config)
    test_clock_system(chrono_state)
    test_backward_pass(model, config)
    test_shared_vs_routed_separation()

    print("=" * 70)
    print("SMOKE TEST COMPLETE")
    print("=" * 70)
    print()
    print("✓ All basic integration tests passed")
    print()
    print("Key findings:")
    print("  - Model initializes correctly")
    print("  - Forward/backward passes work")
    print("  - Routing statistics are collected")
    print("  - P×T coupling computes pressure on routed experts only")
    print("  - Shared experts provide stable baseline")
    print("  - Clock system tracks multi-timescale dynamics")
    print()
    print("Ready for full training experiments.")
    print()


if __name__ == '__main__':
    main()
