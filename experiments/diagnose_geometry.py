"""
Diagnostic script to verify P×T geometry is reaching the router.

Checks:
1. Pressure biases are non-zero
2. Pressure is added to router logits
3. Effective temperatures change over time
4. Routing distributions are affected by geometry
"""

import torch
import numpy as np
from chronomoe.chronovisor_mixtral_bridge import ChronovisorMixtralForCausalLM
from chronomoe.mixtral_core import MixtralConfig


def diagnose_geometry():
    """Run diagnostic checks on ChronoMoE geometry."""

    print("=" * 70)
    print("CHRONOMOE GEOMETRY DIAGNOSTIC")
    print("=" * 70)

    # Create small model
    config = MixtralConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=1024,
        num_layers=2,
        num_experts=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        enable_chronovisor=True,
    )

    model = ChronovisorMixtralForCausalLM(config)
    controller = model.model.controller

    print(f"\nModel: {config.num_layers} layers, {config.num_experts} experts")
    print(f"ChronoMoE enabled: {config.enable_chronovisor}")

    # Run a few forward passes
    batch_size = 2
    seq_len = 8

    print("\n" + "=" * 70)
    print("RUNNING 5 FORWARD PASSES WITH GEOMETRY")
    print("=" * 70)

    for step in range(5):
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        with torch.no_grad():
            logits, chrono_state = model(input_ids, update_chronovisor=True)

        print(f"\n--- Forward Pass {step} ---")

        # Check 1: Pressure biases
        print("\n1. PRESSURE BIASES:")
        for layer_idx in range(config.num_layers):
            lens = controller.lenses[layer_idx]
            # Use expert usage from chrono_state if available
            if chrono_state.expert_usage and layer_idx in chrono_state.expert_usage:
                expert_usage = chrono_state.expert_usage[layer_idx]
                if isinstance(expert_usage, np.ndarray):
                    expert_usage_np = expert_usage
                else:
                    expert_usage_np = expert_usage.cpu().numpy()
                pressure = lens.compute_pressure(expert_usage_np)
            else:
                # Use zero usage as fallback
                pressure = lens.compute_pressure(np.zeros(config.num_experts))

            print(f"   Layer {layer_idx}:")
            print(f"     Pressure range: [{pressure.min():.6f}, {pressure.max():.6f}]")
            print(f"     Pressure mean: {pressure.mean():.6f}")
            print(f"     Pressure std: {pressure.std():.6f}")
            print(f"     Non-zero: {(pressure != 0).any()}")

        # Check 2: Temperature fields
        print("\n2. EFFECTIVE TEMPERATURES:")
        for layer_idx in range(config.num_layers):
            lens = controller.lenses[layer_idx]
            T_eff = lens.temperature_effective

            print(f"   Layer {layer_idx}:")
            print(f"     T_eff range: [{T_eff.min():.6f}, {T_eff.max():.6f}]")
            print(f"     T_eff mean: {T_eff.mean():.6f}")
            print(f"     T_eff std: {T_eff.std():.6f}")

        # Check 3: Structural T
        print("\n3. STRUCTURAL TEMPERATURE:")
        st_diag = controller.get_structural_temperature_diagnostics()
        print(f"   T̄_global mean: {st_diag['mean']:.6f}")
        print(f"   T̄_global variance: {st_diag['variance']:.6f}")
        print(f"   Valleys: {len(st_diag['valleys'])}")
        print(f"   Ridges: {len(st_diag['ridges'])}")

        # Check 4: Expert usage
        print("\n4. EXPERT USAGE:")
        if chrono_state.expert_usage:
            for layer_idx, usage in chrono_state.expert_usage.items():
                if isinstance(usage, np.ndarray):
                    usage_tensor = torch.from_numpy(usage)
                else:
                    usage_tensor = usage
                print(f"   Layer {layer_idx}: {usage_tensor.tolist()}")

        # Check 5: Kuramoto R
        print(f"\n5. KURAMOTO R: {chrono_state.coherence:.6f}")

    # Final check: Verify geometry is being passed to router
    print("\n" + "=" * 70)
    print("ROUTER GEOMETRY CONNECTION CHECK")
    print("=" * 70)

    # Manually inspect a single router call
    layer_0 = model.model.layers[0]
    router = layer_0.moe.router

    # Create dummy input
    hidden = torch.randn(batch_size, seq_len, config.hidden_dim)

    # Get pressure and temperature from controller
    lens = controller.lenses[0]
    # Use zero usage for this test
    pressure = lens.compute_pressure(np.zeros(config.num_experts))
    temperature = lens.temperature_effective

    print(f"\nPressure to be passed to router:")
    print(f"  Shape: {pressure.shape}")
    print(f"  Values: {pressure}")

    print(f"\nTemperature to be passed to router:")
    print(f"  Shape: {temperature.shape}")
    print(f"  Values: {temperature}")

    # Call router WITHOUT geometry
    print("\n--- Router WITHOUT Geometry ---")
    with torch.no_grad():
        weights_no_geo, experts_no_geo = router(hidden)
    print(f"Routing weights (no geometry): {weights_no_geo[0, 0].tolist()}")
    print(f"Selected experts (no geometry): {experts_no_geo[0, 0].tolist()}")

    # Call router WITH geometry
    print("\n--- Router WITH Geometry ---")
    with torch.no_grad():
        pressure_bias = torch.from_numpy(pressure).float()
        temperature_field = torch.from_numpy(temperature).float()
        weights_with_geo, experts_with_geo = router(
            hidden,
            pressure_bias=pressure_bias,
            temperature_field=temperature_field
        )
    print(f"Routing weights (with geometry): {weights_with_geo[0, 0].tolist()}")
    print(f"Selected experts (with geometry): {experts_with_geo[0, 0].tolist()}")

    # Check if routing changed
    weights_changed = not torch.allclose(weights_no_geo, weights_with_geo, atol=1e-6)
    experts_changed = not torch.equal(experts_no_geo, experts_with_geo)

    print(f"\n⚠️  CRITICAL CHECKS:")
    print(f"  Routing weights changed: {weights_changed}")
    print(f"  Selected experts changed: {experts_changed}")

    if weights_changed or experts_changed:
        print("\n✅ GEOMETRY IS AFFECTING ROUTING")
    else:
        print("\n❌ GEOMETRY IS NOT AFFECTING ROUTING")
        print("   → Pressure/Temperature may be too small")
        print("   → Check if geometry is being passed in forward pass")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    diagnose_geometry()
