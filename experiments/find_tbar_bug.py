"""
Find the T̄ Variance Bug

We've proven proto-roles exist (L2 norm: 0.6082), yet T̄_var stayed at exactly 0.
This script investigates where the bug is in the structural temperature system.

Checks:
1. Raw T_fast spread per expert (are upstream signals differentiating?)
2. T̄_global and T̄_local min/max (is variance hidden by normalization?)
3. Reliability spread (are expert performance metrics diverging?)
4. EMA update math (is η effectively zero or T̄ getting averaged?)
"""

import torch
import numpy as np
from pathlib import Path

from chronomoe.chronovisor_mixtral_bridge import ChronovisorMixtralForCausalLM
from chronomoe.mixtral_core import MixtralConfig


def inspect_trained_model():
    """Load trained model and inspect structural temperature state."""

    print("=" * 70)
    print("T̄ VARIANCE BUG INVESTIGATION")
    print("=" * 70)
    print("\nContext: Proto-roles exist (L2=0.6082), but T̄_var=0 throughout training")
    print("Goal: Find where the bug is in the structural temperature system")
    print("=" * 70)

    # Load trained model
    print("\n1. Loading trained model checkpoint...")
    config = MixtralConfig(
        vocab_size=1000,
        hidden_dim=512,
        intermediate_dim=2048,
        num_layers=4,
        num_experts=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        enable_chronovisor=True,
    )

    model = ChronovisorMixtralForCausalLM(config)
    checkpoint_path = Path("proto_role_results/model_trained.pt")

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    controller = model.model.controller

    print(f"✅ Model loaded from {checkpoint_path}")

    # Inspect each layer's lens
    print("\n" + "=" * 70)
    print("LAYER-BY-LAYER INSPECTION")
    print("=" * 70)

    for layer_idx in range(config.num_layers):
        lens = controller.lenses[layer_idx]

        print(f"\n{'=' * 70}")
        print(f"LAYER {layer_idx}")
        print(f"{'=' * 70}")

        # 1. Raw T_fast per expert
        print("\n1. RAW T_FAST PER EXPERT:")
        T_fast = lens.temperature_fast
        print(f"   Values: {T_fast}")
        print(f"   Min: {T_fast.min():.15f}")
        print(f"   Max: {T_fast.max():.15f}")
        print(f"   Mean: {T_fast.mean():.15f}")
        print(f"   Std: {T_fast.std():.15f}")
        print(f"   Variance: {T_fast.var():.15f}")

        # Check if T_fast has differentiated
        if T_fast.std() > 0.05:
            print(f"   ✅ T_fast HAS SPREAD (std={T_fast.std():.6f})")
        else:
            print(f"   ⚠️  T_fast nearly uniform (std={T_fast.std():.6f})")

        # 2. Structural Temperature (hierarchical)
        print("\n2. STRUCTURAL TEMPERATURE T̄:")
        T_bar_hierarchical = lens.structural_T_hierarchical
        print(f"   Values: {T_bar_hierarchical}")
        print(f"   Min: {T_bar_hierarchical.min():.15f}")
        print(f"   Max: {T_bar_hierarchical.max():.15f}")
        print(f"   Spread (max-min): {(T_bar_hierarchical.max() - T_bar_hierarchical.min()):.15f}")
        print(f"   Mean: {T_bar_hierarchical.mean():.15f}")
        print(f"   Std: {T_bar_hierarchical.std():.15f}")
        print(f"   Variance: {T_bar_hierarchical.var():.20f}")

        # Check for the bug
        if T_bar_hierarchical.std() < 1e-10:
            print(f"   🔴 T̄ IS PERFECTLY UNIFORM (std < 1e-10)")
            print(f"   → BUG CONFIRMED: T̄ not diverging despite proto-role formation")
        elif T_bar_hierarchical.std() < 1e-6:
            print(f"   ⚠️  T̄ has tiny variance (std={T_bar_hierarchical.std():.2e})")
            print(f"   → May be print precision issue (was showing as 0.000000)")
        else:
            print(f"   ✅ T̄ has measurable variance")

        # 3. Check if all values are identical
        print("\n3. UNIFORMITY CHECK:")
        unique_values = np.unique(T_bar_hierarchical)
        print(f"   Unique T̄ values: {len(unique_values)}")
        if len(unique_values) == 1:
            print(f"   🔴 ALL T̄ VALUES IDENTICAL: {unique_values[0]:.15f}")
            print(f"   → This is the smoking gun - T̄ not updating per expert")
        else:
            print(f"   ✅ T̄ has {len(unique_values)} distinct values")

        # 4. Local structural T (before hierarchical multiplication)
        print("\n4. LOCAL STRUCTURAL T̄:")
        T_bar_local = lens.structural_T
        print(f"   Values: {T_bar_local}")
        print(f"   Min: {T_bar_local.min():.15f}")
        print(f"   Max: {T_bar_local.max():.15f}")
        print(f"   Std: {T_bar_local.std():.15f}")

        # 5. Effective Temperature (T_fast × T̄)
        print("\n5. EFFECTIVE TEMPERATURE T_eff:")
        T_eff = lens.temperature_effective
        print(f"   Values: {T_eff}")
        print(f"   Min: {T_eff.min():.15f}")
        print(f"   Max: {T_eff.max():.15f}")
        print(f"   Std: {T_eff.std():.15f}")

    # Global structural temperature diagnostics
    print("\n" + "=" * 70)
    print("GLOBAL STRUCTURAL TEMPERATURE DIAGNOSTICS")
    print("=" * 70)

    st_diag = controller.get_structural_temperature_diagnostics()
    print(f"\nT̄_global mean: {st_diag['mean']:.20f}")
    print(f"T̄_global variance: {st_diag['variance']:.25f}")
    print(f"T̄_global min: {st_diag['min']:.20f}")
    print(f"T̄_global max: {st_diag['max']:.20f}")
    print(f"T̄_global spread (max-min): {st_diag['max'] - st_diag['min']:.25f}")
    print(f"Valleys: {len(st_diag['valleys'])}")
    print(f"Ridges: {len(st_diag['ridges'])}")

    # Final diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    # Check each potential bug
    print("\nPotential bugs:")

    print("\n1. PRINT PRECISION ISSUE?")
    if st_diag['variance'] > 1e-10:
        print(f"   ⚠️  Variance exists but is tiny: {st_diag['variance']:.2e}")
        print(f"   → Was being printed as '0.000000' due to format string")
        print(f"   → Need to use higher precision formatting")
    else:
        print(f"   ❌ Variance is literally zero: {st_diag['variance']:.2e}")
        print(f"   → Not a print precision issue")

    print("\n2. NORMALIZATION/AVERAGING BUG?")
    if st_diag['max'] - st_diag['min'] < 1e-10:
        print(f"   🔴 CONFIRMED: All T̄ values identical")
        print(f"   → T̄ being normalized/averaged after update")
        print(f"   → Check for mean() or normalization in update step")
    else:
        print(f"   ✅ T̄ values do differ slightly")

    print("\n3. η EFFECTIVELY ZERO?")
    print(f"   → Check if learning_rate_structural is too small")
    print(f"   → Or if EMA update is being skipped")

    print("\n4. UPDATE NOT REACHING T̄?")
    print(f"   → Check if T̄ update code path is actually executing")
    print(f"   → Add debug prints in TemperatureField.update_structural()")

    print("\n" + "=" * 70)
    print("RECOMMENDED NEXT STEPS")
    print("=" * 70)
    print("\n1. Add debug logging to TemperatureField.update_structural()")
    print("   - Print T̄_before and T̄_after each update")
    print("   - Verify update is actually being called")
    print("   - Check if drift signal is non-zero")
    print("\n2. Check for normalization in update code")
    print("   - Search for .mean() operations on T̄")
    print("   - Look for normalization that flattens variance")
    print("\n3. Increase η temporarily (10x) as test")
    print("   - If T̄ still doesn't move → update not executing")
    print("   - If T̄ moves → η was too small")


if __name__ == "__main__":
    inspect_trained_model()
