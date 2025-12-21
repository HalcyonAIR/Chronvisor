"""
Phase 1.2: Pressure Monotonicity

Tests that pressure functions are monotonic and isolated (no cross-clock leakage).

Success criteria:
- Increasing router_entropy → fast_pressure strictly decreases
- Increasing margin → mid_pressure strictly decreases
- Increasing constraint_penalty → slow_pressure strictly decreases
- Non-swept pressures remain constant
- No sign flips without signal change

What we ignore:
- Exact functional form (tanh vs sigmoid)
- Absolute magnitudes (testing direction only)
- Weight values (testing pressure functions, not composition)
"""

import json
from pathlib import Path

import numpy as np

from chronomoe.pressure import (
    compute_fast_pressure,
    compute_mid_pressure,
    compute_slow_pressure,
)


def check_monotonic(values, pressures, name, increasing=False):
    """
    Check if pressure changes monotonically with input.

    Args:
        values: Input values (increasing)
        pressures: Pressure values
        name: Name for logging
        increasing: If True, pressure should increase; else decrease

    Returns:
        True if monotonic, False otherwise
    """
    for i in range(len(pressures) - 1):
        if increasing:
            if pressures[i + 1] <= pressures[i]:
                print(f"  ✗ {name} not monotonically increasing:")
                print(f"    {values[i]:.2f} → {pressures[i]:+.4f}")
                print(f"    {values[i + 1]:.2f} → {pressures[i + 1]:+.4f}")
                return False
        else:
            if pressures[i + 1] >= pressures[i]:
                print(f"  ✗ {name} not monotonically decreasing:")
                print(f"    {values[i]:.2f} → {pressures[i]:+.4f}")
                print(f"    {values[i + 1]:.2f} → {pressures[i + 1]:+.4f}")
                return False

    return True


def check_constant(pressures, name, tolerance=1e-6):
    """Check if values are constant (within tolerance)."""
    for i in range(len(pressures) - 1):
        if abs(pressures[i + 1] - pressures[i]) > tolerance:
            print(f"  ✗ {name} not constant:")
            print(f"    {pressures[i]:+.4f} → {pressures[i + 1]:+.4f}")
            return False
    return True


def test_fast_pressure_monotonicity():
    """Test that fast pressure decreases with router entropy."""
    print("\nTest 1: Fast pressure vs router_entropy")
    print("-" * 70)

    # Sweep router entropy
    entropy_values = np.linspace(0.0, 2.0, 10)
    fast_pressures = []
    mid_pressures = []
    slow_pressures = []

    # Fixed values for other inputs
    router_margin = 0.8
    delta_R = 0.1
    margin = 1.0
    mid_transition_prob = 0.8
    mid_proximity_meso = 0.2
    slow_confidence = 0.7
    slow_proximity_macro = 0.1
    slow_constraint_penalty = 0.0

    for entropy in entropy_values:
        # Fast pressure (should decrease with entropy)
        fast_p = compute_fast_pressure(entropy, router_margin, delta_R)
        fast_pressures.append(fast_p)

        # Mid pressure (should be constant - doesn't depend on router_entropy)
        mid_p = compute_mid_pressure(
            margin, mid_transition_prob, mid_proximity_meso, delta_R
        )
        mid_pressures.append(mid_p)

        # Slow pressure (should be constant)
        slow_p = compute_slow_pressure(
            slow_confidence, slow_proximity_macro, slow_constraint_penalty
        )
        slow_pressures.append(slow_p)

    # Check monotonicity
    print(f"  Entropy range: [{entropy_values[0]:.2f}, {entropy_values[-1]:.2f}]")
    print(f"  Fast pressure range: [{fast_pressures[0]:+.4f}, {fast_pressures[-1]:+.4f}]")

    fast_monotonic = check_monotonic(
        entropy_values, fast_pressures, "Fast pressure", increasing=False
    )
    assert fast_monotonic, "Fast pressure not monotonically decreasing with entropy!"

    # Check isolation (other pressures should be constant)
    mid_constant = check_constant(mid_pressures, "Mid pressure")
    assert mid_constant, "Mid pressure changed when only entropy varied!"

    slow_constant = check_constant(slow_pressures, "Slow pressure")
    assert slow_constant, "Slow pressure changed when only entropy varied!"

    print("  ✓ Fast pressure monotonically decreases with entropy")
    print("  ✓ Mid pressure unaffected (isolated)")
    print("  ✓ Slow pressure unaffected (isolated)")

    return {
        "sweep": "router_entropy",
        "values": entropy_values.tolist(),
        "fast_pressure": fast_pressures,
        "mid_pressure": mid_pressures,
        "slow_pressure": slow_pressures,
        "monotonic": fast_monotonic,
        "isolated": mid_constant and slow_constant,
    }


def test_mid_pressure_monotonicity():
    """Test that mid pressure decreases with margin (uncertainty)."""
    print("\nTest 2: Mid pressure vs margin")
    print("-" * 70)

    # Sweep margin (note: high margin = low uncertainty)
    # Mid pressure should decrease as margin increases
    margin_values = np.linspace(0.0, 2.0, 10)
    fast_pressures = []
    mid_pressures = []
    slow_pressures = []

    # Fixed values
    router_entropy = 0.5
    router_margin = 0.8
    delta_R = 0.1
    mid_transition_prob = 0.8
    mid_proximity_meso = 0.2
    slow_confidence = 0.7
    slow_proximity_macro = 0.1
    slow_constraint_penalty = 0.0

    for margin in margin_values:
        # Fast pressure (should be constant)
        fast_p = compute_fast_pressure(router_entropy, router_margin, delta_R)
        fast_pressures.append(fast_p)

        # Mid pressure (should decrease with margin)
        # Because formula uses (1.0 - margin), high margin = low pressure
        mid_p = compute_mid_pressure(
            margin, mid_transition_prob, mid_proximity_meso, delta_R
        )
        mid_pressures.append(mid_p)

        # Slow pressure (should be constant)
        slow_p = compute_slow_pressure(
            slow_confidence, slow_proximity_macro, slow_constraint_penalty
        )
        slow_pressures.append(slow_p)

    # Check monotonicity (mid pressure should decrease with margin)
    print(f"  Margin range: [{margin_values[0]:.2f}, {margin_values[-1]:.2f}]")
    print(f"  Mid pressure range: [{mid_pressures[0]:+.4f}, {mid_pressures[-1]:+.4f}]")

    mid_monotonic = check_monotonic(
        margin_values, mid_pressures, "Mid pressure", increasing=False
    )
    assert mid_monotonic, "Mid pressure not monotonically decreasing with margin!"

    # Check isolation
    fast_constant = check_constant(fast_pressures, "Fast pressure")
    assert fast_constant, "Fast pressure changed when only margin varied!"

    slow_constant = check_constant(slow_pressures, "Slow pressure")
    assert slow_constant, "Slow pressure changed when only margin varied!"

    print("  ✓ Mid pressure monotonically decreases with margin")
    print("  ✓ Fast pressure unaffected (isolated)")
    print("  ✓ Slow pressure unaffected (isolated)")

    return {
        "sweep": "margin",
        "values": margin_values.tolist(),
        "fast_pressure": fast_pressures,
        "mid_pressure": mid_pressures,
        "slow_pressure": slow_pressures,
        "monotonic": mid_monotonic,
        "isolated": fast_constant and slow_constant,
    }


def test_slow_pressure_monotonicity():
    """Test that slow pressure decreases with constraint penalty."""
    print("\nTest 3: Slow pressure vs constraint_penalty")
    print("-" * 70)

    # Sweep constraint penalty
    penalty_values = np.linspace(0.0, 3.0, 10)
    fast_pressures = []
    mid_pressures = []
    slow_pressures = []

    # Fixed values
    router_entropy = 0.5
    router_margin = 0.8
    delta_R = 0.1
    margin = 1.0
    mid_transition_prob = 0.8
    mid_proximity_meso = 0.2
    slow_confidence = 0.7
    slow_proximity_macro = 0.1

    for penalty in penalty_values:
        # Fast pressure (should be constant)
        fast_p = compute_fast_pressure(router_entropy, router_margin, delta_R)
        fast_pressures.append(fast_p)

        # Mid pressure (should be constant)
        mid_p = compute_mid_pressure(
            margin, mid_transition_prob, mid_proximity_meso, delta_R
        )
        mid_pressures.append(mid_p)

        # Slow pressure (should decrease with penalty)
        slow_p = compute_slow_pressure(slow_confidence, slow_proximity_macro, penalty)
        slow_pressures.append(slow_p)

    # Check monotonicity
    print(f"  Penalty range: [{penalty_values[0]:.2f}, {penalty_values[-1]:.2f}]")
    print(f"  Slow pressure range: [{slow_pressures[0]:+.4f}, {slow_pressures[-1]:+.4f}]")

    slow_monotonic = check_monotonic(
        penalty_values, slow_pressures, "Slow pressure", increasing=False
    )
    assert slow_monotonic, "Slow pressure not monotonically decreasing with penalty!"

    # Check isolation
    fast_constant = check_constant(fast_pressures, "Fast pressure")
    assert fast_constant, "Fast pressure changed when only penalty varied!"

    mid_constant = check_constant(mid_pressures, "Mid pressure")
    assert mid_constant, "Mid pressure changed when only penalty varied!"

    print("  ✓ Slow pressure monotonically decreases with constraint penalty")
    print("  ✓ Fast pressure unaffected (isolated)")
    print("  ✓ Mid pressure unaffected (isolated)")

    return {
        "sweep": "constraint_penalty",
        "values": penalty_values.tolist(),
        "fast_pressure": fast_pressures,
        "mid_pressure": mid_pressures,
        "slow_pressure": slow_pressures,
        "monotonic": slow_monotonic,
        "isolated": fast_constant and mid_constant,
    }


def test_no_sign_flips():
    """Test that pressures don't flip sign unexpectedly."""
    print("\nTest 4: No unexpected sign flips")
    print("-" * 70)

    # Test with constant inputs - pressure should have constant sign
    num_samples = 100

    fast_pressures = []
    mid_pressures = []
    slow_pressures = []

    for _ in range(num_samples):
        fast_p = compute_fast_pressure(0.5, 0.8, 0.1)
        mid_p = compute_mid_pressure(1.0, 0.8, 0.2, 0.1)
        slow_p = compute_slow_pressure(0.7, 0.1, 0.0)

        fast_pressures.append(fast_p)
        mid_pressures.append(mid_p)
        slow_pressures.append(slow_p)

    # All should be identical
    assert len(set(fast_pressures)) == 1, "Fast pressure varied with constant inputs!"
    assert len(set(mid_pressures)) == 1, "Mid pressure varied with constant inputs!"
    assert len(set(slow_pressures)) == 1, "Slow pressure varied with constant inputs!"

    print(f"  Fast pressure (constant): {fast_pressures[0]:+.4f}")
    print(f"  Mid pressure (constant):  {mid_pressures[0]:+.4f}")
    print(f"  Slow pressure (constant): {slow_pressures[0]:+.4f}")
    print("  ✓ No sign flips with constant inputs")


def test_bounded_outputs():
    """Test that all pressures are bounded to [-1, 1]."""
    print("\nTest 5: Bounded outputs [-1, 1]")
    print("-" * 70)

    # Test extreme inputs
    test_cases = [
        # (router_entropy, router_margin, delta_R, margin, trans_prob, prox_meso, slow_conf, prox_macro, penalty)
        (0.0, 0.0, -1.0, 0.0, 0.0, 10.0, 1.0, 10.0, 10.0),  # Extreme low
        (2.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 0.0),  # Extreme high
        (1.0, 1.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5),  # Moderate
    ]

    all_bounded = True

    for i, inputs in enumerate(test_cases):
        fast_p = compute_fast_pressure(inputs[0], inputs[1], inputs[2])
        mid_p = compute_mid_pressure(inputs[3], inputs[4], inputs[5], inputs[2])
        slow_p = compute_slow_pressure(inputs[6], inputs[7], inputs[8])

        bounded = (
            -1.0 <= fast_p <= 1.0 and -1.0 <= mid_p <= 1.0 and -1.0 <= slow_p <= 1.0
        )

        if not bounded:
            print(f"  ✗ Case {i + 1}: Pressures out of bounds!")
            print(f"    Fast: {fast_p:+.4f}")
            print(f"    Mid:  {mid_p:+.4f}")
            print(f"    Slow: {slow_p:+.4f}")
            all_bounded = False

    if all_bounded:
        print("  ✓ All pressures bounded to [-1, 1] for extreme inputs")

    assert all_bounded, "Some pressures exceeded [-1, 1] bounds!"


def run_all_tests():
    """Run all Phase 1.2 tests."""
    print("=" * 70)
    print("Phase 1.2: Pressure Monotonicity")
    print("=" * 70)

    results = []

    # Run tests
    results.append(test_fast_pressure_monotonicity())
    results.append(test_mid_pressure_monotonicity())
    results.append(test_slow_pressure_monotonicity())
    test_no_sign_flips()
    test_bounded_outputs()

    # Save results
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)

    log_file = output_dir / "phase1_2_monotonicity.json"
    with open(log_file, "w") as f:
        json.dump(
            {"test": "phase1_2_monotonicity", "status": "PASS", "sweeps": results},
            f,
            indent=2,
        )

    print(f"\n✓ Logs saved to {log_file}")

    print("\n" + "=" * 70)
    print("Phase 1.2: PASS")
    print("=" * 70)
    print("\nKey findings:")
    print("  - Fast pressure monotonically decreases with router entropy")
    print("  - Mid pressure monotonically decreases with margin")
    print("  - Slow pressure monotonically decreases with constraint penalty")
    print("  - All pressures isolated (no cross-clock leakage)")
    print("  - No sign flips with constant inputs")
    print("  - All outputs bounded to [-1, 1]")
    print()


if __name__ == "__main__":
    run_all_tests()
