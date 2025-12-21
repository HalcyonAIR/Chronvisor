#!/usr/bin/env python3
"""
Path Wear with Structured Routing (Pragmatic Approach)

Instead of pretrained models, we create structure synthetically:
- Use ChronoMoE-enabled model (routing accessible)
- Create structured inputs (repeated patterns, not random)
- Verify routing has structure (entropy < random baseline)
- Then test: vanilla vs ChronoMoE differential

This tests the hypothesis in a controlled, observable regime.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path
import json

from chronomoe.deepseek_core import DeepSeekConfig
from chronomoe.chronovisor_deepseek_bridge import ChronovisorDeepSeekForCausalLM

from routing_metrics import (
    extract_routing_distribution,
    compute_routing_metrics_suite,
    entropy,
)


def create_structured_inputs(vocab_size: int, length: int = 50):
    """
    Create structured inputs A and B (not random).

    Strategy: Use repeated patterns that should trigger
    consistent routing (if the model has any structure).
    """
    # Pattern A: Repeating sequence [0, 1, 2, 3, 4]
    pattern_A = torch.tensor([i % 5 for i in range(length)]).unsqueeze(0)

    # Pattern B: Similar but shifted [1, 2, 3, 4, 5]
    pattern_B = torch.tensor([(i % 5) + 1 for i in range(length)]).unsqueeze(0)

    return pattern_A, pattern_B


def reset_controller(model):
    """Reset ChronoMoE controller."""
    if hasattr(model, 'model') and hasattr(model.model, 'controller'):
        controller = model.model.controller
        if controller is not None:
            for lens in controller.lenses.values():
                lens.T_bar = np.ones(lens.num_experts)
                lens.expert_usage_ema = np.ones(lens.num_experts) / lens.num_experts


def test_path_wear_structured(
    model,
    input_A: torch.Tensor,
    input_B: torch.Tensor,
    chronovisor_enabled: bool,
    n_repetitions: int = 100,
):
    """
    Test path wear with structured inputs.

    Args:
        chronovisor_enabled: If True, enable ChronoMoE adaptation
    """

    mode_label = "ChronoMoE" if chronovisor_enabled else "Vanilla"

    print(f"\n{'='*70}")
    print(f"PATH WEAR TEST: {mode_label}")
    print(f"{'='*70}\n")

    # Phase 1: Baseline B
    reset_controller(model)

    with torch.no_grad():
        _, chrono_state_virgin, _ = model(input_B, update_chronovisor=chronovisor_enabled)
        routing_B_virgin = extract_routing_distribution(chrono_state_virgin, layer_idx=0)

        if routing_B_virgin is None:
            print(f"ERROR: Could not extract routing")
            return None

        entropy_B_virgin = entropy(routing_B_virgin)
        print(f"Phase 1: B (virgin)")
        print(f"  Entropy: {entropy_B_virgin:.4f}")

    # Phase 2: Establish A's pattern
    reset_controller(model)

    with torch.no_grad():
        _, chrono_state_A, _ = model(input_A, update_chronovisor=chronovisor_enabled)
        routing_A = extract_routing_distribution(chrono_state_A, layer_idx=0)

        entropy_A = entropy(routing_A)
        print(f"\nPhase 2: A (reference)")
        print(f"  Entropy: {entropy_A:.4f}")

    # Phase 3: Wear the path
    print(f"\nPhase 3: Wear path (A × {n_repetitions})")
    reset_controller(model)

    if chronovisor_enabled and model.model.controller:
        T_bar_initial = model.model.controller.lenses[0].T_bar.copy()

    for i in range(n_repetitions):
        with torch.no_grad():
            _ = model(input_A, update_chronovisor=chronovisor_enabled)

    if chronovisor_enabled and model.model.controller:
        T_bar_after = model.model.controller.lenses[0].T_bar.copy()
        T_bar_drift = np.mean(np.abs(T_bar_after - T_bar_initial))
        print(f"  T̄ drift: {T_bar_drift:.6f}")
    else:
        T_bar_drift = 0.0

    # Phase 4: Test B after wear
    print(f"\nPhase 4: B after wear")

    with torch.no_grad():
        _, chrono_state_after, _ = model(input_B, update_chronovisor=chronovisor_enabled)
        routing_B_after = extract_routing_distribution(chrono_state_after, layer_idx=0)

        entropy_B_after = entropy(routing_B_after)
        print(f"  Entropy: {entropy_B_after:.4f}")

    # Compute metrics
    metrics = compute_routing_metrics_suite(
        routing_B_virgin=routing_B_virgin,
        routing_B_after=routing_B_after,
        routing_A=routing_A,
        top_k=6,
    )

    wear_detected = (
        metrics['delta_kl_to_A'] > 0.01 or
        metrics['delta_cos_to_A'] > 0.001 or
        metrics['delta_jaccard_to_A'] > 0.05
    )

    print(f"\nMetrics:")
    print(f"  ΔKL to A:   {metrics['delta_kl_to_A']:+.6f}")
    print(f"  ΔCos to A:  {metrics['delta_cos_to_A']:+.6f}")
    print(f"  ΔJac to A:  {metrics['delta_jaccard_to_A']:+.6f}")
    print(f"  Wear: {'✓ YES' if wear_detected else '○ NO'}")

    return {
        'mode': mode_label,
        'chronovisor_enabled': chronovisor_enabled,
        'T_bar_drift': T_bar_drift,
        'entropy_B_virgin': entropy_B_virgin,
        'entropy_A': entropy_A,
        'metrics': metrics,
        'wear_detected': wear_detected,
    }


def main():
    print("="*70)
    print("PATH WEAR WITH STRUCTURED ROUTING")
    print("="*70)
    print()
    print("Pragmatic approach:")
    print("  1. Create structured inputs (repeated patterns)")
    print("  2. Verify routing has structure (entropy < random)")
    print("  3. Test vanilla (ChronoMoE disabled) as control")
    print("  4. Test ChronoMoE (enabled) for differential")
    print()

    # Config
    vocab_size = 1000
    config = DeepSeekConfig(
        vocab_size=vocab_size,
        hidden_dim=256,
        intermediate_dim=1024,
        num_layers=2,
        num_shared_experts=2,
        num_routed_experts=64,
        num_experts_per_token=6,
        enable_chronovisor=True,  # Will toggle via update_chronovisor parameter
    )

    # Create structured inputs
    input_A, input_B = create_structured_inputs(vocab_size, length=50)

    print(f"Inputs created:")
    print(f"  A: Pattern [0,1,2,3,4] repeated")
    print(f"  B: Pattern [1,2,3,4,5] repeated")
    print()

    # Create model
    torch.manual_seed(42)
    model = ChronovisorDeepSeekForCausalLM(config)
    model.eval()

    # Verify structure exists (baseline check)
    print("="*70)
    print("BASELINE: Verify Structure Exists")
    print("="*70)
    print()

    with torch.no_grad():
        _, chrono_state, _ = model(input_A, update_chronovisor=False)
        routing = extract_routing_distribution(chrono_state, layer_idx=0)

        if routing is not None:
            routing_entropy = entropy(routing)
            max_entropy = np.log(64)
            entropy_fraction = routing_entropy / max_entropy

            print(f"Routing entropy: {routing_entropy:.4f} / {max_entropy:.4f}")
            print(f"Fraction: {entropy_fraction:.2%}")
            print()

            if entropy_fraction < 0.95:
                print("✓ Structure exists (entropy < 95% of max)")
            else:
                print("⚠ Still near-uniform (entropy > 95% of max)")
                print("  Structured inputs didn't help - model inherently flat")
                print()
                print("This means even with repeated patterns, untrained model")
                print("produces uniform routing. Need actual training or pretrained model.")
                print()
                print("Proceeding anyway to demonstrate protocol...")

            print()

    # Test 1: Vanilla (control)
    results_vanilla = test_path_wear_structured(
        model=model,
        input_A=input_A,
        input_B=input_B,
        chronovisor_enabled=False,
        n_repetitions=100,
    )

    # Test 2: ChronoMoE (treatment)
    results_chrono = test_path_wear_structured(
        model=model,
        input_A=input_A,
        input_B=input_B,
        chronovisor_enabled=True,
        n_repetitions=100,
    )

    # Comparative analysis
    print(f"\n{'='*70}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*70}\n")

    if results_vanilla and results_chrono:
        delta_kl = results_chrono['metrics']['delta_kl_to_A'] - results_vanilla['metrics']['delta_kl_to_A']
        delta_cos = results_chrono['metrics']['delta_cos_to_A'] - results_vanilla['metrics']['delta_cos_to_A']
        delta_jac = results_chrono['metrics']['delta_jaccard_to_A'] - results_vanilla['metrics']['delta_jaccard_to_A']

        print(f"Differential (ChronoMoE - Vanilla):")
        print(f"  ΔΔ KL:      {delta_kl:+.6f}")
        print(f"  ΔΔ Cosine:  {delta_cos:+.6f}")
        print(f"  ΔΔ Jaccard: {delta_jac:+.6f}")
        print()

        differential_detected = (
            abs(delta_kl) > 0.01 or
            abs(delta_cos) > 0.001 or
            abs(delta_jac) > 0.05
        )

        if differential_detected:
            print("✓ DIFFERENTIAL DETECTED")
            print("  ChronoMoE creates measurably different routing bias than vanilla")
        else:
            print("○ NO DIFFERENTIAL")
            print("  ChronoMoE and vanilla behave identically")

        print()
        print(f"T̄ drift (ChronoMoE): {results_chrono['T_bar_drift']:.6f}")
        print()

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()

    if results_vanilla['entropy_B_virgin'] > 0.95 * np.log(64):
        print("⚠ CONFOUND: Routing still near-uniform even with structured inputs")
        print()
        print("Conclusion:")
        print("  Untrained models produce flat routing regardless of input structure.")
        print("  Hypothesis test requires trained model with learned routing patterns.")
        print()
        print("Next steps:")
        print("  1. Train small model on toy task")
        print("  2. OR use pretrained MoE from HuggingFace")
        print("  3. OR accept this as extended boundary documentation")
    else:
        print("Protocol validated - proceed to next experiments with trained models")

    # Save results
    output_dir = Path(__file__).parent.parent / "takens_data"
    output_dir.mkdir(exist_ok=True)

    results_json = {
        'vanilla': {k: float(v) if isinstance(v, np.floating) else v
                    for k, v in results_vanilla.items() if k != 'metrics'},
        'chrono': {k: float(v) if isinstance(v, np.floating) else v
                   for k, v in results_chrono.items() if k != 'metrics'},
    }

    with open(output_dir / "path_wear_structured_results.json", 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'path_wear_structured_results.json'}")
    print()


if __name__ == '__main__':
    main()
