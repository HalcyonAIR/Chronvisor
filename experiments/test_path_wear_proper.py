#!/usr/bin/env python3
"""
Path Wear Experiment with Proper Routing Metrics

Fixing the microscope: measuring routing landscape directly,
not outputs or T̄.

Metric suite:
1. ΔKL-to-A: Movement toward A's routing pattern
2. Cosine shift: Correlation change (stable under uniform)
3. Top-k Jaccard: Expert coalition overlap
4. Entropy: Concentration tracking

This is a measurement upgrade, not a new hypothesis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path

from chronomoe.deepseek_core import DeepSeekConfig
from chronomoe.chronovisor_deepseek_bridge import ChronovisorDeepSeekForCausalLM

from chronomoe_telemetry import (
    ChronoMoEObserver,
    TelemetryRecorder,
    StillnessThresholds,
)

from routing_metrics import (
    extract_routing_distribution,
    compute_routing_metrics_suite,
    interpret_routing_metrics,
    print_routing_comparison,
)


def create_paired_inputs(vocab_size: int, length: int = 50):
    """Create similar but distinct inputs A and B."""
    base = torch.randint(0, vocab_size, (length,))
    noise_A = torch.randint(-5, 6, (length,))
    noise_B = torch.randint(-5, 6, (length,))

    input_A = torch.clamp(base + noise_A, 0, vocab_size - 1).unsqueeze(0)
    input_B = torch.clamp(base + noise_B, 0, vocab_size - 1).unsqueeze(0)

    return input_A, input_B


def reset_controller(model):
    """Reset ChronoMoE controller if it exists."""
    if hasattr(model, 'model') and hasattr(model.model, 'controller'):
        controller = model.model.controller
        if controller is not None:
            for lens in controller.lenses.values():
                lens.T_bar = np.ones(lens.num_experts)
                lens.expert_usage_ema = np.ones(lens.num_experts) / lens.num_experts


def test_path_wear_chronomoe(
    model,
    input_A: torch.Tensor,
    input_B: torch.Tensor,
    n_repetitions: int = 100,
    telemetry_recorder=None,
):
    """
    Test path wear with proper routing metrics.

    Only works for ChronoMoE (need chrono_state for routing extraction).
    """

    print(f"\n{'='*70}")
    print("PATH WEAR TEST: CHRONOMOE")
    print(f"{'='*70}\n")

    telemetry_snapshots = []

    # Phase 1: Baseline B (virgin)
    print("Phase 1: Baseline B (virgin)")
    reset_controller(model)

    with torch.no_grad():
        _, chrono_state_virgin, _ = model(input_B, update_chronovisor=True)

        routing_B_virgin = extract_routing_distribution(chrono_state_virgin, layer_idx=0)

        if routing_B_virgin is None:
            print("ERROR: Could not extract routing")
            return None

        if telemetry_recorder:
            snap = telemetry_recorder.capture_snapshot(chrono_state_virgin, "baseline_B")
            telemetry_snapshots.append(snap)
            print(f"  PC1={snap.pc1:.4f}, PC2={snap.pc2:.4f}, entropy={snap.routing_entropy:.4f}")

    # Phase 2: Establish A's pattern
    print("\nPhase 2: Establish A's pattern")
    reset_controller(model)

    with torch.no_grad():
        _, chrono_state_A, _ = model(input_A, update_chronovisor=True)

        routing_A = extract_routing_distribution(chrono_state_A, layer_idx=0)

        if routing_A is None:
            print("ERROR: Could not extract routing")
            return None

        if telemetry_recorder:
            snap = telemetry_recorder.capture_snapshot(chrono_state_A, "establish_A")
            telemetry_snapshots.append(snap)
            print(f"  PC1={snap.pc1:.4f}, PC2={snap.pc2:.4f}, entropy={snap.routing_entropy:.4f}")

    # Phase 3: Wear the path
    print(f"\nPhase 3: Wear path (A × {n_repetitions})")
    reset_controller(model)

    if model.model.controller:
        T_bar_initial = model.model.controller.lenses[0].T_bar.copy()

    for i in range(n_repetitions):
        with torch.no_grad():
            _, chrono_state, _ = model(input_A, update_chronovisor=True)

            if telemetry_recorder and (i + 1) % 25 == 0:
                snap = telemetry_recorder.capture_snapshot(chrono_state, f"wear_{i+1}")
                telemetry_snapshots.append(snap)
                print(f"  Step {i+1}: PC1={snap.pc1:.4f}, PC2={snap.pc2:.4f}, "
                      f"entropy={snap.routing_entropy:.4f}")

    if model.model.controller:
        T_bar_after = model.model.controller.lenses[0].T_bar.copy()
        T_bar_drift = np.mean(np.abs(T_bar_after - T_bar_initial))
        print(f"  T̄ drift: {T_bar_drift:.6f}")
    else:
        T_bar_drift = 0.0

    # Phase 4: Test B after wear (no reset)
    print("\nPhase 4: B after wear (no reset)")

    with torch.no_grad():
        _, chrono_state_after, _ = model(input_B, update_chronovisor=True)

        routing_B_after = extract_routing_distribution(chrono_state_after, layer_idx=0)

        if telemetry_recorder:
            snap = telemetry_recorder.capture_snapshot(chrono_state_after, "B_after_wear")
            telemetry_snapshots.append(snap)
            print(f"  PC1={snap.pc1:.4f}, PC2={snap.pc2:.4f}, entropy={snap.routing_entropy:.4f}")

    # Phase 5: Control - B after reset
    print("\nPhase 5: B after reset (control)")
    reset_controller(model)

    with torch.no_grad():
        _, chrono_state_reset, _ = model(input_B, update_chronovisor=True)

        routing_B_reset = extract_routing_distribution(chrono_state_reset, layer_idx=0)

        if telemetry_recorder:
            snap = telemetry_recorder.capture_snapshot(chrono_state_reset, "B_after_reset")
            telemetry_snapshots.append(snap)
            print(f"  PC1={snap.pc1:.4f}, PC2={snap.pc2:.4f}, entropy={snap.routing_entropy:.4f}")

    # Compute routing metrics suite
    print(f"\n{'='*70}")
    print("ROUTING METRICS (NO RESET)")
    print(f"{'='*70}\n")

    metrics_no_reset = compute_routing_metrics_suite(
        routing_B_virgin=routing_B_virgin,
        routing_B_after=routing_B_after,
        routing_A=routing_A,
        top_k=6,
    )

    print(interpret_routing_metrics(metrics_no_reset))

    # Compute reset control metrics
    print(f"\n{'='*70}")
    print("ROUTING METRICS (AFTER RESET)")
    print(f"{'='*70}\n")

    metrics_after_reset = compute_routing_metrics_suite(
        routing_B_virgin=routing_B_virgin,
        routing_B_after=routing_B_reset,
        routing_A=routing_A,
        top_k=6,
    )

    print(interpret_routing_metrics(metrics_after_reset))

    # Detailed routing comparison
    print_routing_comparison(
        routing_B_virgin=routing_B_virgin,
        routing_B_after=routing_B_after,
        routing_A=routing_A,
        top_k=6,
    )

    # Determine if reset eliminates effect
    wear_no_reset = (
        metrics_no_reset['delta_kl_to_A'] > 0.01 or
        metrics_no_reset['delta_cos_to_A'] > 0.001 or
        metrics_no_reset['delta_jaccard_to_A'] > 0.05
    )

    wear_after_reset = (
        metrics_after_reset['delta_kl_to_A'] > 0.01 or
        metrics_after_reset['delta_cos_to_A'] > 0.001 or
        metrics_after_reset['delta_jaccard_to_A'] > 0.05
    )

    reset_eliminates = wear_no_reset and not wear_after_reset

    return {
        'T_bar_drift': T_bar_drift,
        'metrics_no_reset': metrics_no_reset,
        'metrics_after_reset': metrics_after_reset,
        'wear_detected': wear_no_reset,
        'reset_eliminates': reset_eliminates,
        'telemetry_snapshots': telemetry_snapshots,
        'routing_B_virgin': routing_B_virgin,
        'routing_B_after': routing_B_after,
        'routing_A': routing_A,
    }


def main():
    print("="*70)
    print("PATH WEAR EXPERIMENT (PROPER ROUTING METRICS)")
    print("="*70)
    print()
    print("Fixing the microscope:")
    print("  - Direct routing measurements (not outputs)")
    print("  - Metric suite: ΔKL, Δcosine, ΔJaccard, entropy")
    print("  - ChronoMoE only (need chrono_state for routing)")
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
        enable_chronovisor=True,
    )

    # Create inputs
    input_A, input_B = create_paired_inputs(vocab_size, length=50)
    print(f"Inputs: A={input_A.shape}, B={input_B.shape}\n")

    # Create model
    print("Creating ChronoMoE model...")
    torch.manual_seed(42)
    model = ChronovisorDeepSeekForCausalLM(config)
    model.eval()
    print("Model created\n")

    # Create telemetry
    observer = ChronoMoEObserver()
    telemetry = TelemetryRecorder(
        session_id="path_wear_proper",
        observer=observer,
        stillness_thresholds=StillnessThresholds(fast=0.01, medium=0.02, slow=0.005),
    )

    # Run experiment
    results = test_path_wear_chronomoe(
        model=model,
        input_A=input_A,
        input_B=input_B,
        n_repetitions=100,
        telemetry_recorder=telemetry,
    )

    if results is None:
        print("ERROR: Experiment failed")
        return

    # Final interpretation
    print(f"\n{'='*70}")
    print("FINAL INTERPRETATION")
    print(f"{'='*70}\n")

    metrics = results['metrics_no_reset']

    print(f"T̄ drift: {results['T_bar_drift']:.6f}")
    print(f"Wear detected: {results['wear_detected']}")
    print(f"Reset eliminates: {results['reset_eliminates']}")
    print()

    if results['wear_detected'] and results['reset_eliminates']:
        print("✓ PATH WEAR CONFIRMED")
        print()
        print("Routing moved toward A, and reset eliminated the effect.")
        print("T̄ creates genuine landscape deformation.")
    elif results['wear_detected'] and not results['reset_eliminates']:
        print("⚠ ARTIFACT DETECTED")
        print()
        print("Routing changed but reset didn't eliminate it.")
        print("Effect not due to T̄ persistence.")
    elif results['T_bar_drift'] > 0.01 and not results['wear_detected']:
        print("~ WEAK COUPLING")
        print()
        print("T̄ drifted but routing unchanged.")
        print("Mechanism present but insufficient under current conditions.")
    else:
        print("○ NO WEAR DETECTED")
        print()
        print("Neither T̄ drift nor routing change detected.")

    # Near-uniform warning
    max_entropy = np.log(64)
    if metrics['entropy_B_virgin'] > 0.9 * max_entropy:
        print()
        print("⚠ CONFOUND: Near-uniform routing")
        print(f"  Entropy: {metrics['entropy_B_virgin']:.4f} / {max_entropy:.4f}")
        print("  Small biases won't register in flat landscape.")
        print()
        print("Next step: Entropy-controlled regime")
        print("  - Reduce top-k (force concentration)")
        print("  - Lower router temperature")
        print("  - Inject prior bias")

    print()

    # Save results
    output_dir = Path(__file__).parent.parent / "takens_data"
    output_dir.mkdir(exist_ok=True)

    # Save metrics
    np.savez(
        output_dir / "path_wear_proper_results.npz",
        T_bar_drift=results['T_bar_drift'],
        wear_detected=results['wear_detected'],
        reset_eliminates=results['reset_eliminates'],
        routing_B_virgin=results['routing_B_virgin'],
        routing_B_after=results['routing_B_after'],
        routing_A=results['routing_A'],
        **{f'metric_{k}': v for k, v in metrics.items() if isinstance(v, (int, float))},
    )

    # Save telemetry
    if results['telemetry_snapshots']:
        import json
        snapshots = [s.to_dict() for s in results['telemetry_snapshots']]
        with open(output_dir / "path_wear_proper_telemetry.json", 'w') as f:
            json.dump(snapshots, f, indent=2)

    print(f"Results saved to: {output_dir}")
    print()


if __name__ == '__main__':
    main()
