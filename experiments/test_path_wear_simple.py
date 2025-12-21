#!/usr/bin/env python3
"""
Simplified Path Wear Experiment with Telemetry

Tests: Does ChronoMoE show perturbation from repeated passes?

Simplified approach:
- For ChronoMoE: Track PC dynamics and T̄ drift during wear
- For both: Measure output logit changes (B after wear vs B virgin)

Expected: ChronoMoE shows measurable state changes, vanilla doesn't.
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


def measure_logit_divergence(logits_1: torch.Tensor, logits_2: torch.Tensor) -> float:
    """Measure divergence between two logit distributions."""
    # Average over batch and sequence
    logits_1_mean = logits_1.detach().mean(dim=(0, 1))
    logits_2_mean = logits_2.detach().mean(dim=(0, 1))

    # Cosine distance
    cos_sim = torch.nn.functional.cosine_similarity(
        logits_1_mean.unsqueeze(0),
        logits_2_mean.unsqueeze(0)
    )

    return 1.0 - cos_sim.item()


def test_path_wear(
    model,
    model_type: str,
    input_A: torch.Tensor,
    input_B: torch.Tensor,
    n_repetitions: int = 100,
    telemetry_recorder=None,
):
    """Test if repeated A affects B."""

    print(f"\n{'='*70}")
    print(f"PATH WEAR TEST: {model_type.upper()}")
    print(f"{'='*70}\n")

    telemetry_snapshots = []

    # Phase 1: Baseline B (virgin)
    print("Phase 1: Baseline B (virgin)")
    reset_controller(model)

    with torch.no_grad():
        logits_B_virgin, chrono_state_virgin, _ = model(input_B, update_chronovisor=True)

        if model_type == 'chronovisor' and telemetry_recorder:
            snap = telemetry_recorder.capture_snapshot(chrono_state_virgin, "baseline_B")
            telemetry_snapshots.append(snap)
            print(f"  PC1={snap.pc1:.4f}, PC2={snap.pc2:.4f}, entropy={snap.routing_entropy:.4f}")

    # Phase 2: Wear the path with A
    print(f"\nPhase 2: Wear path (A × {n_repetitions})")
    reset_controller(model)

    if model_type == 'chronovisor' and model.model.controller:
        T_bar_initial = model.model.controller.lenses[0].T_bar.copy()

    for i in range(n_repetitions):
        with torch.no_grad():
            _, chrono_state, _ = model(input_A, update_chronovisor=True)

            if model_type == 'chronovisor' and telemetry_recorder and (i + 1) % 25 == 0:
                snap = telemetry_recorder.capture_snapshot(chrono_state, f"wear_{i+1}")
                telemetry_snapshots.append(snap)
                print(f"  Step {i+1}: PC1={snap.pc1:.4f}, PC2={snap.pc2:.4f}, "
                      f"entropy={snap.routing_entropy:.4f}")

    if model_type == 'chronovisor' and model.model.controller:
        T_bar_after = model.model.controller.lenses[0].T_bar.copy()
        T_bar_drift = np.mean(np.abs(T_bar_after - T_bar_initial))
        print(f"  T̄ drift: {T_bar_drift:.6f}")
    else:
        T_bar_drift = 0.0

    # Phase 3: Test B after wear (no reset)
    print("\nPhase 3: B after wear (no reset)")

    with torch.no_grad():
        logits_B_after, chrono_state_after, _ = model(input_B, update_chronovisor=True)

        if model_type == 'chronovisor' and telemetry_recorder:
            snap = telemetry_recorder.capture_snapshot(chrono_state_after, "B_after_wear")
            telemetry_snapshots.append(snap)
            print(f"  PC1={snap.pc1:.4f}, PC2={snap.pc2:.4f}, entropy={snap.routing_entropy:.4f}")

    # Phase 4: Control - B after reset (ChronoMoE only)
    if model_type == 'chronovisor':
        print("\nPhase 4: B after reset")
        reset_controller(model)

        with torch.no_grad():
            logits_B_reset, chrono_state_reset, _ = model(input_B, update_chronovisor=True)

            if telemetry_recorder:
                snap = telemetry_recorder.capture_snapshot(chrono_state_reset, "B_after_reset")
                telemetry_snapshots.append(snap)
                print(f"  PC1={snap.pc1:.4f}, PC2={snap.pc2:.4f}, entropy={snap.routing_entropy:.4f}")
    else:
        logits_B_reset = logits_B_virgin

    # Analysis: Measure logit divergence
    print(f"\n{'='*70}")
    print(f"ANALYSIS: {model_type.upper()}")
    print(f"{'='*70}\n")

    div_virgin_vs_after = measure_logit_divergence(logits_B_virgin, logits_B_after)
    div_virgin_vs_reset = measure_logit_divergence(logits_B_virgin, logits_B_reset)

    print("Logit divergence (cosine distance):")
    print(f"  B_virgin vs B_after_wear: {div_virgin_vs_after:.6f}")
    print(f"  B_virgin vs B_after_reset: {div_virgin_vs_reset:.6f}")
    print()

    wear_effect = div_virgin_vs_after
    reset_eliminates = div_virgin_vs_reset < 0.01

    print(f"Wear effect: {wear_effect:.6f}")
    print(f"  (Higher = B changed after A repetitions)")
    print()

    if model_type == 'chronovisor':
        print(f"Reset eliminates effect: {reset_eliminates}")
        print()

    return {
        'model_type': model_type,
        'wear_effect': wear_effect,
        'T_bar_drift': T_bar_drift,
        'div_virgin_vs_after': div_virgin_vs_after,
        'div_virgin_vs_reset': div_virgin_vs_reset,
        'reset_eliminates': reset_eliminates,
        'telemetry_snapshots': telemetry_snapshots,
    }


def main():
    print("="*70)
    print("PATH WEAR EXPERIMENT (SIMPLIFIED)")
    print("="*70)
    print()
    print("Question: Does ChronoMoE show state perturbation from repeated passes?")
    print()
    print("Measurement: Logit divergence (B after wear vs B virgin)")
    print()

    # Config
    vocab_size = 1000
    config_shared = {
        'vocab_size': vocab_size,
        'hidden_dim': 256,
        'intermediate_dim': 1024,
        'num_layers': 2,
        'num_shared_experts': 2,
        'num_routed_experts': 64,
        'num_experts_per_token': 6,
    }

    # Create inputs
    input_A, input_B = create_paired_inputs(vocab_size, length=50)
    print(f"Inputs: A={input_A.shape}, B={input_B.shape}\n")

    # Test Vanilla
    print("="*70)
    print("VANILLA (Chronovisor Disabled)")
    print("="*70)

    config_vanilla = DeepSeekConfig(**config_shared, enable_chronovisor=False)
    torch.manual_seed(42)
    model_vanilla = ChronovisorDeepSeekForCausalLM(config_vanilla)
    model_vanilla.eval()

    results_vanilla = test_path_wear(
        model=model_vanilla,
        model_type='vanilla',
        input_A=input_A,
        input_B=input_B,
        n_repetitions=100,
        telemetry_recorder=None,
    )

    # Test ChronoMoE
    print("\n" + "="*70)
    print("CHRONOMOE (Chronovisor Enabled)")
    print("="*70)

    config_chrono = DeepSeekConfig(**config_shared, enable_chronovisor=True)
    torch.manual_seed(42)
    model_chrono = ChronovisorDeepSeekForCausalLM(config_chrono)
    model_chrono.eval()

    observer = ChronoMoEObserver()
    telemetry = TelemetryRecorder(
        session_id="path_wear_chronomoe",
        observer=observer,
        stillness_thresholds=StillnessThresholds(fast=0.01, medium=0.02, slow=0.005),
    )

    results_chrono = test_path_wear(
        model=model_chrono,
        model_type='chronovisor',
        input_A=input_A,
        input_B=input_B,
        n_repetitions=100,
        telemetry_recorder=telemetry,
    )

    # Comparative Analysis
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS")
    print("="*70 + "\n")

    delta_wear = results_chrono['wear_effect'] - results_vanilla['wear_effect']

    print("Wear Effect (Logit Divergence):")
    print(f"  Vanilla:   {results_vanilla['wear_effect']:.6f}")
    print(f"  ChronoMoE: {results_chrono['wear_effect']:.6f}")
    print(f"  Δ (Chrono - Vanilla): {delta_wear:+.6f}")
    print()

    print("Internal State Changes:")
    print(f"  Vanilla T̄ drift:   {results_vanilla['T_bar_drift']:.6f} (N/A)")
    print(f"  ChronoMoE T̄ drift: {results_chrono['T_bar_drift']:.6f}")
    print()

    print("Reset Control (ChronoMoE):")
    print(f"  Reset eliminates effect: {results_chrono['reset_eliminates']}")
    print()

    # Interpretation
    print("="*70)
    print("INTERPRETATION")
    print("="*70 + "\n")

    differential = abs(delta_wear) > 0.001
    chrono_shows_wear = results_chrono['wear_effect'] > 0.01
    vanilla_shows_wear = results_vanilla['wear_effect'] > 0.01

    if differential and chrono_shows_wear and not vanilla_shows_wear:
        print("✓ DIFFERENTIAL EFFECT DETECTED")
        print()
        print("ChronoMoE shows state perturbation from repeated passes.")
        print("Vanilla shows no perturbation.")
        print()
        print("T̄ creates inference-time path dependence.")
    elif chrono_shows_wear and results_chrono['T_bar_drift'] > 0.01:
        print("~ WEAK SIGNAL DETECTED")
        print()
        print("ChronoMoE shows T̄ drift, but output effect is small.")
        print(f"Wear effect: {results_chrono['wear_effect']:.6f}")
        print()
        print("T̄ is changing, but not strongly affecting outputs yet.")
    else:
        print("○ NO DIFFERENTIAL DETECTED")
        print()
        print("Both models behave similarly.")
        print("T̄ mechanism insufficient under these conditions.")

    print()

    # Save results
    output_dir = Path(__file__).parent.parent / "takens_data"
    output_dir.mkdir(exist_ok=True)

    np.savez(
        output_dir / "path_wear_simple_results.npz",
        vanilla_wear=results_vanilla['wear_effect'],
        chrono_wear=results_chrono['wear_effect'],
        delta_wear=delta_wear,
        chrono_T_bar_drift=results_chrono['T_bar_drift'],
        vanilla_div=results_vanilla['div_virgin_vs_after'],
        chrono_div=results_chrono['div_virgin_vs_after'],
    )

    if results_chrono['telemetry_snapshots']:
        import json
        snapshots = [s.to_dict() for s in results_chrono['telemetry_snapshots']]
        with open(output_dir / "path_wear_telemetry.json", 'w') as f:
            json.dump(snapshots, f, indent=2)

    print(f"Results saved to: {output_dir}")
    print()


if __name__ == '__main__':
    main()
