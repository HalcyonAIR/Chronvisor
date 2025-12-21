#!/usr/bin/env python3
"""
Path Wear Experiment with Telemetry Integration

Tests: Does ChronoMoE create differential path wear compared to vanilla?

User's expectation: "The ChronoMoE should show a perturbation at the least
as its leaning towards the last passes each new pass."

Key question: Does repeated inference on A create persistent bias that
affects routing for B in ChronoMoE but not in vanilla?

With telemetry: We can observe PC dynamics, stillness signals, and routing
stability throughout the wear process.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from chronomoe.deepseek_core import DeepSeekConfig
from chronomoe.chronovisor_deepseek_bridge import ChronovisorDeepSeekForCausalLM

from chronomoe_telemetry import (
    ChronoMoEObserver,
    TelemetryRecorder,
    StillnessThresholds,
    TelemetrySnapshot,
)


def extract_routing_distribution(output_tuple, model_type: str) -> np.ndarray:
    """Extract routing distribution from model output."""
    # Both model types return (logits, chrono_state, router_outputs)
    # For vanilla, chrono_state is None
    _, chrono_state, router_outputs = output_tuple

    if model_type == 'chronovisor':
        # Use chrono_state expert usage
        if chrono_state is None or not chrono_state.expert_usage:
            return None
        usage = chrono_state.expert_usage.get(0, None)
        if usage is None:
            return None
        return usage / (usage.sum() + 1e-10)

    elif model_type == 'vanilla':
        # Use router_outputs directly (no chrono_state)
        if router_outputs is None or len(router_outputs) == 0:
            return None

        # Get routing weights from first MoE layer
        first_layer = router_outputs[0]

        # DeepSeek router outputs include routing_weights
        if hasattr(first_layer, 'routing_weights'):
            weights = first_layer.routing_weights
        elif isinstance(first_layer, dict) and 'routing_weights' in first_layer:
            weights = first_layer['routing_weights']
        else:
            # Try to extract from router_logits
            if hasattr(first_layer, 'router_logits'):
                logits = first_layer.router_logits
                if isinstance(logits, torch.Tensor):
                    weights = torch.softmax(logits, dim=-1)
                else:
                    return None
            else:
                return None

        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()

        # Average over batch and sequence dimensions
        while weights.ndim > 1:
            weights = weights.mean(axis=0)

        return weights / (weights.sum() + 1e-10)

    return None


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence KL(P || Q)."""
    p = p + 1e-10
    q = q + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q))


def reset_controller(model):
    """Reset ChronoMoE controller state."""
    if hasattr(model, 'model') and hasattr(model.model, 'controller'):
        controller = model.model.controller
        if controller is not None:
            for lens in controller.lenses.values():
                lens.T_bar = np.ones(lens.num_experts)
                lens.expert_usage_ema = np.ones(lens.num_experts) / lens.num_experts


def create_paired_inputs(vocab_size: int, length: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create similar but distinct inputs A and B."""
    base = torch.randint(0, vocab_size, (length,))
    noise_A = torch.randint(-5, 6, (length,))
    noise_B = torch.randint(-5, 6, (length,))

    input_A = torch.clamp(base + noise_A, 0, vocab_size - 1).unsqueeze(0)
    input_B = torch.clamp(base + noise_B, 0, vocab_size - 1).unsqueeze(0)

    return input_A, input_B


def test_path_wear_with_telemetry(
    model,
    model_type: str,
    input_A: torch.Tensor,
    input_B: torch.Tensor,
    n_repetitions: int = 100,
    telemetry_recorder: TelemetryRecorder = None,
) -> Dict:
    """
    Test path wear with telemetry observation.

    Returns measurements + telemetry snapshots.
    """
    print(f"\n{'='*70}")
    print(f"PATH WEAR TEST: {model_type.upper()}")
    print(f"{'='*70}\n")

    telemetry_snapshots = []

    # Phase 1: Baseline B
    print("Phase 1: Baseline (B without A influence)")
    if model_type == 'chronovisor':
        reset_controller(model)

    with torch.no_grad():
        if model_type == 'chronovisor':
            output = model(input_B, update_chronovisor=True)
            if telemetry_recorder:
                _, chrono_state, _ = output
                snap = telemetry_recorder.capture_snapshot(chrono_state, "baseline_B")
                telemetry_snapshots.append(snap)
        else:
            output = model(input_B)

        routing_B_virgin = extract_routing_distribution(output, model_type)

    if routing_B_virgin is None:
        print("ERROR: Could not extract routing")
        return None

    entropy_B_virgin = -np.sum(routing_B_virgin * np.log(routing_B_virgin + 1e-10))
    print(f"  B (virgin) entropy: {entropy_B_virgin:.4f}")

    # Phase 2: Establish A's pattern
    print("\nPhase 2: Establish A's pattern")
    if model_type == 'chronovisor':
        reset_controller(model)

    with torch.no_grad():
        if model_type == 'chronovisor':
            output = model(input_A, update_chronovisor=True)
            if telemetry_recorder:
                _, chrono_state, _ = output
                snap = telemetry_recorder.capture_snapshot(chrono_state, "establish_A")
                telemetry_snapshots.append(snap)
        else:
            output = model(input_A)

        routing_A = extract_routing_distribution(output, model_type)

    entropy_A = -np.sum(routing_A * np.log(routing_A + 1e-10))
    print(f"  A entropy: {entropy_A:.4f}")

    # Phase 3: Wear the path
    print(f"\nPhase 3: Wear path (A × {n_repetitions})")
    if model_type == 'chronovisor':
        reset_controller(model)
        if model.model.controller:
            T_bar_initial = model.model.controller.lenses[0].T_bar.copy()
    else:
        T_bar_initial = None

    for i in range(n_repetitions):
        with torch.no_grad():
            if model_type == 'chronovisor':
                output = model(input_A, update_chronovisor=True)

                # Capture telemetry periodically
                if telemetry_recorder and (i + 1) % 25 == 0:
                    _, chrono_state, _ = output
                    snap = telemetry_recorder.capture_snapshot(
                        chrono_state,
                        f"wear_step_{i+1}"
                    )
                    telemetry_snapshots.append(snap)
                    print(f"  Step {i+1}: PC1={snap.pc1:.4f}, PC2={snap.pc2:.4f}, "
                          f"entropy={snap.routing_entropy:.4f}")
            else:
                _ = model(input_A)
                if (i + 1) % 25 == 0:
                    print(f"  Step {i+1}: (vanilla - no telemetry)")

    if model_type == 'chronovisor' and model.model.controller:
        T_bar_after = model.model.controller.lenses[0].T_bar.copy()
        T_bar_drift = np.mean(np.abs(T_bar_after - T_bar_initial))
        print(f"  Total T̄ drift: {T_bar_drift:.6f}")
    else:
        T_bar_drift = 0.0

    # Phase 4: Test B after wear (no reset)
    print("\nPhase 4: Test B after wear (no reset)")

    with torch.no_grad():
        if model_type == 'chronovisor':
            output = model(input_B, update_chronovisor=True)
            if telemetry_recorder:
                _, chrono_state, _ = output
                snap = telemetry_recorder.capture_snapshot(chrono_state, "B_after_wear")
                telemetry_snapshots.append(snap)
        else:
            output = model(input_B)

        routing_B_after_wear = extract_routing_distribution(output, model_type)

    entropy_B_after = -np.sum(routing_B_after_wear * np.log(routing_B_after_wear + 1e-10))
    print(f"  B (after wear) entropy: {entropy_B_after:.4f}")

    # Phase 5: Control - reset (ChronoMoE only)
    if model_type == 'chronovisor':
        print("\nPhase 5: Control - B after reset")
        reset_controller(model)

        with torch.no_grad():
            output = model(input_B, update_chronovisor=True)
            if telemetry_recorder:
                _, chrono_state, _ = output
                snap = telemetry_recorder.capture_snapshot(chrono_state, "B_after_reset")
                telemetry_snapshots.append(snap)

            routing_B_after_reset = extract_routing_distribution(output, model_type)

        entropy_B_reset = -np.sum(routing_B_after_reset * np.log(routing_B_after_reset + 1e-10))
        print(f"  B (after reset) entropy: {entropy_B_reset:.4f}")
    else:
        routing_B_after_reset = routing_B_virgin.copy()
        entropy_B_reset = entropy_B_virgin

    # Analysis
    print(f"\n{'='*70}")
    print(f"ANALYSIS: {model_type.upper()}")
    print(f"{'='*70}\n")

    dist_virgin = kl_divergence(routing_B_virgin, routing_A)
    dist_after_wear = kl_divergence(routing_B_after_wear, routing_A)
    dist_after_reset = kl_divergence(routing_B_after_reset, routing_A)

    print("KL Divergence (B || A):")
    print(f"  Virgin:     {dist_virgin:.6f}")
    print(f"  After wear: {dist_after_wear:.6f}")
    print(f"  After reset: {dist_after_reset:.6f}")
    print()

    wear_effect = dist_virgin - dist_after_wear
    print(f"Wear effect: {wear_effect:+.6f}")
    print(f"  (Positive = B moved toward A)")
    print()

    return {
        'model_type': model_type,
        'wear_effect': wear_effect,
        'dist_virgin': dist_virgin,
        'dist_after_wear': dist_after_wear,
        'dist_after_reset': dist_after_reset,
        'T_bar_drift': T_bar_drift,
        'telemetry_snapshots': telemetry_snapshots,
    }


def main():
    print("="*70)
    print("PATH WEAR EXPERIMENT WITH TELEMETRY")
    print("="*70)
    print()
    print("Question: Does ChronoMoE create differential path wear vs vanilla?")
    print()
    print("Expected: ChronoMoE shows perturbation (bias toward recent passes)")
    print("          Vanilla shows no wear (routing unchanged)")
    print()

    # Shared config
    vocab_size = 1000
    hidden_dim = 256
    intermediate_dim = 1024
    num_layers = 2
    num_shared_experts = 2
    num_routed_experts = 64
    num_experts_per_token = 6

    # Create inputs (same for both models)
    input_A, input_B = create_paired_inputs(vocab_size, length=50)
    print(f"Inputs created: A={input_A.shape}, B={input_B.shape}\n")

    # Test Vanilla
    print("="*70)
    print("VANILLA DEEPSEEK")
    print("="*70)

    config_vanilla = DeepSeekConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_layers=num_layers,
        num_shared_experts=num_shared_experts,
        num_routed_experts=num_routed_experts,
        num_experts_per_token=num_experts_per_token,
        enable_chronovisor=False,
    )

    torch.manual_seed(42)
    model_vanilla = ChronovisorDeepSeekForCausalLM(config_vanilla)
    model_vanilla.eval()

    results_vanilla = test_path_wear_with_telemetry(
        model=model_vanilla,
        model_type='vanilla',
        input_A=input_A,
        input_B=input_B,
        n_repetitions=100,
        telemetry_recorder=None,  # No telemetry for vanilla
    )

    # Test ChronoMoE
    print("\n" + "="*70)
    print("CHRONOMOE-ENHANCED DEEPSEEK")
    print("="*70)

    config_chrono = DeepSeekConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_layers=num_layers,
        num_shared_experts=num_shared_experts,
        num_routed_experts=num_routed_experts,
        num_experts_per_token=num_experts_per_token,
        enable_chronovisor=True,
    )

    torch.manual_seed(42)
    model_chrono = ChronovisorDeepSeekForCausalLM(config_chrono)
    model_chrono.eval()

    # Create telemetry recorder for ChronoMoE
    observer = ChronoMoEObserver()
    telemetry = TelemetryRecorder(
        session_id="path_wear_chronomoe",
        observer=observer,
        stillness_thresholds=StillnessThresholds(fast=0.01, medium=0.02, slow=0.005),
    )

    results_chrono = test_path_wear_with_telemetry(
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
    delta_T_bar = results_chrono['T_bar_drift'] - results_vanilla['T_bar_drift']

    print("Wear Effect:")
    print(f"  Vanilla:   {results_vanilla['wear_effect']:+.6f}")
    print(f"  ChronoMoE: {results_chrono['wear_effect']:+.6f}")
    print(f"  Δ (Chrono - Vanilla): {delta_wear:+.6f}")
    print()

    print("T̄ Drift:")
    print(f"  Vanilla:   {results_vanilla['T_bar_drift']:.6f} (N/A)")
    print(f"  ChronoMoE: {results_chrono['T_bar_drift']:.6f}")
    print()

    # Interpretation
    print("="*70)
    print("INTERPRETATION")
    print("="*70 + "\n")

    differential_detected = abs(delta_wear) > 0.01
    chrono_shows_wear = abs(results_chrono['wear_effect']) > 0.05
    vanilla_shows_wear = abs(results_vanilla['wear_effect']) > 0.05

    if differential_detected and chrono_shows_wear and not vanilla_shows_wear:
        print("✓ DIFFERENTIAL PATH WEAR DETECTED")
        print()
        print("ChronoMoE shows wear, vanilla doesn't.")
        print("T̄ creates inference-time path dependence.")
    elif not differential_detected:
        print("○ NO DIFFERENTIAL DETECTED")
        print()
        print("Both models behave similarly.")
        print("T̄ mechanism insufficient under these conditions.")
    else:
        print("~ MIXED RESULTS")
        print()
        print(f"Vanilla wear: {results_vanilla['wear_effect']:+.6f}")
        print(f"ChronoMoE wear: {results_chrono['wear_effect']:+.6f}")

    print()

    # Save results
    output_dir = Path(__file__).parent.parent / "takens_data"
    output_dir.mkdir(exist_ok=True)

    # Save numerical results
    np.savez(
        output_dir / "path_wear_with_telemetry.npz",
        vanilla_wear=results_vanilla['wear_effect'],
        chrono_wear=results_chrono['wear_effect'],
        delta_wear=delta_wear,
        chrono_T_bar_drift=results_chrono['T_bar_drift'],
    )

    # Save telemetry snapshots
    if results_chrono['telemetry_snapshots']:
        import json
        snapshots_data = [s.to_dict() for s in results_chrono['telemetry_snapshots']]
        with open(output_dir / "path_wear_telemetry_snapshots.json", 'w') as f:
            json.dump(snapshots_data, f, indent=2)

        print(f"Results saved to: {output_dir}")
        print(f"  Numerical: path_wear_with_telemetry.npz")
        print(f"  Telemetry: path_wear_telemetry_snapshots.json")
        print()


if __name__ == '__main__':
    main()
