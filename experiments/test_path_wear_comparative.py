#!/usr/bin/env python3
"""
Comparative Path Wear Experiment: Vanilla vs ChronoMoE

Critical insight from user: We have 4 models (2 architectures × 2 configs).
Need to compare vanilla vs ChronoMoE to detect differential effect.

Hypothesis: ChronoMoE creates path wear via T̄ persistence, vanilla doesn't.

Expected outcome:
  - Vanilla: No wear (routing distributions unchanged)
  - ChronoMoE: Perturbation (bias toward recent passes)

If delta_wear > 0 → T̄ works
If delta_wear ≈ 0 → mechanism insufficient (boundary)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

from chronomoe.deepseek_core import DeepSeekConfig, DeepSeekForCausalLM
from chronomoe.chronovisor_deepseek_bridge import ChronovisorDeepSeekForCausalLM


def extract_routing_distribution(output_tuple, model_type: str) -> np.ndarray:
    """
    Extract routing probability distribution.

    Args:
        output_tuple: Model output (vanilla or chronovisor)
        model_type: 'vanilla' or 'chronovisor'
    """
    if model_type == 'chronovisor':
        # ChronovisorDeepSeekForCausalLM returns (logits, chrono_state, router_outputs)
        _, chrono_state, router_outputs = output_tuple
        if chrono_state is None or not chrono_state.expert_usage:
            return None
        # Get expert usage from first layer
        usage = chrono_state.expert_usage.get(0, None)
        if usage is None:
            return None
        usage_normalized = usage / (usage.sum() + 1e-10)
        return usage_normalized

    elif model_type == 'vanilla':
        # DeepSeekForCausalLM returns (logits, router_outputs)
        _, router_outputs = output_tuple
        if router_outputs is None or len(router_outputs) == 0:
            return None

        # Get routing weights from first layer
        first_layer_output = router_outputs[0]
        if hasattr(first_layer_output, 'routing_weights'):
            routing_weights = first_layer_output.routing_weights
        elif isinstance(first_layer_output, dict) and 'routing_weights' in first_layer_output:
            routing_weights = first_layer_output['routing_weights']
        else:
            return None

        # Average over sequence and batch: [batch, seq, num_experts] -> [num_experts]
        if isinstance(routing_weights, torch.Tensor):
            routing_weights = routing_weights.detach().cpu().numpy()

        # Average over batch and sequence dimensions
        while routing_weights.ndim > 1:
            routing_weights = routing_weights.mean(axis=0)

        routing_normalized = routing_weights / (routing_weights.sum() + 1e-10)
        return routing_normalized

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence KL(P || Q)."""
    p = p + 1e-10
    q = q + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q))


def reset_controller(model):
    """Reset chronovisor controller state to baseline (ChronoMoE only)."""
    if hasattr(model, 'model') and hasattr(model.model, 'controller'):
        controller = model.model.controller
        if controller is not None:
            for lens in controller.lenses.values():
                lens.T_bar = np.ones(lens.num_experts)
                lens.expert_usage_ema = np.ones(lens.num_experts) / lens.num_experts


def create_paired_inputs(vocab_size: int, base_length: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create paired inputs A and B that are similar but not identical.

    Strategy: Start with same base, add different noise.
    """
    base = torch.randint(0, vocab_size, (base_length,))

    # A: base + small noise
    noise_A = torch.randint(-5, 6, (base_length,))
    input_A = torch.clamp(base + noise_A, 0, vocab_size - 1)

    # B: base + different small noise
    noise_B = torch.randint(-5, 6, (base_length,))
    input_B = torch.clamp(base + noise_B, 0, vocab_size - 1)

    return input_A.unsqueeze(0), input_B.unsqueeze(0)


def test_path_wear_single_model(
    model,
    model_type: str,
    input_A: torch.Tensor,
    input_B: torch.Tensor,
    n_repetitions: int = 100,
) -> Dict:
    """
    Test path wear for a single model (vanilla or chronovisor).

    Returns measurements for all 3 phases:
      - virgin: B without A influence
      - after_wear: B after A repetitions
      - after_reset: B after controller reset (ChronoMoE only)
    """

    print(f"\n{'='*70}")
    print(f"TESTING: {model_type.upper()}")
    print(f"{'='*70}\n")

    # ============================================
    # PHASE 1: BASELINE - B without any A influence
    # ============================================
    print("Phase 1: Baseline (B without A influence)")

    if model_type == 'chronovisor':
        reset_controller(model)

    with torch.no_grad():
        if model_type == 'chronovisor':
            output_B_virgin = model(input_B, update_chronovisor=True)
        else:
            output_B_virgin = model(input_B)

        routing_B_virgin = extract_routing_distribution(output_B_virgin, model_type)

    if routing_B_virgin is None:
        print("ERROR: Could not extract routing distribution")
        return None

    print(f"  B (virgin) routing captured: {len(routing_B_virgin)} experts")
    print(f"  Entropy: {-np.sum(routing_B_virgin * np.log(routing_B_virgin + 1e-10)):.4f}")

    # ============================================
    # PHASE 2: ESTABLISH A's PATTERN
    # ============================================
    print("\nPhase 2: Establish A's routing pattern")

    if model_type == 'chronovisor':
        reset_controller(model)

    with torch.no_grad():
        if model_type == 'chronovisor':
            output_A = model(input_A, update_chronovisor=True)
        else:
            output_A = model(input_A)

        routing_A = extract_routing_distribution(output_A, model_type)

    print(f"  A routing captured: {len(routing_A)} experts")
    print(f"  Entropy: {-np.sum(routing_A * np.log(routing_A + 1e-10)):.4f}")

    # ============================================
    # PHASE 3: WEAR THE PATH
    # ============================================
    print(f"\nPhase 3: Wear the path (A × {n_repetitions})")

    if model_type == 'chronovisor':
        reset_controller(model)
        # Track T̄ evolution
        T_bar_initial = model.model.controller.lenses[0].T_bar.copy()

    for i in range(n_repetitions):
        with torch.no_grad():
            if model_type == 'chronovisor':
                _ = model(input_A, update_chronovisor=True)
            else:
                _ = model(input_A)

        if (i + 1) % 25 == 0:
            if model_type == 'chronovisor':
                T_bar_current = model.model.controller.lenses[0].T_bar
                T_bar_delta = np.mean(np.abs(T_bar_current - T_bar_initial))
                print(f"  Step {i+1:3d}: T̄ drift = {T_bar_delta:.6f}")
            else:
                print(f"  Step {i+1:3d}: (vanilla - no T̄ tracking)")

    if model_type == 'chronovisor':
        T_bar_after_wear = model.model.controller.lenses[0].T_bar.copy()
        T_bar_total_drift = np.mean(np.abs(T_bar_after_wear - T_bar_initial))
        print(f"  Total T̄ drift: {T_bar_total_drift:.6f}")
    else:
        T_bar_total_drift = 0.0
        print(f"  Total T̄ drift: N/A (vanilla has no T̄)")

    # ============================================
    # PHASE 4: TEST - B after wear (NO RESET)
    # ============================================
    print("\nPhase 4: Test B after path wear (no reset)")

    with torch.no_grad():
        if model_type == 'chronovisor':
            output_B_after_wear = model(input_B, update_chronovisor=True)
        else:
            output_B_after_wear = model(input_B)

        routing_B_after_wear = extract_routing_distribution(output_B_after_wear, model_type)

    print(f"  B (after wear) routing captured")
    print(f"  Entropy: {-np.sum(routing_B_after_wear * np.log(routing_B_after_wear + 1e-10)):.4f}")

    # ============================================
    # PHASE 5: CONTROL - B after RESET (ChronoMoE only)
    # ============================================
    if model_type == 'chronovisor':
        print("\nPhase 5: Control - B after controller reset")
        reset_controller(model)

        with torch.no_grad():
            output_B_after_reset = model(input_B, update_chronovisor=True)
            routing_B_after_reset = extract_routing_distribution(output_B_after_reset, model_type)

        print(f"  B (after reset) routing captured")
        print(f"  Entropy: {-np.sum(routing_B_after_reset * np.log(routing_B_after_reset + 1e-10)):.4f}")
    else:
        # Vanilla doesn't have reset mechanism
        routing_B_after_reset = routing_B_virgin.copy()
        print("\nPhase 5: Control - N/A for vanilla (no controller to reset)")

    # ============================================
    # ANALYSIS
    # ============================================
    print(f"\n{'='*70}")
    print(f"ANALYSIS: {model_type.upper()}")
    print(f"{'='*70}\n")

    dist_virgin = kl_divergence(routing_B_virgin, routing_A)
    dist_after_wear = kl_divergence(routing_B_after_wear, routing_A)
    dist_after_reset = kl_divergence(routing_B_after_reset, routing_A)

    print("KL Divergence (B || A):")
    print(f"  Virgin (no A influence):     {dist_virgin:.6f}")
    print(f"  After wear (with state):     {dist_after_wear:.6f}")
    if model_type == 'chronovisor':
        print(f"  After reset (T̄ cleared):     {dist_after_reset:.6f}")
    print()

    wear_effect = dist_virgin - dist_after_wear
    print(f"Path wear effect: {wear_effect:+.6f}")
    print(f"  (Positive = B moved toward A)")
    print()

    return {
        'model_type': model_type,
        'wear_effect': wear_effect,
        'dist_virgin': dist_virgin,
        'dist_after_wear': dist_after_wear,
        'dist_after_reset': dist_after_reset,
        'T_bar_drift': T_bar_total_drift,
        'routing_A': routing_A,
        'routing_B_virgin': routing_B_virgin,
        'routing_B_after_wear': routing_B_after_wear,
        'routing_B_after_reset': routing_B_after_reset,
    }


def main():
    print("="*70)
    print("COMPARATIVE PATH WEAR EXPERIMENT")
    print("Vanilla DeepSeek vs ChronoMoE-Enhanced DeepSeek")
    print("="*70)
    print()
    print("Hypothesis:")
    print("  ChronoMoE creates path wear via T̄ persistence, vanilla doesn't.")
    print()
    print("Expected outcome:")
    print("  - Vanilla: No wear (routing distributions unchanged)")
    print("  - ChronoMoE: Perturbation (bias toward recent passes)")
    print()
    print("If delta_wear > 0 → T̄ works")
    print("If delta_wear ≈ 0 → mechanism insufficient (boundary)")
    print()

    # Shared config parameters
    vocab_size = 1000
    hidden_dim = 256
    intermediate_dim = 1024
    num_layers = 2
    num_shared_experts = 2
    num_routed_experts = 64
    num_experts_per_token = 6

    # Create vanilla model
    print("="*70)
    print("CREATING VANILLA DEEPSEEK MODEL")
    print("="*70)
    print()

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
    model_vanilla = DeepSeekForCausalLM(config_vanilla)
    model_vanilla.eval()

    print(f"Model created:")
    print(f"  Architecture: DeepSeek-MoE")
    print(f"  Routed experts: {config_vanilla.num_routed_experts}")
    print(f"  Chronovisor: disabled")
    print(f"  Weights: frozen (eval mode)")
    print()

    # Create ChronoMoE model
    print("="*70)
    print("CREATING CHRONOMOE-ENHANCED DEEPSEEK MODEL")
    print("="*70)
    print()

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

    torch.manual_seed(42)  # Same seed for fair comparison
    model_chrono = ChronovisorDeepSeekForCausalLM(config_chrono)
    model_chrono.eval()

    controller = model_chrono.model.controller

    print(f"Model created:")
    print(f"  Architecture: DeepSeek-MoE + ChronoMoE")
    print(f"  Routed experts: {config_chrono.num_routed_experts}")
    print(f"  Chronovisor: enabled")
    if controller is not None:
        print(f"  η_structural_T: {controller.lenses[0].eta_structural_T}")
    print(f"  Weights: frozen (eval mode)")
    print()

    # Create paired inputs (same for both models)
    print("="*70)
    print("CREATING PAIRED INPUTS")
    print("="*70)
    print()

    input_A, input_B = create_paired_inputs(vocab_size, base_length=50)

    print(f"  Input A: {input_A.shape}")
    print(f"  Input B: {input_B.shape}")
    print(f"  Similarity: constructed from same base + different noise")
    print()

    # Run experiments
    results_vanilla = test_path_wear_single_model(
        model=model_vanilla,
        model_type='vanilla',
        input_A=input_A,
        input_B=input_B,
        n_repetitions=100,
    )

    results_chrono = test_path_wear_single_model(
        model=model_chrono,
        model_type='chronovisor',
        input_A=input_A,
        input_B=input_B,
        n_repetitions=100,
    )

    # ============================================
    # COMPARATIVE ANALYSIS
    # ============================================
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS")
    print("="*70 + "\n")

    if results_vanilla is None or results_chrono is None:
        print("ERROR: Could not complete analysis")
        return

    delta_wear = results_chrono['wear_effect'] - results_vanilla['wear_effect']
    delta_T_bar_drift = results_chrono['T_bar_drift'] - results_vanilla['T_bar_drift']

    print("Path Wear Effect:")
    print(f"  Vanilla:   {results_vanilla['wear_effect']:+.6f}")
    print(f"  ChronoMoE: {results_chrono['wear_effect']:+.6f}")
    print(f"  Δ (Chrono - Vanilla): {delta_wear:+.6f}")
    print()

    print("T̄ Drift:")
    print(f"  Vanilla:   {results_vanilla['T_bar_drift']:.6f} (N/A)")
    print(f"  ChronoMoE: {results_chrono['T_bar_drift']:.6f}")
    print(f"  Δ: {delta_T_bar_drift:.6f}")
    print()

    print("KL Divergence (B || A):")
    print(f"  Vanilla virgin:        {results_vanilla['dist_virgin']:.6f}")
    print(f"  Vanilla after wear:    {results_vanilla['dist_after_wear']:.6f}")
    print(f"  ChronoMoE virgin:      {results_chrono['dist_virgin']:.6f}")
    print(f"  ChronoMoE after wear:  {results_chrono['dist_after_wear']:.6f}")
    print(f"  ChronoMoE after reset: {results_chrono['dist_after_reset']:.6f}")
    print()

    # ============================================
    # INTERPRETATION
    # ============================================
    print("="*70)
    print("INTERPRETATION")
    print("="*70 + "\n")

    differential_detected = abs(delta_wear) > 0.01
    chrono_shows_wear = results_chrono['wear_effect'] > 0.05
    vanilla_shows_wear = results_vanilla['wear_effect'] > 0.05

    if differential_detected and chrono_shows_wear and not vanilla_shows_wear:
        print("✓ DIFFERENTIAL PATH WEAR DETECTED")
        print()
        print("Outcome A: ChronoMoE shows wear, vanilla doesn't")
        print()
        print("Conclusion:")
        print("  T̄ creates inference-time path dependence. Repeated traversal")
        print("  through region A deforms the routing landscape via geological")
        print("  adaptation, biasing future routing toward A-like patterns.")
        print()
        print("Interpretation:")
        print("  - T̄ acts as a resistance field")
        print("  - Repeated inference 'wears trails' in the landscape")
        print("  - Novel inputs snap to worn trails")
        print("  - This is differential: ChronoMoE only")
        print()
        print("Publishable claim:")
        print('  "Chronovisor control layer creates history-biased routing')
        print('   trajectories without weight updates, demonstrating inference-time')
        print('   path dependence via slow geological adaptation (T̄)."')

    elif not differential_detected:
        print("○ NO DIFFERENTIAL DETECTED")
        print()
        print("Outcome B: Both models behave similarly")
        print()
        print("Conclusion:")
        print("  T̄ mechanism insufficient to create detectable differential")
        print("  path wear under these conditions.")
        print()
        print("Possible reasons:")
        print("  - η_structural_T too small (0.015)")
        print("  - n_repetitions too few (100)")
        print("  - Measurement insensitive (KL of endpoints vs trajectory ease)")
        print()
        print("Next steps:")
        print("  - Increase η_structural_T to 0.05-0.1")
        print("  - Increase n_repetitions to 500-1000")
        print("  - Measure convergence speed, curvature")
        print("  - Accept boundary and document")

    elif chrono_shows_wear and vanilla_shows_wear:
        print("⚠ BOTH MODELS SHOW WEAR")
        print()
        print("Outcome C: Artifact - both show similar wear")
        print()
        print("Conclusion:")
        print("  The apparent bias is not specific to T̄ persistence.")
        print()
        print("Possible causes:")
        print("  - Input similarity too high (A and B route similarly)")
        print("  - Model architecture creates path dependence without T̄")
        print("  - Random routing bias")
        print()
        print("Fix:")
        print("  - Use more distinct A/B pairs")
        print("  - Verify no architectural memory")

    else:
        print("~ UNEXPECTED PATTERN")
        print()
        print(f"ChronoMoE wear: {results_chrono['wear_effect']:+.6f}")
        print(f"Vanilla wear: {results_vanilla['wear_effect']:+.6f}")
        print()
        print("Investigate manually.")

    print()
    print("="*70 + "\n")

    # Save results
    output_dir = Path(__file__).parent.parent / "takens_data"
    output_dir.mkdir(exist_ok=True)

    np.savez(
        output_dir / "path_wear_comparative_results.npz",
        delta_wear=delta_wear,
        delta_T_bar_drift=delta_T_bar_drift,
        vanilla_wear_effect=results_vanilla['wear_effect'],
        chrono_wear_effect=results_chrono['wear_effect'],
        vanilla_routing_A=results_vanilla['routing_A'],
        vanilla_routing_B_virgin=results_vanilla['routing_B_virgin'],
        vanilla_routing_B_after_wear=results_vanilla['routing_B_after_wear'],
        chrono_routing_A=results_chrono['routing_A'],
        chrono_routing_B_virgin=results_chrono['routing_B_virgin'],
        chrono_routing_B_after_wear=results_chrono['routing_B_after_wear'],
        chrono_routing_B_after_reset=results_chrono['routing_B_after_reset'],
        chrono_T_bar_drift=results_chrono['T_bar_drift'],
    )

    print(f"Results saved to: {output_dir / 'path_wear_comparative_results.npz'}")


if __name__ == '__main__':
    main()
