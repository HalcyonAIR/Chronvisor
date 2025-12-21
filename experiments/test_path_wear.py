#!/usr/bin/env python3
"""
Path Wear Experiment: Testing Inference-Time Resistance Fields

Hypothesis: Repeated inference through region A deforms the routing landscape
(via T̄ or similar persistent state) such that novel input B is biased toward
A-like routing patterns, even with frozen weights.

This tests path dependence, not memory.
This tests resistance fields, not storage.

Critical control: Reset controller between A and B.
If effect disappears on reset → it's real persistence via T̄, not artifact.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

from chronomoe.deepseek_core import DeepSeekConfig
from chronomoe.chronovisor_deepseek_bridge import ChronovisorDeepSeekForCausalLM
from generate_long_geeky_conversations import LongGeekyConversationGenerator


def extract_routing_distribution(chrono_state) -> np.ndarray:
    """Extract routing probability distribution from chronovisor state."""
    if chrono_state is None or not chrono_state.expert_usage:
        return None

    # Get expert usage from first layer as proxy for routing distribution
    usage = chrono_state.expert_usage.get(0, None)
    if usage is None:
        return None

    # Normalize to probability distribution
    usage_normalized = usage / (usage.sum() + 1e-10)
    return usage_normalized


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence KL(P || Q)."""
    # Add small epsilon to avoid log(0)
    p = p + 1e-10
    q = q + 1e-10

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    return np.sum(p * np.log(p / q))


def reset_controller(model):
    """Reset chronovisor controller state to baseline."""
    # ChronovisorDeepSeekForCausalLM has model.model.controller
    controller = getattr(model.model if hasattr(model, 'model') else model, 'controller', None)
    if controller is not None:
        for lens in controller.lenses.values():
            # Reset T̄ to baseline (1.0 = no bias)
            lens.T_bar = np.ones(lens.num_experts)
            # Reset EMA tracking
            lens.expert_usage_ema = np.ones(lens.num_experts) / lens.num_experts


def create_paired_inputs(vocab_size: int, base_length: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create paired inputs A and B that are similar but not identical.

    Strategy: Start with same base, add different noise.
    This ensures structural similarity while being distinct.
    """
    # Create base sequence
    base = torch.randint(0, vocab_size, (base_length,))

    # A: base + small noise
    noise_A = torch.randint(-5, 6, (base_length,))
    input_A = torch.clamp(base + noise_A, 0, vocab_size - 1)

    # B: base + different small noise
    noise_B = torch.randint(-5, 6, (base_length,))
    input_B = torch.clamp(base + noise_B, 0, vocab_size - 1)

    return input_A.unsqueeze(0), input_B.unsqueeze(0)


def test_path_wear(
    model,
    input_A: torch.Tensor,
    input_B: torch.Tensor,
    n_repetitions: int = 100,
) -> Dict:
    """
    Test if repeated inference creates routing bias via persistent state.

    Critical control: reset controller between A and B.
    If bias disappears on reset → it's real persistence, not artifact.

    Returns:
        Dictionary with all measurements and interpretation.
    """

    print(f"\n{'='*70}")
    print("PATH WEAR EXPERIMENT")
    print(f"{'='*70}\n")

    print(f"Testing: Does repeated inference on A bias routing for B?")
    print(f"Repetitions: {n_repetitions}")
    print(f"Input length: {input_A.shape[1]}")
    print()

    # ============================================
    # PHASE 1: BASELINE - B without any A influence
    # ============================================
    print("Phase 1: Baseline (B without A influence)")
    reset_controller(model)

    with torch.no_grad():
        _, state_B_virgin, _ = model(input_B, update_chronovisor=True)
        routing_B_virgin = extract_routing_distribution(state_B_virgin)

    if routing_B_virgin is None:
        print("ERROR: Could not extract routing distribution")
        return None

    print(f"  B (virgin) routing captured: {len(routing_B_virgin)} experts")
    print(f"  Entropy: {-np.sum(routing_B_virgin * np.log(routing_B_virgin + 1e-10)):.4f}")

    # ============================================
    # PHASE 2: ESTABLISH A's PATTERN
    # ============================================
    print("\nPhase 2: Establish A's routing pattern")
    reset_controller(model)

    with torch.no_grad():
        _, state_A, _ = model(input_A, update_chronovisor=True)
        routing_A = extract_routing_distribution(state_A)

    print(f"  A routing captured: {len(routing_A)} experts")
    print(f"  Entropy: {-np.sum(routing_A * np.log(routing_A + 1e-10)):.4f}")

    # ============================================
    # PHASE 3: WEAR THE PATH
    # ============================================
    print(f"\nPhase 3: Wear the path (A × {n_repetitions})")
    reset_controller(model)

    # Track T̄ evolution during wear
    T_bar_initial = model.model.controller.lenses[0].T_bar.copy()

    for i in range(n_repetitions):
        with torch.no_grad():
            _ = model(input_A, update_chronovisor=True)

        if (i + 1) % 25 == 0:
            T_bar_current = model.model.controller.lenses[0].T_bar
            T_bar_delta = np.mean(np.abs(T_bar_current - T_bar_initial))
            print(f"  Step {i+1:3d}: T̄ drift = {T_bar_delta:.6f}")

    T_bar_after_wear = model.model.controller.lenses[0].T_bar.copy()
    T_bar_total_drift = np.mean(np.abs(T_bar_after_wear - T_bar_initial))
    print(f"  Total T̄ drift: {T_bar_total_drift:.6f}")

    # ============================================
    # PHASE 4: TEST - B after wear (NO RESET)
    # ============================================
    print("\nPhase 4: Test B after path wear (no controller reset)")

    with torch.no_grad():
        _, state_B_after_wear, _ = model(input_B, update_chronovisor=True)
        routing_B_after_wear = extract_routing_distribution(state_B_after_wear)

    print(f"  B (after wear) routing captured")
    print(f"  Entropy: {-np.sum(routing_B_after_wear * np.log(routing_B_after_wear + 1e-10)):.4f}")

    # ============================================
    # PHASE 5: CRITICAL CONTROL - B after RESET
    # ============================================
    print("\nPhase 5: Control - B after controller reset")
    reset_controller(model)

    with torch.no_grad():
        _, state_B_after_reset, _ = model(input_B, update_chronovisor=True)
        routing_B_after_reset = extract_routing_distribution(state_B_after_reset)

    print(f"  B (after reset) routing captured")
    print(f"  Entropy: {-np.sum(routing_B_after_reset * np.log(routing_B_after_reset + 1e-10)):.4f}")

    # ============================================
    # ANALYSIS
    # ============================================
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}\n")

    # KL divergence: how close is B to A under each condition?
    dist_virgin = kl_divergence(routing_B_virgin, routing_A)
    dist_after_wear = kl_divergence(routing_B_after_wear, routing_A)
    dist_after_reset = kl_divergence(routing_B_after_reset, routing_A)

    print("KL Divergence (B || A):")
    print(f"  Virgin (no A influence):     {dist_virgin:.6f}")
    print(f"  After wear (with T̄ state):   {dist_after_wear:.6f}")
    print(f"  After reset (T̄ cleared):     {dist_after_reset:.6f}")
    print()

    # Path wear effect
    wear_effect = dist_virgin - dist_after_wear
    print(f"Path wear effect: {wear_effect:+.6f}")
    print(f"  (Positive = B moved toward A)")
    print()

    # Control validation
    reset_similarity = abs(dist_after_reset - dist_virgin)
    reset_eliminates = reset_similarity < 0.01

    print(f"Reset eliminates effect: {reset_eliminates}")
    print(f"  Reset similarity to virgin: {reset_similarity:.6f}")
    print()

    # ============================================
    # INTERPRETATION
    # ============================================
    print(f"{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}\n")

    wear_detected = wear_effect > 0.05

    if wear_detected and reset_eliminates:
        print("✓ PATH WEAR DETECTED")
        print()
        print("Outcome A: Strong wear + reset eliminates it")
        print()
        print("Conclusion:")
        print("  Inference-time dynamics create path dependence via persistent")
        print("  controller state (T̄). Repeated traversal through region A deforms")
        print("  the routing landscape, biasing future routing toward A-like patterns.")
        print()
        print("Interpretation:")
        print("  - T̄ acts as a resistance field")
        print("  - Repeated inference 'wears trails' in the landscape")
        print("  - Novel inputs snap to worn trails")
        print("  - This is path dependence, not memory")
        print()
        print("Publishable claim:")
        print('  "Repeated inference deforms the routing landscape through slow')
        print('   adaptation of control variables, creating history-biased')
        print('   trajectories without weight updates."')

    elif not wear_detected:
        print("○ NO PATH WEAR DETECTED")
        print()
        print("Outcome B: No significant wear effect")
        print()
        print("Conclusion:")
        print("  Current T̄ implementation insufficient to create detectable")
        print("  resistance field under these conditions.")
        print()
        print("Possible reasons:")
        print("  - η_structural_T too small (T̄ adapts too slowly)")
        print("  - n_repetitions too few (need more wear)")
        print("  - Input similarity too low (A and B too different)")
        print()
        print("Next steps:")
        print("  - Increase η_structural_T")
        print("  - Increase n_repetitions")
        print("  - Test with more similar A/B pairs")

    elif wear_detected and not reset_eliminates:
        print("⚠ ARTIFACT DETECTED")
        print()
        print("Outcome D: Strong wear but reset doesn't eliminate")
        print()
        print("Conclusion:")
        print("  The apparent bias is not due to T̄ persistence.")
        print()
        print("Possible causes:")
        print("  - Dataset leakage (A and B route similarly by construction)")
        print("  - Model memorized patterns during training")
        print("  - Random routing bias, not history-dependent")
        print()
        print("Fix:")
        print("  - Use truly novel inputs")
        print("  - Verify no correlation between A and B routing")

    else:
        print("~ WEAK SIGNAL DETECTED")
        print()
        print("Outcome C: Weak wear, reset eliminates")
        print()
        print("Conclusion:")
        print("  Signal exists but small. Real dynamical effect present.")
        print()
        print("Next steps:")
        print("  - Increase n_repetitions (try 500-1000)")
        print("  - Amplify signal with stronger A/B similarity")
        print("  - Measure cumulative effect over longer sequences")

    print()
    print(f"{'='*70}\n")

    # Return results
    return {
        'wear_detected': wear_detected,
        'wear_effect': wear_effect,
        'reset_eliminates': reset_eliminates,
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
    print("PATH WEAR EXPERIMENT: Testing Inference-Time Resistance Fields")
    print("="*70)
    print()
    print("Hypothesis:")
    print("  Repeated inference through region A deforms the routing landscape")
    print("  (via T̄) such that novel input B is biased toward A-like routing,")
    print("  even with frozen weights.")
    print()
    print("Control:")
    print("  Reset controller between A and B.")
    print("  If effect disappears → it's real persistence via T̄, not artifact.")
    print()

    # Create model with chronovisor enabled
    config = DeepSeekConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=1024,
        num_layers=2,
        num_shared_experts=2,
        num_routed_experts=64,
        num_experts_per_token=6,
        enable_chronovisor=True,
    )

    torch.manual_seed(42)
    model = ChronovisorDeepSeekForCausalLM(config)
    model.eval()  # Frozen weights, no training

    controller = model.model.controller

    print(f"Model created:")
    print(f"  Architecture: DeepSeek-MoE")
    print(f"  Routed experts: {config.num_routed_experts}")
    print(f"  Chronovisor: {'enabled' if controller is not None else 'disabled'}")
    if controller is not None:
        print(f"  η_structural_T: {controller.lenses[0].eta_structural_T}")
    print(f"  Weights: frozen (eval mode)")
    print()

    # Create paired inputs
    print("Creating paired inputs...")
    input_A, input_B = create_paired_inputs(config.vocab_size, base_length=50)

    print(f"  Input A: {input_A.shape}")
    print(f"  Input B: {input_B.shape}")
    print(f"  Similarity: constructed from same base + different noise")
    print()

    # Run experiment
    results = test_path_wear(
        model=model,
        input_A=input_A,
        input_B=input_B,
        n_repetitions=100,
    )

    # Save results
    if results is not None:
        output_dir = Path(__file__).parent.parent / "takens_data"
        output_dir.mkdir(exist_ok=True)

        np.savez(
            output_dir / "path_wear_results.npz",
            **{k: v for k, v in results.items() if isinstance(v, (np.ndarray, float, bool))}
        )

        print(f"Results saved to: {output_dir / 'path_wear_results.npz'}")


if __name__ == '__main__':
    main()
