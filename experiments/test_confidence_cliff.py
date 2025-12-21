#!/usr/bin/env python3
"""
Find the Confidence Cliff: Real → Noise Interpolation

Gradually mix real tokens with random noise at controlled ratios:
  0% noise (pure real) → Expert 7
  100% noise (pure random) → Expert 2

Question: Where does the attractor flip?

The transition point is the router's confidence boundary.
This tells us how robust the in-distribution basin is.

Also tracking:
- Margin evolution (should drop as we approach cliff)
- Wear magnitude (should increase as margin drops)
- Expert distribution (smooth transition or sharp flip?)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path
import json
import random

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("ERROR: transformers not installed")
    sys.exit(1)

from chronovisor_switch_adapter import wrap_switch_model_with_chronovisor


def create_noise_interpolated_input(tokenizer, text, noise_fraction, seed=42):
    """
    Create input with noise_fraction of tokens replaced by random.

    Args:
        noise_fraction: 0.0 (pure real) to 1.0 (pure random)

    Returns:
        inputs dict with mixed tokens
    """
    # Tokenize real text
    real_inputs = tokenizer(text, return_tensors="pt")
    token_ids = real_inputs['input_ids'][0].tolist()

    # Decide which positions to replace
    random.seed(seed)
    num_tokens = len(token_ids)
    num_to_replace = int(num_tokens * noise_fraction)

    # Random positions to replace
    positions_to_replace = set(random.sample(range(num_tokens), num_to_replace))

    # Replace selected positions with random tokens
    vocab_size = len(tokenizer)
    mixed_ids = []

    for i, token_id in enumerate(token_ids):
        if i in positions_to_replace:
            mixed_ids.append(random.randint(0, vocab_size - 1))
        else:
            mixed_ids.append(token_id)

    return {
        'input_ids': torch.tensor([mixed_ids]),
        'attention_mask': torch.ones_like(torch.tensor([mixed_ids]))
    }


def extract_routing_with_margin(model, inputs, layer_idx=1):
    """Extract routing with margin computation."""
    captured_outputs = []

    def hook_fn(module, input, output):
        router_mask, router_probs, router_logits = output
        captured_outputs.append(router_logits.detach().cpu().numpy())

    target_block = model.encoder.block[layer_idx]
    router_module = target_block.layer[-1].mlp.router
    handle = router_module.register_forward_hook(hook_fn)

    try:
        decoder_input_ids = inputs['input_ids'].clone()

        with torch.no_grad():
            _ = model(
                input_ids=inputs['input_ids'],
                decoder_input_ids=decoder_input_ids,
            )

        logits = captured_outputs[0].squeeze(0)  # [seq_len, num_experts]

        # Compute probs
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)

        # Compute margin
        sorted_logits = np.sort(logits, axis=-1)[:, ::-1]
        margins = sorted_logits[:, 0] - sorted_logits[:, 1]
        avg_margin = float(np.mean(margins))

        return probs, logits, avg_margin

    finally:
        handle.remove()


def analyze_routing_state(model, adapter, inputs, n_repetitions=100, layer_idx=1):
    """
    Analyze routing before and after wear.

    Returns:
        - initial_margin
        - initial_entropy
        - final_entropy
        - delta_entropy
        - top_expert
        - top_expert_mass
        - expert_distribution (final)
    """
    # Reset adapter
    adapter.reset()
    adapter.enable_adaptation()

    # Measure initial
    routing_initial, logits_initial, margin_initial = extract_routing_with_margin(
        model, inputs, layer_idx=layer_idx
    )

    # Run wear
    for i in range(n_repetitions):
        with torch.no_grad():
            decoder_input_ids = inputs['input_ids'].clone()
            _ = model(
                input_ids=inputs['input_ids'],
                decoder_input_ids=decoder_input_ids,
            )

    # Measure final
    routing_final, logits_final, margin_final = extract_routing_with_margin(
        model, inputs, layer_idx=layer_idx
    )

    # Aggregate
    avg_routing_initial = routing_initial.mean(axis=0)
    avg_routing_final = routing_final.mean(axis=0)

    entropy_initial = -np.sum(avg_routing_initial * np.log(avg_routing_initial + 1e-10))
    entropy_final = -np.sum(avg_routing_final * np.log(avg_routing_final + 1e-10))

    top_expert = int(np.argmax(avg_routing_final))
    top_expert_mass = float(avg_routing_final[top_expert])

    return {
        'initial_margin': margin_initial,
        'initial_entropy': float(entropy_initial),
        'final_entropy': float(entropy_final),
        'delta_entropy': float(entropy_final - entropy_initial),
        'top_expert': top_expert,
        'top_expert_mass': top_expert_mass,
        'expert_distribution': avg_routing_final.tolist(),
    }


def main():
    print("="*70)
    print("FIND THE CONFIDENCE CLIFF: Real → Noise Interpolation")
    print("="*70)
    print()
    print("Hypothesis: Attractor flips from Expert 7 (in-dist) to Expert 2 (OOD)")
    print("at a critical noise threshold.")
    print()

    if not HAS_TRANSFORMERS:
        print("ERROR: transformers not available")
        return

    # Load model
    print("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/switch-base-8",
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
    print("✓ Model loaded\n")

    # Wrap with ChronoMoE
    adapters = wrap_switch_model_with_chronovisor(
        model,
        layer_indices=[1],
        eta_structural_T=0.05,
    )
    adapter = adapters[1]
    print()

    # Test text
    text = "Machine learning models process data efficiently."

    # Noise fractions to test (fine grid near expected transition)
    noise_fractions = [
        0.0,   # Pure real
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,   # Pure random
    ]

    print(f"Testing {len(noise_fractions)} noise levels...")
    print()

    results = []

    for noise_frac in noise_fractions:
        print(f"Noise fraction: {noise_frac:.1f}")

        # Create interpolated input
        inputs = create_noise_interpolated_input(tokenizer, text, noise_frac, seed=42)

        # Analyze
        result = analyze_routing_state(model, adapter, inputs, n_repetitions=100, layer_idx=1)

        result['noise_fraction'] = noise_frac

        print(f"  Initial margin: {result['initial_margin']:.4f}")
        print(f"  Δ Entropy:      {result['delta_entropy']:.4f}")
        print(f"  Top expert:     {result['top_expert']} ({result['top_expert_mass']:.3f})")
        print()

        results.append(result)

    # Analysis
    print("="*70)
    print("ANALYSIS: Finding the Cliff")
    print("="*70)
    print()

    # Table
    print(f"{'Noise %':<10} {'Margin':<10} {'Wear':<10} {'Top Expert':<12} {'Mass':<8}")
    print("-"*70)

    for r in results:
        noise_pct = int(r['noise_fraction'] * 100)
        print(f"{noise_pct:<10} {r['initial_margin']:<10.4f} {abs(r['delta_entropy']):<10.4f} "
              f"{r['top_expert']:<12} {r['top_expert_mass']:<8.3f}")

    print()

    # Find transition point
    experts_by_noise = [(r['noise_fraction'], r['top_expert']) for r in results]

    # Check if transition occurs
    initial_expert = experts_by_noise[0][1]
    final_expert = experts_by_noise[-1][1]

    if initial_expert != final_expert:
        print(f"✓ ATTRACTOR TRANSITION DETECTED")
        print(f"  0% noise → Expert {initial_expert}")
        print(f"  100% noise → Expert {final_expert}")
        print()

        # Find transition point
        for i in range(len(experts_by_noise) - 1):
            noise1, expert1 = experts_by_noise[i]
            noise2, expert2 = experts_by_noise[i + 1]

            if expert1 != expert2:
                print(f"  Transition occurs between {noise1*100:.0f}% and {noise2*100:.0f}% noise")
                print(f"    {noise1*100:.0f}%: Expert {expert1}")
                print(f"    {noise2*100:.0f}%: Expert {expert2}")
                print()

                # Compute margin at transition
                margin1 = results[i]['initial_margin']
                margin2 = results[i+1]['initial_margin']
                print(f"  Margin drops from {margin1:.4f} to {margin2:.4f}")
                print(f"  Cliff steepness: {(margin1 - margin2)/(noise2 - noise1):.4f} margin/noise")
                break
    else:
        print("~ NO TRANSITION")
        print(f"  Same expert ({initial_expert}) across all noise levels")

    print()

    # Check margin evolution
    margins = [r['initial_margin'] for r in results]
    noise_levels = [r['noise_fraction'] for r in results]

    # Compute slope
    if len(margins) > 1:
        # Linear fit
        slope = (margins[-1] - margins[0]) / (noise_levels[-1] - noise_levels[0])
        print(f"Margin evolution:")
        print(f"  0% → 100% noise: {margins[0]:.4f} → {margins[-1]:.4f}")
        print(f"  Slope: {slope:.4f} margin/noise")
        print()

    # Check wear evolution
    wear_mags = [abs(r['delta_entropy']) for r in results]

    print(f"Wear evolution:")
    print(f"  0% noise:   {wear_mags[0]:.4f}")
    print(f"  100% noise: {wear_mags[-1]:.4f}")
    print(f"  Ratio:      {wear_mags[-1] / wear_mags[0]:.2f}x")
    print()

    # Check if transition is smooth or sharp
    # Look at expert distribution across noise levels
    print("Expert distribution evolution:")
    expert_2_mass = [r['expert_distribution'][2] for r in results]
    expert_7_mass = [r['expert_distribution'][7] for r in results]

    print(f"\n{'Noise %':<10} {'Expert 2':<12} {'Expert 7':<12}")
    print("-"*40)
    for i, r in enumerate(results):
        noise_pct = int(r['noise_fraction'] * 100)
        print(f"{noise_pct:<10} {expert_2_mass[i]:<12.4f} {expert_7_mass[i]:<12.4f}")

    print()

    # Check if transition is smooth (gradual shift) or sharp (flip)
    # Compute rate of change in expert 7 mass
    if len(expert_7_mass) > 1:
        max_drop = max(abs(expert_7_mass[i] - expert_7_mass[i+1])
                      for i in range(len(expert_7_mass) - 1))
        avg_drop = np.mean([abs(expert_7_mass[i] - expert_7_mass[i+1])
                           for i in range(len(expert_7_mass) - 1)])

        print(f"Expert 7 mass change:")
        print(f"  Max step drop: {max_drop:.4f}")
        print(f"  Avg step drop: {avg_drop:.4f}")
        print(f"  Ratio: {max_drop / avg_drop:.2f}x")
        print()

        if max_drop > 3 * avg_drop:
            print("✓ SHARP TRANSITION (cliff)")
            print("  Expert preference flips abruptly")
        else:
            print("~ SMOOTH TRANSITION (gradient)")
            print("  Expert preference shifts gradually")

    print()

    # Save results
    output_dir = Path(__file__).parent.parent / "takens_data"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "confidence_cliff_results.json", 'w') as f:
        json.dump({
            'text': text,
            'noise_fractions': noise_fractions,
            'results': results,
        }, f, indent=2)

    print(f"Results saved to: {output_dir / 'confidence_cliff_results.json'}")
    print()

    print("="*70)
    print("INTERPRETATION")
    print("="*70)
    print()
    print("The confidence cliff marks the router's OOD boundary.")
    print("Below this threshold: confident in-distribution routing")
    print("Above this threshold: collapse to OOD attractor")
    print()
    print("This transition point is a direct measure of routing robustness.")
    print()


if __name__ == '__main__':
    main()
