#!/usr/bin/env python3
"""
Yield Point Mapping: Stress-Testing the Cliff

Question: Can we deform the separatrix with higher η?

η sweep: 0.05, 0.10, 0.15, 0.20, 0.25
Fixed: n_repetitions = 100

Three diagnostics (in order):
1. Transition point movement (7→4 and 4→2 boundaries)
2. Expert 4 widening (ambiguity band expansion)
3. Margin-noise curve shape change

Safety check: Pathological collapse
- Does 0-20% noise still route to Expert 7 at high η?
- If not, we've tilted the entire landscape into one well

Expected:
- η=0.10: Deeper basins, no boundary movement
- η=0.15: Expert 4 widening
- Cliff movement: Via uncertainty widening, not direct 7→2 shift

This is yield-point mapping, not vibe check.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path
import json
import random
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("ERROR: transformers not installed")
    sys.exit(1)

from chronovisor_switch_adapter import wrap_switch_model_with_chronovisor


def create_noise_interpolated_input(tokenizer, text, noise_fraction, seed=42):
    """Create input with noise_fraction of tokens replaced by random."""
    real_inputs = tokenizer(text, return_tensors="pt")
    token_ids = real_inputs['input_ids'][0].tolist()

    random.seed(seed)
    num_tokens = len(token_ids)
    num_to_replace = int(num_tokens * noise_fraction)
    positions_to_replace = set(random.sample(range(num_tokens), num_to_replace))

    vocab_size = len(tokenizer)
    mixed_ids = [
        random.randint(0, vocab_size - 1) if i in positions_to_replace else token_id
        for i, token_id in enumerate(token_ids)
    ]

    return {
        'input_ids': torch.tensor([mixed_ids]),
        'attention_mask': torch.ones_like(torch.tensor([mixed_ids]))
    }


def extract_routing_with_margin(model, inputs, layer_idx=1):
    """Extract routing with margin."""
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

        logits = captured_outputs[0].squeeze(0)
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)

        sorted_logits = np.sort(logits, axis=-1)[:, ::-1]
        margins = sorted_logits[:, 0] - sorted_logits[:, 1]
        avg_margin = float(np.mean(margins))

        return probs, logits, avg_margin

    finally:
        handle.remove()


def run_noise_ladder_at_eta(model, adapter, tokenizer, text, eta, n_repetitions=100):
    """
    Run noise ladder at specified η.

    Returns results for each noise level.
    """
    # Update η
    adapter.lens.eta_structural_T = eta

    # Noise levels: 0% to 60% in 5% steps
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    results = []

    for noise_frac in noise_levels:
        # Reset adapter for fresh run
        adapter.reset()
        adapter.enable_adaptation()

        inputs = create_noise_interpolated_input(tokenizer, text, noise_frac, seed=42)

        # Measure initial
        routing_initial, logits_initial, margin_initial = extract_routing_with_margin(
            model, inputs, layer_idx=1
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
            model, inputs, layer_idx=1
        )

        # Aggregate
        avg_routing_initial = routing_initial.mean(axis=0)
        avg_routing_final = routing_final.mean(axis=0)

        entropy_initial = -np.sum(avg_routing_initial * np.log(avg_routing_initial + 1e-10))
        entropy_final = -np.sum(avg_routing_final * np.log(avg_routing_final + 1e-10))

        top_expert = int(np.argmax(avg_routing_final))

        results.append({
            'noise_fraction': noise_frac,
            'margin_initial': margin_initial,
            'margin_final': margin_final,
            'entropy_initial': float(entropy_initial),
            'entropy_final': float(entropy_final),
            'delta_entropy': float(entropy_final - entropy_initial),
            'top_expert': top_expert,
            'expert_distribution': avg_routing_final.tolist(),
        })

    return results


def analyze_boundaries(results):
    """
    Find transition boundaries (7→4 and 4→2).

    Returns:
        - boundary_7_to_4: noise fraction where Expert 7 loses dominance
        - boundary_4_to_2: noise fraction where Expert 4 loses to Expert 2
        - expert_4_width: width of Expert 4 region
    """
    experts = [r['top_expert'] for r in results]
    noise_levels = [r['noise_fraction'] for r in results]

    # Find 7→4 boundary
    boundary_7_to_4 = None
    for i in range(len(experts) - 1):
        if experts[i] == 7 and experts[i+1] != 7:
            boundary_7_to_4 = noise_levels[i]
            break

    # Find 4→2 boundary
    boundary_4_to_2 = None
    for i in range(len(experts) - 1):
        if experts[i] == 4 and experts[i+1] == 2:
            boundary_4_to_2 = noise_levels[i]
            break

    # Expert 4 width
    expert_4_levels = [noise_levels[i] for i, e in enumerate(experts) if e == 4]
    if expert_4_levels:
        expert_4_width = max(expert_4_levels) - min(expert_4_levels) + 0.05  # Include endpoint
    else:
        expert_4_width = 0.0

    return {
        'boundary_7_to_4': boundary_7_to_4,
        'boundary_4_to_2': boundary_4_to_2,
        'expert_4_width': expert_4_width,
    }


def check_pathological_collapse(results):
    """
    Check if low-noise (in-distribution) still routes to Expert 7.

    Pathological collapse: everything goes to Expert 2 regardless of noise.
    """
    low_noise_experts = [r['top_expert'] for r in results if r['noise_fraction'] <= 0.20]

    if not low_noise_experts:
        return None

    expert_7_count = sum(1 for e in low_noise_experts if e == 7)
    is_healthy = expert_7_count >= len(low_noise_experts) * 0.5

    return {
        'low_noise_expert_7_fraction': expert_7_count / len(low_noise_experts),
        'is_healthy': is_healthy,
    }


def main():
    print("="*70)
    print("YIELD POINT MAPPING: Stress-Testing the Cliff")
    print("="*70)
    print()
    print("Question: Can we deform the separatrix with higher η?")
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
        eta_structural_T=0.05,  # Will be updated for each test
    )
    adapter = adapters[1]
    print()

    # Test text
    text = "Machine learning models process data efficiently."

    # η values to test
    eta_values = [0.05, 0.10, 0.15, 0.20, 0.25]

    print(f"Testing η values: {eta_values}")
    print(f"Fixed: n_repetitions = 100")
    print()

    # Run experiments
    all_results = {}

    for eta in eta_values:
        print(f"Running η = {eta:.2f}...")
        results = run_noise_ladder_at_eta(
            model, adapter, tokenizer, text,
            eta=eta,
            n_repetitions=100
        )
        all_results[eta] = results
        print(f"  Complete.")

    print()

    # ================================================================
    # DIAGNOSTIC 1: Transition Point Movement
    # ================================================================
    print("="*70)
    print("DIAGNOSTIC 1: Transition Point Movement")
    print("="*70)
    print()

    boundaries_by_eta = {}

    print(f"{'η':<8} {'7→4 boundary':<15} {'4→2 boundary':<15} {'Expert 4 width':<15}")
    print("-"*70)

    for eta in eta_values:
        boundaries = analyze_boundaries(all_results[eta])
        boundaries_by_eta[eta] = boundaries

        b_7_4 = f"{boundaries['boundary_7_to_4']*100:.0f}%" if boundaries['boundary_7_to_4'] else "N/A"
        b_4_2 = f"{boundaries['boundary_4_to_2']*100:.0f}%" if boundaries['boundary_4_to_2'] else "N/A"
        width = f"{boundaries['expert_4_width']*100:.0f}%"

        print(f"{eta:<8.2f} {b_7_4:<15} {b_4_2:<15} {width:<15}")

    print()

    # Check for movement
    boundary_7_4_values = [b['boundary_7_to_4'] for b in boundaries_by_eta.values() if b['boundary_7_to_4'] is not None]
    if len(set(boundary_7_4_values)) > 1:
        print("✓ BOUNDARY MOVEMENT DETECTED (7→4)")
        print(f"  Range: {min(boundary_7_4_values)*100:.0f}% - {max(boundary_7_4_values)*100:.0f}%")
    else:
        print("○ No boundary movement (7→4)")

    print()

    # ================================================================
    # DIAGNOSTIC 2: Expert 4 Widening
    # ================================================================
    print("="*70)
    print("DIAGNOSTIC 2: Expert 4 Widening")
    print("="*70)
    print()

    expert_4_widths = [b['expert_4_width'] for b in boundaries_by_eta.values()]

    print(f"{'η':<8} {'Expert 4 width':<15}")
    print("-"*30)
    for eta, width in zip(eta_values, expert_4_widths):
        print(f"{eta:<8.2f} {width*100:<15.0f}%")

    print()

    if max(expert_4_widths) > min(expert_4_widths) * 1.5:
        print("✓ EXPERT 4 WIDENING")
        print(f"  Width grows from {min(expert_4_widths)*100:.0f}% to {max(expert_4_widths)*100:.0f}%")
        print("  Ambiguity band expanding with η")
    else:
        print("○ Expert 4 width stable")

    print()

    # ================================================================
    # DIAGNOSTIC 3: Margin-Noise Curve Shape
    # ================================================================
    print("="*70)
    print("DIAGNOSTIC 3: Margin-Noise Curve Shape")
    print("="*70)
    print()

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for eta in eta_values:
        results = all_results[eta]
        noise_levels = [r['noise_fraction'] * 100 for r in results]
        margins = [r['margin_initial'] for r in results]

        ax1.plot(noise_levels, margins, marker='o', label=f'η={eta:.2f}')

    ax1.set_xlabel('Noise %')
    ax1.set_ylabel('Margin (top1 - top2 logit)')
    ax1.set_title('Margin vs Noise (Initial)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Expert progression
    for eta in eta_values:
        results = all_results[eta]
        noise_levels = [r['noise_fraction'] * 100 for r in results]
        experts = [r['top_expert'] for r in results]

        ax2.plot(noise_levels, experts, marker='s', label=f'η={eta:.2f}', linewidth=2)

    ax2.set_xlabel('Noise %')
    ax2.set_ylabel('Top Expert')
    ax2.set_title('Expert Selection vs Noise')
    ax2.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = Path(__file__).parent.parent / "takens_data"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "yield_point_mapping.png", dpi=150)
    print(f"Plot saved to: {output_dir / 'yield_point_mapping.png'}")
    print()

    # ================================================================
    # SAFETY CHECK: Pathological Collapse
    # ================================================================
    print("="*70)
    print("SAFETY CHECK: Pathological Collapse")
    print("="*70)
    print()

    print(f"{'η':<8} {'In-dist → Expert 7':<20} {'Status':<15}")
    print("-"*50)

    for eta in eta_values:
        collapse_check = check_pathological_collapse(all_results[eta])

        if collapse_check:
            fraction = collapse_check['low_noise_expert_7_fraction']
            status = "✓ Healthy" if collapse_check['is_healthy'] else "⚠ Collapsed"
            print(f"{eta:<8.2f} {fraction*100:<20.0f}% {status:<15}")
        else:
            print(f"{eta:<8.2f} {'N/A':<20} {'?':<15}")

    print()

    # Check if any η caused collapse
    collapsed_etas = [
        eta for eta in eta_values
        if check_pathological_collapse(all_results[eta]) and
           not check_pathological_collapse(all_results[eta])['is_healthy']
    ]

    if collapsed_etas:
        print(f"⚠ PATHOLOGICAL COLLAPSE at η ≥ {min(collapsed_etas):.2f}")
        print("  In-distribution inputs no longer route to Expert 7")
        print("  Landscape tilted into garbage can")
    else:
        print("✓ No pathological collapse")
        print("  All η values preserve in-distribution routing")

    print()

    # ================================================================
    # SUMMARY
    # ================================================================
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()

    # Yield point determination
    if len(set(boundary_7_4_values)) > 1:
        print("✓ YIELD POINT DETECTED")
        print(f"  Separatrix deforms with η")
        yield_eta = min([eta for eta, b in boundaries_by_eta.items()
                        if b['boundary_7_to_4'] != boundaries_by_eta[0.05]['boundary_7_to_4']])
        print(f"  Yield begins at η ≈ {yield_eta:.2f}")
    elif max(expert_4_widths) > min(expert_4_widths) * 1.5:
        print("~ PARTIAL YIELD (Expert 4 widening)")
        print("  Boundary softening but not moving")
    else:
        print("○ ELASTIC REGIME")
        print("  Separatrix rigid up to η = 0.25")
        print("  Below yield point")

    print()

    # Save results
    with open(output_dir / "yield_point_mapping_results.json", 'w') as f:
        json.dump({
            'text': text,
            'eta_values': eta_values,
            'n_repetitions': 100,
            'all_results': {str(eta): results for eta, results in all_results.items()},
            'boundaries_by_eta': {str(eta): boundaries for eta, boundaries in boundaries_by_eta.items()},
        }, f, indent=2)

    print(f"Results saved to: {output_dir / 'yield_point_mapping_results.json'}")
    print()


if __name__ == '__main__':
    main()
