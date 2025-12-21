#!/usr/bin/env python3
"""
Hysteresis Across the Competence Cliff: Phase Diagram

Endpoints: A = 20% noise (Expert 7), B = 60% noise (Expert 2)
Ramps: 5% steps (fine-grained)

At each step, TWO passes:
1. Stiffness pass: No accumulation, measure instantaneous geometry
2. Wear pass: With repetitions, measure trail effects

Control experiment in 30-50% band:
- Fresh run (T̄ reset)
- Carry-forward run (T̄ persists from previous step)

Hysteresis markers:
- Does transition point shift with direction?
- Does Expert 4 width change?
- Expected: 7 holds longer going up, 2 holds longer going down

This distinguishes "grass bends" (instant) from "grass stays bent" (trail).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path
import json
import random
from collections import defaultdict

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
    """Extract routing with margin (stiffness proxy)."""
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

        # Compute probs
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)

        # Compute margin (stiffness proxy)
        sorted_logits = np.sort(logits, axis=-1)[:, ::-1]
        margins = sorted_logits[:, 0] - sorted_logits[:, 1]
        avg_margin = float(np.mean(margins))

        return probs, logits, avg_margin

    finally:
        handle.remove()


def measure_stiffness_only(model, inputs, layer_idx=1):
    """
    Pass 1: Stiffness measurement without accumulation.

    Returns instantaneous geometry.
    """
    routing, logits, margin = extract_routing_with_margin(model, inputs, layer_idx)

    avg_routing = routing.mean(axis=0)
    entropy = -np.sum(avg_routing * np.log(avg_routing + 1e-10))
    top_expert = int(np.argmax(avg_routing))

    return {
        'margin': margin,
        'entropy': float(entropy),
        'top_expert': top_expert,
        'expert_distribution': avg_routing.tolist(),
    }


def measure_wear(model, adapter, inputs, n_repetitions=100, layer_idx=1):
    """
    Pass 2: Wear measurement with repetitions.

    Returns trail effects.
    """
    # Measure initial
    routing_initial, logits_initial, margin_initial = extract_routing_with_margin(
        model, inputs, layer_idx
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
        model, inputs, layer_idx
    )

    # Aggregate
    avg_routing_initial = routing_initial.mean(axis=0)
    avg_routing_final = routing_final.mean(axis=0)

    entropy_initial = -np.sum(avg_routing_initial * np.log(avg_routing_initial + 1e-10))
    entropy_final = -np.sum(avg_routing_final * np.log(avg_routing_final + 1e-10))

    top_expert = int(np.argmax(avg_routing_final))

    return {
        'initial_margin': margin_initial,
        'final_margin': margin_final,
        'delta_entropy': float(entropy_final - entropy_initial),
        'top_expert': top_expert,
        'expert_distribution': avg_routing_final.tolist(),
    }


def run_ramp(model, adapter, tokenizer, text, noise_levels, direction, reset_between_steps=True):
    """
    Run noise ramp in specified direction.

    Args:
        direction: 'up' (A→B) or 'down' (B→A)
        reset_between_steps: If False, carry T̄ forward
    """
    results = []

    # Reset at start
    adapter.reset()
    adapter.enable_adaptation()

    for i, noise_frac in enumerate(noise_levels):
        # Reset between steps if requested
        if reset_between_steps and i > 0:
            adapter.reset()
            adapter.enable_adaptation()

        inputs = create_noise_interpolated_input(tokenizer, text, noise_frac, seed=42)

        # Pass 1: Stiffness (instantaneous)
        stiffness = measure_stiffness_only(model, inputs, layer_idx=1)

        # Pass 2: Wear (with accumulation)
        wear = measure_wear(model, adapter, inputs, n_repetitions=100, layer_idx=1)

        # Get T̄ state
        T_bar_state = adapter.lens.T_bar.copy()

        results.append({
            'noise_fraction': noise_frac,
            'stiffness': stiffness,
            'wear': wear,
            'T_bar_range': [float(T_bar_state.min()), float(T_bar_state.max())],
            'T_bar_mean': float(T_bar_state.mean()),
        })

    return results


def main():
    print("="*70)
    print("HYSTERESIS ACROSS COMPETENCE CLIFF: Phase Diagram")
    print("="*70)
    print()
    print("Testing for 'grass stays bent' vs 'grass bends'")
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

    # Noise levels: 20% → 60% in 5% steps
    noise_levels_up = [0.20 + 0.05 * i for i in range(9)]  # 20%, 25%, ..., 60%
    noise_levels_down = noise_levels_up[::-1]  # Reverse for down ramp

    print(f"Ramp range: {noise_levels_up[0]*100:.0f}% → {noise_levels_up[-1]*100:.0f}%")
    print(f"Step size: 5%")
    print(f"Critical zone: 30-50% (cliff region)")
    print()

    # ================================================================
    # EXPERIMENT 1: A→B ramp (up) with fresh T̄ at each step
    # ================================================================
    print("="*70)
    print("EXPERIMENT 1: A→B Ramp (Fresh T̄)")
    print("="*70)
    print()

    results_up_fresh = run_ramp(
        model, adapter, tokenizer, text,
        noise_levels_up,
        direction='up',
        reset_between_steps=True
    )

    print("Complete.\n")

    # ================================================================
    # EXPERIMENT 2: B→A ramp (down) with fresh T̄ at each step
    # ================================================================
    print("="*70)
    print("EXPERIMENT 2: B→A Ramp (Fresh T̄)")
    print("="*70)
    print()

    results_down_fresh = run_ramp(
        model, adapter, tokenizer, text,
        noise_levels_down,
        direction='down',
        reset_between_steps=True
    )

    print("Complete.\n")

    # ================================================================
    # EXPERIMENT 3: A→B ramp with T̄ carry-forward
    # ================================================================
    print("="*70)
    print("EXPERIMENT 3: A→B Ramp (Carry-forward T̄)")
    print("="*70)
    print()

    results_up_carry = run_ramp(
        model, adapter, tokenizer, text,
        noise_levels_up,
        direction='up',
        reset_between_steps=False
    )

    print("Complete.\n")

    # ================================================================
    # EXPERIMENT 4: B→A ramp with T̄ carry-forward
    # ================================================================
    print("="*70)
    print("EXPERIMENT 4: B→A Ramp (Carry-forward T̄)")
    print("="*70)
    print()

    results_down_carry = run_ramp(
        model, adapter, tokenizer, text,
        noise_levels_down,
        direction='down',
        reset_between_steps=False
    )

    print("Complete.\n")

    # ================================================================
    # ANALYSIS
    # ================================================================
    print("="*70)
    print("ANALYSIS: Phase Diagram")
    print("="*70)
    print()

    # Table for fresh T̄ (instantaneous boundary)
    print("FRESH T̄ (Instantaneous Geometry):")
    print(f"\n{'Noise %':<10} {'Up→Expert':<12} {'Up Margin':<12} {'Down→Expert':<12} {'Down Margin':<12}")
    print("-"*70)

    for i, noise_frac in enumerate(noise_levels_up):
        noise_pct = int(noise_frac * 100)
        up_expert = results_up_fresh[i]['wear']['top_expert']
        up_margin = results_up_fresh[i]['stiffness']['margin']

        # Find corresponding down result
        down_idx = len(noise_levels_down) - 1 - i
        down_expert = results_down_fresh[down_idx]['wear']['top_expert']
        down_margin = results_down_fresh[down_idx]['stiffness']['margin']

        match = "✓" if up_expert == down_expert else "✗"
        print(f"{noise_pct:<10} {up_expert:<12} {up_margin:<12.4f} {down_expert:<12} {down_margin:<12.4f} {match}")

    print()

    # Check for hysteresis in fresh runs
    expert_mismatches = sum(
        1 for i in range(len(noise_levels_up))
        if results_up_fresh[i]['wear']['top_expert'] !=
           results_down_fresh[len(noise_levels_down) - 1 - i]['wear']['top_expert']
    )

    if expert_mismatches > 0:
        print(f"⚠ {expert_mismatches} expert mismatches in FRESH runs")
        print("  (Direction affects outcome even without T̄ trail)")
    else:
        print("✓ Fresh runs identical up/down (no directional bias)")

    print()

    # Table for carry-forward T̄ (trail effects)
    print("CARRY-FORWARD T̄ (Trail Effects):")
    print(f"\n{'Noise %':<10} {'Up→Expert':<12} {'T̄ drift':<12} {'Down→Expert':<12} {'T̄ drift':<12}")
    print("-"*70)

    for i, noise_frac in enumerate(noise_levels_up):
        noise_pct = int(noise_frac * 100)
        up_expert = results_up_carry[i]['wear']['top_expert']
        up_T_bar = results_up_carry[i]['T_bar_mean']

        down_idx = len(noise_levels_down) - 1 - i
        down_expert = results_down_carry[down_idx]['wear']['top_expert']
        down_T_bar = results_down_carry[down_idx]['T_bar_mean']

        match = "✓" if up_expert == down_expert else "✗"
        print(f"{noise_pct:<10} {up_expert:<12} {up_T_bar:<12.4f} {down_expert:<12} {down_T_bar:<12.4f} {match}")

    print()

    # Detect hysteresis loop
    expert_mismatches_carry = sum(
        1 for i in range(len(noise_levels_up))
        if results_up_carry[i]['wear']['top_expert'] !=
           results_down_carry[len(noise_levels_down) - 1 - i]['wear']['top_expert']
    )

    if expert_mismatches_carry > expert_mismatches:
        print(f"✓ HYSTERESIS DETECTED")
        print(f"  Fresh: {expert_mismatches} mismatches")
        print(f"  Carry: {expert_mismatches_carry} mismatches")
        print(f"  T̄ trail creates {expert_mismatches_carry - expert_mismatches} additional divergences")
    else:
        print(f"○ NO TRAIL HYSTERESIS")
        print(f"  Direction effects same with/without T̄ carry-forward")

    print()

    # Expert 4 width analysis
    def find_expert_4_range(results):
        """Find noise range where Expert 4 dominates."""
        expert_4_ranges = []
        for r in results:
            if r['wear']['top_expert'] == 4:
                expert_4_ranges.append(r['noise_fraction'])
        if expert_4_ranges:
            return (min(expert_4_ranges), max(expert_4_ranges))
        return None

    expert_4_up_fresh = find_expert_4_range(results_up_fresh)
    expert_4_down_fresh = find_expert_4_range(results_down_fresh)
    expert_4_up_carry = find_expert_4_range(results_up_carry)
    expert_4_down_carry = find_expert_4_range(results_down_carry)

    print("Expert 4 (ambiguity basin) width:")
    if expert_4_up_fresh:
        print(f"  Up (fresh):   {expert_4_up_fresh[0]*100:.0f}%-{expert_4_up_fresh[1]*100:.0f}%")
    if expert_4_down_fresh:
        print(f"  Down (fresh): {expert_4_down_fresh[0]*100:.0f}%-{expert_4_down_fresh[1]*100:.0f}%")
    if expert_4_up_carry:
        print(f"  Up (carry):   {expert_4_up_carry[0]*100:.0f}%-{expert_4_up_carry[1]*100:.0f}%")
    if expert_4_down_carry:
        print(f"  Down (carry): {expert_4_down_carry[0]*100:.0f}%-{expert_4_down_carry[1]*100:.0f}%")

    print()

    # Transition point analysis
    def find_7_to_2_transition(results):
        """Find where Expert 7 loses to Expert 2."""
        for i in range(len(results) - 1):
            curr_expert = results[i]['wear']['top_expert']
            next_expert = results[i+1]['wear']['top_expert']

            if curr_expert == 7 and next_expert != 7:
                return results[i]['noise_fraction']
        return None

    transition_up_fresh = find_7_to_2_transition(results_up_fresh)
    transition_down_fresh = find_7_to_2_transition(results_down_fresh[::-1])  # Reverse
    transition_up_carry = find_7_to_2_transition(results_up_carry)
    transition_down_carry = find_7_to_2_transition(results_down_carry[::-1])

    print("Transition point (Expert 7 → other):")
    if transition_up_fresh:
        print(f"  Up (fresh):   {transition_up_fresh*100:.0f}%")
    if transition_down_fresh:
        print(f"  Down (fresh): {transition_down_fresh*100:.0f}%")
    if transition_up_carry:
        print(f"  Up (carry):   {transition_up_carry*100:.0f}%")
    if transition_down_carry:
        print(f"  Down (carry): {transition_down_carry*100:.0f}%")

    print()

    if transition_up_carry and transition_down_carry:
        delta = abs(transition_up_carry - transition_down_carry)
        print(f"Loop width: {delta*100:.0f}% noise")

        if delta > 0.05:  # More than one step
            print("✓ GRASS STAYS BENT")
            print("  Transition point depends on approach direction")
        else:
            print("~ GRASS BENDS (minimal trail)")

    # Save results
    output_dir = Path(__file__).parent.parent / "takens_data"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "hysteresis_phase_diagram.json", 'w') as f:
        json.dump({
            'text': text,
            'noise_levels': noise_levels_up,
            'results_up_fresh': results_up_fresh,
            'results_down_fresh': results_down_fresh,
            'results_up_carry': results_up_carry,
            'results_down_carry': results_down_carry,
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'hysteresis_phase_diagram.json'}")
    print()


if __name__ == '__main__':
    main()
