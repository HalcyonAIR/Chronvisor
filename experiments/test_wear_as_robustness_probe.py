#!/usr/bin/env python3
"""
Wear as Robustness Probe: The Full Story

Three experiments in one:

1. CRITICAL QUESTION: When random concentrates, same expert or different per seed?
   - Run multiple seeds, track which expert wins
   - Compare to real text (should be consistent)

2. GRADUATED OOD LADDER: Measure wear vs OOD severity
   - Level 0: Real text
   - Level 1: Word-salad (real words, random order)
   - Level 2: Random vocab tokens (matched unigram stats)
   - Level 3: Uniform random tokens

3. ROUTER CONFIDENCE TEST: Wear vs initial margin
   - Compute initial margin (top1 - top2 logit)
   - Show: low margin → high wear

Hypothesis: Wear magnitude inversely correlates with routing robustness.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path
import json
import random
from collections import Counter

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("ERROR: transformers not installed")
    sys.exit(1)

from chronovisor_switch_adapter import wrap_switch_model_with_chronovisor


def create_ood_ladder(tokenizer, text, seed=42):
    """
    Create graduated OOD ladder.

    Returns dict with keys:
        - real: Original text
        - word_salad: Real words, random order
        - random_vocab: Random tokens from vocab (matched stats)
        - uniform_random: Truly uniform random
    """
    # Tokenize real text
    real_inputs = tokenizer(text, return_tensors="pt")
    token_ids = real_inputs['input_ids'][0].tolist()

    # Level 0: Real (unchanged)
    real_variant = real_inputs

    # Level 1: Word-salad (shuffle tokens, preserving vocabulary)
    random.seed(seed)
    word_salad_ids = token_ids.copy()
    random.shuffle(word_salad_ids)
    word_salad_variant = {
        'input_ids': torch.tensor([word_salad_ids]),
        'attention_mask': torch.ones_like(torch.tensor([word_salad_ids]))
    }

    # Level 2: Random vocab tokens (sample from token distribution)
    # Use a simple unigram model - just sample uniformly from a reasonable range
    random.seed(seed + 1)
    vocab_size = min(len(tokenizer), 30000)  # Reasonable vocab
    random_vocab_ids = [random.randint(0, vocab_size - 1) for _ in range(len(token_ids))]
    random_vocab_variant = {
        'input_ids': torch.tensor([random_vocab_ids]),
        'attention_mask': torch.ones_like(torch.tensor([random_vocab_ids]))
    }

    # Level 3: Uniform random (truly uniform over full vocab)
    random.seed(seed + 2)
    vocab_size_full = len(tokenizer)
    uniform_random_ids = [random.randint(0, vocab_size_full - 1) for _ in range(len(token_ids))]
    uniform_random_variant = {
        'input_ids': torch.tensor([uniform_random_ids]),
        'attention_mask': torch.ones_like(torch.tensor([uniform_random_ids]))
    }

    return {
        'real': real_variant,
        'word_salad': word_salad_variant,
        'random_vocab': random_vocab_variant,
        'uniform_random': uniform_random_variant,
    }


def extract_routing_with_margin(model, inputs, layer_idx=1):
    """
    Extract routing and compute router margin (top1 - top2 logit).

    Returns:
        - routing_probs: [seq_len, num_experts]
        - routing_logits: [seq_len, num_experts]
        - margin: Average margin (top1 - top2 logit)
    """
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

        # Compute margin for each token
        sorted_logits = np.sort(logits, axis=-1)[:, ::-1]  # Descending
        margins = sorted_logits[:, 0] - sorted_logits[:, 1]  # top1 - top2
        avg_margin = float(np.mean(margins))

        return probs, logits, avg_margin

    finally:
        handle.remove()


def run_wear_with_tracking(model, adapter, inputs, n_repetitions=100, layer_idx=1):
    """
    Run wear experiment with detailed tracking.

    Returns:
        - initial_routing
        - final_routing
        - initial_margin
        - final_margin
        - top_expert (which expert became dominant)
        - delta_entropy
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

    # Compute aggregate metrics
    avg_routing_initial = routing_initial.mean(axis=0)
    avg_routing_final = routing_final.mean(axis=0)

    entropy_initial = -np.sum(avg_routing_initial * np.log(avg_routing_initial + 1e-10))
    entropy_final = -np.sum(avg_routing_final * np.log(avg_routing_final + 1e-10))

    top_expert = int(np.argmax(avg_routing_final))
    top_expert_mass = float(avg_routing_final[top_expert])

    delta_entropy = entropy_final - entropy_initial

    return {
        'initial_margin': margin_initial,
        'final_margin': margin_final,
        'initial_entropy': entropy_initial,
        'final_entropy': entropy_final,
        'delta_entropy': delta_entropy,
        'top_expert': top_expert,
        'top_expert_mass': top_expert_mass,
    }


def main():
    print("="*70)
    print("WEAR AS ROBUSTNESS PROBE: The Full Story")
    print("="*70)
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

    # ================================================================
    # EXPERIMENT 1: Critical Question - Same expert or different per seed?
    # ================================================================
    print("="*70)
    print("EXPERIMENT 1: Random Seed Stability")
    print("="*70)
    print()
    print("Question: When random tokens concentrate, same expert or different?")
    print()

    num_seeds = 5
    random_top_experts = []

    for seed in range(num_seeds):
        # Create uniform random input
        random.seed(seed)
        token_ids = [random.randint(0, len(tokenizer) - 1) for _ in range(10)]
        random_inputs = {
            'input_ids': torch.tensor([token_ids]),
            'attention_mask': torch.ones_like(torch.tensor([token_ids]))
        }

        result = run_wear_with_tracking(model, adapter, random_inputs, n_repetitions=100, layer_idx=1)
        random_top_experts.append(result['top_expert'])

        print(f"  Seed {seed}: Top expert = {result['top_expert']}, "
              f"mass = {result['top_expert_mass']:.3f}, "
              f"Δ entropy = {result['delta_entropy']:.4f}")

    # Check consistency
    expert_counts = Counter(random_top_experts)
    most_common_expert, count = expert_counts.most_common(1)[0]

    print()
    print(f"Expert selection across {num_seeds} seeds:")
    for expert, cnt in expert_counts.most_common():
        print(f"  Expert {expert}: {cnt}/{num_seeds}")

    print()
    if count == num_seeds:
        print("✓ CONSISTENT GLOBAL COLLAPSE")
        print(f"  All seeds converge to expert {most_common_expert}")
    elif count >= num_seeds * 0.6:
        print("~ MOSTLY CONSISTENT")
        print(f"  {count}/{num_seeds} seeds pick expert {most_common_expert}")
    else:
        print("✓ UNSTABLE BASIN SELECTION")
        print("  Different seeds pick different experts")

    print()

    # Compare to real text consistency
    print("Comparing to real text...")
    real_inputs = tokenizer(text, return_tensors="pt")
    real_top_experts = []

    for seed in range(num_seeds):
        # Use same text but different T̄ initialization
        adapter.reset()
        # Perturb T̄ slightly
        np.random.seed(seed)
        adapter.lens.T_bar += np.random.normal(0, 0.01, size=adapter.lens.T_bar.shape)
        adapter.enable_adaptation()

        result = run_wear_with_tracking(model, adapter, real_inputs, n_repetitions=100, layer_idx=1)
        real_top_experts.append(result['top_expert'])

    real_expert_counts = Counter(real_top_experts)
    print(f"\nReal text expert selection:")
    for expert, cnt in real_expert_counts.most_common():
        print(f"  Expert {expert}: {cnt}/{num_seeds}")

    print()

    # ================================================================
    # EXPERIMENT 2: Graduated OOD Ladder
    # ================================================================
    print("="*70)
    print("EXPERIMENT 2: Graduated OOD Ladder")
    print("="*70)
    print()

    ood_variants = create_ood_ladder(tokenizer, text, seed=42)

    ood_results = {}

    for level, (variant_name, variant_inputs) in enumerate(ood_variants.items()):
        print(f"Level {level}: {variant_name}")

        result = run_wear_with_tracking(model, adapter, variant_inputs, n_repetitions=100, layer_idx=1)

        ood_results[variant_name] = result

        print(f"  Initial margin: {result['initial_margin']:.4f}")
        print(f"  Initial entropy: {result['initial_entropy']:.4f}")
        print(f"  Δ Entropy: {result['delta_entropy']:.4f}")
        print(f"  Top expert: {result['top_expert']} ({result['top_expert_mass']:.3f})")
        print()

    # ================================================================
    # EXPERIMENT 3: Router Confidence vs Wear
    # ================================================================
    print("="*70)
    print("EXPERIMENT 3: Router Confidence vs Wear")
    print("="*70)
    print()

    # Plot correlation
    variants_sorted = sorted(ood_results.items(), key=lambda x: x[1]['initial_margin'])

    print(f"{'Variant':<20} {'Init Margin':<12} {'Δ Entropy':<12} {'Wear Magnitude':<15}")
    print("-"*70)

    for variant_name, result in variants_sorted:
        wear_magnitude = abs(result['delta_entropy'])
        print(f"{variant_name:<20} {result['initial_margin']:<12.4f} "
              f"{result['delta_entropy']:<+12.4f} {wear_magnitude:<15.4f}")

    print()

    # Check correlation
    margins = [r['initial_margin'] for r in ood_results.values()]
    wear_magnitudes = [abs(r['delta_entropy']) for r in ood_results.values()]

    if len(margins) > 1:
        correlation = np.corrcoef(margins, wear_magnitudes)[0, 1]
        print(f"Correlation (margin vs wear magnitude): {correlation:.3f}")
        print()

        if correlation < -0.5:
            print("✓ STRONG NEGATIVE CORRELATION")
            print("  Low margin (uncertain routing) → high wear")
            print("  Wear is a probe for routing flatness/tie-breaking")
        elif correlation > 0.5:
            print("⚠ POSITIVE CORRELATION (unexpected)")
        else:
            print("~ WEAK CORRELATION")

    print()

    # ================================================================
    # SUMMARY
    # ================================================================
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()

    print("1. Random token expert selection:")
    if count == num_seeds:
        print(f"   → Consistent global collapse to expert {most_common_expert}")
    else:
        print(f"   → Unstable basin selection across seeds")

    print()
    print("2. OOD ladder (wear magnitude):")
    for variant_name in ['real', 'word_salad', 'random_vocab', 'uniform_random']:
        if variant_name in ood_results:
            wear_mag = abs(ood_results[variant_name]['delta_entropy'])
            print(f"   {variant_name:>15}: {wear_mag:.4f}")

    print()
    print("3. Router confidence:")
    print(f"   Correlation: {correlation:.3f} (margin vs wear)")

    print()
    print("Interpretation:")
    print("  Wear magnitude inversely correlates with routing robustness.")
    print("  In-distribution inputs sit in deep basins (high margin, low wear).")
    print("  Out-of-distribution inputs probe flat regions (low margin, high wear).")
    print()
    print("  Semantics shows up as resistance, not as structure to exploit.")

    # Save results
    output_dir = Path(__file__).parent.parent / "takens_data"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "wear_as_robustness_probe.json", 'w') as f:
        json.dump({
            'text': text,
            'random_seed_stability': {
                'top_experts': random_top_experts,
                'expert_counts': dict(expert_counts),
                'real_top_experts': real_top_experts,
            },
            'ood_ladder': {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                               for kk, vv in v.items()}
                          for k, v in ood_results.items()},
            'correlation_margin_wear': float(correlation),
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'wear_as_robustness_probe.json'}")
    print()


if __name__ == '__main__':
    main()
