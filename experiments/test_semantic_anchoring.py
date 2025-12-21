#!/usr/bin/env python3
"""
Semantic Anchoring Test: The Killer Experiment

Compare wear on:
  A: Real text prompts (semantic structure)
  B: Shuffled tokens (preserves unigram stats, destroys semantics)
  C: Random tokens (pure noise)

Hypothesis:
- If wear is just "winner-take-more," A, B, C will concentrate similarly
- If wear is anchored to representation geometry, A will show different,
  more stable and selective concentration than B/C

Fixed: η=0.05, n_repetitions=100 (detection regime)
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


def create_input_variants(tokenizer, text):
    """
    Create three variants of input:
      - Real: Original text
      - Shuffled: Tokens shuffled (destroys semantics, preserves vocabulary)
      - Random: Random tokens from vocabulary
    """
    # Tokenize real text
    real_inputs = tokenizer(text, return_tensors="pt")
    token_ids = real_inputs['input_ids'][0].tolist()

    # Variant A: Real (unchanged)
    real_variant = real_inputs

    # Variant B: Shuffled tokens
    shuffled_ids = token_ids.copy()
    random.seed(42)  # Reproducible shuffle
    random.shuffle(shuffled_ids)
    shuffled_variant = {
        'input_ids': torch.tensor([shuffled_ids]),
        'attention_mask': torch.ones_like(torch.tensor([shuffled_ids]))
    }

    # Variant C: Random tokens (matching length)
    vocab_size = len(tokenizer)
    random.seed(43)  # Different seed for random
    random_ids = [random.randint(0, vocab_size - 1) for _ in range(len(token_ids))]
    random_variant = {
        'input_ids': torch.tensor([random_ids]),
        'attention_mask': torch.ones_like(torch.tensor([random_ids]))
    }

    return {
        'real': real_variant,
        'shuffled': shuffled_variant,
        'random': random_variant,
    }


def extract_routing(model, inputs, layer_idx=1):
    """Extract routing from Switch Transformer."""
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
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)

        return probs, logits

    finally:
        handle.remove()


def compute_entropy(probs):
    """Shannon entropy"""
    return -np.sum(probs * np.log(probs + 1e-10))


def compute_concentration(routing_probs):
    """
    Measure concentration: How much mass is in top-k experts?

    Returns:
        - top1_mass: Probability mass in top-1 expert
        - top2_mass: Probability mass in top-2 experts
        - entropy: Shannon entropy
    """
    avg_probs = routing_probs.mean(axis=0)  # Average over sequence

    sorted_probs = np.sort(avg_probs)[::-1]

    return {
        'top1_mass': float(sorted_probs[0]),
        'top2_mass': float(sorted_probs[0] + sorted_probs[1]),
        'entropy': float(compute_entropy(avg_probs)),
    }


def run_wear_experiment(model, adapter, inputs, n_repetitions=100, layer_idx=1):
    """
    Run wear experiment on given input.

    Returns:
        - initial_concentration
        - final_concentration
        - delta metrics
    """
    # Reset adapter
    adapter.reset()
    adapter.enable_adaptation()

    # Measure initial routing
    routing_initial, _ = extract_routing(model, inputs, layer_idx=layer_idx)
    concentration_initial = compute_concentration(routing_initial)

    # Run wear
    for i in range(n_repetitions):
        with torch.no_grad():
            decoder_input_ids = inputs['input_ids'].clone()
            _ = model(
                input_ids=inputs['input_ids'],
                decoder_input_ids=decoder_input_ids,
            )

    # Measure final routing
    routing_final, _ = extract_routing(model, inputs, layer_idx=layer_idx)
    concentration_final = compute_concentration(routing_final)

    # Compute deltas
    delta_top1 = concentration_final['top1_mass'] - concentration_initial['top1_mass']
    delta_top2 = concentration_final['top2_mass'] - concentration_initial['top2_mass']
    delta_entropy = concentration_final['entropy'] - concentration_initial['entropy']

    return {
        'initial': concentration_initial,
        'final': concentration_final,
        'delta_top1_mass': float(delta_top1),
        'delta_top2_mass': float(delta_top2),
        'delta_entropy': float(delta_entropy),
    }


def main():
    print("="*70)
    print("SEMANTIC ANCHORING TEST: The Killer Experiment")
    print("="*70)
    print()
    print("Comparing wear on:")
    print("  A: Real text (semantic structure)")
    print("  B: Shuffled tokens (destroys semantics)")
    print("  C: Random tokens (pure noise)")
    print()
    print("Hypothesis: If wear is anchored to representation geometry,")
    print("A will show different concentration pattern than B/C")
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

    # Wrap with ChronoMoE at detection-regime η
    adapters = wrap_switch_model_with_chronovisor(
        model,
        layer_indices=[1],
        eta_structural_T=0.05,  # Detection regime
    )
    adapter = adapters[1]
    print()

    # Test texts
    test_texts = {
        'Text1': "The cat sat on the mat. It was a sunny day.",
        'Text2': "Machine learning models process data efficiently.",
    }

    results = {}

    for text_label, text in test_texts.items():
        print(f"{'='*70}")
        print(f"TEXT: {text_label}")
        print(f"{'='*70}")
        print(f"\"{text}\"")
        print()

        # Create variants
        variants = create_input_variants(tokenizer, text)

        # Show token representations
        print("Token variants:")
        print(f"  Real:     {tokenizer.convert_ids_to_tokens(variants['real']['input_ids'][0])[:10]}")
        print(f"  Shuffled: {tokenizer.convert_ids_to_tokens(variants['shuffled']['input_ids'][0])[:10]}")
        print(f"  Random:   {tokenizer.convert_ids_to_tokens(variants['random']['input_ids'][0])[:10]}")
        print()

        # Run experiments
        variant_results = {}

        for variant_label, variant_inputs in variants.items():
            print(f"Running {variant_label}...")

            result = run_wear_experiment(
                model, adapter, variant_inputs,
                n_repetitions=100,
                layer_idx=1
            )

            variant_results[variant_label] = result

            print(f"  Initial: Top1={result['initial']['top1_mass']:.3f}, "
                  f"Entropy={result['initial']['entropy']:.3f}")
            print(f"  Final:   Top1={result['final']['top1_mass']:.3f}, "
                  f"Entropy={result['final']['entropy']:.3f}")
            print(f"  Δ Top1:  {result['delta_top1_mass']:+.4f}")
            print(f"  Δ Entropy: {result['delta_entropy']:+.4f}")
            print()

        results[text_label] = variant_results

    # Analysis
    print("="*70)
    print("ANALYSIS")
    print("="*70)
    print()

    # Average across texts
    avg_delta_by_variant = {
        'real': np.mean([results[t]['real']['delta_top1_mass'] for t in test_texts.keys()]),
        'shuffled': np.mean([results[t]['shuffled']['delta_top1_mass'] for t in test_texts.keys()]),
        'random': np.mean([results[t]['random']['delta_top1_mass'] for t in test_texts.keys()]),
    }

    avg_entropy_delta_by_variant = {
        'real': np.mean([results[t]['real']['delta_entropy'] for t in test_texts.keys()]),
        'shuffled': np.mean([results[t]['shuffled']['delta_entropy'] for t in test_texts.keys()]),
        'random': np.mean([results[t]['random']['delta_entropy'] for t in test_texts.keys()]),
    }

    print("Average Δ Top-1 mass:")
    for variant, delta in avg_delta_by_variant.items():
        print(f"  {variant:>10}: {delta:+.4f}")

    print()
    print("Average Δ Entropy:")
    for variant, delta in avg_entropy_delta_by_variant.items():
        print(f"  {variant:>10}: {delta:+.4f}")

    print()

    # Interpretation
    real_vs_shuffled_ratio = (
        abs(avg_delta_by_variant['real']) / (abs(avg_delta_by_variant['shuffled']) + 1e-10)
    )
    real_vs_random_ratio = (
        abs(avg_delta_by_variant['real']) / (abs(avg_delta_by_variant['random']) + 1e-10)
    )

    print(f"Concentration ratio:")
    print(f"  Real / Shuffled: {real_vs_shuffled_ratio:.2f}x")
    print(f"  Real / Random:   {real_vs_random_ratio:.2f}x")
    print()

    # Verdict
    if real_vs_shuffled_ratio > 1.5 or real_vs_random_ratio > 1.5:
        print("✓ SEMANTIC ANCHORING DETECTED")
        print()
        print("Real text produces stronger/different concentration than")
        print("shuffled or random tokens. Wear is exploiting learned")
        print("representation geometry, not just statistical winner-take-more.")
    elif real_vs_shuffled_ratio < 0.67 or real_vs_random_ratio < 0.67:
        print("⚠ ANTI-SEMANTIC PATTERN")
        print()
        print("Real text produces weaker concentration than noise.")
        print("Unexpected - may indicate semantic structure resists wear.")
    else:
        print("~ NEUTRAL / WINNER-TAKE-MORE")
        print()
        print("Real, shuffled, and random produce similar concentration.")
        print("Wear may be purely statistical (winner-take-more) without")
        print("exploiting semantic structure.")

    print()

    # Save results
    output_dir = Path(__file__).parent.parent / "takens_data"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "semantic_anchoring_results.json", 'w') as f:
        json.dump({
            'test_texts': test_texts,
            'results': results,
            'avg_delta_by_variant': {k: float(v) for k, v in avg_delta_by_variant.items()},
            'avg_entropy_delta_by_variant': {k: float(v) for k, v in avg_entropy_delta_by_variant.items()},
            'real_vs_shuffled_ratio': float(real_vs_shuffled_ratio),
            'real_vs_random_ratio': float(real_vs_random_ratio),
        }, f, indent=2)

    print(f"Results saved to: {output_dir / 'semantic_anchoring_results.json'}")
    print()


if __name__ == '__main__':
    main()
