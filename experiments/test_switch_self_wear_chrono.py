#!/usr/bin/env python3
"""
Self-Wear Test with ChronoMoE: A×N → A′

Compare vanilla vs ChronoMoE-enabled routing evolution.

Protocol:
1. Vanilla: Run A×100, measure drift (should be ~0)
2. ChronoMoE: Run A×100 with T̄ adaptation, measure drift
3. Differential: ChronoMoE_drift - Vanilla_drift

Looking for: ΔKL > 0.01 or ΔEntropy < -0.05 (concentration)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path
import json

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("ERROR: transformers not installed")
    sys.exit(1)

from chronovisor_switch_adapter import wrap_switch_model_with_chronovisor


def extract_switch_routing(model, inputs, layer_idx=1):
    """
    Extract routing from Switch Transformer.

    IMPORTANT: This uses a hook on the router's OUTPUT (not classifier),
    so it captures the wrapped/biased routing when ChronoMoE is enabled.
    """
    captured_router_outputs = []

    def hook_fn(module, input, output):
        # output is (router_mask, router_probs, router_logits)
        # We want router_logits (the full distribution)
        router_mask, router_probs, router_logits = output
        captured_router_outputs.append(router_logits.detach())

    target_block = model.encoder.block[layer_idx]
    router_module = target_block.layer[-1].mlp.router

    # Hook the router's forward output (not classifier)
    handle = router_module.register_forward_hook(hook_fn)

    try:
        decoder_input_ids = inputs['input_ids'].clone()

        with torch.no_grad():
            _ = model(
                input_ids=inputs['input_ids'],
                decoder_input_ids=decoder_input_ids,
            )

        logits = captured_router_outputs[0].squeeze(0)
        probs = torch.softmax(logits, dim=-1)

        return probs.detach().cpu().numpy(), logits.detach().cpu().numpy()

    finally:
        handle.remove()


def compute_kl_divergence(p, q):
    """KL(p || q)"""
    return np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))


def compute_entropy(probs):
    """Shannon entropy"""
    return -np.sum(probs * np.log(probs + 1e-10))


def compute_jaccard_similarity(p, q, k=2):
    """Jaccard similarity of top-k expert sets"""
    top_k_p = set(np.argsort(p)[-k:])
    top_k_q = set(np.argsort(q)[-k:])
    intersection = len(top_k_p & top_k_q)
    union = len(top_k_p | top_k_q)
    return intersection / union if union > 0 else 0.0


def analyze_self_wear(routing_A0, routing_An):
    """Analyze how A's routing changed."""
    # Average over sequence
    avg_A0 = routing_A0.mean(axis=0)
    avg_An = routing_An.mean(axis=0)

    kl_div = compute_kl_divergence(avg_An, avg_A0)
    entropy_A0 = compute_entropy(avg_A0)
    entropy_An = compute_entropy(avg_An)
    delta_entropy = entropy_An - entropy_A0
    jaccard = compute_jaccard_similarity(avg_A0, avg_An, k=2)

    # Per-token analysis
    seq_len_0 = routing_A0.shape[0]
    seq_len_n = routing_An.shape[0]

    token_entropies_A0 = [compute_entropy(routing_A0[t]) for t in range(seq_len_0)]
    token_entropies_An = [compute_entropy(routing_An[t]) for t in range(seq_len_n)]

    return {
        'kl_divergence': float(kl_div),
        'entropy_A0': float(entropy_A0),
        'entropy_An': float(entropy_An),
        'delta_entropy': float(delta_entropy),
        'jaccard_similarity': float(jaccard),
        'token_entropy_mean_A0': float(np.mean(token_entropies_A0)),
        'token_entropy_mean_An': float(np.mean(token_entropies_An)),
        'token_entropy_variance_A0': float(np.var(token_entropies_A0)),
        'token_entropy_variance_An': float(np.var(token_entropies_An)),
    }


def main():
    print("="*70)
    print("SELF-WEAR TEST: ChronoMoE vs Vanilla")
    print("="*70)
    print()

    if not HAS_TRANSFORMERS:
        print("ERROR: transformers not available")
        return

    # Load model
    print("Loading google/switch-base-8...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/switch-base-8",
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
    print("✓ Model loaded\n")

    # Wrap with ChronoMoE
    print("Wrapping model with ChronoMoE adapters...")
    adapters = wrap_switch_model_with_chronovisor(
        model,
        layer_indices=[1],  # Just layer 1 for now
        eta_structural_T=0.015,
    )
    print()

    # Test text
    text_A = "The cat sat on the mat. It was a sunny day."
    print(f"Input A: \"{text_A}\"")
    print()

    inputs = tokenizer(text_A, return_tensors="pt")
    n_repetitions = 100

    # ================================================================
    # Phase 1: Baseline A₀
    # ================================================================
    print("="*70)
    print("PHASE 1: Baseline A₀ (single pass)")
    print("="*70)
    print()

    # Disable ChronoMoE for baseline
    for adapter in adapters.values():
        adapter.disable_all()

    routing_probs_A0, _ = extract_switch_routing(model, inputs, layer_idx=1)

    avg_routing_A0 = routing_probs_A0.mean(axis=0)
    entropy_A0 = compute_entropy(avg_routing_A0)
    max_entropy = np.log(8)

    print(f"Sequence-averaged entropy: {entropy_A0:.4f} / {max_entropy:.4f} ({entropy_A0/max_entropy:.1%})")

    # Per-token entropy
    token_entropies = [compute_entropy(routing_probs_A0[t]) for t in range(routing_probs_A0.shape[0])]
    print(f"Per-token entropy: {np.mean(token_entropies):.4f} ± {np.std(token_entropies):.4f}")
    print()

    # ================================================================
    # Phase 2: VANILLA (A×100, no adaptation)
    # ================================================================
    print("="*70)
    print("PHASE 2: Vanilla (A×100, no ChronoMoE)")
    print("="*70)
    print()

    for adapter in adapters.values():
        adapter.disable_all()

    print(f"Running A × {n_repetitions} times...")
    for i in range(n_repetitions):
        with torch.no_grad():
            decoder_input_ids = inputs['input_ids'].clone()
            _ = model(
                input_ids=inputs['input_ids'],
                decoder_input_ids=decoder_input_ids,
            )

    routing_probs_An_vanilla, _ = extract_switch_routing(model, inputs, layer_idx=1)
    analysis_vanilla = analyze_self_wear(routing_probs_A0, routing_probs_An_vanilla)

    print("Vanilla Results:")
    print(f"  KL(A₁₀₀ || A₀):     {analysis_vanilla['kl_divergence']:.6f}")
    print(f"  ΔEntropy:           {analysis_vanilla['delta_entropy']:+.6f}")
    print(f"  Jaccard similarity: {analysis_vanilla['jaccard_similarity']:.4f}")
    print(f"  Per-token Δentropy: {analysis_vanilla['token_entropy_mean_An'] - analysis_vanilla['token_entropy_mean_A0']:+.6f}")
    print()

    # ================================================================
    # Phase 3: ChronoMoE (A×100 with T̄ adaptation)
    # ================================================================
    print("="*70)
    print("PHASE 3: ChronoMoE (A×100 with T̄ adaptation)")
    print("="*70)
    print()

    # Reset adapters
    for adapter in adapters.values():
        adapter.reset()
        adapter.enable_adaptation()

    # Get initial T̄
    initial_T_bar = adapters[1].lens.T_bar.copy()
    print(f"Initial T̄: {initial_T_bar}")
    print()

    print(f"Running A × {n_repetitions} times with ChronoMoE enabled...")
    for i in range(n_repetitions):
        with torch.no_grad():
            decoder_input_ids = inputs['input_ids'].clone()
            _ = model(
                input_ids=inputs['input_ids'],
                decoder_input_ids=decoder_input_ids,
            )

        if (i + 1) % 25 == 0:
            state = adapters[1].get_state()
            print(f"  Pass {i+1}: T̄ range [{state['T_bar'].min():.4f}, {state['T_bar'].max():.4f}]")

    # Get final T̄
    final_T_bar = adapters[1].lens.T_bar
    T_bar_drift = np.mean(np.abs(final_T_bar - initial_T_bar))

    print()
    print(f"Final T̄: {final_T_bar}")
    print(f"T̄ drift: {T_bar_drift:.6f}")
    print()

    # Measure routing after ChronoMoE
    routing_probs_An_chrono, _ = extract_switch_routing(model, inputs, layer_idx=1)
    analysis_chrono = analyze_self_wear(routing_probs_A0, routing_probs_An_chrono)

    print("ChronoMoE Results:")
    print(f"  KL(A₁₀₀ || A₀):     {analysis_chrono['kl_divergence']:.6f}")
    print(f"  ΔEntropy:           {analysis_chrono['delta_entropy']:+.6f}")
    print(f"  Jaccard similarity: {analysis_chrono['jaccard_similarity']:.4f}")
    print(f"  Per-token Δentropy: {analysis_chrono['token_entropy_mean_An'] - analysis_chrono['token_entropy_mean_A0']:+.6f}")
    print()

    # ================================================================
    # Differential Analysis
    # ================================================================
    print("="*70)
    print("DIFFERENTIAL: ChronoMoE - Vanilla")
    print("="*70)
    print()

    delta_kl = analysis_chrono['kl_divergence'] - analysis_vanilla['kl_divergence']
    delta_entropy = analysis_chrono['delta_entropy'] - analysis_vanilla['delta_entropy']
    delta_jaccard = analysis_chrono['jaccard_similarity'] - analysis_vanilla['jaccard_similarity']

    print(f"ΔΔ KL divergence:   {delta_kl:+.6f}")
    print(f"ΔΔ Entropy:         {delta_entropy:+.6f}")
    print(f"ΔΔ Jaccard:         {delta_jaccard:+.6f}")
    print()

    # Detection thresholds
    wear_detected = (
        abs(delta_kl) > 0.01 or
        abs(delta_entropy) > 0.05 or
        abs(delta_jaccard) > 0.1
    )

    if wear_detected:
        print("✓ DIFFERENTIAL DETECTED")
        print("  ChronoMoE creates measurably different routing evolution than vanilla")
    else:
        print("○ NO DIFFERENTIAL")
        print("  ChronoMoE and vanilla behave identically")

    print()
    print(f"T̄ mechanism status:")
    print(f"  Drift: {T_bar_drift:.6f} ({'active' if T_bar_drift > 0.001 else 'inactive'})")
    print()

    # ================================================================
    # Summary
    # ================================================================
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()

    print(f"Baseline:")
    print(f"  Sequence-averaged entropy: {entropy_A0:.4f}")
    print(f"  Per-token entropy: {np.mean(token_entropies):.4f}")
    print()

    print(f"After {n_repetitions} passes:")
    print(f"  Vanilla drift:    KL={analysis_vanilla['kl_divergence']:.6f}")
    print(f"  ChronoMoE drift:  KL={analysis_chrono['kl_divergence']:.6f}")
    print(f"  Differential:     ΔΔ KL={delta_kl:+.6f}")
    print()

    if wear_detected:
        print("✓ Self-wear detected with ChronoMoE")
        print()
        print("Next: Test cross-wear (A×N → B)")
    else:
        print("⊘ No self-wear detected")
        print()
        print("Possible reasons:")
        print("  1. η too weak (try 0.05, 0.1)")
        print("  2. n_repetitions too few (try 500, 1000)")
        print("  3. Sequence averaging washes out per-token effects")
        print("  4. Pretrained routing too stable for weak perturbations")

    print()

    # Save results
    output_dir = Path(__file__).parent.parent / "takens_data"
    output_dir.mkdir(exist_ok=True)

    results = {
        'text': text_A,
        'num_repetitions': n_repetitions,
        'eta_structural_T': 0.015,
        'baseline': {
            'sequence_averaged_entropy': float(entropy_A0),
            'per_token_entropy_mean': float(np.mean(token_entropies)),
            'per_token_entropy_std': float(np.std(token_entropies)),
        },
        'vanilla': analysis_vanilla,
        'chronomoe': analysis_chrono,
        'differential': {
            'delta_kl': float(delta_kl),
            'delta_entropy': float(delta_entropy),
            'delta_jaccard': float(delta_jaccard),
            'T_bar_drift': float(T_bar_drift),
            'wear_detected': wear_detected,
        },
    }

    with open(output_dir / "switch_self_wear_chrono_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_dir / 'switch_self_wear_chrono_results.json'}")
    print()


if __name__ == '__main__':
    main()
