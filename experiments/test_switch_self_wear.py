#!/usr/bin/env python3
"""
Self-Wear Test: A×N → A′

Before testing cross-input transfer (A×N → B), first test:
Does repeated inference amplify self-consistency?

Hypothesis:
- Run A once → capture routing_A₀
- Run A×N with ChronoMoE → capture routing_Aₙ
- Measure: ΔKL(routing_A₀, routing_Aₙ), entropy change, coalition stability

If self-wear exists, routing should become MORE concentrated on its
own pattern (lower entropy, higher consistency).

This tests the fundamental mechanism before cross-input effects.
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


def extract_switch_routing(model, inputs, layer_idx=1):
    """
    Extract routing from Switch Transformer using forward hooks.

    Returns:
        routing_probs: [seq_len, num_experts]
        router_logits: [seq_len, num_experts]
    """
    captured_router_logits = []

    def hook_fn(module, input, output):
        hidden_states = input[0]
        with torch.no_grad():
            router_logits = module.classifier(hidden_states)
            captured_router_logits.append(router_logits.detach())

    if layer_idx == 0:
        raise ValueError("Layer 0 is dense. Use layer_idx >= 1")

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

        if len(captured_router_logits) == 0:
            raise ValueError("No router logits captured")

        logits = captured_router_logits[0].squeeze(0)
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


def analyze_self_wear(
    routing_A0: np.ndarray,
    routing_An: np.ndarray,
    num_experts: int = 8,
):
    """
    Analyze self-wear: how did A's routing change after N repetitions?

    Returns:
        - ΔKL: KL divergence change (higher = more drift)
        - ΔEntropy: Entropy change (negative = more concentrated)
        - ΔJaccard: Coalition stability (lower = more drift)
        - Per-token analysis
    """
    seq_len_0, _ = routing_A0.shape
    seq_len_n, _ = routing_An.shape

    # Average over sequence for aggregate measures
    avg_A0 = routing_A0.mean(axis=0)
    avg_An = routing_An.mean(axis=0)

    # 1. KL divergence (how much did distribution change?)
    kl_div = compute_kl_divergence(avg_An, avg_A0)

    # 2. Entropy (did routing become more concentrated?)
    entropy_A0 = compute_entropy(avg_A0)
    entropy_An = compute_entropy(avg_An)
    delta_entropy = entropy_An - entropy_A0

    # 3. Jaccard similarity (did top experts change?)
    jaccard = compute_jaccard_similarity(avg_A0, avg_An, k=2)

    # 4. Per-token analysis (variance across tokens)
    token_entropies_A0 = [compute_entropy(routing_A0[t]) for t in range(seq_len_0)]
    token_entropies_An = [compute_entropy(routing_An[t]) for t in range(seq_len_n)]

    return {
        'kl_divergence': float(kl_div),
        'entropy_A0': float(entropy_A0),
        'entropy_An': float(entropy_An),
        'delta_entropy': float(delta_entropy),
        'jaccard_similarity': float(jaccard),
        'token_entropy_variance_A0': float(np.var(token_entropies_A0)),
        'token_entropy_variance_An': float(np.var(token_entropies_An)),
        'delta_variance': float(np.var(token_entropies_An) - np.var(token_entropies_A0)),
    }


def main():
    print("="*70)
    print("SELF-WEAR TEST: A×N → A′")
    print("="*70)
    print()
    print("Testing: Does repeated inference amplify self-consistency?")
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

    # Test text
    text_A = "The cat sat on the mat. It was a sunny day."

    print(f"Input A: \"{text_A}\"")
    print()

    # Tokenize
    inputs = tokenizer(text_A, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    print(f"Tokens ({len(tokens)}): {tokens}")
    print()

    # ================================================================
    # Phase 1: Baseline A₀ (single pass)
    # ================================================================
    print("="*70)
    print("PHASE 1: Baseline A₀ (single pass)")
    print("="*70)
    print()

    routing_probs_A0, router_logits_A0 = extract_switch_routing(
        model, inputs, layer_idx=1
    )

    avg_routing_A0 = routing_probs_A0.mean(axis=0)
    entropy_A0 = compute_entropy(avg_routing_A0)
    max_entropy = np.log(8)

    print(f"Entropy: {entropy_A0:.4f} / {max_entropy:.4f} ({entropy_A0/max_entropy:.1%})")
    print(f"Top-3 experts:")
    top3 = np.argsort(avg_routing_A0)[::-1][:3]
    for i, expert_idx in enumerate(top3):
        print(f"  {i+1}. Expert {expert_idx}: {avg_routing_A0[expert_idx]:.2%}")
    print()

    # ================================================================
    # Phase 2: VANILLA Control (A×N without any adaptation)
    # ================================================================
    print("="*70)
    print("PHASE 2: VANILLA Control (A×100, no adaptation)")
    print("="*70)
    print()
    print("Running A × 100 times (vanilla model, stateless)...")

    n_repetitions = 100
    for i in range(n_repetitions):
        with torch.no_grad():
            decoder_input_ids = inputs['input_ids'].clone()
            _ = model(
                input_ids=inputs['input_ids'],
                decoder_input_ids=decoder_input_ids,
            )

    print("Measuring A after 100 repetitions...")

    routing_probs_An_vanilla, router_logits_An_vanilla = extract_switch_routing(
        model, inputs, layer_idx=1
    )

    analysis_vanilla = analyze_self_wear(routing_probs_A0, routing_probs_An_vanilla)

    print()
    print("Vanilla (No Adaptation) Results:")
    print(f"  KL(A₁₀₀ || A₀):     {analysis_vanilla['kl_divergence']:.6f}")
    print(f"  ΔEntropy:           {analysis_vanilla['delta_entropy']:+.6f}")
    print(f"  Jaccard similarity: {analysis_vanilla['jaccard_similarity']:.4f}")
    print(f"  ΔVariance:          {analysis_vanilla['delta_variance']:+.6f}")
    print()

    if abs(analysis_vanilla['kl_divergence']) > 0.001:
        print("⚠ WARNING: Vanilla model shows drift! Model may have non-determinism.")
    else:
        print("✓ Vanilla baseline: No drift (expected for stateless pretrained model)")
    print()

    # ================================================================
    # Phase 3: ChronoMoE-Enabled (future)
    # ================================================================
    print("="*70)
    print("PHASE 3: ChronoMoE-Enabled (Future)")
    print("="*70)
    print()
    print("This phase requires:")
    print("  1. ChronoMoE adapter wrapping Switch router")
    print("  2. T̄ adaptation during A×N")
    print("  3. Measure routing_Aₙ with geological bias")
    print()
    print("Not implemented yet - vanilla baseline establishes control.")
    print()

    # ================================================================
    # Summary
    # ================================================================
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()

    print(f"Baseline entropy: {entropy_A0:.4f} ({entropy_A0/max_entropy:.1%} of max)")
    print()

    print("Vanilla (no adaptation):")
    print(f"  Drift:  {analysis_vanilla['kl_divergence']:.6f} (should be ~0)")
    print()

    print("Interpretation:")
    if abs(analysis_vanilla['kl_divergence']) < 0.001:
        print("✓ Vanilla baseline stable - ready for ChronoMoE differential test")
        print()
        print("Next steps:")
        print("  1. Implement ChronoMoE adapter for Switch router")
        print("  2. Run A×N with T̄ adaptation enabled")
        print("  3. Measure differential: ChronoMoE drift - Vanilla drift")
        print("  4. Look for ΔKL > 0.01 or ΔEntropy < -0.05")
    else:
        print("⚠ Unexpected vanilla drift - investigate before proceeding")
    print()

    # Save results
    output_dir = Path(__file__).parent.parent / "takens_data"
    output_dir.mkdir(exist_ok=True)

    results = {
        'text': text_A,
        'num_repetitions': n_repetitions,
        'baseline': {
            'entropy': float(entropy_A0),
            'max_entropy': float(max_entropy),
            'routing_distribution': avg_routing_A0.tolist(),
        },
        'vanilla': analysis_vanilla,
    }

    with open(output_dir / "switch_self_wear_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_dir / 'switch_self_wear_results.json'}")
    print()

    print("="*70)
    print("SELF-WEAR BASELINE COMPLETE")
    print("="*70)
    print()


if __name__ == '__main__':
    main()
