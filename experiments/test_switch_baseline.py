#!/usr/bin/env python3
"""
Switch-base-8 Baseline: Characterize Learned Routing Structure

Pre-test before any ChronoMoE injection.

Measures:
1. Entropy distribution (per-token)
2. Top-2 gaps (logit₁ - logit₂) - curvature to deform
3. Variance across tokens
4. Expert usage patterns

Goal: Establish baseline landscape before testing deformation.
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

    Switch Transformers: Block 0 is dense, Blocks 1+ have MoE routing.
    Default layer_idx=1 (first MoE layer).

    Returns:
        routing_probs: [seq_len, num_experts]
        router_logits: [seq_len, num_experts]
    """
    # Storage for captured routing
    captured_router_logits = []

    def hook_fn(module, input, output):
        """Capture router logits during forward pass."""
        # Router returns (router_probs, selected_experts)
        # But we need to access the router.classifier output
        hidden_states = input[0]  # [batch, seq_len, hidden_dim]

        # Get router logits directly
        with torch.no_grad():
            router_logits = module.classifier(hidden_states)  # [batch, seq_len, num_experts]
            captured_router_logits.append(router_logits.detach())

    # Find the router in the specified layer
    # Layer 0 is dense, layer 1+ have MoE
    if layer_idx == 0:
        raise ValueError("Layer 0 is dense (no router). Use layer_idx >= 1 for MoE layers.")

    target_block = model.encoder.block[layer_idx]
    router_module = target_block.layer[-1].mlp.router

    # Register hook
    handle = router_module.register_forward_hook(hook_fn)

    try:
        # Run forward pass
        decoder_input_ids = inputs['input_ids'].clone()

        with torch.no_grad():
            _ = model(
                input_ids=inputs['input_ids'],
                decoder_input_ids=decoder_input_ids,
            )

        # Extract captured logits
        if len(captured_router_logits) == 0:
            raise ValueError("No router logits captured")

        logits = captured_router_logits[0]  # [batch, seq_len, num_experts]

        # Remove batch dimension
        logits = logits.squeeze(0)  # [seq_len, num_experts]

        # Compute probabilities
        probs = torch.softmax(logits, dim=-1)

        return probs.detach().cpu().numpy(), logits.detach().cpu().numpy()

    finally:
        # Clean up hook
        handle.remove()


def analyze_routing_structure(routing_probs, router_logits):
    """
    Analyze routing structure in detail.

    Returns dict with:
    - per-token entropy
    - mean/std entropy
    - top-2 gaps
    - expert usage distribution
    """
    seq_len, num_experts = routing_probs.shape

    # 1. Per-token entropy
    entropies = []
    for t in range(seq_len):
        p = routing_probs[t]
        H = -np.sum(p * np.log(p + 1e-10))
        entropies.append(H)

    entropies = np.array(entropies)

    # 2. Top-2 gaps (curvature measure)
    top2_gaps = []
    for t in range(seq_len):
        logits = router_logits[t]
        sorted_logits = np.sort(logits)[::-1]
        gap = sorted_logits[0] - sorted_logits[1]  # logit₁ - logit₂
        top2_gaps.append(gap)

    top2_gaps = np.array(top2_gaps)

    # 3. Expert usage (averaged over sequence)
    expert_usage = routing_probs.mean(axis=0)

    # 4. Per-token top expert
    top_experts = np.argmax(routing_probs, axis=1)
    expert_frequencies = np.bincount(top_experts, minlength=num_experts) / seq_len

    return {
        'entropy_per_token': entropies,
        'entropy_mean': float(np.mean(entropies)),
        'entropy_std': float(np.std(entropies)),
        'entropy_max': float(np.log(num_experts)),
        'top2_gaps': top2_gaps,
        'top2_gap_mean': float(np.mean(top2_gaps)),
        'top2_gap_std': float(np.std(top2_gaps)),
        'expert_usage_mean': expert_usage,
        'expert_frequencies': expert_frequencies,
        'num_experts': num_experts,
        'seq_len': seq_len,
    }


def main():
    print("="*70)
    print("SWITCH-BASE-8 BASELINE: Characterize Routing Structure")
    print("="*70)
    print()

    if not HAS_TRANSFORMERS:
        print("ERROR: transformers not available")
        return

    # Load model
    print("Loading google/switch-base-8...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/switch-base-8",
        device_map="cpu",  # Use CPU for now (slower but safer)
    )
    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
    print("✓ Model loaded\n")

    # Test texts (coherent, not random)
    texts = {
        'A': "The cat sat on the mat. It was a sunny day.",
        'B': "The dog lay on the rug. It was a cloudy day.",
        'C': "Machine learning models process data efficiently.",
    }

    print("Test texts:")
    for label, text in texts.items():
        print(f"  {label}: \"{text}\"")
    print()

    # Analyze each text
    results = {}

    for label, text in texts.items():
        print(f"{'='*70}")
        print(f"ANALYZING TEXT {label}")
        print(f"{'='*70}\n")

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        print(f"Tokens ({len(tokens)}): {tokens[:10]}...")
        print()

        # Extract routing (layer 1 - first MoE layer)
        routing_probs, router_logits = extract_switch_routing(
            model, inputs, layer_idx=1
        )

        # Analyze structure
        analysis = analyze_routing_structure(routing_probs, router_logits)

        print(f"Routing Structure:")
        print(f"  Entropy: {analysis['entropy_mean']:.4f} ± {analysis['entropy_std']:.4f}")
        print(f"  Max possible: {analysis['entropy_max']:.4f}")
        print(f"  Fraction: {analysis['entropy_mean']/analysis['entropy_max']:.2%}")
        print()

        print(f"  Top-2 gap: {analysis['top2_gap_mean']:.4f} ± {analysis['top2_gap_std']:.4f}")
        print(f"    (Higher = sharper routing = more curvature)")
        print()

        # Expert usage
        print(f"  Expert usage (top 5):")
        top5_experts = np.argsort(analysis['expert_frequencies'])[::-1][:5]
        for i, expert_idx in enumerate(top5_experts):
            freq = analysis['expert_frequencies'][expert_idx]
            print(f"    {i+1}. Expert {expert_idx}: {freq:.2%}")
        print()

        # Per-token entropy distribution
        print(f"  Entropy distribution:")
        print(f"    Min: {np.min(analysis['entropy_per_token']):.4f}")
        print(f"    25%: {np.percentile(analysis['entropy_per_token'], 25):.4f}")
        print(f"    50%: {np.percentile(analysis['entropy_per_token'], 50):.4f}")
        print(f"    75%: {np.percentile(analysis['entropy_per_token'], 75):.4f}")
        print(f"    Max: {np.max(analysis['entropy_per_token']):.4f}")
        print()

        results[label] = analysis

    # Summary
    print("="*70)
    print("SUMMARY: Baseline Routing Characteristics")
    print("="*70)
    print()

    print(f"{'Text':<6} {'Entropy':<20} {'Top-2 Gap':<20} {'Structure':<10}")
    print("-" * 70)

    for label, analysis in results.items():
        entropy = analysis['entropy_mean']
        entropy_max = analysis['entropy_max']
        entropy_frac = entropy / entropy_max

        gap = analysis['top2_gap_mean']

        if entropy_frac < 0.5:
            structure = "Strong"
        elif entropy_frac < 0.7:
            structure = "Moderate"
        else:
            structure = "Weak"

        print(f"{label:<6} {entropy:.4f} / {entropy_max:.4f}  "
              f"{gap:.4f} ± {analysis['top2_gap_std']:.4f}   {structure:<10}")

    print()

    # Interpretation
    print("="*70)
    print("INTERPRETATION")
    print("="*70)
    print()

    mean_entropy = np.mean([r['entropy_mean'] for r in results.values()])
    mean_gap = np.mean([r['top2_gap_mean'] for r in results.values()])

    entropy_frac = mean_entropy / np.log(8)

    print(f"Average routing entropy: {mean_entropy:.4f} ({entropy_frac:.1%} of max)")
    print(f"Average top-2 gap: {mean_gap:.4f}")
    print()

    if entropy_frac < 0.5:
        print("✓ STRONG STRUCTURE DETECTED")
        print("  Routing is concentrated with clear expert specialization.")
        print("  There is significant curvature to deform.")
        print()
        print("Proceed to self-wear test (A×N → A′).")
    elif entropy_frac < 0.7:
        print("~ MODERATE STRUCTURE")
        print("  Routing has some concentration but is not sharp.")
        print("  Wear may exist but be second-order.")
        print()
        print("Proceed with caution - track variance metrics.")
    else:
        print("⚠ WEAK STRUCTURE")
        print("  Routing is still relatively flat.")
        print("  May need different text domain or longer sequences.")
        print()
        print("Consider alternative texts before proceeding.")

    print()

    # Save results
    output_dir = Path(__file__).parent.parent / "takens_data"
    output_dir.mkdir(exist_ok=True)

    results_serializable = {}
    for label, analysis in results.items():
        results_serializable[label] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in analysis.items()
        }

    with open(output_dir / "switch_baseline_structure.json", 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"Results saved to: {output_dir / 'switch_baseline_structure.json'}")
    print()

    print("="*70)
    print("BASELINE COMPLETE")
    print("="*70)
    print()
    print("Next: Implement self-wear test (A×N → A′)")
    print()


if __name__ == '__main__':
    main()
