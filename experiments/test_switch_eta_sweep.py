#!/usr/bin/env python3
"""
η Sweep: Tuning the Geological Gain

Following Halcyon's mic preamp analogy:
- Band A (linear regime): η = 0.0025, 0.005, 0.01
- Band B (amplification): η = 0.015, 0.02, 0.03
- Band C (stress test): η = 0.05, 0.075, 0.10

Plus η=0.0 as sanity anchor.

Fixed: n_repetitions = 100 (no time compression confound)

Looking for:
1. Linear scaling in Band A
2. Clear detection in Band B
3. Saturation/oscillation in Band C
4. Smooth monotone entropy decrease (wear) vs oscillation (too hot)
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
    """Extract routing from Switch Transformer (captures ChronoMoE bias)."""
    captured_router_outputs = []

    def hook_fn(module, input, output):
        router_mask, router_probs, router_logits = output
        captured_router_outputs.append(router_logits.detach())

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


def analyze_self_wear(routing_A0, routing_An):
    """Analyze how A's routing changed."""
    avg_A0 = routing_A0.mean(axis=0)
    avg_An = routing_An.mean(axis=0)

    kl_div = compute_kl_divergence(avg_An, avg_A0)
    entropy_A0 = compute_entropy(avg_A0)
    entropy_An = compute_entropy(avg_An)
    delta_entropy = entropy_An - entropy_A0

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
        'token_entropy_mean_A0': float(np.mean(token_entropies_A0)),
        'token_entropy_mean_An': float(np.mean(token_entropies_An)),
        'token_entropy_std_A0': float(np.std(token_entropies_A0)),
        'token_entropy_std_An': float(np.std(token_entropies_An)),
    }


def run_eta_experiment(model, tokenizer, inputs, eta, n_repetitions=100, layer_idx=1):
    """
    Run single η experiment.

    Returns:
        results dict with T̄ drift, ΔKL, ΔEntropy, etc.
    """
    print(f"  η = {eta:.4f}")

    # Wrap model with ChronoMoE at this η
    # First remove any existing wrappers (reset)
    # (In practice we'd need to reload model, but for now we'll reset adapters)

    # Get adapter (should already exist from initial wrapping)
    adapter = model._chronovisor_adapters.get(layer_idx)
    if adapter is None:
        raise ValueError(f"No adapter found for layer {layer_idx}")

    # Update η
    adapter.lens.eta_structural_T = eta

    # Reset adapter
    adapter.reset()

    if eta == 0.0:
        # Sanity anchor: no adaptation
        adapter.disable_all()
    else:
        # Enable adaptation
        adapter.enable_adaptation()

    # Measure baseline
    routing_probs_A0, _ = extract_switch_routing(model, inputs, layer_idx=layer_idx)

    # Run A×N
    initial_T_bar = adapter.lens.T_bar.copy()

    for i in range(n_repetitions):
        with torch.no_grad():
            decoder_input_ids = inputs['input_ids'].clone()
            _ = model(
                input_ids=inputs['input_ids'],
                decoder_input_ids=decoder_input_ids,
            )

    final_T_bar = adapter.lens.T_bar
    T_bar_drift = np.mean(np.abs(final_T_bar - initial_T_bar))

    # Measure after
    routing_probs_An, _ = extract_switch_routing(model, inputs, layer_idx=layer_idx)

    # Analyze
    analysis = analyze_self_wear(routing_probs_A0, routing_probs_An)

    # Check for oscillation (entropy should decrease smoothly for wear)
    # Track if entropy is oscillating by checking variance of per-token entropies
    entropy_variance_ratio = analysis['token_entropy_std_An'] / (analysis['token_entropy_std_A0'] + 1e-10)

    return {
        'eta': float(eta),
        'T_bar_drift': float(T_bar_drift),
        'T_bar_range': [float(final_T_bar.min()), float(final_T_bar.max())],
        'kl_divergence': analysis['kl_divergence'],
        'delta_entropy': analysis['delta_entropy'],
        'token_delta_entropy': analysis['token_entropy_mean_An'] - analysis['token_entropy_mean_A0'],
        'entropy_variance_ratio': float(entropy_variance_ratio),
        'stability': 'smooth' if entropy_variance_ratio < 1.2 else 'oscillating',
    }


def main():
    print("="*70)
    print("η SWEEP: Tuning the Geological Gain")
    print("="*70)
    print()
    print("Following Halcyon's mic preamp analogy:")
    print("  Band A (linear regime):  η = 0.0, 0.0025, 0.005, 0.01")
    print("  Band B (amplification):  η = 0.015, 0.02, 0.03")
    print("  Band C (stress test):    η = 0.05, 0.075, 0.10")
    print()
    print(f"Fixed: n_repetitions = 100")
    print()

    if not HAS_TRANSFORMERS:
        print("ERROR: transformers not available")
        return

    # Load model once
    print("Loading google/switch-base-8...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/switch-base-8",
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
    print("✓ Model loaded\n")

    # Wrap with ChronoMoE (will be reconfigured for each η)
    print("Wrapping model with ChronoMoE adapters...")
    adapters = wrap_switch_model_with_chronovisor(
        model,
        layer_indices=[1],
        eta_structural_T=0.015,  # Default, will be overridden
    )

    # Store adapters on model for easy access
    model._chronovisor_adapters = adapters
    print()

    # Test input
    text_A = "The cat sat on the mat. It was a sunny day."
    inputs = tokenizer(text_A, return_tensors="pt")

    # η values to sweep
    eta_values = [
        # Band A: Linear regime
        0.0,     # Sanity anchor
        0.0025,
        0.005,
        0.01,
        # Band B: Amplification
        0.015,   # Current baseline
        0.02,
        0.03,
        # Band C: Stress test
        0.05,
        0.075,
        0.10,
    ]

    # Run sweep
    results = []

    print("="*70)
    print("RUNNING SWEEP")
    print("="*70)
    print()

    for eta in eta_values:
        result = run_eta_experiment(
            model, tokenizer, inputs, eta,
            n_repetitions=100,
            layer_idx=1
        )
        results.append(result)

        # Print summary
        print(f"    T̄ drift:    {result['T_bar_drift']:.6f}")
        print(f"    ΔKL:        {result['kl_divergence']:+.6f}")
        print(f"    ΔEntropy:   {result['delta_entropy']:+.6f}")
        print(f"    Stability:  {result['stability']}")
        print()

    # Analysis
    print("="*70)
    print("ANALYSIS")
    print("="*70)
    print()

    # Table
    print(f"{'η':<8} {'T̄ drift':<12} {'ΔKL':<12} {'ΔEntropy':<12} {'Stability':<12}")
    print("-"*70)

    for r in results:
        print(f"{r['eta']:<8.4f} {r['T_bar_drift']:<12.6f} {r['kl_divergence']:<+12.6f} "
              f"{r['delta_entropy']:<+12.6f} {r['stability']:<12}")

    print()

    # Band analysis
    band_a = [r for r in results if 0.0 <= r['eta'] <= 0.01]
    band_b = [r for r in results if 0.015 <= r['eta'] <= 0.03]
    band_c = [r for r in results if 0.05 <= r['eta'] <= 0.10]

    print("Band A (linear regime):")
    if len(band_a) > 1:
        # Check linearity: ΔKL vs η
        eta_a = [r['eta'] for r in band_a if r['eta'] > 0]
        kl_a = [r['kl_divergence'] for r in band_a if r['eta'] > 0]

        if len(eta_a) > 1:
            # Fit line
            slope_kl = (kl_a[-1] - kl_a[0]) / (eta_a[-1] - eta_a[0]) if eta_a[-1] != eta_a[0] else 0

            print(f"  ΔKL/η slope: {slope_kl:.4f}")
            print(f"  Linearity: {'good' if all(r['stability'] == 'smooth' for r in band_a) else 'unstable'}")
    print()

    print("Band B (amplification):")
    if len(band_b) > 0:
        max_kl = max(r['kl_divergence'] for r in band_b)
        print(f"  Max ΔKL: {max_kl:.6f}")
        print(f"  Detection: {'YES' if max_kl > 0.01 else 'weak signal'}")
    print()

    print("Band C (stress test):")
    if len(band_c) > 0:
        oscillating = sum(1 for r in band_c if r['stability'] == 'oscillating')
        print(f"  Oscillating: {oscillating}/{len(band_c)}")

        # Check if ΔKL saturates
        kl_c = [r['kl_divergence'] for r in band_c]
        if len(kl_c) > 1:
            kl_growth = (kl_c[-1] - kl_c[0]) / (band_c[-1]['eta'] - band_c[0]['eta'])
            print(f"  ΔKL growth rate: {kl_growth:.4f}")
            print(f"  Saturation: {'YES' if kl_growth < 0.01 else 'NO (still linear)'}")
    print()

    # Recommendations
    print("="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print()

    # Find optimal η (max signal without oscillation)
    smooth_results = [r for r in results if r['stability'] == 'smooth' and r['eta'] > 0]
    if smooth_results:
        optimal = max(smooth_results, key=lambda r: abs(r['kl_divergence']))
        print(f"Optimal η (max signal, stable): {optimal['eta']:.4f}")
        print(f"  ΔKL: {optimal['kl_divergence']:.6f}")
        print(f"  ΔEntropy: {optimal['delta_entropy']:.6f}")
        print()

    # Detection threshold
    detected = [r for r in results if abs(r['kl_divergence']) > 0.01 and r['eta'] > 0]
    if detected:
        print(f"✓ Detection achieved at η ≥ {min(r['eta'] for r in detected):.4f}")
    else:
        print("⚠ No η value achieved detection threshold (ΔKL > 0.01)")
        print(f"  Max ΔKL: {max(r['kl_divergence'] for r in results):.6f}")
        print("  Consider: longer n_repetitions or different regime")

    print()

    # Save results
    output_dir = Path(__file__).parent.parent / "takens_data"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "switch_eta_sweep_results.json", 'w') as f:
        json.dump({
            'eta_values': eta_values,
            'n_repetitions': 100,
            'results': results,
            'text': text_A,
        }, f, indent=2)

    print(f"Results saved to: {output_dir / 'switch_eta_sweep_results.json'}")
    print()

    print("="*70)
    print("SWEEP COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
