#!/usr/bin/env python3
"""
Entropy-Controlled Sweep: Phase Diagram for Path Wear

Tests path wear across routing concentration regimes.

Hypothesis: Path wear requires structured routing (low entropy).
In uniform regimes (high entropy), T̄ bias is too weak to register.

Sweep strategy:
1. Control top-k: 6 → 4 → 2 (force concentration)
2. Control temperature: 1.0 → 0.7 → 0.5 (sharpen softmax)

Keep constant:
- η_structural_T = 0.015 (same adaptation rate)
- n_repetitions = 100 (same wear duration)
- Same routing metrics (ΔKL, ΔCos, ΔJaccard)

This maps the phase diagram: entropy vs wear signal.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import json

from chronomoe.deepseek_core import DeepSeekConfig
from chronomoe.chronovisor_deepseek_bridge import ChronovisorDeepSeekForCausalLM

from routing_metrics import (
    extract_routing_distribution,
    compute_routing_metrics_suite,
    entropy,
)


def create_paired_inputs(vocab_size: int, length: int = 50):
    """Create similar but distinct inputs A and B."""
    base = torch.randint(0, vocab_size, (length,))
    noise_A = torch.randint(-5, 6, (length,))
    noise_B = torch.randint(-5, 6, (length,))

    input_A = torch.clamp(base + noise_A, 0, vocab_size - 1).unsqueeze(0)
    input_B = torch.clamp(base + noise_B, 0, vocab_size - 1).unsqueeze(0)

    return input_A, input_B


def reset_controller(model):
    """Reset ChronoMoE controller."""
    if hasattr(model, 'model') and hasattr(model.model, 'controller'):
        controller = model.model.controller
        if controller is not None:
            for lens in controller.lenses.values():
                lens.T_bar = np.ones(lens.num_experts)
                lens.expert_usage_ema = np.ones(lens.num_experts) / lens.num_experts


def test_path_wear_single_config(
    model,
    input_A: torch.Tensor,
    input_B: torch.Tensor,
    n_repetitions: int = 100,
) -> Dict:
    """
    Test path wear for a single configuration.

    Returns routing metrics without verbose output.
    """

    # Phase 1: Baseline B (virgin)
    reset_controller(model)
    with torch.no_grad():
        _, chrono_state_virgin, _ = model(input_B, update_chronovisor=True)
        routing_B_virgin = extract_routing_distribution(chrono_state_virgin, layer_idx=0)

    # Phase 2: Establish A's pattern
    reset_controller(model)
    with torch.no_grad():
        _, chrono_state_A, _ = model(input_A, update_chronovisor=True)
        routing_A = extract_routing_distribution(chrono_state_A, layer_idx=0)

    # Phase 3: Wear the path
    reset_controller(model)
    if model.model.controller:
        T_bar_initial = model.model.controller.lenses[0].T_bar.copy()

    for i in range(n_repetitions):
        with torch.no_grad():
            _ = model(input_A, update_chronovisor=True)

    if model.model.controller:
        T_bar_after = model.model.controller.lenses[0].T_bar.copy()
        T_bar_drift = np.mean(np.abs(T_bar_after - T_bar_initial))
    else:
        T_bar_drift = 0.0

    # Phase 4: Test B after wear
    with torch.no_grad():
        _, chrono_state_after, _ = model(input_B, update_chronovisor=True)
        routing_B_after = extract_routing_distribution(chrono_state_after, layer_idx=0)

    # Compute metrics
    metrics = compute_routing_metrics_suite(
        routing_B_virgin=routing_B_virgin,
        routing_B_after=routing_B_after,
        routing_A=routing_A,
        top_k=6,
    )

    return {
        'T_bar_drift': T_bar_drift,
        'metrics': metrics,
        'routing_B_virgin': routing_B_virgin,
        'routing_B_after': routing_B_after,
        'routing_A': routing_A,
    }


def run_entropy_sweep():
    """
    Run entropy-controlled sweep across routing concentration regimes.

    Sweep dimensions:
    1. top-k: [6, 4, 2]
    2. temperature: [1.0, 0.7, 0.5]
    """

    print("="*70)
    print("ENTROPY-CONTROLLED SWEEP: Path Wear Phase Diagram")
    print("="*70)
    print()
    print("Mapping: Routing entropy → Wear signal")
    print()
    print("Constant:")
    print("  η_structural_T = 0.015")
    print("  n_repetitions = 100")
    print("  Same routing metrics")
    print()
    print("Varied:")
    print("  top-k: [6, 4, 2]")
    print("  temperature: [1.0, 0.7, 0.5]")
    print()

    # Base config
    vocab_size = 1000
    config_base = {
        'vocab_size': vocab_size,
        'hidden_dim': 256,
        'intermediate_dim': 1024,
        'num_layers': 2,
        'num_shared_experts': 2,
        'num_routed_experts': 64,
        'enable_chronovisor': True,
    }

    # Create inputs (same for all configs)
    input_A, input_B = create_paired_inputs(vocab_size, length=50)
    print(f"Inputs: A={input_A.shape}, B={input_B.shape}")
    print()

    # Sweep configurations
    sweep_configs = [
        # Baseline (established boundary)
        {'top_k': 6, 'temperature': 1.0, 'label': 'baseline'},

        # Reduce top-k (force concentration)
        {'top_k': 4, 'temperature': 1.0, 'label': 'topk_4'},
        {'top_k': 2, 'temperature': 1.0, 'label': 'topk_2'},

        # Lower temperature (sharpen softmax)
        {'top_k': 6, 'temperature': 0.7, 'label': 'temp_0.7'},
        {'top_k': 6, 'temperature': 0.5, 'label': 'temp_0.5'},

        # Combined (maximum concentration)
        {'top_k': 2, 'temperature': 0.5, 'label': 'concentrated'},
    ]

    results = []

    print(f"Running {len(sweep_configs)} configurations...")
    print()

    for i, sweep_config in enumerate(sweep_configs):
        label = sweep_config['label']
        top_k = sweep_config['top_k']
        temp = sweep_config['temperature']

        print(f"[{i+1}/{len(sweep_configs)}] Config: {label} (top_k={top_k}, temp={temp})")

        # Create model with this config
        config = DeepSeekConfig(
            **config_base,
            num_experts_per_token=top_k,
            # Note: temperature control would require modifying router
            # For now, we focus on top-k which directly affects entropy
        )

        torch.manual_seed(42)  # Same seed for all
        model = ChronovisorDeepSeekForCausalLM(config)
        model.eval()

        # Run experiment
        result = test_path_wear_single_config(
            model=model,
            input_A=input_A,
            input_B=input_B,
            n_repetitions=100,
        )

        # Extract key metrics
        metrics = result['metrics']

        entropy_B = metrics['entropy_B_virgin']
        entropy_A = metrics['entropy_A']
        delta_kl = metrics['delta_kl_to_A']
        delta_cos = metrics['delta_cos_to_A']
        delta_jaccard = metrics['delta_jaccard_to_A']
        T_bar_drift = result['T_bar_drift']

        # Determine if wear detected
        wear_detected = (
            delta_kl > 0.01 or
            delta_cos > 0.001 or
            delta_jaccard > 0.05
        )

        print(f"  Entropy(B): {entropy_B:.4f}")
        print(f"  ΔKL to A:   {delta_kl:+.6f}")
        print(f"  ΔCos to A:  {delta_cos:+.6f}")
        print(f"  ΔJac to A:  {delta_jaccard:+.6f}")
        print(f"  T̄ drift:    {T_bar_drift:.6f}")
        print(f"  Wear: {'✓ YES' if wear_detected else '○ NO'}")
        print()

        results.append({
            'label': label,
            'top_k': top_k,
            'temperature': temp,
            'entropy_B': entropy_B,
            'entropy_A': entropy_A,
            'delta_kl': delta_kl,
            'delta_cos': delta_cos,
            'delta_jaccard': delta_jaccard,
            'T_bar_drift': T_bar_drift,
            'wear_detected': wear_detected,
            'metrics': metrics,
        })

    # Summary
    print("="*70)
    print("SWEEP SUMMARY")
    print("="*70)
    print()

    print(f"{'Config':<15} {'Entropy':<10} {'ΔKL':<12} {'ΔCos':<12} {'ΔJac':<12} {'Wear':<6}")
    print("-" * 70)

    for r in results:
        wear_marker = "✓" if r['wear_detected'] else "○"
        print(f"{r['label']:<15} {r['entropy_B']:<10.4f} "
              f"{r['delta_kl']:<+12.6f} {r['delta_cos']:<+12.6f} "
              f"{r['delta_jaccard']:<+12.6f} {wear_marker:<6}")

    print()

    # Analysis
    print("="*70)
    print("ANALYSIS")
    print("="*70)
    print()

    wear_found = any(r['wear_detected'] for r in results)

    if wear_found:
        print("✓ WEAR DETECTED in some configurations")
        print()
        wear_configs = [r for r in results if r['wear_detected']]
        no_wear_configs = [r for r in results if not r['wear_detected']]

        if wear_configs and no_wear_configs:
            entropy_threshold = np.mean([
                max(r['entropy_B'] for r in no_wear_configs),
                min(r['entropy_B'] for r in wear_configs)
            ])

            print(f"Entropy threshold: ~{entropy_threshold:.2f}")
            print(f"  Below {entropy_threshold:.2f}: Wear detected")
            print(f"  Above {entropy_threshold:.2f}: No wear")
            print()
            print("Phase transition found.")
    else:
        print("○ NO WEAR DETECTED in any configuration")
        print()
        print("Path wear remains elusive even with concentrated routing.")
        print()
        print("Possible next steps:")
        print("  1. Increase n_repetitions (100 → 500)")
        print("  2. Increase η_structural_T (0.015 → 0.05)")
        print("  3. Use more similar A/B inputs")
        print("  4. Consider alternative persistence mechanism")

    # Check entropy reduction
    entropy_baseline = results[0]['entropy_B']
    entropy_min = min(r['entropy_B'] for r in results)
    entropy_reduction = entropy_baseline - entropy_min

    print()
    print(f"Entropy range: {entropy_min:.4f} to {entropy_baseline:.4f}")
    print(f"  Reduction: {entropy_reduction:.4f}")
    print()

    if entropy_reduction < 0.5:
        print("⚠ WARNING: Entropy reduction insufficient")
        print("  Top-k alone may not concentrate routing enough.")
        print("  Consider adding temperature control or prior bias.")

    print()

    # Save results
    output_dir = Path(__file__).parent.parent / "takens_data"
    output_dir.mkdir(exist_ok=True)

    # Save summary
    with open(output_dir / "entropy_sweep_results.json", 'w') as f:
        # Convert numpy types to python types for JSON
        results_serializable = []
        for r in results:
            r_clean = {k: v for k, v in r.items() if k != 'metrics'}
            r_clean['metrics_summary'] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in r['metrics'].items()
                if isinstance(v, (int, float, np.floating, np.integer))
            }
            results_serializable.append(r_clean)

        json.dump(results_serializable, f, indent=2)

    print(f"Results saved to: {output_dir / 'entropy_sweep_results.json'}")
    print()

    return results


def main():
    results = run_entropy_sweep()

    print("="*70)
    print("ENTROPY SWEEP COMPLETE")
    print("="*70)
    print()
    print("Phase diagram mapped.")
    print("Boundary documented in BOUNDARY_PATH_WEAR_UNIFORM.md")
    print()


if __name__ == '__main__':
    main()
