#!/usr/bin/env python3
"""
η=0.02 Retuning Test

Tests whether increasing η from 0.015 → 0.02 recovers the stable basin
under responsive EMA-based geological memory.

Hypothesis: The operating regime shifted when we corrected expert-usage
dynamics from fossilized counters to responsive EMA. Faster geological
adaptation (higher η) may recover beneficial regime.

Expected: If η=0.02 recovers ≥50% robustness with negative Δloss,
confirms that basin exists but requires retuning under corrected dynamics.
"""

import sys
import os

# Add both src and experiments to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch
import numpy as np
from pathlib import Path

from chronomoe.mixtral_core import MixtralConfig
from chronomoe.chronovisor_mixtral_bridge import ChronovisorMixtralForCausalLM
from generate_long_geeky_conversations import LongGeekyConversationGenerator
from analyze_turn_usage import TurnUsageAnalyzer


class LongConversationDataset:
    """Simple dataset wrapper for long conversations with turn boundaries."""

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def run_training(config, enable_chronovisor, eta, pressure_scale, seed, num_conversations=10, num_epochs=50):
    """Run training - matches ablation_study.py exactly."""
    torch.manual_seed(seed + 1000)
    np.random.seed(seed + 1000)

    model = ChronovisorMixtralForCausalLM(config)

    if enable_chronovisor:
        controller = model.model.controller
        controller.eta_structural_T_local = eta
        controller.eta_structural_T_global = eta / 2.0
        controller.pressure_scale = pressure_scale

        for lens in controller.lenses.values():
            lens.eta_structural_T = eta

    print(f"  Generating {num_conversations} long conversations...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = LongGeekyConversationGenerator(vocab_size=config.vocab_size, min_length=500, max_length=1000)
    dataset_dict = gen.generate_dataset(num_conversations=num_conversations)
    conversations = dataset_dict['sequences']

    print(f"  Average conversation length: {dataset_dict['avg_length']:.0f} tokens")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dataset = LongConversationDataset(conversations)

    model.train()
    losses = []
    tbar_vars = []

    print(f"  Training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        for conv_idx in range(len(dataset)):
            batch = dataset[conv_idx]

            input_ids = torch.from_numpy(batch["input_ids"]).long().unsqueeze(0)
            labels = torch.from_numpy(batch["labels"]).long().unsqueeze(0)

            optimizer.zero_grad()
            logits, chrono_state = model(input_ids, update_chronovisor=enable_chronovisor)

            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            loss = torch.nn.functional.cross_entropy(logits_flat, labels_flat, ignore_index=-100)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if chrono_state and hasattr(chrono_state, 'T_bar') and chrono_state.T_bar is not None:
                T_bar = chrono_state.T_bar
                if isinstance(T_bar, np.ndarray) and len(T_bar) > 0:
                    tbar_vars.append(T_bar.var())
                else:
                    tbar_vars.append(0.0)
            else:
                tbar_vars.append(0.0)

    # Post-training analysis with correct separation metric
    model.eval()
    analyzer = TurnUsageAnalyzer(model, num_turns=7)

    for conv_idx in range(len(dataset)):
        batch = dataset[conv_idx]
        batch_expanded = {
            "input_ids": torch.from_numpy(batch["input_ids"]).long().unsqueeze(0),
            "labels": torch.from_numpy(batch["labels"]).long().unsqueeze(0),
            "turn_boundaries": [batch["turn_boundaries"]]
        }
        analyzer.analyze_batch(batch_expanded)

    usage_matrix = analyzer.get_usage_matrix(layer_idx=0)
    usage_normalized = usage_matrix / (usage_matrix.sum(axis=1, keepdims=True) + 1e-9)
    expert_variance = usage_normalized.var(axis=0)
    turn_separation = expert_variance.sum()

    final_loss = np.mean(losses[-50:])
    final_tbar_var = tbar_vars[-1] if tbar_vars else 0.0

    return {
        'final_loss': final_loss,
        'final_sep': turn_separation,
        'T_bar_variance': final_tbar_var,
        'losses': losses,
    }


def main():
    print("=" * 70)
    print("η=0.02 RETUNING TEST: Responsive Memory Calibration")
    print("=" * 70)
    print()
    print("Testing whether η=0.02 recovers stable basin under corrected")
    print("EMA-based geological dynamics (vs η=0.015 which degraded).")
    print()

    config = MixtralConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=1024,
        num_layers=2,
        num_experts=8,
        num_experts_per_token=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        max_seq_length=2048,
        enable_chronovisor=True,
    )

    # NEW PARAMETER: η=0.02 (was 0.015)
    eta = 0.02
    pressure_scale = 0.5
    seeds = [42, 12345]

    print(f"Configuration:")
    print(f"  Layers: {config.num_layers}")
    print(f"  Experts: {config.num_experts}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  η (ADJUSTED): {eta} (was 0.015)")
    print(f"  Pressure scale: {pressure_scale}")
    print(f"  Seeds: {seeds}")
    print()

    # Run frozen baseline
    print("=" * 70)
    print("FROZEN BASELINE (no Chronovisor)")
    print("=" * 70)

    frozen_config = MixtralConfig(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        intermediate_dim=config.intermediate_dim,
        num_layers=config.num_layers,
        num_experts=config.num_experts,
        num_experts_per_token=config.num_experts_per_token,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_seq_length=config.max_seq_length,
        enable_chronovisor=False,
    )

    frozen_result = run_training(
        frozen_config,
        enable_chronovisor=False,
        eta=0.0,
        pressure_scale=0.0,
        seed=42,
        num_conversations=10,
        num_epochs=50
    )

    print(f"\nFrozen baseline:")
    print(f"  Final loss: {frozen_result['final_loss']:.6f}")
    print(f"  Final separation: {frozen_result['final_sep']:.6f}")
    print()

    # Run Chronovisor with η=0.02
    print("=" * 70)
    print("CHRONOVISOR WITH η=0.02, P=0.5")
    print("=" * 70)

    results = []
    for seed in seeds:
        print(f"\nSeed {seed}:")
        result = run_training(
            config,
            enable_chronovisor=True,
            eta=eta,
            pressure_scale=pressure_scale,
            seed=seed,
            num_conversations=10,
            num_epochs=50
        )

        # Compute deltas
        delta_loss = (result['final_loss'] - frozen_result['final_loss']) / frozen_result['final_loss'] * 100

        if frozen_result['final_sep'] > 0:
            delta_sep = (result['final_sep'] - frozen_result['final_sep']) / frozen_result['final_sep'] * 100
        else:
            delta_sep = result['final_sep'] * 100

        result['delta_loss'] = delta_loss
        result['delta_sep'] = delta_sep
        result['seed'] = seed
        results.append(result)

        print(f"  Final loss: {result['final_loss']:.6f} (Δ: {delta_loss:+.2f}%)")
        print(f"  Final separation: {result['final_sep']:.6f} (Δ: {delta_sep:+.2f}%)")
        print(f"  T̄ variance: {result['T_bar_variance']:.6f}")

        is_pareto = delta_loss < 0 and delta_sep > 0
        print(f"  Pareto-better: {'✓' if is_pareto else '✗'}")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    delta_losses = [r['delta_loss'] for r in results]
    delta_seps = [r['delta_sep'] for r in results]
    T_bar_vars = [r['T_bar_variance'] for r in results]

    pareto_count = sum(1 for r in results if r['delta_loss'] < 0 and r['delta_sep'] > 0)
    robustness = pareto_count / len(results) * 100

    print(f"\nη=0.02 Results:")
    print(f"  Robustness: {pareto_count}/{len(results)} seeds ({robustness:.0f}%)")
    print(f"  Δ Loss: {np.mean(delta_losses):+.2f}% ± {np.std(delta_losses):.2f}%")
    print(f"  Δ Sep: {np.mean(delta_seps):+.2f}% ± {np.std(delta_seps):.2f}%")
    print(f"  T̄ variance: {np.mean(T_bar_vars):.6f} ± {np.std(T_bar_vars):.6f}")

    # Comparison
    print()
    print("COMPARISON TO η=0.015 (Previous Test):")
    print("-" * 70)
    print("Metric              η=0.015       η=0.02        Outcome")
    print("-" * 70)

    eta_015_results = {
        'robustness': 0.0,
        'delta_loss': 4.44,
        'delta_sep': -15.38,
        'T_bar_var': 0.002130,
    }

    def improvement_status(old_val, new_val, metric_type):
        if metric_type == 'robustness':
            if new_val >= 50:
                return "✓ RECOVERED"
            elif new_val > old_val:
                return "~ BETTER"
            else:
                return "✗ SAME"
        elif metric_type == 'delta_loss':
            if new_val < 0:
                return "✓ IMPROVED"
            elif new_val < old_val:
                return "~ BETTER"
            else:
                return "✗ WORSE"
        elif metric_type == 'delta_sep':
            if new_val > 0:
                return "✓ IMPROVED"
            elif new_val > old_val:
                return "~ BETTER"
            else:
                return "✗ WORSE"
        else:  # T_bar_var
            if abs(new_val - 0.002) < abs(old_val - 0.002):
                return "✓ CLOSER"
            else:
                return "~ SAME"

    rob_status = improvement_status(eta_015_results['robustness'], robustness, 'robustness')
    loss_status = improvement_status(eta_015_results['delta_loss'], np.mean(delta_losses), 'delta_loss')
    sep_status = improvement_status(eta_015_results['delta_sep'], np.mean(delta_seps), 'delta_sep')
    tbar_status = improvement_status(eta_015_results['T_bar_var'], np.mean(T_bar_vars), 'T_bar_var')

    print(f"Robustness         {eta_015_results['robustness']:6.0f}%       {robustness:6.0f}%       {rob_status}")
    print(f"Δ Loss             {eta_015_results['delta_loss']:+6.2f}%       {np.mean(delta_losses):+6.2f}%       {loss_status}")
    print(f"Δ Sep              {eta_015_results['delta_sep']:+6.2f}%       {np.mean(delta_seps):+6.2f}%       {sep_status}")
    print(f"T̄ var             {eta_015_results['T_bar_var']:7.4f}      {np.mean(T_bar_vars):7.4f}      {tbar_status}")
    print("-" * 70)

    # Verdict
    print()
    basin_recovered = robustness >= 50 and np.mean(delta_losses) < 0
    partial_recovery = robustness >= 50 or np.mean(delta_losses) < 0

    if basin_recovered:
        print("✓ BASIN RECOVERED AT η=0.02")
        print("  Responsive memory requires faster geological adaptation.")
        print("  Operating point confirmed: η=0.02, P=0.5")
        print()
        print("Next: Proceed to scaling test (4L/16E) with η=0.02")
    elif partial_recovery:
        print("~ PARTIAL RECOVERY")
        print("  Some improvement but not full basin.")
        print("  May need further tuning or different pressure scale.")
        print()
        print("Options:")
        print("  A. Try η=0.025 (push harder)")
        print("  B. Try η=0.01 (opposite direction)")
        print("  C. Adjust pressure (P=0.3 or P=0.7)")
    else:
        print("⚠ NO RECOVERY")
        print("  η adjustment alone doesn't recover basin.")
        print("  Basin may require multi-parameter retuning or different approach.")
        print()
        print("Recommendation: Document regime shift and proceed with new baseline.")

    print()
    print("=" * 70)

    # Save results
    output_dir = Path(__file__).parent.parent / "eta_020_test_results"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "eta_020_summary.txt"
    with open(output_file, 'w') as f:
        f.write(f"η=0.02 Retuning Test Results\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(f"Configuration: η={eta}, P={pressure_scale}\n")
        f.write(f"Seeds: {seeds}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Robustness: {pareto_count}/{len(results)} ({robustness:.0f}%)\n")
        f.write(f"  Δ Loss: {np.mean(delta_losses):+.2f}% ± {np.std(delta_losses):.2f}%\n")
        f.write(f"  Δ Sep: {np.mean(delta_seps):+.2f}% ± {np.std(delta_seps):.2f}%\n")
        f.write(f"  T̄ variance: {np.mean(T_bar_vars):.6f} ± {np.std(T_bar_vars):.6f}\n\n")
        f.write(f"Status: {'RECOVERED' if basin_recovered else ('PARTIAL' if partial_recovery else 'NO RECOVERY')}\n")

    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
