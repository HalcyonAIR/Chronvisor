#!/usr/bin/env python3
"""
Continuity Check: Re-validate stable basin post-EMA changes.

Validates that η=0.015, P=0.5 still achieves stable behavior with current code.
Compares against validated baseline from ablation study.

Expected results (from ablation study):
- Robustness: 3/3 seeds (100%)
- Δ Loss: -0.4% ± 0.3%
- Δ Sep: +6.9% ± 2.0%
- T̄ variance: ~0.002

This test uses 2 seeds for quick continuity verification.
"""

import sys
import os

# Add both src and experiments to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch
import numpy as np
from pathlib import Path
import time

from chronomoe.mixtral_core import MixtralConfig
from chronomoe.chronovisor_mixtral_bridge import ChronovisorMixtralForCausalLM

# Import conversation generator and analyzer
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
    """Run training with specified configuration - MATCHES ablation_study.py exactly."""
    torch.manual_seed(seed + 1000)
    np.random.seed(seed + 1000)

    # Create model
    model = ChronovisorMixtralForCausalLM(config)

    # Configure Chronovisor if enabled
    if enable_chronovisor:
        controller = model.model.controller
        controller.eta_structural_T_local = eta
        controller.eta_structural_T_global = eta / 2.0
        controller.pressure_scale = pressure_scale

        # Also set on lenses (matching ablation_study.py line 75)
        for lens in controller.lenses.values():
            lens.eta_structural_T = eta

    # Generate long conversations
    print(f"  Generating {num_conversations} long conversations...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = LongGeekyConversationGenerator(vocab_size=config.vocab_size, min_length=500, max_length=1000)
    dataset_dict = gen.generate_dataset(num_conversations=num_conversations)
    conversations = dataset_dict['sequences']

    print(f"  Average conversation length: {dataset_dict['avg_length']:.0f} tokens")

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dataset = LongConversationDataset(conversations)

    # Metrics tracking
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

            # Track T̄ (matching ablation_study.py lines 107-114)
            if chrono_state and hasattr(chrono_state, 'T_bar') and chrono_state.T_bar is not None:
                T_bar = chrono_state.T_bar
                if isinstance(T_bar, np.ndarray) and len(T_bar) > 0:
                    tbar_vars.append(T_bar.var())
                else:
                    tbar_vars.append(0.0)
            else:
                tbar_vars.append(0.0)

    # Post-training analysis - CORRECT SEPARATION METRIC (matching ablation_study.py lines 117-132)
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

    # Final metrics
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
    print("CONTINUITY CHECK: Stable Basin Post-EMA Validation")
    print("=" * 70)
    print()
    print("Re-validating η=0.015, P=0.5 with current codebase.")
    print("Expected: 100% seed robustness, Δloss ≈ -0.4%, Δsep ≈ +7%")
    print()

    # Configuration
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

    # Stable basin parameters
    eta = 0.015
    pressure_scale = 0.5

    # Test with 2 seeds for quick continuity check
    seeds = [42, 12345]

    print(f"Configuration:")
    print(f"  Layers: {config.num_layers}")
    print(f"  Experts: {config.num_experts}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  η (structural T learning rate): {eta}")
    print(f"  Pressure scale: {pressure_scale}")
    print(f"  Seeds: {seeds}")
    print()

    # Run frozen baseline (seed 42 only)
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

    print(f"Frozen baseline:")
    print(f"  Final loss: {frozen_result['final_loss']:.6f}")
    print(f"  Final separation: {frozen_result['final_sep']:.6f}")
    print()

    # Run Chronovisor tests
    print("=" * 70)
    print("CHRONOVISOR WITH STABLE BASIN (η=0.015, P=0.5)")
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

        # Safe delta_sep calculation (avoid divide by zero)
        if frozen_result['final_sep'] > 0:
            delta_sep = (result['final_sep'] - frozen_result['final_sep']) / frozen_result['final_sep'] * 100
        else:
            # If frozen baseline has zero separation, just report absolute improvement
            delta_sep = result['final_sep'] * 100  # As percentage points

        result['delta_loss'] = delta_loss
        result['delta_sep'] = delta_sep
        result['seed'] = seed
        results.append(result)

        print(f"  Final loss: {result['final_loss']:.6f} (Δ: {delta_loss:+.2f}%)")
        print(f"  Final separation: {result['final_sep']:.6f} (Δ: {delta_sep:+.2f}%)")
        print(f"  T̄ variance: {result['T_bar_variance']:.6f}")

        # Check Pareto-better
        is_pareto = delta_loss < 0 and delta_sep > 0
        print(f"  Pareto-better: {'✓' if is_pareto else '✗'}")

    # Summary statistics
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    delta_losses = [r['delta_loss'] for r in results]
    delta_seps = [r['delta_sep'] for r in results]
    T_bar_vars = [r['T_bar_variance'] for r in results]

    pareto_count = sum(1 for r in results if r['delta_loss'] < 0 and r['delta_sep'] > 0)
    robustness = pareto_count / len(results) * 100

    print(f"\nRobustness: {pareto_count}/{len(results)} seeds ({robustness:.0f}%)")
    print(f"Δ Loss: {np.mean(delta_losses):.2f}% ± {np.std(delta_losses):.2f}%")
    print(f"Δ Sep: {np.mean(delta_seps):.2f}% ± {np.std(delta_seps):.2f}%")
    print(f"T̄ variance: {np.mean(T_bar_vars):.6f} ± {np.std(T_bar_vars):.6f}")

    # Compare against validated baseline
    print()
    print("COMPARISON TO VALIDATED BASELINE:")
    print("-" * 70)
    print("                 Expected        Current         Status")
    print("-" * 70)

    # Expected values from ablation study
    expected = {
        'robustness': 100.0,
        'delta_loss': -0.4,
        'delta_loss_std': 0.3,
        'delta_sep': 6.9,
        'delta_sep_std': 2.0,
        'T_bar_var': 0.002,
    }

    def status_check(expected_val, current_val, tolerance=0.5):
        """Check if current is within reasonable range of expected."""
        # For robustness, we have fewer seeds so accept 50-100%
        if 'robustness' in str(expected_val):
            return "✓" if current_val >= 50.0 else "⚠"
        # For other metrics, use tolerance
        diff = abs(current_val - expected_val)
        if diff <= tolerance * abs(expected_val):
            return "✓"
        elif diff <= 2 * tolerance * abs(expected_val):
            return "~"
        else:
            return "⚠"

    rob_status = status_check(expected['robustness'], robustness)
    loss_status = status_check(expected['delta_loss'], np.mean(delta_losses))
    sep_status = status_check(expected['delta_sep'], np.mean(delta_seps))
    tbar_status = status_check(expected['T_bar_var'], np.mean(T_bar_vars))

    print(f"Robustness      {expected['robustness']:6.0f}%       {robustness:6.0f}%        {rob_status}")
    print(f"Δ Loss          {expected['delta_loss']:+6.2f}%       {np.mean(delta_losses):+6.2f}%        {loss_status}")
    print(f"Δ Sep           {expected['delta_sep']:+6.2f}%       {np.mean(delta_seps):+6.2f}%        {sep_status}")
    print(f"T̄ var          {expected['T_bar_var']:7.4f}      {np.mean(T_bar_vars):7.4f}       {tbar_status}")
    print("-" * 70)

    # Overall verdict
    all_good = all([
        robustness >= 50.0,  # At least 50% with 2 seeds
        abs(np.mean(delta_losses) - expected['delta_loss']) <= 1.0,
        abs(np.mean(delta_seps) - expected['delta_sep']) <= 5.0,
        abs(np.mean(T_bar_vars) - expected['T_bar_var']) <= 0.003,
    ])

    print()
    if all_good:
        print("✓ CONTINUITY CHECK PASSED")
        print("  Stable basin behavior confirmed with current codebase.")
        print("  Results are consistent with validated baseline.")
    else:
        print("⚠ CONTINUITY CHECK: REVIEW NEEDED")
        print("  Some metrics deviate from validated baseline.")
        print("  May indicate code changes affecting behavior.")

    print()
    print("=" * 70)

    # Save results
    output_dir = Path(__file__).parent.parent / "continuity_check_results"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "continuity_check_summary.txt"
    with open(output_file, 'w') as f:
        f.write(f"Continuity Check: Stable Basin Post-EMA Validation\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(f"Configuration: η={eta}, P={pressure_scale}\n")
        f.write(f"Seeds: {seeds}\n\n")
        f.write(f"Robustness: {pareto_count}/{len(results)} ({robustness:.0f}%)\n")
        f.write(f"Δ Loss: {np.mean(delta_losses):.2f}% ± {np.std(delta_losses):.2f}%\n")
        f.write(f"Δ Sep: {np.mean(delta_seps):.2f}% ± {np.std(delta_seps):.2f}%\n")
        f.write(f"T̄ variance: {np.mean(T_bar_vars):.6f} ± {np.std(T_bar_vars):.6f}\n\n")
        f.write(f"Status: {'PASSED' if all_good else 'REVIEW NEEDED'}\n")

    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
