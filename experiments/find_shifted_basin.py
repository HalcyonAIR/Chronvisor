#!/usr/bin/env python3
"""
Find Shifted Basin: Tight 2×2 Grid Search

After correcting expert-usage dynamics (symmetric trust, EMA, fresh temperature),
the optimal operating point shifted. This tight grid finds or kills the hypothesis
that a basin exists near the original regime.

Grid:
  η ∈ {0.015, 0.03}
  P ∈ {0.3, 0.7}

Seeds: [42, 12345] (same as continuity checks)

Success criterion: ≥1 seed Pareto-better in any cell
Scoring: Best cell = highest Pareto-better count, then lowest loss variance

If found: Re-run 3-seed ablation (baseline/P-only/T-only/full P×T)
If not found: Document basin moved outside this neighborhood, decide next step
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch
import numpy as np
from pathlib import Path
from itertools import product

from chronomoe.mixtral_core import MixtralConfig
from chronomoe.chronovisor_mixtral_bridge import ChronovisorMixtralForCausalLM
from generate_long_geeky_conversations import LongGeekyConversationGenerator
from analyze_turn_usage import TurnUsageAnalyzer


class LongConversationDataset:
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

    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = LongGeekyConversationGenerator(vocab_size=config.vocab_size, min_length=500, max_length=1000)
    dataset_dict = gen.generate_dataset(num_conversations=num_conversations)
    conversations = dataset_dict['sequences']

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dataset = LongConversationDataset(conversations)

    model.train()
    losses = []
    tbar_vars = []

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

    # Post-training analysis - correct separation metric
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
    print("FIND SHIFTED BASIN: 2×2 Grid Search")
    print("=" * 70)
    print()
    print("Searching for stable basin after correcting geological dynamics.")
    print("Grid: η ∈ {0.015, 0.03} × P ∈ {0.3, 0.7}")
    print("Seeds: [42, 12345]")
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

    # Grid parameters
    eta_values = [0.015, 0.03]
    pressure_values = [0.3, 0.7]
    seeds = [42, 12345]

    print(f"Total cells: {len(eta_values) * len(pressure_values)} = {len(eta_values)} x {len(pressure_values)}")
    print(f"Runs per cell: {len(seeds)}")
    print(f"Total runs: {len(eta_values) * len(pressure_values) * len(seeds)} (+ 1 frozen baseline)")
    print()

    # Run frozen baseline ONCE
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

    # Grid search
    grid_results = {}

    for eta, pressure_scale in product(eta_values, pressure_values):
        cell_key = (eta, pressure_scale)

        print("=" * 70)
        print(f"CELL: η={eta:.3f}, P={pressure_scale:.1f}")
        print("=" * 70)

        cell_results = []

        for seed in seeds:
            print(f"\n  Seed {seed}:")
            result = run_training(
                config,
                enable_chronovisor=True,
                eta=eta,
                pressure_scale=pressure_scale,
                seed=seed,
                num_conversations=10,
                num_epochs=50
            )

            # Compute deltas (guard against divide-by-zero)
            delta_loss = (result['final_loss'] - frozen_result['final_loss']) / frozen_result['final_loss'] * 100

            if frozen_result['final_sep'] > 1e-6:
                delta_sep = (result['final_sep'] - frozen_result['final_sep']) / frozen_result['final_sep'] * 100
            else:
                # If frozen baseline has near-zero separation, use absolute improvement
                delta_sep = (result['final_sep'] - frozen_result['final_sep']) * 1000  # Scale for visibility

            result['delta_loss'] = delta_loss
            result['delta_sep'] = delta_sep
            result['seed'] = seed
            result['is_pareto'] = (delta_loss < 0 and delta_sep > 0)

            cell_results.append(result)

            print(f"    Loss: {result['final_loss']:.6f} (Δ: {delta_loss:+.2f}%)")
            print(f"    Sep: {result['final_sep']:.6f} (Δ: {delta_sep:+.2f}%)")
            print(f"    T̄ var: {result['T_bar_variance']:.6f}")
            print(f"    Pareto-better: {'✓' if result['is_pareto'] else '✗'}")

        grid_results[cell_key] = cell_results

        # Cell summary
        pareto_count = sum(1 for r in cell_results if r['is_pareto'])
        print(f"\n  Cell summary: {pareto_count}/{len(seeds)} seeds Pareto-better")

    # Overall analysis
    print()
    print("=" * 70)
    print("GRID SEARCH RESULTS")
    print("=" * 70)
    print()

    # Score each cell
    cell_scores = []
    for cell_key, cell_results in grid_results.items():
        eta, pressure = cell_key
        pareto_count = sum(1 for r in cell_results if r['is_pareto'])

        delta_losses = [r['delta_loss'] for r in cell_results]
        delta_seps = [r['delta_sep'] for r in cell_results]
        T_bar_vars = [r['T_bar_variance'] for r in cell_results]

        avg_delta_loss = np.mean(delta_losses)
        avg_delta_sep = np.mean(delta_seps)
        loss_variance = np.var(delta_losses)

        cell_scores.append({
            'eta': eta,
            'pressure': pressure,
            'pareto_count': pareto_count,
            'avg_delta_loss': avg_delta_loss,
            'avg_delta_sep': avg_delta_sep,
            'loss_variance': loss_variance,
            'avg_T_bar_var': np.mean(T_bar_vars),
        })

    # Sort by: 1) Pareto count (desc), 2) Loss variance (asc)
    cell_scores.sort(key=lambda x: (-x['pareto_count'], x['loss_variance']))

    print("Cell Rankings:")
    print("-" * 70)
    print("Rank  η      P     Pareto  Δ Loss    Δ Sep     T̄ var")
    print("-" * 70)

    for i, score in enumerate(cell_scores, 1):
        pareto_str = f"{score['pareto_count']}/{len(seeds)}"
        print(f"{i:2d}.  {score['eta']:.3f}  {score['pressure']:.1f}   {pareto_str:5s}   "
              f"{score['avg_delta_loss']:+6.2f}%  {score['avg_delta_sep']:+7.2f}%  {score['avg_T_bar_var']:.6f}")

    print("-" * 70)

    # Find best cell
    best_cell = cell_scores[0]
    basin_found = best_cell['pareto_count'] >= 1

    print()
    if basin_found:
        print(f"✓ BASIN FOUND")
        print(f"  Best cell: η={best_cell['eta']:.3f}, P={best_cell['pressure']:.1f}")
        print(f"  Pareto-better: {best_cell['pareto_count']}/{len(seeds)} seeds")
        print(f"  Avg Δ Loss: {best_cell['avg_delta_loss']:+.2f}%")
        print(f"  Avg Δ Sep: {best_cell['avg_delta_sep']:+.2f}%")
        print()
        print("RECOMMENDATION:")
        print(f"  1. Re-run 3-seed ablation at η={best_cell['eta']:.3f}, P={best_cell['pressure']:.1f}")
        print(f"  2. Validate with (baseline / P-only / T-only / full P×T)")
        print(f"  3. If ablation confirms, proceed to scaling test")
    else:
        print(f"⚠ NO BASIN FOUND")
        print(f"  Best cell: η={best_cell['eta']:.3f}, P={best_cell['pressure']:.1f}")
        print(f"  Pareto-better: {best_cell['pareto_count']}/{len(seeds)} seeds (need ≥1)")
        print(f"  Avg Δ Loss: {best_cell['avg_delta_loss']:+.2f}%")
        print(f"  Avg Δ Sep: {best_cell['avg_delta_sep']:+.2f}%")
        print()
        print("INTERPRETATION:")

        # Check if trending toward basin
        any_negative_loss = any(s['avg_delta_loss'] < 0 for s in cell_scores)
        any_positive_sep = any(s['avg_delta_sep'] > 0 for s in cell_scores)

        if any_negative_loss or any_positive_sep:
            print("  Some cells show partial improvement (negative loss OR positive sep).")
            print("  Basin may be nearby but outside this grid.")
            print()
            print("OPTIONS:")
            print("  A. Expand grid: η ∈ {0.01, 0.02, 0.03, 0.04}, P ∈ {0.2, 0.5, 0.8}")
            print("  B. Follow gradient: Test η=0.04, P=0.3 (extrapolate best trend)")
            print("  C. Document: Basin moved outside local neighborhood, broaden search")
        else:
            print("  No cells show improvement in either metric.")
            print("  Basin likely doesn't exist in this parameter regime.")
            print()
            print("RECOMMENDATION:")
            print("  Document regime shift honestly and either:")
            print("  1. Broaden search significantly (full grid sweep)")
            print("  2. Proceed without P×T coupling validation")
            print("  3. Investigate whether EMA correction requires different approach")

    print()
    print("=" * 70)

    # Save results
    output_dir = Path(__file__).parent.parent / "grid_search_results"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "2x2_grid_summary.txt"
    with open(output_file, 'w') as f:
        f.write("2×2 Grid Search Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Grid: η ∈ {eta_values}, P ∈ {pressure_values}\n")
        f.write(f"Seeds: {seeds}\n\n")
        f.write("Cell Rankings:\n")
        f.write("-" * 70 + "\n")
        for i, score in enumerate(cell_scores, 1):
            f.write(f"{i}. η={score['eta']:.3f}, P={score['pressure']:.1f}: "
                   f"{score['pareto_count']}/{len(seeds)} Pareto-better, "
                   f"Δloss={score['avg_delta_loss']:+.2f}%, "
                   f"Δsep={score['avg_delta_sep']:+.2f}%\n")
        f.write("\n")
        f.write(f"Basin found: {basin_found}\n")
        if basin_found:
            f.write(f"Best cell: η={best_cell['eta']:.3f}, P={best_cell['pressure']:.1f}\n")

    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
