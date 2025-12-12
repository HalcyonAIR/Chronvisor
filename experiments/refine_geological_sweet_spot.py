"""
Refined Geological Sweet Spot Sweep

Dense grid around η=0.01, P=0.5 to map the attractor neighborhood.

Grid:
- η ∈ {0.008, 0.01, 0.012, 0.015}
- P ∈ {0.3, 0.5, 0.7, 1.0}
- 500 steps (geological timescale)
"""

import os
import torch
import numpy as np
from torch.optim import AdamW
from chronomoe.chronovisor_mixtral_bridge import ChronovisorMixtralForCausalLM
from chronomoe.mixtral_core import MixtralConfig
from experiments.conversational_dataset import ConversationalDataset
from experiments.analyze_turn_usage import TurnUsageAnalyzer, ThreeDomainDatasetPyTorch


def run_single_config(
    enable_chronovisor: bool,
    eta_local: float,
    pressure_scale: float,
    sequences,
    num_steps: int = 500,
):
    """Run single configuration and return metrics."""

    config = MixtralConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=1024,
        num_layers=2,
        num_experts=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        enable_chronovisor=enable_chronovisor,
    )

    model = ChronovisorMixtralForCausalLM(config)

    if enable_chronovisor:
        controller = model.model.controller
        controller.eta_structural_T_local = eta_local
        controller.eta_structural_T_global = eta_local / 2
        controller.pressure_scale = pressure_scale

        for lens in controller.lenses.values():
            lens.eta_structural_T = eta_local

    optimizer = AdamW(model.parameters(), lr=1e-4)
    dataset = ThreeDomainDatasetPyTorch(sequences, vocab_size=1000)

    # Training
    model.train()
    losses = []
    tbar_vars = []

    for step in range(num_steps):
        batch_idx = step % len(dataset)
        batch = dataset[batch_idx]

        input_ids = batch["input_ids"].unsqueeze(0)
        labels = batch["labels"].unsqueeze(0)

        optimizer.zero_grad()
        logits, chrono_state = model(input_ids, update_chronovisor=enable_chronovisor)

        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        loss = torch.nn.functional.cross_entropy(logits_flat, labels_flat, ignore_index=-100)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Track T̄
        if chrono_state and hasattr(chrono_state, 'T_bar') and chrono_state.T_bar is not None:
            T_bar = chrono_state.T_bar
            if isinstance(T_bar, np.ndarray) and len(T_bar) > 0:
                tbar_vars.append(T_bar.var())
            else:
                tbar_vars.append(0.0)
        else:
            tbar_vars.append(0.0)

    # Post-training analysis
    model.eval()
    analyzer = TurnUsageAnalyzer(model, num_turns=7)

    for i in range(len(dataset)):
        batch = dataset[i]
        batch_expanded = {
            "input_ids": batch["input_ids"].unsqueeze(0),
            "labels": batch["labels"].unsqueeze(0),
            "turn_boundaries": [batch.get("turn_boundaries", [])]
        }
        analyzer.analyze_batch(batch_expanded)

    usage_matrix = analyzer.get_usage_matrix(layer_idx=0)
    usage_normalized = usage_matrix / usage_matrix.sum(axis=1, keepdims=True)
    expert_variance = usage_normalized.var(axis=0)
    turn_separation = expert_variance.sum()

    # Final metrics
    final_loss = np.mean(losses[-50:])
    final_tbar_var = tbar_vars[-1] if tbar_vars else 0.0
    max_tbar_var = max(tbar_vars) if tbar_vars else 0.0

    return {
        "final_loss": final_loss,
        "turn_separation": turn_separation,
        "tbar_variance_final": final_tbar_var,
        "tbar_variance_max": max_tbar_var,
    }


def main():
    print("="*70)
    print("REFINED GEOLOGICAL SWEET SPOT SWEEP")
    print("="*70)
    print("\nDense grid around η=0.01, P=0.5 to map attractor neighborhood\n")

    # Generate dataset
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_gen = ConversationalDataset(seq_length=128, vocab_size=1000)
    result = dataset_gen.generate_dataset(num_sequences=15, balanced=False)
    sequences = result["sequences"]
    print(f"✅ Dataset generated (seed={seed})\n")

    # Dense grid around sweet spot (cartography, not optimization)
    # Wider η range to capture curvature, higher P based on seed 12345 signal
    eta_locals = [0.0075, 0.01, 0.015, 0.02]
    pressure_scales = [0.5, 0.7, 1.0, 1.5]
    num_steps = 500

    print(f"Grid: {len(eta_locals)} × {len(pressure_scales)} = {len(eta_locals) * len(pressure_scales)} configs")
    print(f"  η_local ∈ {eta_locals}")
    print(f"  pressure_scale ∈ {pressure_scales}")
    print(f"  steps = {num_steps}\n")

    # Run frozen baseline
    print("Running frozen baseline...")
    frozen = run_single_config(
        enable_chronovisor=False,
        eta_local=0.01,
        pressure_scale=0.5,
        sequences=sequences,
        num_steps=num_steps,
    )
    print(f"✅ Frozen: loss={frozen['final_loss']:.6f}, sep={frozen['turn_separation']:.6f}\n")

    # Run sweep
    results = []

    for eta_local in eta_locals:
        for pressure_scale in pressure_scales:
            print(f"Running η={eta_local:.3f}, P={pressure_scale:.1f}...", end=" ")

            live = run_single_config(
                enable_chronovisor=True,
                eta_local=eta_local,
                pressure_scale=pressure_scale,
                sequences=sequences,
                num_steps=num_steps,
            )

            delta_loss = live['final_loss'] - frozen['final_loss']
            delta_sep = live['turn_separation'] - frozen['turn_separation']
            delta_loss_pct = (delta_loss / frozen['final_loss']) * 100
            delta_sep_pct = (delta_sep / frozen['turn_separation']) * 100

            result = {
                'eta': eta_local,
                'pressure': pressure_scale,
                'loss': live['final_loss'],
                'separation': live['turn_separation'],
                'tbar_var': live['tbar_variance_final'],
                'delta_loss': delta_loss,
                'delta_loss_pct': delta_loss_pct,
                'delta_sep': delta_sep,
                'delta_sep_pct': delta_sep_pct,
            }
            results.append(result)

            print(f"Δloss={delta_loss_pct:+.1f}%, Δsep={delta_sep_pct:+.1f}%, T̄={live['tbar_variance_final']:.6f}")

    # Analysis
    print(f"\n{'='*70}")
    print("REFINED SWEET SPOT RESULTS")
    print("="*70)
    print(f"\nFrozen: loss={frozen['final_loss']:.6f}, sep={frozen['turn_separation']:.6f}\n")
    print(f"{'η':>6s} {'P':>6s} {'Loss':>10s} {'Δ Loss':>9s} {'Sep':>10s} {'Δ Sep':>9s} {'T̄ var':>10s}")
    print("-"*70)

    for r in results:
        marker = "  🎯" if r['delta_loss'] < 0 and r['delta_sep'] > 0 else "    "
        print(f"{r['eta']:6.3f} {r['pressure']:6.1f} "
              f"{r['loss']:10.6f} {r['delta_loss_pct']:+8.1f}% "
              f"{r['separation']:10.6f} {r['delta_sep_pct']:+8.1f}% "
              f"{r['tbar_var']:10.6f}{marker}")

    # Find Pareto-better configs
    pareto = [r for r in results if r['delta_loss'] < 0 and r['delta_sep'] > 0]

    print(f"\n{'='*70}")
    print("PARETO-BETTER REGION")
    print("="*70)

    if pareto:
        print(f"\n✅ Found {len(pareto)}/{len(results)} Pareto-better configs\n")

        # Find best on each dimension
        best_loss = min(pareto, key=lambda r: r['delta_loss_pct'])
        best_sep = max(pareto, key=lambda r: r['delta_sep_pct'])
        best_combined = min(pareto, key=lambda r: -r['delta_sep_pct'] + r['delta_loss_pct'])

        print("Best loss improvement:")
        print(f"  η={best_loss['eta']:.3f}, P={best_loss['pressure']:.1f}")
        print(f"    Δloss = {best_loss['delta_loss_pct']:+.1f}%, Δsep = {best_loss['delta_sep_pct']:+.1f}%")

        print("\nBest separation improvement:")
        print(f"  η={best_sep['eta']:.3f}, P={best_sep['pressure']:.1f}")
        print(f"    Δloss = {best_sep['delta_loss_pct']:+.1f}%, Δsep = {best_sep['delta_sep_pct']:+.1f}%")

        print("\nBest combined (max Δsep, min Δloss):")
        print(f"  η={best_combined['eta']:.3f}, P={best_combined['pressure']:.1f}")
        print(f"    Δloss = {best_combined['delta_loss_pct']:+.1f}%, Δsep = {best_combined['delta_sep_pct']:+.1f}%")
        print(f"    T̄ var = {best_combined['tbar_var']:.6f}")

    else:
        print("\n⚠️  No Pareto-better configs in this region")

    # Geological activity heatmap data
    print(f"\n{'='*70}")
    print("GEOLOGICAL ACTIVITY MAP")
    print("="*70)

    print(f"\nT̄ Variance Heatmap:\n")
    header = "η \\ P"
    print(f"{header:>8s}", end="")
    for p in pressure_scales:
        print(f"{p:>10.1f}", end="")
    print()
    print("-"*70)

    for eta in eta_locals:
        print(f"{eta:8.3f}", end="")
        for p in pressure_scales:
            r = next((r for r in results if r['eta'] == eta and r['pressure'] == p), None)
            if r:
                print(f"{r['tbar_var']:10.6f}", end="")
        print()

    # Save results
    os.makedirs("refined_sweet_spot_results", exist_ok=True)

    with open("refined_sweet_spot_results/summary.txt", "w") as f:
        f.write("Refined Geological Sweet Spot - Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Grid centered on η=0.01, P=0.5\n")
        f.write(f"Frozen baseline: loss={frozen['final_loss']:.6f}, sep={frozen['turn_separation']:.6f}\n\n")
        f.write(f"{'η':>6s} {'P':>6s} {'Loss':>10s} {'Δ Loss':>9s} {'Sep':>10s} {'Δ Sep':>9s} {'T̄ var':>10s}\n")
        f.write("-"*70 + "\n")

        for r in results:
            marker = " *" if r['delta_loss'] < 0 and r['delta_sep'] > 0 else ""
            f.write(f"{r['eta']:6.3f} {r['pressure']:6.1f} "
                   f"{r['loss']:10.6f} {r['delta_loss_pct']:+8.1f}% "
                   f"{r['separation']:10.6f} {r['delta_sep_pct']:+8.1f}% "
                   f"{r['tbar_var']:10.6f}{marker}\n")

        if pareto:
            f.write(f"\nPareto-better configs: {len(pareto)}/{len(results)}\n\n")
            f.write(f"Best combined: η={best_combined['eta']:.3f}, P={best_combined['pressure']:.1f}\n")
            f.write(f"  Δloss = {best_combined['delta_loss_pct']:+.1f}%\n")
            f.write(f"  Δsep  = {best_combined['delta_sep_pct']:+.1f}%\n")

    print(f"\n✅ Results saved to refined_sweet_spot_results/summary.txt")


if __name__ == "__main__":
    main()
