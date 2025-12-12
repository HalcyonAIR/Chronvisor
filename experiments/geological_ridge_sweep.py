"""
Geological Ridge Sweep

First ridge sweep with T̄ properly exported and active.

Tests a small focused grid:
- η ∈ {0.01, 0.02} (geological timescale)
- P ∈ {0.5, 1.0, 1.5} (pressure range)
- 500 steps (long enough for geology to breathe)

Compares frozen vs live at each point to measure geological contribution.
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
        controller.eta_structural_T_global = eta_local / 2  # Global slower than local
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

        # Track T̄ variance
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
        "loss_history": losses,
        "tbar_var_history": tbar_vars,
    }


def main():
    print("="*70)
    print("GEOLOGICAL RIDGE SWEEP")
    print("="*70)
    print("\nFirst ridge sweep with T̄ properly exported and active")
    print("Testing geological coupling contribution\n")

    # Generate fixed dataset
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_gen = ConversationalDataset(seq_length=128, vocab_size=1000)
    result = dataset_gen.generate_dataset(num_sequences=15, balanced=False)
    sequences = result["sequences"]
    print(f"✅ Dataset generated (seed={seed})\n")

    # Grid parameters
    eta_locals = [0.01, 0.02]
    pressure_scales = [0.5, 1.0, 1.5]
    num_steps = 500

    print(f"Grid: {len(eta_locals)} × {len(pressure_scales)} = {len(eta_locals) * len(pressure_scales)} configs")
    print(f"  η_local ∈ {eta_locals}")
    print(f"  pressure_scale ∈ {pressure_scales}")
    print(f"  steps = {num_steps}")
    print(f"\nRunning frozen baseline first...\n")

    # Run frozen baseline once
    frozen = run_single_config(
        enable_chronovisor=False,
        eta_local=0.01,
        pressure_scale=1.0,
        sequences=sequences,
        num_steps=num_steps,
    )
    print(f"✅ Frozen baseline: loss={frozen['final_loss']:.6f}, sep={frozen['turn_separation']:.6f}\n")

    # Run geological sweep
    results = []

    for eta_local in eta_locals:
        for pressure_scale in pressure_scales:
            print(f"Running η={eta_local:.3f}, P={pressure_scale:.1f}...")

            live = run_single_config(
                enable_chronovisor=True,
                eta_local=eta_local,
                pressure_scale=pressure_scale,
                sequences=sequences,
                num_steps=num_steps,
            )

            # Compute deltas from frozen
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
                'tbar_var_max': live['tbar_variance_max'],
                'delta_loss': delta_loss,
                'delta_loss_pct': delta_loss_pct,
                'delta_sep': delta_sep,
                'delta_sep_pct': delta_sep_pct,
            }
            results.append(result)

            print(f"  Loss: {live['final_loss']:.6f} (Δ {delta_loss_pct:+.1f}%)")
            print(f"  Sep:  {live['turn_separation']:.6f} (Δ {delta_sep_pct:+.1f}%)")
            print(f"  T̄ var: {live['tbar_variance_final']:.6f} (max: {live['tbar_variance_max']:.6f})")
            print()

    # Analysis
    print("="*70)
    print("GEOLOGICAL RIDGE RESULTS")
    print("="*70)
    print(f"\nFrozen baseline: loss={frozen['final_loss']:.6f}, sep={frozen['turn_separation']:.6f}\n")
    print(f"{'η':>6s} {'P':>6s} {'Loss':>10s} {'Δ Loss':>9s} {'Sep':>10s} {'Δ Sep':>9s} {'T̄ var':>10s}")
    print("-"*70)

    for r in results:
        print(f"{r['eta']:6.3f} {r['pressure']:6.1f} "
              f"{r['loss']:10.6f} {r['delta_loss_pct']:+8.1f}% "
              f"{r['separation']:10.6f} {r['delta_sep_pct']:+8.1f}% "
              f"{r['tbar_var']:10.6f}")

    # Find Pareto-better configs
    print(f"\n{'='*70}")
    print("PARETO-BETTER CONFIGURATIONS")
    print("="*70)

    pareto = [r for r in results if r['delta_loss'] < 0 and r['delta_sep'] > 0]

    if pareto:
        print(f"\n✅ Found {len(pareto)} Pareto-better config(s):\n")
        for r in pareto:
            print(f"  η={r['eta']:.3f}, P={r['pressure']:.1f}")
            print(f"    Δ loss = {r['delta_loss_pct']:+.1f}% (better)")
            print(f"    Δ sep  = {r['delta_sep_pct']:+.1f}% (better)")
            print(f"    T̄ var  = {r['tbar_var']:.6f}")
            print()
    else:
        print("\n⚠️  No Pareto-better configurations found")
        print("    (None with both Δloss < 0 AND Δsep > 0)")

        # Find best compromise
        print("\n  Best loss improvement:")
        best_loss = min(results, key=lambda r: r['delta_loss'])
        print(f"    η={best_loss['eta']:.3f}, P={best_loss['pressure']:.1f}: Δloss={best_loss['delta_loss_pct']:+.1f}%, Δsep={best_loss['delta_sep_pct']:+.1f}%")

        print("\n  Best separation improvement:")
        best_sep = max(results, key=lambda r: r['delta_sep'])
        print(f"    η={best_sep['eta']:.3f}, P={best_sep['pressure']:.1f}: Δloss={best_sep['delta_loss_pct']:+.1f}%, Δsep={best_sep['delta_sep_pct']:+.1f}%")

    # Check geological activity
    print(f"\n{'='*70}")
    print("GEOLOGICAL ACTIVITY CHECK")
    print("="*70)

    active_geology = [r for r in results if r['tbar_var'] > 0.0001]

    if active_geology:
        print(f"\n✅ {len(active_geology)}/{len(results)} configs show active geology (Var(T̄) > 0.0001)")
        avg_var = np.mean([r['tbar_var'] for r in active_geology])
        print(f"   Average T̄ variance: {avg_var:.6f}")
    else:
        print("\n❌ No configs show significant geological activity")
        print("   500 steps may be too short, or η too slow")

    # Save results
    os.makedirs("geological_ridge_results", exist_ok=True)

    with open("geological_ridge_results/summary.txt", "w") as f:
        f.write("Geological Ridge Sweep - Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Configuration: 500 steps, seed={seed}\n")
        f.write(f"Frozen baseline: loss={frozen['final_loss']:.6f}, sep={frozen['turn_separation']:.6f}\n\n")
        f.write(f"{'η':>6s} {'P':>6s} {'Loss':>10s} {'Δ Loss':>9s} {'Sep':>10s} {'Δ Sep':>9s} {'T̄ var':>10s}\n")
        f.write("-"*70 + "\n")

        for r in results:
            f.write(f"{r['eta']:6.3f} {r['pressure']:6.1f} "
                   f"{r['loss']:10.6f} {r['delta_loss_pct']:+8.1f}% "
                   f"{r['separation']:10.6f} {r['delta_sep_pct']:+8.1f}% "
                   f"{r['tbar_var']:10.6f}\n")

        if pareto:
            f.write(f"\n✅ Pareto-better configs:\n")
            for r in pareto:
                f.write(f"  η={r['eta']:.3f}, P={r['pressure']:.1f}: Δloss={r['delta_loss_pct']:+.1f}%, Δsep={r['delta_sep_pct']:+.1f}%\n")
        else:
            f.write(f"\n⚠️  No Pareto-better configs found\n")

    print(f"\n✅ Results saved to geological_ridge_results/summary.txt")


if __name__ == "__main__":
    main()
