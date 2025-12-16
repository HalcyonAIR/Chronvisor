"""
Refined Ridge Sweep

Zoom in on the ridge region around η=0.005, P=1.0 to characterize
the attractor line with higher resolution.

Grid: 4×4 around the sweet spot
- η ∈ {0.003, 0.004, 0.006, 0.008}
- P ∈ {0.7, 0.9, 1.1, 1.3}
"""

import os
import torch
import numpy as np
from torch.optim import AdamW
from chronomoe.chronovisor_mixtral_bridge import ChronovisorMixtralForCausalLM
from chronomoe.mixtral_core import MixtralConfig
from experiments.conversational_dataset import ConversationalDataset
from experiments.analyze_turn_usage import TurnUsageAnalyzer, ThreeDomainDatasetPyTorch


def train_single_config(
    eta_local: float,
    pressure_scale: float,
    sequences,
    num_steps: int = 250,
    save_dir: str = "refined_ridge_results",
):
    """Train model with given coupling parameters."""

    print(f"\n{'='*70}")
    print(f"Configuration: η_local={eta_local:.4f}, pressure_scale={pressure_scale:.2f}")
    print(f"{'='*70}\n")

    # Create model
    config = MixtralConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=1024,
        num_layers=2,
        num_experts=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        enable_chronovisor=True,
    )

    model = ChronovisorMixtralForCausalLM(config)

    # Modify coupling parameters in controller
    controller = model.model.controller
    controller.eta_structural_T_local = eta_local
    controller.eta_structural_T_global = 0.005  # Keep global fixed
    controller.pressure_scale = pressure_scale

    # Update lens parameters
    for lens in controller.lenses.values():
        lens.eta_structural_T = eta_local

    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Create dataset
    dataset = ThreeDomainDatasetPyTorch(sequences, vocab_size=1000)

    # Training loop
    model.train()
    losses = []

    for step in range(num_steps):
        batch_idx = step % len(dataset)
        batch = dataset[batch_idx]

        input_ids = batch["input_ids"].unsqueeze(0)
        labels = batch["labels"].unsqueeze(0)

        optimizer.zero_grad()
        logits, chrono_state = model(input_ids, update_chronovisor=True)

        # Compute loss
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        loss = torch.nn.functional.cross_entropy(logits_flat, labels_flat, ignore_index=-100)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 50 == 0:
            print(f"Step {step+1}/{num_steps} - Loss: {loss.item():.4f}")

    # Post-training analysis
    model.eval()
    analyzer = TurnUsageAnalyzer(model, num_turns=7)

    print("\nRunning turn usage analysis...")
    for i in range(len(dataset)):
        batch = dataset[i]
        batch_expanded = {
            "input_ids": batch["input_ids"].unsqueeze(0),
            "labels": batch["labels"].unsqueeze(0),
            "turn_boundaries": [batch.get("turn_boundaries", [])]
        }
        analyzer.analyze_batch(batch_expanded)

    usage_matrix = analyzer.get_usage_matrix(layer_idx=0)

    # Normalize to get probabilities
    usage_normalized = usage_matrix / usage_matrix.sum(axis=1, keepdims=True)

    # Turn separation: variance of each expert's usage across turns
    expert_variance = usage_normalized.var(axis=0)
    turn_separation = expert_variance.sum()

    # Expert utilization
    threshold = usage_matrix.mean() * 0.5
    experts_used = (usage_matrix.sum(axis=0) > threshold).sum()

    # Final metrics
    final_loss = np.mean(losses[-50:])

    metrics = {
        "eta_local": eta_local,
        "pressure_scale": pressure_scale,
        "final_loss": final_loss,
        "turn_separation": turn_separation,
        "experts_used": experts_used,
    }

    # Save per-run results
    os.makedirs(save_dir, exist_ok=True)
    filename = f"eta{eta_local:.4f}_pressure{pressure_scale:.2f}.txt"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, "w") as f:
        f.write(f"Refined Ridge Sweep - Single Run\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  η_local:        {eta_local:.4f}\n")
        f.write(f"  pressure_scale: {pressure_scale:.2f}\n\n")
        f.write(f"Final Metrics:\n")
        f.write(f"  Loss:           {final_loss:.6f}\n")
        f.write(f"  Turn separation: {turn_separation:.6f}\n")
        f.write(f"  Experts used:   {experts_used}/8\n")

    print(f"\n✅ Results saved to {filepath}")
    print(f"   Loss: {final_loss:.6f}, Turn sep: {turn_separation:.6f}")

    return metrics


def main():
    """Run refined ridge sweep."""

    print("="*70)
    print("REFINED RIDGE SWEEP")
    print("="*70)
    print("\nZooming in on ridge region around η=0.005, P=1.0\n")

    # Generate fixed dataset (same as original sweep for comparison)
    print("Generating dataset (15 sequences, seq_len=128)...")
    dataset_gen = ConversationalDataset(seq_length=128, vocab_size=1000)
    result = dataset_gen.generate_dataset(num_sequences=15, balanced=False)
    sequences = result["sequences"]
    print(f"✅ Dataset generated: {len(sequences)} sequences\n")

    # Refined grid around the ridge
    eta_locals = [0.003, 0.004, 0.006, 0.008]
    pressure_scales = [0.7, 0.9, 1.1, 1.3]

    print(f"Refined grid: {len(eta_locals)} × {len(pressure_scales)} = {len(eta_locals) * len(pressure_scales)} configurations")
    print(f"  η_local ∈ {eta_locals}")
    print(f"  pressure_scale ∈ {pressure_scales}")
    print(f"\nStarting sweep...\n")

    # Run sweep
    all_results = []
    frozen_baseline_loss = 3.015188  # From original sweep
    frozen_baseline_sep = 0.039921

    for eta_local in eta_locals:
        for pressure_scale in pressure_scales:
            metrics = train_single_config(
                eta_local=eta_local,
                pressure_scale=pressure_scale,
                sequences=sequences,
                num_steps=250,
                save_dir="refined_ridge_results",
            )

            # Add deltas from frozen baseline
            metrics["loss_delta_pct"] = ((metrics["final_loss"] - frozen_baseline_loss)
                                         / frozen_baseline_loss * 100)
            metrics["separation_delta_pct"] = ((metrics["turn_separation"] - frozen_baseline_sep)
                                               / frozen_baseline_sep * 100)

            all_results.append(metrics)

    # Generate summary table
    print("\n" + "="*70)
    print("REFINED RIDGE SUMMARY")
    print("="*70)

    summary_path = "refined_ridge_results/summary.txt"
    with open(summary_path, "w") as f:
        f.write("Refined Ridge Sweep - Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Frozen Baseline (from original sweep):\n")
        f.write(f"  Loss:            {frozen_baseline_loss:.6f}\n")
        f.write(f"  Turn separation: {frozen_baseline_sep:.6f}\n\n")
        f.write("Refined Grid Results:\n\n")
        f.write(f"{'η_local':>8s} {'P_scale':>8s} {'Loss':>10s} {'Δ Loss':>9s} {'Turn Sep':>10s} {'Δ Sep':>8s}\n")
        f.write("-"*70 + "\n")

        for r in all_results:
            f.write(f"{r['eta_local']:8.4f} {r['pressure_scale']:8.2f} "
                   f"{r['final_loss']:10.6f} {r['loss_delta_pct']:+8.1f}% "
                   f"{r['turn_separation']:10.6f} {r['separation_delta_pct']:+7.1f}%\n")

        # Find ridge line (max turn separation at each pressure level)
        f.write("\n" + "="*70 + "\n")
        f.write("Ridge Line Analysis:\n\n")
        f.write("At each pressure level, which η maximizes turn separation?\n\n")

        for p in pressure_scales:
            configs_at_p = [r for r in all_results if r['pressure_scale'] == p]
            best = max(configs_at_p, key=lambda r: r['turn_separation'])
            f.write(f"P={p:.1f}: η={best['eta_local']:.4f} "
                   f"(sep={best['turn_separation']:.6f}, loss={best['final_loss']:.6f})\n")

        # Find sweet spot (best loss/sep trade-off)
        f.write("\n" + "="*70 + "\n")
        f.write("Sweet Spot:\n\n")

        # Filter: sep > 5% above frozen
        candidates = [r for r in all_results if r['separation_delta_pct'] > 5.0]
        if candidates:
            best_tradeoff = min(candidates, key=lambda r: r['loss_delta_pct'])
            f.write(f"Best loss/structure trade-off:\n")
            f.write(f"  η={best_tradeoff['eta_local']:.4f}, P={best_tradeoff['pressure_scale']:.2f}\n")
            f.write(f"  Δ loss = {best_tradeoff['loss_delta_pct']:+.1f}%\n")
            f.write(f"  Δ sep = {best_tradeoff['separation_delta_pct']:+.1f}%\n")

    # Print summary to console
    print(f"\n{'η_local':>8s} {'P_scale':>8s} {'Loss':>10s} {'Δ Loss':>9s} {'Turn Sep':>10s} {'Δ Sep':>8s}")
    print("-"*70)
    for r in all_results:
        print(f"{r['eta_local']:8.4f} {r['pressure_scale']:8.2f} "
              f"{r['final_loss']:10.6f} {r['loss_delta_pct']:+8.1f}% "
              f"{r['turn_separation']:10.6f} {r['separation_delta_pct']:+7.1f}%")

    print(f"\n✅ Summary saved to {summary_path}")
    print("\nAll results saved to refined_ridge_results/")
    print("\nReady to visualize refined ridge with higher resolution heatmap.")


if __name__ == "__main__":
    main()
