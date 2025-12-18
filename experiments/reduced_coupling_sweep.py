"""
Reduced-Coupling Sweep

Dial down geology's coupling strength (η_local, pressure_scale) to find the
sweet spot where loss equalizes with frozen baseline while maintaining elevated
turn separation.

Grid: 3×3 over (η_local, pressure_scale)
Baseline: Frozen model from keystone experiment
Output: One file per run + summary table
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
    save_dir: str = "coupling_sweep_results",
):
    """
    Train model with given coupling parameters.

    Returns:
        dict with metrics: final_loss, turn_separation, tbar_variance, etc.
    """

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
    tbar_vars = []

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

        # Track T̄ variance
        if chrono_state and hasattr(chrono_state, 'T_bar'):
            T_bar = chrono_state.T_bar
            if isinstance(T_bar, np.ndarray) and len(T_bar) > 0:
                tbar_vars.append(T_bar.var())
            else:
                tbar_vars.append(0.0)
        else:
            tbar_vars.append(0.0)

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

    # Expert utilization: count experts with >50% of mean usage
    threshold = usage_matrix.mean() * 0.5
    experts_used = (usage_matrix.sum(axis=0) > threshold).sum()

    # Final metrics
    final_loss = np.mean(losses[-50:])  # Average last 50 steps
    final_tbar_var = tbar_vars[-1] if tbar_vars else 0.0

    metrics = {
        "eta_local": eta_local,
        "pressure_scale": pressure_scale,
        "final_loss": final_loss,
        "turn_separation": turn_separation,
        "tbar_variance": final_tbar_var,
        "experts_used": experts_used,
        "loss_history": losses,
        "tbar_var_history": tbar_vars,
    }

    # Save per-run results
    os.makedirs(save_dir, exist_ok=True)
    filename = f"eta{eta_local:.4f}_pressure{pressure_scale:.2f}.txt"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, "w") as f:
        f.write(f"Reduced-Coupling Sweep - Single Run\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  η_local:        {eta_local:.4f}\n")
        f.write(f"  η_global:       0.005 (fixed)\n")
        f.write(f"  pressure_scale: {pressure_scale:.2f}\n")
        f.write(f"  num_steps:      {num_steps}\n")
        f.write(f"  num_sequences:  {len(dataset)}\n\n")
        f.write(f"Final Metrics:\n")
        f.write(f"  Loss:           {final_loss:.6f}\n")
        f.write(f"  Turn separation: {turn_separation:.6f}\n")
        f.write(f"  T̄ variance:     {final_tbar_var:.6f}\n")
        f.write(f"  Experts used:   {experts_used}/8\n\n")
        f.write(f"Turn × Expert Usage Matrix:\n")
        turn_names = ['Inquiry', 'Premise', 'Complic', 'Contra', 'Except', 'Concess', 'Synth']
        f.write(f"         ")
        for expert_idx in range(8):
            f.write(f"  E{expert_idx}  ")
        f.write(f"\n")
        f.write(f"-" * 70 + "\n")
        for turn_idx in range(7):
            f.write(f"{turn_names[turn_idx]:8s} ")
            for expert_idx in range(8):
                val = usage_normalized[turn_idx, expert_idx]
                f.write(f"{val:6.3f} ")
            f.write(f"\n")

    print(f"\n✅ Results saved to {filepath}")
    print(f"   Loss: {final_loss:.6f}")
    print(f"   Turn separation: {turn_separation:.6f}")

    return metrics


def run_frozen_baseline(sequences, num_steps: int = 250):
    """
    Run frozen model (update_chronovisor=False) to establish baseline.
    """

    print(f"\n{'='*70}")
    print(f"FROZEN BASELINE (update_chronovisor=False)")
    print(f"{'='*70}\n")

    config = MixtralConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=1024,
        num_layers=2,
        num_experts=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        enable_chronovisor=False,  # Frozen
    )

    model = ChronovisorMixtralForCausalLM(config)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    dataset = ThreeDomainDatasetPyTorch(sequences, vocab_size=1000)

    model.train()
    losses = []

    for step in range(num_steps):
        batch_idx = step % len(dataset)
        batch = dataset[batch_idx]

        input_ids = batch["input_ids"].unsqueeze(0)
        labels = batch["labels"].unsqueeze(0)

        optimizer.zero_grad()
        logits, _ = model(input_ids, update_chronovisor=False)

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
    usage_normalized = usage_matrix / usage_matrix.sum(axis=1, keepdims=True)
    expert_variance = usage_normalized.var(axis=0)
    turn_separation = expert_variance.sum()

    final_loss = np.mean(losses[-50:])

    print(f"\n✅ Frozen baseline complete")
    print(f"   Loss: {final_loss:.6f}")
    print(f"   Turn separation: {turn_separation:.6f}")

    return {
        "final_loss": final_loss,
        "turn_separation": turn_separation,
    }


def main():
    """Run reduced-coupling sweep."""

    print("="*70)
    print("REDUCED-COUPLING SWEEP")
    print("="*70)
    print("\nGoal: Find coupling strength where loss equalizes with frozen")
    print("      while maintaining elevated turn separation.\n")

    # Generate fixed dataset (same across all runs)
    print("Generating dataset (15 sequences, seq_len=128)...")
    dataset_gen = ConversationalDataset(seq_length=128, vocab_size=1000)
    result = dataset_gen.generate_dataset(num_sequences=15, balanced=False)
    sequences = result["sequences"]
    print(f"✅ Dataset generated: {len(sequences)} sequences\n")

    # Run frozen baseline first
    frozen_metrics = run_frozen_baseline(sequences, num_steps=250)

    # Grid parameters
    eta_locals = [0.001, 0.005, 0.01]
    pressure_scales = [0.1, 0.5, 1.0]

    print(f"\nGrid: {len(eta_locals)} × {len(pressure_scales)} = {len(eta_locals) * len(pressure_scales)} configurations")
    print(f"  η_local ∈ {eta_locals}")
    print(f"  pressure_scale ∈ {pressure_scales}")
    print(f"\nStarting sweep...\n")

    # Run sweep
    all_results = []

    for eta_local in eta_locals:
        for pressure_scale in pressure_scales:
            metrics = train_single_config(
                eta_local=eta_local,
                pressure_scale=pressure_scale,
                sequences=sequences,
                num_steps=250,
                save_dir="coupling_sweep_results",
            )

            # Add deltas from frozen baseline
            metrics["loss_delta_pct"] = ((metrics["final_loss"] - frozen_metrics["final_loss"])
                                         / frozen_metrics["final_loss"] * 100)
            metrics["separation_delta_pct"] = ((metrics["turn_separation"] - frozen_metrics["turn_separation"])
                                               / frozen_metrics["turn_separation"] * 100)

            all_results.append(metrics)

    # Generate summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)

    summary_path = "coupling_sweep_results/summary.txt"
    with open(summary_path, "w") as f:
        f.write("Reduced-Coupling Sweep - Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Frozen Baseline:\n")
        f.write(f"  Loss:            {frozen_metrics['final_loss']:.6f}\n")
        f.write(f"  Turn separation: {frozen_metrics['turn_separation']:.6f}\n\n")
        f.write("Grid Results:\n\n")
        f.write(f"{'η_local':>8s} {'P_scale':>8s} {'Loss':>10s} {'Δ Loss':>9s} {'Turn Sep':>10s} {'Δ Sep':>8s} {'T̄ Var':>8s}\n")
        f.write("-"*70 + "\n")

        for r in all_results:
            f.write(f"{r['eta_local']:8.4f} {r['pressure_scale']:8.2f} "
                   f"{r['final_loss']:10.6f} {r['loss_delta_pct']:+8.1f}% "
                   f"{r['turn_separation']:10.6f} {r['separation_delta_pct']:+7.1f}% "
                   f"{r['tbar_variance']:8.6f}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("Key Questions:\n\n")
        f.write("Q1: Which config has loss closest to frozen baseline?\n")
        best_loss = min(all_results, key=lambda r: abs(r['loss_delta_pct']))
        f.write(f"    → η={best_loss['eta_local']:.4f}, P={best_loss['pressure_scale']:.2f} "
               f"(Δ loss = {best_loss['loss_delta_pct']:+.1f}%)\n\n")

        f.write("Q2: Which config maximizes turn separation?\n")
        best_sep = max(all_results, key=lambda r: r['turn_separation'])
        f.write(f"    → η={best_sep['eta_local']:.4f}, P={best_sep['pressure_scale']:.2f} "
               f"(sep = {best_sep['turn_separation']:.6f})\n\n")

        f.write("Q3: Best loss/structure trade-off (minimize loss penalty while keeping separation >5% above frozen)?\n")
        candidates = [r for r in all_results if r['separation_delta_pct'] > 5.0]
        if candidates:
            best_tradeoff = min(candidates, key=lambda r: r['loss_delta_pct'])
            f.write(f"    → η={best_tradeoff['eta_local']:.4f}, P={best_tradeoff['pressure_scale']:.2f} "
                   f"(Δ loss = {best_tradeoff['loss_delta_pct']:+.1f}%, Δ sep = {best_tradeoff['separation_delta_pct']:+.1f}%)\n\n")
        else:
            f.write(f"    → No config meets criteria (all have Δ sep ≤ 5%)\n\n")

    # Print summary to console
    print(f"\n{'η_local':>8s} {'P_scale':>8s} {'Loss':>10s} {'Δ Loss':>9s} {'Turn Sep':>10s} {'Δ Sep':>8s}")
    print("-"*66)
    for r in all_results:
        print(f"{r['eta_local']:8.4f} {r['pressure_scale']:8.2f} "
              f"{r['final_loss']:10.6f} {r['loss_delta_pct']:+8.1f}% "
              f"{r['turn_separation']:10.6f} {r['separation_delta_pct']:+7.1f}%")

    print(f"\n✅ Summary saved to {summary_path}")
    print("\nAll results saved to coupling_sweep_results/")


if __name__ == "__main__":
    main()
