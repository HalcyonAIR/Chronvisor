"""
Geological Seed-Robustness Test

Test η=0.01, P=0.5 (the sweet spot) across multiple seeds
to confirm geological coupling is general, not seed-dependent.

Tests 5 different seeds:
- Original (42)
- 12345, 67890, 11111, 99999
"""

import torch
import numpy as np
from torch.optim import AdamW
from chronomoe.chronovisor_mixtral_bridge import ChronovisorMixtralForCausalLM
from chronomoe.mixtral_core import MixtralConfig
from experiments.conversational_dataset import ConversationalDataset
from experiments.analyze_turn_usage import TurnUsageAnalyzer, ThreeDomainDatasetPyTorch


def run_config(enable_chronovisor, eta_local, pressure_scale, sequences, seed):
    """Run single configuration with given seed."""

    # Set seed for model init
    torch.manual_seed(seed + 1000)  # Offset to ensure different init from dataset
    np.random.seed(seed + 1000)

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
    num_steps = 500

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
        "tbar_variance": final_tbar_var,
        "tbar_variance_max": max_tbar_var,
    }


def main():
    print("="*70)
    print("GEOLOGICAL SEED-ROBUSTNESS TEST")
    print("="*70)
    print("\nTesting η=0.01, P=0.5 across 5 seeds to confirm generality\n")

    # Test configuration
    eta_local = 0.01
    pressure_scale = 0.5
    seeds = [42, 12345, 67890, 11111, 99999]

    print(f"Configuration: η={eta_local:.3f}, P={pressure_scale:.1f}")
    print(f"Seeds: {seeds}\n")

    results = []

    for seed in seeds:
        print(f"{'='*70}")
        print(f"Seed: {seed}")
        print(f"{'='*70}\n")

        # Generate dataset with this seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        dataset_gen = ConversationalDataset(seq_length=128, vocab_size=1000)
        result = dataset_gen.generate_dataset(num_sequences=15, balanced=False)
        sequences = result["sequences"]

        # Run frozen baseline
        print("Running frozen baseline...")
        frozen = run_config(
            enable_chronovisor=False,
            eta_local=eta_local,
            pressure_scale=pressure_scale,
            sequences=sequences,
            seed=seed,
        )
        print(f"  Loss: {frozen['final_loss']:.6f}, Sep: {frozen['turn_separation']:.6f}")

        # Run live geology
        print("Running live geology...")
        live = run_config(
            enable_chronovisor=True,
            eta_local=eta_local,
            pressure_scale=pressure_scale,
            sequences=sequences,
            seed=seed,
        )
        print(f"  Loss: {live['final_loss']:.6f}, Sep: {live['turn_separation']:.6f}, T̄: {live['tbar_variance']:.6f}")

        # Compute deltas
        delta_loss = live['final_loss'] - frozen['final_loss']
        delta_sep = live['turn_separation'] - frozen['turn_separation']
        delta_loss_pct = (delta_loss / frozen['final_loss']) * 100
        delta_sep_pct = (delta_sep / frozen['turn_separation']) * 100

        is_pareto = delta_loss < 0 and delta_sep > 0

        result_data = {
            'seed': seed,
            'frozen_loss': frozen['final_loss'],
            'frozen_sep': frozen['turn_separation'],
            'live_loss': live['final_loss'],
            'live_sep': live['turn_separation'],
            'tbar_var': live['tbar_variance'],
            'delta_loss': delta_loss,
            'delta_sep': delta_sep,
            'delta_loss_pct': delta_loss_pct,
            'delta_sep_pct': delta_sep_pct,
            'is_pareto': is_pareto,
        }
        results.append(result_data)

        print(f"\n  Δ Loss: {delta_loss_pct:+.1f}%")
        print(f"  Δ Sep:  {delta_sep_pct:+.1f}%")
        print(f"  Pareto: {'✅ YES' if is_pareto else '❌ NO'}\n")

    # Summary analysis
    print("="*70)
    print("SEED-ROBUSTNESS SUMMARY")
    print("="*70)

    print(f"\n{'Seed':>8s} {'Δ Loss':>10s} {'Δ Sep':>10s} {'T̄ var':>10s} {'Pareto':>8s}")
    print("-"*70)

    for r in results:
        pareto_marker = "✅" if r['is_pareto'] else "❌"
        print(f"{r['seed']:8d} {r['delta_loss_pct']:+9.1f}% {r['delta_sep_pct']:+9.1f}% "
              f"{r['tbar_var']:10.6f} {pareto_marker:>8s}")

    # Statistics
    pareto_count = sum(1 for r in results if r['is_pareto'])
    avg_delta_loss = np.mean([r['delta_loss_pct'] for r in results])
    avg_delta_sep = np.mean([r['delta_sep_pct'] for r in results])
    avg_tbar_var = np.mean([r['tbar_var'] for r in results])

    std_delta_loss = np.std([r['delta_loss_pct'] for r in results])
    std_delta_sep = np.std([r['delta_sep_pct'] for r in results])

    print(f"\n{'='*70}")
    print("STATISTICS")
    print("="*70)

    print(f"\nPareto-better seeds: {pareto_count}/{len(seeds)} ({pareto_count/len(seeds)*100:.0f}%)")

    print(f"\nΔ Loss (mean ± std): {avg_delta_loss:+.1f}% ± {std_delta_loss:.1f}%")
    print(f"Δ Sep  (mean ± std): {avg_delta_sep:+.1f}% ± {std_delta_sep:.1f}%")
    print(f"T̄ var  (mean):       {avg_tbar_var:.6f}")

    # Verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print("="*70)

    if pareto_count == len(seeds):
        print(f"\n✅ PERFECT ROBUSTNESS")
        print(f"   All {len(seeds)} seeds show Pareto-better performance")
        print(f"   Geological coupling is STABLE and GENERAL")
    elif pareto_count >= len(seeds) * 0.8:
        print(f"\n✅ STRONG ROBUSTNESS")
        print(f"   {pareto_count}/{len(seeds)} seeds show Pareto-better performance")
        print(f"   Geological coupling is RELIABLE")
    elif pareto_count >= len(seeds) * 0.6:
        print(f"\n⚠️  MODERATE ROBUSTNESS")
        print(f"   {pareto_count}/{len(seeds)} seeds show Pareto-better performance")
        print(f"   Effect is real but has variance")
    else:
        print(f"\n❌ WEAK ROBUSTNESS")
        print(f"   Only {pareto_count}/{len(seeds)} seeds show Pareto-better performance")
        print(f"   Sweet spot may be seed-dependent")

    # Check geological activity
    if all(r['tbar_var'] > 0.0001 for r in results):
        print(f"\n✅ Geology is active in all seeds (all Var(T̄) > 0.0001)")
    else:
        inactive = sum(1 for r in results if r['tbar_var'] <= 0.0001)
        print(f"\n⚠️  Geology inactive in {inactive}/{len(seeds)} seeds")

    # Save results
    with open("geological_seed_robustness.txt", "w") as f:
        f.write("Geological Seed-Robustness Test\n")
        f.write("="*70 + "\n\n")
        f.write(f"Configuration: η={eta_local:.3f}, P={pressure_scale:.1f}\n")
        f.write(f"Seeds tested: {len(seeds)}\n\n")
        f.write(f"{'Seed':>8s} {'Δ Loss':>10s} {'Δ Sep':>10s} {'T̄ var':>10s} {'Pareto':>8s}\n")
        f.write("-"*70 + "\n")

        for r in results:
            pareto_marker = "YES" if r['is_pareto'] else "NO"
            f.write(f"{r['seed']:8d} {r['delta_loss_pct']:+9.1f}% {r['delta_sep_pct']:+9.1f}% "
                   f"{r['tbar_var']:10.6f} {pareto_marker:>8s}\n")

        f.write(f"\nPareto-better: {pareto_count}/{len(seeds)} ({pareto_count/len(seeds)*100:.0f}%)\n")
        f.write(f"Δ Loss: {avg_delta_loss:+.1f}% ± {std_delta_loss:.1f}%\n")
        f.write(f"Δ Sep:  {avg_delta_sep:+.1f}% ± {std_delta_sep:.1f}%\n")
        f.write(f"T̄ var:  {avg_tbar_var:.6f}\n")

    print(f"\n✅ Results saved to geological_seed_robustness.txt")


if __name__ == "__main__":
    main()
