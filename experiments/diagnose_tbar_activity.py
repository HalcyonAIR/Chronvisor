"""
Is T̄ Alive? Diagnostic

Crank η_T high and log T̄ evolution to confirm the slow field is updating.

Test configurations:
1. High η (0.05) - should see rapid T̄ evolution
2. Medium η (0.01) - baseline
3. Low η (0.001) - very slow evolution

For each: Track min/max/variance/gradients of T̄ over 500-1000 steps.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.optim import AdamW
from chronomoe.chronovisor_mixtral_bridge import ChronovisorMixtralForCausalLM
from chronomoe.mixtral_core import MixtralConfig
from experiments.conversational_dataset import ConversationalDataset
from experiments.analyze_turn_usage import ThreeDomainDatasetPyTorch


def run_diagnostic(eta_local, num_steps=1000):
    """Run training and track T̄ statistics."""

    print(f"\n{'='*70}")
    print(f"Testing η_local = {eta_local:.4f}")
    print(f"{'='*70}\n")

    # Fixed seed for comparison
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate dataset
    dataset_gen = ConversationalDataset(seq_length=128, vocab_size=1000)
    result = dataset_gen.generate_dataset(num_sequences=15, balanced=False)
    sequences = result["sequences"]
    dataset = ThreeDomainDatasetPyTorch(sequences, vocab_size=1000)

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

    # Set coupling parameters
    controller = model.model.controller
    controller.eta_structural_T_local = eta_local
    controller.eta_structural_T_global = 0.005
    controller.pressure_scale = 1.0

    for lens in controller.lenses.values():
        lens.eta_structural_T = eta_local

    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Tracking arrays
    tbar_stats = {
        'min': [],
        'max': [],
        'mean': [],
        'var': [],
        'range': [],
        'step': [],
    }

    losses = []

    # Training loop
    model.train()
    for step in range(num_steps):
        batch_idx = step % len(dataset)
        batch = dataset[batch_idx]

        input_ids = batch["input_ids"].unsqueeze(0)
        labels = batch["labels"].unsqueeze(0)

        optimizer.zero_grad()
        logits, chrono_state = model(input_ids, update_chronovisor=True)

        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        loss = torch.nn.functional.cross_entropy(logits_flat, labels_flat, ignore_index=-100)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Track T̄ statistics
        if chrono_state and hasattr(chrono_state, 'T_bar'):
            T_bar = chrono_state.T_bar
            if isinstance(T_bar, np.ndarray) and len(T_bar) > 0:
                tbar_stats['min'].append(T_bar.min())
                tbar_stats['max'].append(T_bar.max())
                tbar_stats['mean'].append(T_bar.mean())
                tbar_stats['var'].append(T_bar.var())
                tbar_stats['range'].append(T_bar.max() - T_bar.min())
                tbar_stats['step'].append(step)

        if (step + 1) % 100 == 0:
            recent_loss = np.mean(losses[-50:])
            if tbar_stats['var']:
                tbar_var = tbar_stats['var'][-1]
                tbar_range = tbar_stats['range'][-1]
                print(f"Step {step+1}/{num_steps} - Loss: {recent_loss:.4f}, "
                      f"T̄ var: {tbar_var:.6f}, T̄ range: {tbar_range:.6f}")
            else:
                print(f"Step {step+1}/{num_steps} - Loss: {recent_loss:.4f}, T̄: NOT TRACKED")

    # Analysis
    print(f"\n{'='*70}")
    print(f"T̄ Activity Summary (η={eta_local:.4f})")
    print(f"{'='*70}\n")

    if tbar_stats['var']:
        initial_var = tbar_stats['var'][0]
        final_var = tbar_stats['var'][-1]
        max_var = max(tbar_stats['var'])

        initial_range = tbar_stats['range'][0]
        final_range = tbar_stats['range'][-1]
        max_range = max(tbar_stats['range'])

        print(f"T̄ Variance:")
        print(f"  Initial: {initial_var:.6f}")
        print(f"  Final:   {final_var:.6f}")
        print(f"  Max:     {max_var:.6f}")
        print(f"  Change:  {final_var - initial_var:+.6f}")

        print(f"\nT̄ Range (max - min):")
        print(f"  Initial: {initial_range:.6f}")
        print(f"  Final:   {final_range:.6f}")
        print(f"  Max:     {max_range:.6f}")
        print(f"  Change:  {final_range - initial_range:+.6f}")

        print(f"\nT̄ Mean:")
        print(f"  Initial: {tbar_stats['mean'][0]:.6f}")
        print(f"  Final:   {tbar_stats['mean'][-1]:.6f}")

        # Verdict
        is_alive = final_var > 0.0001 or (final_var - initial_var) > 0.00001

        if is_alive:
            print(f"\n✅ T̄ IS ALIVE")
            print(f"   Variance evolved: {initial_var:.6f} → {final_var:.6f}")
        else:
            print(f"\n❌ T̄ IS FROZEN")
            print(f"   Variance stuck at: {final_var:.6f}")
    else:
        print("❌ T̄ was never tracked (not accessible)")

    return tbar_stats, losses


def main():
    print("="*70)
    print("IS T̄ ALIVE? DIAGNOSTIC")
    print("="*70)
    print("\nCranking η_T and logging T̄ evolution to confirm slow field updates.\n")

    # Test three configurations
    configs = [
        ("High η (fast evolution)", 0.05),
        ("Baseline η", 0.01),
        ("Low η (slow evolution)", 0.001),
    ]

    all_results = {}

    for name, eta in configs:
        print(f"\n{'#'*70}")
        print(f"# {name}: η = {eta:.4f}")
        print(f"{'#'*70}")

        tbar_stats, losses = run_diagnostic(eta, num_steps=1000)
        all_results[name] = {
            'eta': eta,
            'tbar_stats': tbar_stats,
            'losses': losses,
        }

    # Generate plots
    print(f"\n{'='*70}")
    print("GENERATING PLOTS")
    print("="*70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: T̄ Variance over time
    ax = axes[0, 0]
    for name, data in all_results.items():
        if data['tbar_stats']['var']:
            ax.plot(data['tbar_stats']['step'], data['tbar_stats']['var'],
                   label=f"{name} (η={data['eta']:.4f})", linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Var(T̄)')
    ax.set_title('T̄ Variance Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: T̄ Range over time
    ax = axes[0, 1]
    for name, data in all_results.items():
        if data['tbar_stats']['range']:
            ax.plot(data['tbar_stats']['step'], data['tbar_stats']['range'],
                   label=f"{name} (η={data['eta']:.4f})", linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('T̄ Range (max - min)')
    ax.set_title('T̄ Range Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Loss over time
    ax = axes[1, 0]
    for name, data in all_results.items():
        ax.plot(data['losses'], label=f"{name} (η={data['eta']:.4f})", linewidth=2, alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: T̄ Min/Max bounds
    ax = axes[1, 1]
    for name, data in all_results.items():
        if data['tbar_stats']['min'] and data['tbar_stats']['max']:
            steps = data['tbar_stats']['step']
            ax.fill_between(steps, data['tbar_stats']['min'], data['tbar_stats']['max'],
                           alpha=0.3, label=f"{name} (η={data['eta']:.4f})")
    ax.set_xlabel('Step')
    ax.set_ylabel('T̄ Bounds')
    ax.set_title('T̄ Min/Max Envelope')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('T̄ Activity Diagnostic: Is the Slow Field Breathing?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = 'tbar_activity_diagnostic.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plots saved to {plot_path}")

    # Final summary
    print(f"\n{'='*70}")
    print("DIAGNOSTIC SUMMARY")
    print("="*70)

    for name, data in all_results.items():
        stats = data['tbar_stats']
        if stats['var']:
            initial_var = stats['var'][0]
            final_var = stats['var'][-1]
            delta_var = final_var - initial_var

            print(f"\n{name} (η={data['eta']:.4f}):")
            print(f"  Var(T̄): {initial_var:.6f} → {final_var:.6f} (Δ = {delta_var:+.6f})")

            if final_var > 0.0001:
                print(f"  Status: ✅ ALIVE (variance evolved)")
            else:
                print(f"  Status: ❌ FROZEN (variance ~0)")

    print(f"\n{'='*70}")
    print("Next steps:")
    print("  1. If all configs show Var(T̄) ≈ 0, check temperature computation")
    print("  2. If high-η shows activity, retune η_local upward")
    print("  3. If variance grows, geology is breathing - extend horizon")
    print("="*70)


if __name__ == "__main__":
    main()
