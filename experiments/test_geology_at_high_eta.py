"""
High-Speed Geological Test

Confirm T̄ evolution at η=0.1 (10x baseline) over 1000 steps.
This should show clear, measurable geological breathing.
"""

import torch
import numpy as np
from torch.optim import AdamW
from chronomoe.chronovisor_mixtral_bridge import ChronovisorMixtralForCausalLM
from chronomoe.mixtral_core import MixtralConfig
from experiments.conversational_dataset import ConversationalDataset
from experiments.analyze_turn_usage import ThreeDomainDatasetPyTorch


def main():
    print("="*70)
    print("HIGH-SPEED GEOLOGICAL TEST")
    print("="*70)
    print("\nTesting η_local = 0.1 over 1000 steps")
    print("Expected: Strong T̄ evolution, clear geological breathing\n")

    # Fixed seed
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

    # Set high coupling
    controller = model.model.controller
    controller.eta_structural_T_local = 0.1  # 10x baseline
    controller.eta_structural_T_global = 0.01  # 2x baseline
    controller.pressure_scale = 1.0

    for lens in controller.lenses.values():
        lens.eta_structural_T = 0.1

    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Tracking
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
    num_steps = 1000

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

        # Track T̄
        if chrono_state and hasattr(chrono_state, 'T_bar') and chrono_state.T_bar is not None:
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
                tbar_mean = tbar_stats['mean'][-1]
                print(f"Step {step+1}/{num_steps} - Loss: {recent_loss:.4f}, "
                      f"T̄ mean: {tbar_mean:.4f}, var: {tbar_var:.6f}, range: {tbar_range:.6f}")

    # Final analysis
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}\n")

    if tbar_stats['var']:
        initial_var = tbar_stats['var'][0]
        final_var = tbar_stats['var'][-1]
        max_var = max(tbar_stats['var'])

        initial_range = tbar_stats['range'][0]
        final_range = tbar_stats['range'][-1]
        max_range = max(tbar_stats['range'])

        initial_mean = tbar_stats['mean'][0]
        final_mean = tbar_stats['mean'][-1]

        print(f"T̄ Global Statistics (η=0.1, 1000 steps):")
        print(f"\n  Variance:")
        print(f"    Initial: {initial_var:.6f}")
        print(f"    Final:   {final_var:.6f}")
        print(f"    Max:     {max_var:.6f}")
        print(f"    Change:  {final_var - initial_var:+.6f} ({((final_var - initial_var)/initial_var*100) if initial_var > 0 else float('inf'):+.1f}%)")

        print(f"\n  Range (max - min):")
        print(f"    Initial: {initial_range:.6f}")
        print(f"    Final:   {final_range:.6f}")
        print(f"    Max:     {max_range:.6f}")
        print(f"    Change:  {final_range - initial_range:+.6f}")

        print(f"\n  Mean:")
        print(f"    Initial: {initial_mean:.6f}")
        print(f"    Final:   {final_mean:.6f}")
        print(f"    Change:  {final_mean - initial_mean:+.6f} ({((final_mean - initial_mean)/initial_mean*100):+.1f}%)")

        # Check for strong evolution
        if final_var > 0.001 and (final_var - initial_var) > 0.001:
            print(f"\n✅ STRONG GEOLOGICAL ACTIVITY")
            print(f"   Variance grew significantly: {initial_var:.6f} → {final_var:.6f}")
            print(f"   Geology is breathing and evolving at η=0.1")
        elif final_var > 0.0001:
            print(f"\n⚠️  MODERATE GEOLOGICAL ACTIVITY")
            print(f"   Variance: {final_var:.6f}")
        else:
            print(f"\n❌ WEAK GEOLOGICAL ACTIVITY")
            print(f"   Variance: {final_var:.6f} (too low)")

        # Save summary
        with open("high_eta_geology_test.txt", "w") as f:
            f.write("High-Speed Geological Test Results\n")
            f.write("="*70 + "\n\n")
            f.write(f"Configuration: η_local=0.1, η_global=0.01, 1000 steps\n\n")
            f.write(f"T̄ Variance: {initial_var:.6f} → {final_var:.6f} (Δ = {final_var - initial_var:+.6f})\n")
            f.write(f"T̄ Range:    {initial_range:.6f} → {final_range:.6f} (Δ = {final_range - initial_range:+.6f})\n")
            f.write(f"T̄ Mean:     {initial_mean:.6f} → {final_mean:.6f} (Δ = {final_mean - initial_mean:+.6f})\n\n")
            f.write(f"Final Loss:  {np.mean(losses[-50:]):.6f}\n\n")
            if final_var > 0.001:
                f.write("✅ Strong geological activity confirmed\n")
            else:
                f.write("⚠️  Geological activity present but weak\n")

        print(f"\n✅ Results saved to high_eta_geology_test.txt")

    else:
        print("❌ T̄ was never tracked (no data)")


if __name__ == "__main__":
    main()
