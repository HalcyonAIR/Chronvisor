"""
Quick test to verify structural temperature system is now working.

After the fix, we should see:
- T_fast varies per expert (not all 1.0)
- structural_T diverges via EMA (not all 1.0)
- T̄ variance > 0 after a few hundred steps
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from chronomoe.chronovisor_mixtral_bridge import ChronovisorMixtralForCausalLM
from chronomoe.mixtral_core import MixtralConfig
from chronomoe.training import ChronoMoELoss
from experiments.conversational_dataset import ConversationalDataset
from experiments.train_conversational import ThreeDomainDatasetPyTorch


def test_geology_awakening():
    """
    Test that T_fast and structural_T now diverge during training.
    """

    print("=" * 70)
    print("TESTING GEOLOGY FIX")
    print("=" * 70)
    print("\nExpected after fix:")
    print("  - T_fast should vary per expert (not all 1.0)")
    print("  - structural_T should diverge via EMA")
    print("  - T̄ variance should be > 0 after ~100-200 steps")
    print("=" * 70)

    # Create small dataset
    print("\n1. Generating dataset...")
    dataset_gen = ConversationalDataset(seq_length=128, vocab_size=1000)
    sequences = dataset_gen.generate_dataset(num_sequences=100, balanced=True)["sequences"]
    dataset = ThreeDomainDatasetPyTorch(sequences, vocab_size=1000)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Create model
    print("\n2. Creating model...")
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
    controller = model.model.controller

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = ChronoMoELoss(lambda_balance=0.01, lambda_coherence=0.001, lambda_valley=0.0001)

    print("\n3. Running 200 training steps...")
    print("\nCheckpoint | Loss   | R     | T̄_var (15 decimals)        | T_fast spread")
    print("-" * 80)

    model.train()
    train_iter = iter(loader)

    for step in range(200):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        logits, chrono_state = model(input_ids, update_chronovisor=True)

        loss, _ = loss_fn.compute(logits, labels, chrono_state, controller)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Check at key checkpoints
        if step in [0, 10, 50, 100, 150, 200-1]:
            st_diag = controller.get_structural_temperature_diagnostics()
            lens_0 = controller.lenses[0]

            T_fast_spread = lens_0.temperature_fast.max() - lens_0.temperature_fast.min()

            print(f"Step {step:3d}   | {loss.item():.3f} | {chrono_state.coherence:.3f} | "
                  f"{st_diag['variance']:.15f} | {T_fast_spread:.6f}")

            # Detailed inspection at final checkpoint
            if step == 199:
                print("\n" + "=" * 70)
                print("FINAL STATE INSPECTION (Layer 0)")
                print("=" * 70)

                print(f"\nT_fast per expert:")
                print(f"  Values: {lens_0.temperature_fast}")
                print(f"  Min: {lens_0.temperature_fast.min():.6f}")
                print(f"  Max: {lens_0.temperature_fast.max():.6f}")
                print(f"  Spread: {T_fast_spread:.6f}")
                print(f"  Std: {lens_0.temperature_fast.std():.6f}")

                print(f"\nstructural_T per expert:")
                print(f"  Values: {lens_0.structural_T}")
                print(f"  Min: {lens_0.structural_T.min():.6f}")
                print(f"  Max: {lens_0.structural_T.max():.6f}")
                print(f"  Spread: {lens_0.structural_T.max() - lens_0.structural_T.min():.6f}")
                print(f"  Std: {lens_0.structural_T.std():.6f}")

                print(f"\nT̄_hierarchical per expert:")
                print(f"  Values: {lens_0.structural_T_hierarchical}")
                print(f"  Variance: {lens_0.structural_T_hierarchical.var():.15f}")

                print(f"\nGlobal structural T̄:")
                print(f"  Mean: {st_diag['mean']:.15f}")
                print(f"  Variance: {st_diag['variance']:.15f}")
                print(f"  Min: {st_diag.get('min', 'N/A')}")
                print(f"  Max: {st_diag.get('max', 'N/A')}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    final_diag = controller.get_structural_temperature_diagnostics()
    final_T_fast_std = controller.lenses[0].temperature_fast.std()

    if final_diag['variance'] > 1e-10:
        print(f"\n✅ GEOLOGY IS AWAKE!")
        print(f"   T̄ variance: {final_diag['variance']:.15f} (was 0.0)")
        print(f"   T_fast std: {final_T_fast_std:.6f} (was 0.0)")
        print(f"\n   The structural temperature system is now updating!")
    elif final_T_fast_std > 0.01:
        print(f"\n⚠️  PARTIAL SUCCESS")
        print(f"   T_fast has spread (std={final_T_fast_std:.6f})")
        print(f"   But T̄ variance still very small ({final_diag['variance']:.2e})")
        print(f"\n   May need more steps for EMA to show variance")
    else:
        print(f"\n❌ GEOLOGY STILL FROZEN")
        print(f"   T_fast std: {final_T_fast_std:.6f} (still uniform!)")
        print(f"   T̄ variance: {final_diag['variance']:.2e}")
        print(f"\n   The fix may not be working correctly")


if __name__ == "__main__":
    test_geology_awakening()
