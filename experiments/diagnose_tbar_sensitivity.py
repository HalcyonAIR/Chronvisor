"""
Diagnostic: Why is T̄ variance stuck at zero?

Three hypotheses:
1. World too balanced - T_fast identical across experts (upstream issue)
2. Print precision - Real variance exists (1e-7) but shows as "0.000000"
3. Wiring shorted - T̄ accidentally normalized/averaged after update

Checks at multiple checkpoints:
- Raw T_fast per expert (not T_eff)
- Raw T̄_global and T̄_local values (min/max, not just variance)
- Reliability values per expert
- Expert usage to see if differentiation happening upstream
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Dict

from chronomoe.chronovisor_mixtral_bridge import ChronovisorMixtralForCausalLM
from chronomoe.mixtral_core import MixtralConfig
from experiments.conversational_dataset import ConversationalDataset, ALL_CONVERSATIONAL_DOMAINS as ALL_DOMAINS


class ThreeDomainDatasetPyTorch(Dataset):
    """PyTorch Dataset wrapper for 3-domain synthetic data."""

    def __init__(self, sequences: List[Dict], vocab_size: int):
        self.sequences = sequences
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = self.sequences[idx]
        token_ids = torch.tensor(item["token_ids"], dtype=torch.long)
        labels = torch.cat([token_ids[1:], torch.tensor([-100])])
        return {
            "input_ids": token_ids,
            "labels": labels,
            "domain": item["domain"]
        }


def diagnose_tbar_sensitivity():
    """Run diagnostic on T̄ sensitivity issue."""

    print("=" * 70)
    print("T̄ SENSITIVITY DIAGNOSTIC")
    print("=" * 70)
    print("\nHypotheses:")
    print("1. World too balanced - T_fast uniform across experts")
    print("2. Print precision - Variance < 1e-6 shows as '0.000000'")
    print("3. Wiring shorted - T_fast diverges but T̄ gets flattened")
    print("=" * 70)

    # Create dataset
    print("\n📊 Generating conversational dataset...")
    dataset_gen = ConversationalDataset(seq_length=128, vocab_size=1000)
    full_dataset = dataset_gen.generate_dataset(num_sequences=1000, balanced=True)

    train_sequences = full_dataset["sequences"]
    train_dataset = ThreeDomainDatasetPyTorch(train_sequences, vocab_size=1000)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Create model
    print("\n🏗️  Creating model...")
    config = MixtralConfig(
        vocab_size=1000,
        hidden_dim=512,
        intermediate_dim=2048,
        num_layers=4,
        num_experts=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        enable_chronovisor=True,
    )

    model = ChronovisorMixtralForCausalLM(config)
    controller = model.model.controller

    print(f"   {config.num_layers} layers, {config.num_experts} experts")
    print(f"   ChronoMoE enabled: {config.enable_chronovisor}")

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # Checkpoints to inspect
    checkpoints = [0, 50, 100, 200, 500]

    print("\n" + "=" * 70)
    print("TRAINING WITH DIAGNOSTICS")
    print("=" * 70)

    model.train()
    train_iter = iter(train_loader)

    for step in range(max(checkpoints) + 1):
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Forward pass
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        logits, chrono_state = model(input_ids, update_chronovisor=True)

        # Compute loss
        loss = loss_fn(
            logits.view(-1, config.vocab_size),
            labels.view(-1)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Diagnostics at checkpoints
        if step in checkpoints:
            print(f"\n{'=' * 70}")
            print(f"CHECKPOINT: Step {step}")
            print(f"{'=' * 70}")
            print(f"Loss: {loss.item():.4f}")
            print(f"Kuramoto R: {chrono_state.coherence:.6f}")

            # Check each layer
            for layer_idx in range(config.num_layers):
                lens = controller.lenses[layer_idx]

                print(f"\n--- Layer {layer_idx} ---")

                # 1. Raw T_fast per expert (THE KEY CHECK)
                print("\n1. RAW T_FAST PER EXPERT:")
                T_fast = lens.temperature_fast
                print(f"   T_fast values: {T_fast}")
                print(f"   T_fast range: [{T_fast.min():.6f}, {T_fast.max():.6f}]")
                print(f"   T_fast mean: {T_fast.mean():.6f}")
                print(f"   T_fast std: {T_fast.std():.6f}")
                print(f"   T_fast variance: {T_fast.var():.10f}")  # High precision

                # 2. Structural Temperature (hierarchical)
                print("\n2. STRUCTURAL TEMPERATURE T̄:")
                T_bar_hierarchical = lens.temperature_structural
                print(f"   T̄_hierarchical values: {T_bar_hierarchical}")
                print(f"   T̄_hierarchical range: [{T_bar_hierarchical.min():.10f}, {T_bar_hierarchical.max():.10f}]")
                print(f"   T̄_hierarchical mean: {T_bar_hierarchical.mean():.10f}")
                print(f"   T̄_hierarchical std: {T_bar_hierarchical.std():.10f}")
                print(f"   T̄_hierarchical variance: {T_bar_hierarchical.var():.15f}")  # VERY high precision

                # 3. T_effective (T_fast × T̄)
                print("\n3. EFFECTIVE TEMPERATURE T_eff:")
                T_eff = lens.temperature_effective
                print(f"   T_eff values: {T_eff}")
                print(f"   T_eff range: [{T_eff.min():.6f}, {T_eff.max():.6f}]")
                print(f"   T_eff variance: {T_eff.var():.10f}")

                # 4. Reliability per expert
                print("\n4. RELIABILITY PER EXPERT:")
                reliability = lens.reliability
                print(f"   Reliability values: {reliability}")
                print(f"   Reliability range: [{reliability.min():.6f}, {reliability.max():.6f}]")
                print(f"   Reliability std: {reliability.std():.6f}")

                # 5. Expert usage (if available)
                if chrono_state.expert_usage and layer_idx in chrono_state.expert_usage:
                    usage = chrono_state.expert_usage[layer_idx]
                    if isinstance(usage, np.ndarray):
                        usage_array = usage
                    else:
                        usage_array = usage.cpu().numpy()
                    print("\n5. EXPERT USAGE:")
                    print(f"   Usage: {usage_array}")
                    print(f"   Usage std: {usage_array.std():.6f}")

            # Global structural temperature diagnostics
            print(f"\n{'=' * 70}")
            print("GLOBAL STRUCTURAL TEMPERATURE:")
            st_diag = controller.get_structural_temperature_diagnostics()
            print(f"   T̄_global mean: {st_diag['mean']:.15f}")
            print(f"   T̄_global variance: {st_diag['variance']:.15f}")
            print(f"   T̄_global min: {st_diag['min']:.15f}")
            print(f"   T̄_global max: {st_diag['max']:.15f}")
            print(f"   Spread (max-min): {st_diag['max'] - st_diag['min']:.15f}")
            print(f"   Valleys: {len(st_diag['valleys'])}")
            print(f"   Ridges: {len(st_diag['ridges'])}")

        # Progress logging
        if step % 50 == 0 and step not in checkpoints:
            print(f"Step {step:5d} | Loss: {loss.item():.4f} | R: {chrono_state.coherence:.3f}")

    # Final analysis
    print("\n" + "=" * 70)
    print("DIAGNOSIS:")
    print("=" * 70)

    # Check hypothesis 1: World too balanced
    print("\n1. WORLD TOO BALANCED?")
    print("   Check if T_fast spread is consistently < 0.01")
    print("   If YES → Upstream signals (reliability, drift) not differentiating")

    # Check hypothesis 2: Print precision
    print("\n2. PRINT PRECISION ISSUE?")
    print("   Check if T̄ variance is in 1e-7 to 1e-6 range")
    print("   If YES → Real variance exists but was hidden in .6f formatting")

    # Check hypothesis 3: Wiring shorted
    print("\n3. WIRING SHORTED?")
    print("   Check if T_fast has persistent asymmetry but T̄ stays uniform")
    print("   If T_fast std > 0.05 for 500+ steps but T̄ max-min < 1e-6:")
    print("   → Normalization/averaging undoing divergence in update")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    diagnose_tbar_sensitivity()
