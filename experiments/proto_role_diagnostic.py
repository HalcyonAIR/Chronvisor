"""
Complete Proto-Role Diagnostic (Halcyon's Specification)

Workflow:
1. Analyze UNTRAINED model (baseline)
2. Run training with checkpoint saving
3. Analyze TRAINED model
4. Compare turn-level usage patterns

Answer: Did proto-roles form during training?
- If trained model shows DIFFERENT turn preferences → proto-roles exist, T̄_var=0 is wiring issue
- If trained model shows SAME patterns as baseline → no specialization, T̄_var=0 is correct
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Dict

from chronomoe.chronovisor_mixtral_bridge import ChronovisorMixtralForCausalLM
from chronomoe.mixtral_core import MixtralConfig
from chronomoe.training import TrainingConfig, ChronoMoELoss
from experiments.conversational_dataset import ConversationalDataset, ALL_CONVERSATIONAL_DOMAINS as ALL_DOMAINS
from experiments.analyze_turn_usage import TurnUsageAnalyzer, ThreeDomainDatasetPyTorch


def run_quick_training(model, train_loader, config, num_steps=1000):
    """Run quick training and return trained model."""

    print("\n" + "=" * 70)
    print(f"TRAINING FOR {num_steps} STEPS")
    print("=" * 70)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = ChronoMoELoss(
        lambda_balance=config.lambda_balance,
        lambda_coherence=config.lambda_coherence,
        lambda_valley=config.lambda_valley,
    )

    model.train()
    train_iter = iter(train_loader)
    controller = model.model.controller

    for step in range(num_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        logits, chrono_state = model(input_ids, update_chronovisor=True)

        loss, loss_components = loss_fn.compute(
            logits=logits,
            labels=labels,
            chrono_state=chrono_state,
            controller=controller,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            st_diag = controller.get_structural_temperature_diagnostics()
            print(f"Step {step:5d} | Loss: {loss.item():.4f} | R: {chrono_state.coherence:.3f} | T̄_var: {st_diag['variance']:.15f}")

    model.eval()
    return model


def analyze_model(model, test_loader, label="Model"):
    """Analyze turn-level expert usage for a model."""

    print(f"\n{'=' * 70}")
    print(f"ANALYZING: {label}")
    print(f"{'=' * 70}")

    analyzer = TurnUsageAnalyzer(model, num_turns=7)

    for batch_idx, batch in enumerate(test_loader):
        analyzer.analyze_batch(batch)

        if (batch_idx + 1) % 50 == 0:
            print(f"   Processed {(batch_idx + 1) * 4} samples...")

    # Get usage matrix for layer 0
    usage = analyzer.get_usage_matrix(layer_idx=0, normalize=True)

    # Print summary
    analyzer.print_usage_summary(layer_idx=0)

    # Diagnose specialization
    analyzer.diagnose_specialization(layer_idx=0, threshold=0.01)

    return usage, analyzer


def compare_models(untrained_usage, trained_usage):
    """Compare turn-level usage between untrained and trained models."""

    print("\n" + "=" * 70)
    print("COMPARISON: UNTRAINED vs TRAINED")
    print("=" * 70)

    # Compute difference matrix
    diff = trained_usage - untrained_usage

    # Compute L2 norm of difference (measure of how much changed)
    diff_norm = np.linalg.norm(diff)

    turn_names = ["Inquiry", "Premise", "Complication", "Contradiction",
                  "Exception", "Concession", "Synthesis"]

    print("\nChange in expert preferences per turn (trained - untrained):")
    for turn_idx, turn_name in enumerate(turn_names):
        turn_diff = diff[turn_idx]
        max_change_expert = np.argmax(np.abs(turn_diff))
        max_change = turn_diff[max_change_expert]

        print(f"\n{turn_name}:")
        print(f"  Max change: Expert {max_change_expert} ({max_change:+.3f})")
        print(f"  Change vector: {' '.join([f'{d:+.2f}' for d in turn_diff])}")

    print(f"\nOverall change (L2 norm): {diff_norm:.4f}")

    # Verdict
    print(f"\n{'=' * 70}")
    print("VERDICT:")
    print(f"{'=' * 70}")

    if diff_norm > 0.5:
        print(f"✅ STRONG PROTO-ROLE FORMATION")
        print(f"   L2 norm: {diff_norm:.4f} (threshold: 0.5)")
        print("\n   → Training caused significant shift in turn-level preferences")
        print("   → Experts learned phase-specific roles")
        print("   → T̄_var=0 is a WIRING/LOGGING ISSUE, not conceptual")
        print("\n   Next: Investigate structural T update logic with high-precision logging")
    elif diff_norm > 0.2:
        print(f"⚠️  MODERATE PROTO-ROLE FORMATION")
        print(f"   L2 norm: {diff_norm:.4f} (threshold: 0.2)")
        print("\n   → Some shift in preferences, but weak")
        print("   → May need longer training or stronger geometry influence")
    else:
        print(f"❌ NO PROTO-ROLE FORMATION")
        print(f"   L2 norm: {diff_norm:.4f} (threshold: 0.2)")
        print("\n   → Expert usage barely changed from random initialization")
        print("   → Router hasn't learned meaningful phase specialization")
        print("   → T̄_var=0 is CORRECT - world too flat for geometry")
        print("\n   Next: Increase geometry influence (α_P, α_T) or add more data asymmetry")


def main():
    """Run complete proto-role diagnostic."""

    print("=" * 70)
    print("PROTO-ROLE DIAGNOSTIC: FULL WORKFLOW")
    print("=" * 70)
    print("\nQuestion: Did proto-roles form during training?")
    print("Method: Compare turn-level expert usage before and after training")
    print("=" * 70)

    # Setup
    output_dir = Path("proto_role_results")
    output_dir.mkdir(exist_ok=True)

    # Generate dataset
    print("\n1. Generating conversational dataset...")
    dataset_gen = ConversationalDataset(seq_length=128, vocab_size=1000)
    full_dataset = dataset_gen.generate_dataset(num_sequences=5000, balanced=True)

    train_sequences = full_dataset["sequences"][500:]
    test_sequences = full_dataset["sequences"][:500]

    train_dataset = ThreeDomainDatasetPyTorch(train_sequences, vocab_size=1000)
    test_dataset = ThreeDomainDatasetPyTorch(test_sequences, vocab_size=1000)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Create model
    print("\n2. Creating model...")
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
    print(f"   Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # Analyze UNTRAINED model
    print("\n3. Analyzing UNTRAINED model (baseline)...")
    model.eval()
    untrained_usage, untrained_analyzer = analyze_model(model, test_loader, label="UNTRAINED (Baseline)")

    # Save untrained heatmap
    untrained_analyzer.plot_usage_heatmap(str(output_dir / "untrained_turn_usage.png"))

    # Train model
    print("\n4. Training model...")
    train_config = TrainingConfig(
        learning_rate=1e-4,
        max_steps=1000,
        batch_size=4,
        lambda_balance=0.01,
        lambda_coherence=0.001,
        lambda_valley=0.0001,
        fast_geology=True,
        log_every_n_steps=100,
        save_every_n_steps=1000,
        use_coherence_gating=False,
    )

    model = run_quick_training(model, train_loader, train_config, num_steps=1000)

    # Save trained model
    checkpoint_path = output_dir / "model_trained.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\n💾 Model checkpoint saved: {checkpoint_path}")

    # Analyze TRAINED model
    print("\n5. Analyzing TRAINED model...")
    model.eval()
    trained_usage, trained_analyzer = analyze_model(model, test_loader, label="TRAINED (After 1000 steps)")

    # Save trained heatmap
    trained_analyzer.plot_usage_heatmap(str(output_dir / "trained_turn_usage.png"))

    # Compare
    print("\n6. Comparing untrained vs trained...")
    compare_models(untrained_usage, trained_usage)

    print(f"\n{'=' * 70}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nResults saved to: {output_dir}/")
    print("  - untrained_turn_usage.png")
    print("  - trained_turn_usage.png")
    print("  - model_trained.pt")


if __name__ == "__main__":
    main()
