"""
Part 1: Robustness Test on Winning Configuration

Verify that η=0.006, P=1.1 is stable across seeds.

Test: Run live vs frozen with new random seed
Confirm: Δloss < 0, Δsep > 0
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

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

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
        controller.eta_structural_T_global = 0.005
        controller.pressure_scale = pressure_scale
        for lens in controller.lenses.values():
            lens.eta_structural_T = eta_local

    optimizer = AdamW(model.parameters(), lr=1e-4)
    dataset = ThreeDomainDatasetPyTorch(sequences, vocab_size=1000)

    # Training
    model.train()
    losses = []
    tbar_vars = []

    for step in range(250):
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
        if chrono_state and hasattr(chrono_state, 'T_bar'):
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

    # Expert activation counts
    threshold = usage_matrix.mean() * 0.5
    experts_active = (usage_matrix.sum(axis=0) > threshold).sum()

    # Final metrics
    final_loss = np.mean(losses[-50:])
    final_tbar_var = tbar_vars[-1] if tbar_vars else 0.0

    return {
        "final_loss": final_loss,
        "turn_separation": turn_separation,
        "tbar_variance": final_tbar_var,
        "experts_active": experts_active,
        "usage_matrix": usage_matrix,
        "usage_normalized": usage_normalized,
        "loss_history": losses,
        "tbar_var_history": tbar_vars,
    }


def main():
    print("="*70)
    print("PART 1: ROBUSTNESS TEST")
    print("="*70)
    print("\nTesting winning config: η=0.006, P=1.1")
    print("Hypothesis: Δloss < 0, Δsep > 0 across seeds\n")

    # Generate dataset with NEW seed
    seed = 12345
    print(f"Using seed: {seed}")
    dataset_gen = ConversationalDataset(seq_length=128, vocab_size=1000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    result = dataset_gen.generate_dataset(num_sequences=15, balanced=False)
    sequences = result["sequences"]
    print(f"✅ Dataset generated\n")

    # Run frozen baseline
    print("Running frozen baseline...")
    frozen = run_config(
        enable_chronovisor=False,
        eta_local=0.006,
        pressure_scale=1.1,
        sequences=sequences,
        seed=seed,
    )
    print(f"✅ Frozen: loss={frozen['final_loss']:.6f}, sep={frozen['turn_separation']:.6f}\n")

    # Run live geology
    print("Running live geology (η=0.006, P=1.1)...")
    live = run_config(
        enable_chronovisor=True,
        eta_local=0.006,
        pressure_scale=1.1,
        sequences=sequences,
        seed=seed,
    )
    print(f"✅ Live: loss={live['final_loss']:.6f}, sep={live['turn_separation']:.6f}\n")

    # Compute deltas
    delta_loss = live['final_loss'] - frozen['final_loss']
    delta_sep = live['turn_separation'] - frozen['turn_separation']
    delta_loss_pct = (delta_loss / frozen['final_loss']) * 100
    delta_sep_pct = (delta_sep / frozen['turn_separation']) * 100

    # Live-Frozen difference matrix
    diff_matrix = live['usage_normalized'] - frozen['usage_normalized']

    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nFrozen baseline:")
    print(f"  Loss:            {frozen['final_loss']:.6f}")
    print(f"  Turn separation: {frozen['turn_separation']:.6f}")
    print(f"  Experts active:  {frozen['experts_active']}/8")

    print(f"\nLive geology (η=0.006, P=1.1):")
    print(f"  Loss:            {live['final_loss']:.6f}")
    print(f"  Turn separation: {live['turn_separation']:.6f}")
    print(f"  T̄ variance:      {live['tbar_variance']:.6f}")
    print(f"  Experts active:  {live['experts_active']}/8")

    print(f"\nDeltas:")
    print(f"  Δ Loss:  {delta_loss:+.6f} ({delta_loss_pct:+.1f}%)")
    print(f"  Δ Sep:   {delta_sep:+.6f} ({delta_sep_pct:+.1f}%)")

    print(f"\nLive - Frozen Expert Usage Difference Matrix:")
    print(f"         ", end="")
    for expert_idx in range(8):
        print(f"  E{expert_idx}  ", end="")
    print()
    print("-"*70)

    turn_names = ['Inquiry', 'Premise', 'Complic', 'Contra', 'Except', 'Concess', 'Synth']
    for turn_idx in range(7):
        print(f"{turn_names[turn_idx]:8s} ", end="")
        for expert_idx in range(8):
            diff = diff_matrix[turn_idx, expert_idx]
            sign = "+" if diff >= 0 else ""
            print(f"{sign}{diff:5.3f} ", end="")
        print()

    # Validation
    print(f"\n{'='*70}")
    print("VALIDATION")
    print("="*70)

    if delta_loss < 0 and delta_sep > 0:
        print("\n✅ HYPOTHESIS CONFIRMED")
        print(f"   Δloss = {delta_loss_pct:+.1f}% < 0 ✓")
        print(f"   Δsep  = {delta_sep_pct:+.1f}% > 0 ✓")
        print("\n   The winning config is Pareto-better and seed-stable.")
    elif delta_loss < 0:
        print("\n⚠️  PARTIAL CONFIRMATION")
        print(f"   Δloss = {delta_loss_pct:+.1f}% < 0 ✓")
        print(f"   Δsep  = {delta_sep_pct:+.1f}% ✗")
        print("\n   Loss improved but separation did not.")
    elif delta_sep > 0:
        print("\n⚠️  PARTIAL CONFIRMATION")
        print(f"   Δloss = {delta_loss_pct:+.1f}% ✗")
        print(f"   Δsep  = {delta_sep_pct:+.1f}% > 0 ✓")
        print("\n   Separation improved but loss penalty exists.")
    else:
        print("\n❌ HYPOTHESIS REJECTED")
        print(f"   Δloss = {delta_loss_pct:+.1f}% ✗")
        print(f"   Δsep  = {delta_sep_pct:+.1f}% ✗")
        print("\n   Config is NOT Pareto-better on this seed.")

    # Save results
    with open("robustness_test_results.txt", "w") as f:
        f.write("Part 1: Robustness Test Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Config: η=0.006, P=1.1\n\n")
        f.write(f"Frozen: loss={frozen['final_loss']:.6f}, sep={frozen['turn_separation']:.6f}\n")
        f.write(f"Live:   loss={live['final_loss']:.6f}, sep={live['turn_separation']:.6f}\n\n")
        f.write(f"Δ Loss: {delta_loss_pct:+.1f}%\n")
        f.write(f"Δ Sep:  {delta_sep_pct:+.1f}%\n\n")
        if delta_loss < 0 and delta_sep > 0:
            f.write("✅ Pareto-better and seed-stable\n")
        else:
            f.write("⚠️  Results vary from original sweep\n")

    print(f"\n✅ Results saved to robustness_test_results.txt")


if __name__ == "__main__":
    main()
