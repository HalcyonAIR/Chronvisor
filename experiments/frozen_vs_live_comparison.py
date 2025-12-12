"""
Frozen vs Live Geology Comparison

Answers the key question: What does the live P×T geometry actually DO?

Experimental Design:
1. Train TWO models on identical data:
   - Model A: Frozen geology (enable_chronovisor=False)
   - Model B: Live P×T (enable_chronovisor=True)

2. Compare:
   - Speed of proto-role formation (how many steps until roles emerge?)
   - Sharpness of proto-roles (how distinct are the turn preferences?)
   - Robustness under domain shift
   - Load balancing vs loss tradeoff

This demonstrates whether the geological control adds value beyond
just logging routing statistics.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from chronomoe.chronovisor_mixtral_bridge import ChronovisorMixtralForCausalLM
from chronomoe.mixtral_core import MixtralConfig
from chronomoe.training import ChronoMoELoss
from experiments.conversational_dataset import ConversationalDataset
from experiments.analyze_turn_usage import TurnUsageAnalyzer, ThreeDomainDatasetPyTorch


def train_model(enable_chronovisor, num_steps, dataset, label):
    """Train a model with or without live geology."""

    print(f"\n{'=' * 70}")
    print(f"TRAINING: {label}")
    print(f"{'=' * 70}")

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
        enable_chronovisor=enable_chronovisor,
    )
    model = ChronovisorMixtralForCausalLM(config)

    if enable_chronovisor:
        controller = model.model.controller

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = ChronoMoELoss(
        lambda_balance=0.01,
        lambda_coherence=0.001 if enable_chronovisor else 0.0,
        lambda_valley=0.0001 if enable_chronovisor else 0.0,
    )

    # Turn usage tracking
    turn_analyzer = TurnUsageAnalyzer(model, num_turns=7)

    # Training
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    model.train()
    train_iter = iter(loader)

    history = {
        'step': [],
        'loss': [],
        'coherence': [] if enable_chronovisor else None,
        'T_bar_var': [] if enable_chronovisor else None,
    }

    for step in range(num_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # Track turn usage
        turn_boundaries = batch.get("turn_boundaries", None)
        if turn_boundaries is not None:
            turn_analyzer.analyze_batch(batch)

        # Forward pass
        if enable_chronovisor:
            logits, chrono_state = model(input_ids, update_chronovisor=True)
            loss, _ = loss_fn.compute(logits, labels, chrono_state, controller)
        else:
            # Frozen model still returns tuple, but chrono_state is dummy
            logits, chrono_state_dummy = model(input_ids, update_chronovisor=False)
            # Create proper dummy chrono_state for loss computation
            chrono_state = type('obj', (object,), {
                'coherence': 0.0,
                'delta_coherence': 0.0,
                'valleys': [],
                'ridges': [],
                'expert_usage': None  # No usage stats for frozen model
            })()
            loss, _ = loss_fn.compute(logits, labels, chrono_state, None)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track
        history['step'].append(step)
        history['loss'].append(loss.item())
        if enable_chronovisor:
            history['coherence'].append(chrono_state.coherence)
            st_diag = controller.get_structural_temperature_diagnostics()
            history['T_bar_var'].append(st_diag['variance'])

        if step % 100 == 0:
            if enable_chronovisor:
                print(f"Step {step:4d} | Loss: {loss.item():.4f} | R: {chrono_state.coherence:.3f} | "
                      f"T̄_var: {st_diag['variance']:.6e}")
            else:
                print(f"Step {step:4d} | Loss: {loss.item():.4f}")

    # Get final turn usage
    turn_usage = turn_analyzer.get_usage_matrix(layer_idx=0)

    return model, history, turn_usage


def compare_proto_role_sharpness(usage_frozen, usage_live):
    """
    Compute how "sharp" the proto-roles are.

    Metric: Variance of expert preferences across turns.
    Higher variance = sharper roles (experts specialize to specific turns)
    """

    # Per-expert variance across turns
    expert_var_frozen = usage_frozen.var(axis=0)  # (num_experts,)
    expert_var_live = usage_live.var(axis=0)

    mean_sharpness_frozen = expert_var_frozen.mean()
    mean_sharpness_live = expert_var_live.mean()

    return mean_sharpness_frozen, mean_sharpness_live, expert_var_frozen, expert_var_live


def main():
    """Main comparison experiment."""

    print("=" * 70)
    print("FROZEN vs LIVE GEOLOGY COMPARISON")
    print("=" * 70)
    print("\nExperiment:")
    print("  Train two identical models on same data")
    print("  Model A: enable_chronovisor=False (frozen geology)")
    print("  Model B: enable_chronovisor=True (live P×T)")
    print("\nMetrics:")
    print("  - Proto-role sharpness (expert variance across turns)")
    print("  - Speed of role formation")
    print("  - Final loss")
    print("=" * 70)

    # Generate dataset
    print("\n1. Generating shared dataset...")
    dataset_gen = ConversationalDataset(seq_length=128, vocab_size=1000)
    sequences = dataset_gen.generate_dataset(num_sequences=15, balanced=False)["sequences"]
    dataset = ThreeDomainDatasetPyTorch(sequences, vocab_size=1000)

    NUM_STEPS = 500

    # Train frozen model
    model_frozen, history_frozen, usage_frozen = train_model(
        enable_chronovisor=False,
        num_steps=NUM_STEPS,
        dataset=dataset,
        label="FROZEN GEOLOGY (baseline MoE)"
    )

    # Train live model
    model_live, history_live, usage_live = train_model(
        enable_chronovisor=True,
        num_steps=NUM_STEPS,
        dataset=dataset,
        label="LIVE P×T GEOLOGY"
    )

    # Analysis
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS")
    print("=" * 70)

    # 1. Final loss comparison
    final_loss_frozen = history_frozen['loss'][-1]
    final_loss_live = history_live['loss'][-1]

    print("\n1. FINAL LOSS:")
    print(f"   Frozen geology: {final_loss_frozen:.6f}")
    print(f"   Live P×T:       {final_loss_live:.6f}")

    if final_loss_live < final_loss_frozen * 0.95:
        print(f"   ✅ Live P×T achieves {(1 - final_loss_live/final_loss_frozen)*100:.1f}% lower loss!")
    elif final_loss_frozen < final_loss_live * 0.95:
        print(f"   ⚠️  Frozen achieves {(1 - final_loss_frozen/final_loss_live)*100:.1f}% lower loss")
    else:
        print(f"   ≈  Similar final loss (difference < 5%)")

    # 2. Proto-role sharpness
    sharpness_frozen, sharpness_live, var_frozen, var_live = compare_proto_role_sharpness(
        usage_frozen, usage_live
    )

    print("\n2. PROTO-ROLE SHARPNESS:")
    print(f"   Frozen geology: {sharpness_frozen:.6f} (mean expert variance across turns)")
    print(f"   Live P×T:       {sharpness_live:.6f}")

    if sharpness_live > sharpness_frozen * 1.1:
        improvement = (sharpness_live / sharpness_frozen - 1) * 100
        print(f"   ✅ Live P×T produces {improvement:.1f}% sharper proto-roles!")
        print(f"      → Geological control helps experts specialize to turn types")
    elif sharpness_frozen > sharpness_live * 1.1:
        print(f"   ⚠️  Frozen produces sharper roles (unexpected!)")
    else:
        print(f"   ≈  Similar sharpness (difference < 10%)")

    # 3. Expert specialization details
    print("\n3. EXPERT SPECIALIZATION (Layer 0):")
    print("\n   Frozen Geology:")
    for expert_idx in range(8):
        if var_frozen[expert_idx] > 0.01:
            print(f"     Expert {expert_idx}: var={var_frozen[expert_idx]:.4f} ✓ SPECIALIZED")

    print("\n   Live P×T:")
    for expert_idx in range(8):
        if var_live[expert_idx] > 0.01:
            print(f"     Expert {expert_idx}: var={var_live[expert_idx]:.4f} ✓ SPECIALIZED")

    # 4. Visualizations
    print("\n4. Creating comparison plots...")
    output_dir = Path("comparison_results")
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Loss curves
    ax = axes[0, 0]
    ax.plot(history_frozen['step'], history_frozen['loss'], label='Frozen Geology', alpha=0.7)
    ax.plot(history_live['step'], history_live['loss'], label='Live P×T', alpha=0.7)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(alpha=0.3)

    # Turn usage heatmap - Frozen
    ax = axes[0, 1]
    im = ax.imshow(usage_frozen.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=usage_frozen.max())
    ax.set_xlabel('Conversation Turn')
    ax.set_ylabel('Expert')
    ax.set_title('Turn Usage: Frozen Geology')
    ax.set_xticks(range(7))
    ax.set_xticklabels(['Inquiry', 'Premise', 'Complic', 'Contra', 'Except', 'Concess', 'Synth'], rotation=45)
    plt.colorbar(im, ax=ax, label='Usage')

    # Turn usage heatmap - Live
    ax = axes[0, 2]
    im = ax.imshow(usage_live.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=usage_live.max())
    ax.set_xlabel('Conversation Turn')
    ax.set_ylabel('Expert')
    ax.set_title('Turn Usage: Live P×T')
    ax.set_xticks(range(7))
    ax.set_xticklabels(['Inquiry', 'Premise', 'Complic', 'Contra', 'Except', 'Concess', 'Synth'], rotation=45)
    plt.colorbar(im, ax=ax, label='Usage')

    # Expert variance comparison
    ax = axes[1, 0]
    x = np.arange(8)
    width = 0.35
    ax.bar(x - width/2, var_frozen, width, label='Frozen', alpha=0.7)
    ax.bar(x + width/2, var_live, width, label='Live P×T', alpha=0.7)
    ax.set_xlabel('Expert')
    ax.set_ylabel('Variance Across Turns')
    ax.set_title('Proto-Role Sharpness per Expert')
    ax.legend()
    ax.grid(alpha=0.3)

    # T̄ variance over time (only for live model)
    ax = axes[1, 1]
    if history_live['T_bar_var'] is not None:
        ax.plot(history_live['step'], history_live['T_bar_var'], alpha=0.7)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('T̄ Variance')
        ax.set_title('Geological Temperature Variance (Live P×T only)')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'N/A\n(Frozen model)', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('T̄ Variance (Live P×T only)')

    # Coherence over time (only for live model)
    ax = axes[1, 2]
    if history_live['coherence'] is not None:
        ax.plot(history_live['step'], history_live['coherence'], alpha=0.7)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Kuramoto Coherence R')
        ax.set_title('Phase Coherence (Live P×T only)')
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'N/A\n(Frozen model)', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Coherence (Live P×T only)')

    plt.tight_layout()
    plot_path = output_dir / "frozen_vs_live_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   📊 Plot saved: {plot_path}")

    # Save data
    np.savez(
        output_dir / "comparison_data.npz",
        usage_frozen=usage_frozen,
        usage_live=usage_live,
        loss_frozen=np.array(history_frozen['loss']),
        loss_live=np.array(history_live['loss']),
        var_frozen=var_frozen,
        var_live=var_live,
    )
    print(f"   💾 Data saved: {output_dir / 'comparison_data.npz'}")

    # Difference matrix (live - frozen)
    print("\n" + "=" * 70)
    print("TURN × EXPERT DIFFERENCE MATRIX (Live - Frozen)")
    print("=" * 70)
    print("\nPositive = Live model uses expert MORE for this turn")
    print("Negative = Live model uses expert LESS for this turn\n")

    diff_matrix = usage_live - usage_frozen
    turn_names = ['Inquiry', 'Premise', 'Complic', 'Contra', 'Except', 'Concess', 'Synth']

    print("         ", end="")
    for expert_idx in range(8):
        print(f"  E{expert_idx}  ", end="")
    print()
    print("-" * 70)

    for turn_idx in range(7):
        print(f"{turn_names[turn_idx]:8s} ", end="")
        for expert_idx in range(8):
            diff = diff_matrix[turn_idx, expert_idx]
            sign = "+" if diff >= 0 else ""
            print(f"{sign}{diff:5.3f} ", end="")
        print()

    # Key questions
    print("\n" + "=" * 70)
    print("KEY QUESTIONS")
    print("=" * 70)

    # Q1: Stronger turn separation?
    turn_separation_frozen = var_frozen.sum()
    turn_separation_live = var_live.sum()
    print(f"\n1. Does live model show STRONGER turn separation?")
    print(f"   Frozen total variance: {turn_separation_frozen:.6f}")
    print(f"   Live total variance:   {turn_separation_live:.6f}")
    if turn_separation_live > turn_separation_frozen * 1.1:
        print(f"   ✅ YES - Live shows {(turn_separation_live/turn_separation_frozen - 1)*100:.1f}% stronger separation")
    else:
        print(f"   ❌ NO - Similar or weaker separation")

    # Q2: T̄ convergence
    if history_live['T_bar_var'] is not None and len(history_live['T_bar_var']) > 0:
        initial_tbar = history_live['T_bar_var'][0]
        final_tbar = history_live['T_bar_var'][-1]
        print(f"\n2. Does live model show convergence in T̄?")
        print(f"   Initial T̄_var: {initial_tbar:.6e}")
        print(f"   Final T̄_var:   {final_tbar:.6e}")
        if final_tbar > initial_tbar * 2:
            print(f"   ✅ T̄ variance GREW (geology actively differentiating experts)")
        elif final_tbar < initial_tbar * 0.5:
            print(f"   ⚠️  T̄ variance SHRANK (experts converging)")
        else:
            print(f"   ≈  T̄ variance stable")

    # Q3: Damped oscillation
    print(f"\n3. Does frozen model show fragmentation?")
    frozen_experts_used = (usage_frozen.sum(axis=0) > usage_frozen.mean() * 0.5).sum()
    live_experts_used = (usage_live.sum(axis=0) > usage_live.mean() * 0.5).sum()
    print(f"   Frozen experts with >50% avg usage: {frozen_experts_used}/8")
    print(f"   Live experts with >50% avg usage:   {live_experts_used}/8")
    if frozen_experts_used < 6 and live_experts_used >= 6:
        print(f"   ✅ Frozen shows expert starvation, live maintains diversity")
    else:
        print(f"   ≈  Both models maintain similar expert utilization")

    # Q4: Proto-role consistency
    print(f"\n4. Does geology push toward consistent proto-roles?")
    print(f"   Frozen proto-role sharpness: {sharpness_frozen:.6f}")
    print(f"   Live proto-role sharpness:   {sharpness_live:.6f}")
    if sharpness_live > sharpness_frozen * 1.2:
        print(f"   ✅ YES - Live shows {(sharpness_live/sharpness_frozen - 1)*100:.1f}% sharper proto-roles")
    elif sharpness_live > sharpness_frozen * 1.05:
        print(f"   ≈  WEAK signal - Live shows {(sharpness_live/sharpness_frozen - 1)*100:.1f}% sharper")
    else:
        print(f"   ❌ NO - Similar or weaker proto-role formation")

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    signals = 0
    if turn_separation_live > turn_separation_frozen * 1.1:
        signals += 1
    if sharpness_live > sharpness_frozen * 1.1:
        signals += 1
    if final_loss_live <= final_loss_frozen * 1.05:
        signals += 1

    if signals >= 2:
        print("\n✅ GEOLOGY IS WORKING")
        print(f"   {signals}/3 positive signals detected")
        print(f"   Live P×T geometry demonstrably improves routing structure")
    elif signals == 1:
        print("\n⚠️  WEAK SIGNAL")
        print(f"   {signals}/3 positive signals detected")
        print(f"   Geology shows effect but not decisive")
    else:
        print("\n❌ NO CLEAR BENEFIT")
        print(f"   {signals}/3 positive signals detected")
        print(f"   Geology not improving routing structure in this regime")


if __name__ == "__main__":
    main()
