"""
Temperature Intervention Experiment

Tests whether ChronoMoE's geological temperature system exhibits self-correction.

Experimental Design:
1. Train normally for N steps (baseline)
2. Artificially heat a single expert for M steps (perturbation)
3. Remove intervention and continue training (recovery)
4. Measure:
   - Usage reaction during heating (should decrease)
   - Temperature self-correction after removal (should cool back down)
   - Valley self-stabilization (other experts adjust)

This demonstrates that the P×T geometry actively controls routing,
not just logs it.
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
from experiments.train_conversational import ThreeDomainDatasetPyTorch


def temperature_intervention_experiment():
    """
    Main intervention experiment.

    Timeline:
    - Steps 0-200: Normal training (baseline)
    - Steps 200-400: Heat Expert 0 by +1.0 (intervention)
    - Steps 400-600: Normal training (recovery)
    """

    print("=" * 70)
    print("TEMPERATURE INTERVENTION EXPERIMENT")
    print("=" * 70)
    print("\nTimeline:")
    print("  Steps 0-200:   Normal training (BASELINE)")
    print("  Steps 200-400: Heat Expert 0 by +1.0 (INTERVENTION)")
    print("  Steps 400-600: Normal training (RECOVERY)")
    print("=" * 70)

    # Create dataset
    print("\n1. Generating dataset...")
    dataset_gen = ConversationalDataset(seq_length=128, vocab_size=1000)
    sequences = dataset_gen.generate_dataset(num_sequences=200, balanced=True)["sequences"]
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
    loss_fn = ChronoMoELoss(
        lambda_balance=0.01,
        lambda_coherence=0.001,
        lambda_valley=0.0001
    )

    # Tracking
    history = {
        'step': [],
        'loss': [],
        'coherence': [],
        'expert_0_usage': [],
        'expert_0_T_fast': [],
        'expert_0_T_structural': [],
        'T_bar_var': [],
        'phase': [],  # 'baseline', 'intervention', 'recovery'
    }

    print("\n3. Running intervention experiment...")
    print("\nStep | Phase       | Loss   | R     | E0_usage | E0_T_fast | E0_T_struct | T̄_var")
    print("-" * 95)

    model.train()
    train_iter = iter(loader)

    BASELINE_END = 200
    INTERVENTION_END = 400
    TOTAL_STEPS = 600

    INTERVENTION_TARGET_EXPERT = 0
    INTERVENTION_DELTA_T = 1.0  # Add 1.0 to expert 0's temperature

    for step in range(TOTAL_STEPS):
        # Determine phase
        if step < BASELINE_END:
            phase = 'baseline'
        elif step < INTERVENTION_END:
            phase = 'intervention'
        else:
            phase = 'recovery'

        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # Apply intervention if in intervention phase
        if phase == 'intervention':
            # Heat expert 0 by adding to its fast temperature
            for layer_idx, lens in controller.lenses.items():
                lens.temperature_fast[INTERVENTION_TARGET_EXPERT] += INTERVENTION_DELTA_T
                # Recompute effective temperature
                lens.temperature_effective = lens.temperature_fast * lens.structural_T_hierarchical

        # Forward pass
        logits, chrono_state = model(input_ids, update_chronovisor=True)

        # Compute loss
        loss, _ = loss_fn.compute(logits, labels, chrono_state, controller)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track metrics
        lens_0 = controller.lenses[0]
        expert_0_usage = controller.expert_usage[0][INTERVENTION_TARGET_EXPERT]
        expert_0_T_fast = lens_0.temperature_fast[INTERVENTION_TARGET_EXPERT]
        expert_0_T_struct = lens_0.structural_T[INTERVENTION_TARGET_EXPERT]
        st_diag = controller.get_structural_temperature_diagnostics()

        history['step'].append(step)
        history['loss'].append(loss.item())
        history['coherence'].append(chrono_state.coherence)
        history['expert_0_usage'].append(expert_0_usage)
        history['expert_0_T_fast'].append(expert_0_T_fast)
        history['expert_0_T_structural'].append(expert_0_T_struct)
        history['T_bar_var'].append(st_diag['variance'])
        history['phase'].append(phase)

        # Log at key checkpoints
        if step % 50 == 0 or step in [BASELINE_END-1, BASELINE_END, INTERVENTION_END-1, INTERVENTION_END]:
            print(f"{step:4d} | {phase:11s} | {loss.item():6.3f} | {chrono_state.coherence:.3f} | "
                  f"{expert_0_usage:8.1f} | {expert_0_T_fast:9.4f} | {expert_0_T_struct:11.4f} | "
                  f"{st_diag['variance']:.6e}")

    # Analysis
    print("\n" + "=" * 70)
    print("INTERVENTION ANALYSIS")
    print("=" * 70)

    # Extract phases
    baseline_usage = [history['expert_0_usage'][i] for i, p in enumerate(history['phase']) if p == 'baseline']
    intervention_usage = [history['expert_0_usage'][i] for i, p in enumerate(history['phase']) if p == 'intervention']
    recovery_usage = [history['expert_0_usage'][i] for i, p in enumerate(history['phase']) if p == 'recovery']

    baseline_T_fast = [history['expert_0_T_fast'][i] for i, p in enumerate(history['phase']) if p == 'baseline']
    intervention_T_fast = [history['expert_0_T_fast'][i] for i, p in enumerate(history['phase']) if p == 'intervention']
    recovery_T_fast = [history['expert_0_T_fast'][i] for i, p in enumerate(history['phase']) if p == 'recovery']

    print("\n1. USAGE RESPONSE TO HEATING:")
    print(f"   Baseline usage (mean):       {np.mean(baseline_usage):.2f}")
    print(f"   Intervention usage (mean):   {np.mean(intervention_usage):.2f}")
    print(f"   Recovery usage (mean):       {np.mean(recovery_usage):.2f}")

    usage_drop = np.mean(baseline_usage) - np.mean(intervention_usage)
    if usage_drop > 0:
        print(f"   ✅ Usage DECREASED by {usage_drop:.2f} during heating (expected!)")
        print(f"      → High temperature makes routing diffuse (exploration)")
    else:
        print(f"   ⚠️  Usage did not decrease during heating ({usage_drop:.2f})")

    print("\n2. TEMPERATURE SELF-CORRECTION:")
    print(f"   Baseline T_fast (mean):      {np.mean(baseline_T_fast):.4f}")
    print(f"   Intervention T_fast (mean):  {np.mean(intervention_T_fast):.4f}")
    print(f"   Recovery T_fast (mean):      {np.mean(recovery_T_fast):.4f}")

    recovery_happened = np.mean(recovery_T_fast) < np.mean(intervention_T_fast)
    if recovery_happened:
        cooling = np.mean(intervention_T_fast) - np.mean(recovery_T_fast)
        print(f"   ✅ Temperature COOLED by {cooling:.4f} after intervention removed")
        print(f"      → Geological EMA self-corrected back toward equilibrium")
    else:
        print(f"   ⚠️  Temperature did not self-correct after intervention")

    print("\n3. STRUCTURAL TEMPERATURE INTEGRATION:")
    baseline_T_struct = [history['expert_0_T_structural'][i] for i, p in enumerate(history['phase']) if p == 'baseline']
    recovery_T_struct = [history['expert_0_T_structural'][i] for i, p in enumerate(history['phase']) if p == 'recovery']

    structural_changed = abs(np.mean(recovery_T_struct) - np.mean(baseline_T_struct))
    print(f"   Baseline T̄ (mean):           {np.mean(baseline_T_struct):.4f}")
    print(f"   Recovery T̄ (mean):           {np.mean(recovery_T_struct):.4f}")
    print(f"   Change in T̄:                 {structural_changed:.4f}")

    if structural_changed > 0.01:
        print(f"   ✅ Structural T̄ integrated the perturbation (geological memory)")
    else:
        print(f"   ⚠️  Structural T̄ barely changed (may need more intervention steps)")

    # Visualization
    print("\n4. Creating plots...")
    output_dir = Path("intervention_results")
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Expert 0 usage over time
    ax = axes[0, 0]
    ax.plot(history['step'], history['expert_0_usage'], linewidth=1, alpha=0.7)
    ax.axvline(BASELINE_END, color='red', linestyle='--', label='Intervention Start')
    ax.axvline(INTERVENTION_END, color='green', linestyle='--', label='Intervention End')
    ax.axhspan(0, 1000, xmin=BASELINE_END/TOTAL_STEPS, xmax=INTERVENTION_END/TOTAL_STEPS,
               alpha=0.1, color='red', label='Heated Period')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Expert 0 Usage (Cumulative)')
    ax.set_title('Expert 0 Usage During Temperature Intervention')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Expert 0 T_fast over time
    ax = axes[0, 1]
    ax.plot(history['step'], history['expert_0_T_fast'], linewidth=1, alpha=0.7)
    ax.axvline(BASELINE_END, color='red', linestyle='--')
    ax.axvline(INTERVENTION_END, color='green', linestyle='--')
    ax.axhspan(0, 10, xmin=BASELINE_END/TOTAL_STEPS, xmax=INTERVENTION_END/TOTAL_STEPS,
               alpha=0.1, color='red')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Expert 0 T_fast')
    ax.set_title('Expert 0 Fast Temperature (with +1.0 boost during intervention)')
    ax.grid(alpha=0.3)

    # Plot 3: Expert 0 structural T over time
    ax = axes[1, 0]
    ax.plot(history['step'], history['expert_0_T_structural'], linewidth=1, alpha=0.7)
    ax.axvline(BASELINE_END, color='red', linestyle='--')
    ax.axvline(INTERVENTION_END, color='green', linestyle='--')
    ax.axhspan(0, 10, xmin=BASELINE_END/TOTAL_STEPS, xmax=INTERVENTION_END/TOTAL_STEPS,
               alpha=0.1, color='red')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Expert 0 T̄ (structural)')
    ax.set_title('Expert 0 Structural Temperature (geological memory via EMA)')
    ax.grid(alpha=0.3)

    # Plot 4: Global T̄ variance
    ax = axes[1, 1]
    ax.plot(history['step'], history['T_bar_var'], linewidth=1, alpha=0.7)
    ax.axvline(BASELINE_END, color='red', linestyle='--')
    ax.axvline(INTERVENTION_END, color='green', linestyle='--')
    ax.axhspan(0, 1, xmin=BASELINE_END/TOTAL_STEPS, xmax=INTERVENTION_END/TOTAL_STEPS,
               alpha=0.1, color='red')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('T̄ Variance')
    ax.set_title('Global Structural Temperature Variance')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "temperature_intervention.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   📊 Plot saved: {plot_path}")

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if usage_drop > 0 and recovery_happened:
        print("\n✅ VALLEY SELF-CORRECTION DEMONSTRATED!")
        print(f"   - Heating expert decreased its usage (T↑ → routing diffuses)")
        print(f"   - Removing heat allowed temperature to self-correct")
        print(f"   - The P×T geometry actively controls routing, not just logs it")
        print(f"\n   This is geological self-regulation in action.")
    elif usage_drop > 0:
        print("\n⚠️  PARTIAL SUCCESS")
        print(f"   - Heating did affect usage (good!)")
        print(f"   - But temperature didn't fully self-correct yet")
        print(f"   - May need longer recovery period for EMA to equilibrate")
    else:
        print("\n❌ INTERVENTION HAD NO EFFECT")
        print(f"   - Heating did not change usage as expected")
        print(f"   - Temperature may not be properly coupled to routing")
        print(f"   - Check if temperature is being applied to router logits")


if __name__ == "__main__":
    temperature_intervention_experiment()
