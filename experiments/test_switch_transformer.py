#!/usr/bin/env python3
"""
Switch Transformer Test: Top-1 Routing with P×T Coupling

Tests whether P×T coupling generalizes to Switch Transformer's top-1 routing.

Hypothesis: P×T coupling should work with top-1 routing, demonstrating
generalization across routing mechanisms (top-1 vs top-k).

Key differences from Mixtral:
- Top-1 routing (each token → 1 expert) vs top-2 in Mixtral
- Load balancing auxiliary loss
- Typically uses more experts (but we'll use 8 for comparison)

Scientific value:
- Tests routing mechanism generalization
- Different load balancing dynamics (aux loss vs implicit)
- Stronger expert specialization pressure (top-1)

If successful, demonstrates P×T coupling is routing-agnostic.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from pathlib import Path

from chronomoe.switch_core import SwitchConfig
from chronomoe.chronovisor_switch_bridge import ChronovisorSwitchForCausalLM

# Import conversation generator
from generate_long_geeky_conversations import LongGeekyConversationGenerator


def run_training(config, enable_chronovisor, eta, pressure_scale, seed, num_conversations=10, num_epochs=50):
    """Run training with specified configuration."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model
    model = ChronovisorSwitchForCausalLM(config)

    # Configure Chronovisor if enabled
    if enable_chronovisor:
        model.model.controller.eta_structural_T_local = eta
        model.model.controller.eta_structural_T_global = eta / 2.0
        model.model.controller.pressure_scale = pressure_scale

    # Generate long conversations
    print(f"  Generating {num_conversations} long conversations...")
    gen = LongGeekyConversationGenerator(vocab_size=config.vocab_size)
    dataset = gen.generate_dataset(num_conversations=num_conversations)
    conversations = [seq['input_ids'].tolist() for seq in dataset['sequences']]

    avg_len = np.mean([len(c) for c in conversations])
    print(f"  Average conversation length: {avg_len:.0f} tokens")

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Metrics tracking
    losses = []
    aux_losses = []
    turn_separations = []
    coherence_history = []
    T_bar_history = []

    print(f"  Training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_aux_loss = 0.0
        epoch_sep = 0.0

        for conv_tokens in conversations:
            input_ids = torch.tensor(conv_tokens[:-1]).unsqueeze(0)
            target_ids = torch.tensor(conv_tokens[1:]).unsqueeze(0)

            # Forward pass (includes auxiliary loss)
            total_loss, aux_loss = model.compute_loss(input_ids, target_ids)

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Track losses
            ce_loss = total_loss.item() - aux_loss.item()
            epoch_loss += ce_loss
            epoch_aux_loss += aux_loss.item()

            # Get chronovisor state for metrics
            _, chrono_state, _ = model(input_ids, update_chronovisor=False)

            # Track coherence and T̄
            if enable_chronovisor and chrono_state is not None:
                coherence_history.append(chrono_state.coherence)
                if chrono_state.T_bar is not None:
                    T_bar_history.append(chrono_state.T_bar.copy())

            # Compute turn separation
            if enable_chronovisor and chrono_state is not None:
                for layer_idx, usage in chrono_state.expert_usage.items():
                    if usage.sum() > 0:
                        epoch_sep += np.var(usage)

        losses.append(epoch_loss / len(conversations))
        aux_losses.append(epoch_aux_loss / len(conversations))
        turn_separations.append(epoch_sep / len(conversations))

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{num_epochs}: loss={losses[-1]:.4f}, aux={aux_losses[-1]:.6f}, sep={turn_separations[-1]:.6f}")

    # Return final metrics
    final_loss = np.mean(losses[-10:])
    final_aux_loss = np.mean(aux_losses[-10:])
    final_sep = np.mean(turn_separations[-10:])

    # T̄ variance
    T_bar_var = 0.0
    if T_bar_history:
        T_bar_means = [np.mean(t) for t in T_bar_history[-50:]]
        T_bar_var = np.var(T_bar_means)

    return {
        'final_loss': final_loss,
        'final_aux_loss': final_aux_loss,
        'final_sep': final_sep,
        'T_bar_variance': T_bar_var,
        'losses': losses,
        'aux_losses': aux_losses,
        'separations': turn_separations,
        'coherence': coherence_history,
    }


def main():
    print("=" * 70)
    print("SWITCH TRANSFORMER: Top-1 Routing with P×T Coupling")
    print("=" * 70)
    print()
    print("Testing P×T coupling generalization to Switch Transformer's")
    print("top-1 routing mechanism.")
    print()

    # Switch configuration (comparable to Mixtral baseline)
    switch_config = SwitchConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=1024,
        num_layers=2,
        num_experts=8,
        num_experts_per_token=1,  # TOP-1 ROUTING (key difference)
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        max_seq_length=2048,
        enable_chronovisor=True,
        router_aux_loss_coef=0.01,  # Load balancing loss
    )

    # Stable basin parameters (transfer from Mixtral)
    eta = 0.015
    pressure_scale = 0.5

    # Test with 3 seeds
    seeds = [42, 12345, 67890]

    print("ARCHITECTURE COMPARISON:")
    print("-" * 70)
    print(f"                    Mixtral (baseline)   Switch (test)")
    print("-" * 70)
    print(f"Layers              2                    {switch_config.num_layers}")
    print(f"Experts             8                    {switch_config.num_experts}")
    print(f"Routing strategy    top-2                top-{switch_config.num_experts_per_token}")
    print(f"Hidden dim          256                  {switch_config.hidden_dim}")
    print(f"Load balancing      implicit             auxiliary loss")
    print("-" * 70)
    print()

    print(f"P×T Configuration (transfer from Mixtral):")
    print(f"  η (structural T learning rate): {eta}")
    print(f"  Pressure scale: {pressure_scale}")
    print(f"  Seeds: {seeds}")
    print()

    # Run frozen baseline for Switch
    print("=" * 70)
    print("STEP 1: Frozen Baseline (Switch, no Chronovisor)")
    print("=" * 70)

    frozen_config = SwitchConfig(
        vocab_size=switch_config.vocab_size,
        hidden_dim=switch_config.hidden_dim,
        intermediate_dim=switch_config.intermediate_dim,
        num_layers=switch_config.num_layers,
        num_experts=switch_config.num_experts,
        num_experts_per_token=switch_config.num_experts_per_token,
        num_attention_heads=switch_config.num_attention_heads,
        num_key_value_heads=switch_config.num_key_value_heads,
        head_dim=switch_config.head_dim,
        max_seq_length=switch_config.max_seq_length,
        enable_chronovisor=False,
        router_aux_loss_coef=switch_config.router_aux_loss_coef,
    )

    frozen_result = run_training(
        frozen_config,
        enable_chronovisor=False,
        eta=0.0,
        pressure_scale=0.0,
        seed=42,
        num_conversations=10,
        num_epochs=50
    )

    print(f"\nFrozen baseline (Switch top-1):")
    print(f"  Final loss: {frozen_result['final_loss']:.6f}")
    print(f"  Final aux loss: {frozen_result['final_aux_loss']:.6f}")
    print(f"  Final separation: {frozen_result['final_sep']:.6f}")
    print()

    # Run Chronovisor tests on Switch
    print("=" * 70)
    print("STEP 2: Chronovisor with Stable Basin (η=0.015, P=0.5)")
    print("=" * 70)

    results = []
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] Seed {seed}:")
        result = run_training(
            switch_config,
            enable_chronovisor=True,
            eta=eta,
            pressure_scale=pressure_scale,
            seed=seed,
            num_conversations=10,
            num_epochs=50
        )

        # Compute deltas
        delta_loss = (result['final_loss'] - frozen_result['final_loss']) / frozen_result['final_loss'] * 100
        delta_sep = (result['final_sep'] - frozen_result['final_sep']) / frozen_result['final_sep'] * 100

        result['delta_loss'] = delta_loss
        result['delta_sep'] = delta_sep
        result['seed'] = seed
        results.append(result)

        print(f"  Final loss: {result['final_loss']:.6f} (Δ: {delta_loss:+.2f}%)")
        print(f"  Final aux loss: {result['final_aux_loss']:.6f}")
        print(f"  Final separation: {result['final_sep']:.6f} (Δ: {delta_sep:+.2f}%)")
        print(f"  T̄ variance: {result['T_bar_variance']:.6f}")

        # Check Pareto-better
        is_pareto = delta_loss < 0 and delta_sep > 0
        print(f"  Pareto-better: {'✓' if is_pareto else '✗'}")

    # Summary statistics
    print()
    print("=" * 70)
    print("SWITCH TRANSFORMER RESULTS")
    print("=" * 70)

    delta_losses = [r['delta_loss'] for r in results]
    delta_seps = [r['delta_sep'] for r in results]
    T_bar_vars = [r['T_bar_variance'] for r in results]

    pareto_count = sum(1 for r in results if r['delta_loss'] < 0 and r['delta_sep'] > 0)
    robustness = pareto_count / len(results) * 100

    print(f"\nSwitch Transformer (top-1 routing) Results:")
    print(f"  Robustness: {pareto_count}/{len(results)} seeds ({robustness:.0f}%)")
    print(f"  Δ Loss: {np.mean(delta_losses):+.2f}% ± {np.std(delta_losses):.2f}%")
    print(f"  Δ Sep: {np.mean(delta_seps):+.2f}% ± {np.std(delta_seps):.2f}%")
    print(f"  T̄ variance: {np.mean(T_bar_vars):.6f} ± {np.std(T_bar_vars):.6f}")

    # Compare to Mixtral baseline
    print()
    print("COMPARISON TO MIXTRAL BASELINE (top-2 routing):")
    print("-" * 70)
    print("Metric              Mixtral (top-2)   Switch (top-1)   Status")
    print("-" * 70)

    # Expected from Mixtral validation
    mixtral_expected = {
        'robustness': 100.0,
        'delta_loss': -0.4,
        'delta_sep': 6.9,
        'T_bar_var': 0.002,
    }

    def generalization_status(mixtral_val, switch_val, metric_type='robustness'):
        """Determine if Switch results show similar behavior to Mixtral."""
        if metric_type == 'robustness':
            if switch_val >= 67.0:
                return "✓ GOOD"
            elif switch_val >= 33.0:
                return "~ PARTIAL"
            else:
                return "✗ POOR"
        elif metric_type in ['delta_loss', 'delta_sep']:
            # Should have similar sign and rough magnitude
            same_sign = (mixtral_val * switch_val) > 0
            if same_sign and abs(switch_val - mixtral_val) <= 5.0:
                return "✓ GOOD"
            elif same_sign:
                return "~ PARTIAL"
            else:
                return "✗ DIFFERENT"
        else:  # T_bar_var
            if switch_val > 0.001:
                return "✓ GOOD"
            elif switch_val > 0.0001:
                return "~ PARTIAL"
            else:
                return "✗ POOR"

    rob_status = generalization_status(mixtral_expected['robustness'], robustness, 'robustness')
    loss_status = generalization_status(mixtral_expected['delta_loss'], np.mean(delta_losses), 'delta_loss')
    sep_status = generalization_status(mixtral_expected['delta_sep'], np.mean(delta_seps), 'delta_sep')
    tbar_status = generalization_status(mixtral_expected['T_bar_var'], np.mean(T_bar_vars), 'T_bar_var')

    print(f"Robustness         {mixtral_expected['robustness']:6.0f}%          {robustness:6.0f}%       {rob_status}")
    print(f"Δ Loss             {mixtral_expected['delta_loss']:+6.2f}%          {np.mean(delta_losses):+6.2f}%       {loss_status}")
    print(f"Δ Sep              {mixtral_expected['delta_sep']:+6.2f}%          {np.mean(delta_seps):+6.2f}%       {sep_status}")
    print(f"T̄ var             {mixtral_expected['T_bar_var']:7.4f}         {np.mean(T_bar_vars):7.4f}      {tbar_status}")
    print("-" * 70)

    # Overall verdict
    generalization_success = (
        robustness >= 67.0 and
        np.mean(delta_losses) < 0 and
        np.mean(delta_seps) > 0 and
        np.mean(T_bar_vars) > 0.001
    )

    print()
    if generalization_success:
        print("✓ GENERALIZATION SUCCESS")
        print("  P×T coupling works with Switch Transformer's top-1 routing.")
        print("  Demonstrates routing mechanism independence.")
        print()
        print("Key findings:")
        print("  - Stable basin transfers from Mixtral (top-2) to Switch (top-1)")
        print("  - Similar improvements in loss and separation")
        print("  - Geological temperature evolves comparably")
        print("  - Load balancing auxiliary loss integrates smoothly")
    else:
        generalization_partial = robustness >= 33.0
        if generalization_partial:
            print("~ PARTIAL GENERALIZATION")
            print("  Some positive effects but different behavior from Mixtral.")
            print("  May require routing-specific tuning.")
            print()
            if np.mean(delta_seps) < 0:
                print("Note: Negative separation may indicate top-1 routing requires")
                print("      different pressure dynamics (stronger balancing).")
        else:
            print("✗ GENERALIZATION FAILED")
            print("  P×T coupling doesn't transfer cleanly to top-1 routing.")
            print("  Requires investigation or routing-specific modifications.")

    print()
    print("=" * 70)

    # Save results
    output_dir = Path(__file__).parent.parent / "switch_test_results"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "switch_transformer_summary.txt"
    with open(output_file, 'w') as f:
        f.write(f"Switch Transformer Test: Top-1 Routing with P×T Coupling\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(f"Configuration: η={eta}, P={pressure_scale}\n")
        f.write(f"Routing: top-1 (vs Mixtral top-2)\n")
        f.write(f"Seeds: {seeds}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Robustness: {pareto_count}/{len(results)} ({robustness:.0f}%)\n")
        f.write(f"  Δ Loss: {np.mean(delta_losses):+.2f}% ± {np.std(delta_losses):.2f}%\n")
        f.write(f"  Δ Sep: {np.mean(delta_seps):+.2f}% ± {np.std(delta_seps):.2f}%\n")
        f.write(f"  T̄ variance: {np.mean(T_bar_vars):.6f} ± {np.std(T_bar_vars):.6f}\n\n")
        f.write(f"Generalization Status: {'SUCCESS' if generalization_success else ('PARTIAL' if generalization_partial else 'FAILED')}\n")

        if generalization_success:
            f.write(f"\nConclusion: P×T coupling generalizes to top-1 routing.\n")
            f.write(f"Demonstrates routing mechanism independence.\n")

    print(f"Results saved to: {output_file}")

    # Scientific interpretation
    print()
    print("SCIENTIFIC INTERPRETATION:")
    print("-" * 70)
    if generalization_success:
        print("This result demonstrates that P×T coupling is a general mechanism")
        print("for MoE routing control, not specific to Mixtral's top-k approach.")
        print()
        print("The stable basin (η=0.015, P=0.5) transfers across:")
        print("  ✓ Different routing strategies (top-1 vs top-2)")
        print("  ✓ Different load balancing approaches (aux loss vs implicit)")
        print("  ✓ Different specialization pressures (top-1 is more aggressive)")
        print()
        print("This strengthens the publication claim of architectural generality.")
    else:
        print("Top-1 routing may require different P×T parameters or adaptations.")
        print("This doesn't invalidate the mechanism, but suggests routing-specific")
        print("tuning may be beneficial.")

    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
