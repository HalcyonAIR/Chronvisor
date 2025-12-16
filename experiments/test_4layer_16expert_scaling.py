#!/usr/bin/env python3
"""
Architectural Scaling Test: 4-layer, 16-expert Mixtral

Tests whether the stable basin (η=0.015, P=0.5) transfers to a larger
architecture with more layers and experts.

Hypothesis: P×T coupling should scale, possibly with minor tuning.

Architecture changes from validated baseline:
- Layers: 2 → 4 (2× depth)
- Experts: 8 → 16 (2× breadth)
- Same hidden_dim=256 (keep model size reasonable)
- Same top-2 routing

If successful, demonstrates that P×T coupling is not specific to small models.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from pathlib import Path

from chronomoe.mixtral_core import MixtralConfig
from chronomoe.chronovisor_mixtral_bridge import ChronovisorMixtralForCausalLM

# Import conversation generator
from generate_long_geeky_conversations import LongGeekyConversationGenerator


def run_training(config, enable_chronovisor, eta, pressure_scale, seed, num_conversations=10, num_epochs=50):
    """Run training with specified configuration."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model
    model = ChronovisorMixtralForCausalLM(config)

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
    turn_separations = []
    coherence_history = []
    T_bar_history = []

    print(f"  Training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_sep = 0.0

        for conv_tokens in conversations:
            input_ids = torch.tensor(conv_tokens[:-1]).unsqueeze(0)
            target_ids = torch.tensor(conv_tokens[1:]).unsqueeze(0)

            # Forward pass
            logits, chrono_state = model(input_ids, update_chronovisor=enable_chronovisor)

            # Compute loss
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, config.vocab_size),
                target_ids.view(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

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
        turn_separations.append(epoch_sep / len(conversations))

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{num_epochs}: loss={losses[-1]:.4f}, sep={turn_separations[-1]:.6f}")

    # Return final metrics
    final_loss = np.mean(losses[-10:])
    final_sep = np.mean(turn_separations[-10:])

    # T̄ variance
    T_bar_var = 0.0
    if T_bar_history:
        T_bar_means = [np.mean(t) for t in T_bar_history[-50:]]
        T_bar_var = np.var(T_bar_means)

    return {
        'final_loss': final_loss,
        'final_sep': final_sep,
        'T_bar_variance': T_bar_var,
        'losses': losses,
        'separations': turn_separations,
        'coherence': coherence_history,
    }


def main():
    print("=" * 70)
    print("SCALING TEST: 4-Layer, 16-Expert Mixtral with P×T Coupling")
    print("=" * 70)
    print()
    print("Testing whether stable basin (η=0.015, P=0.5) transfers to larger")
    print("architecture with 2× layers and 2× experts.")
    print()

    # SCALED configuration (4 layers, 16 experts)
    scaled_config = MixtralConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=1024,
        num_layers=4,           # ← SCALED from 2
        num_experts=16,         # ← SCALED from 8
        num_experts_per_token=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        max_seq_length=2048,
        enable_chronovisor=True,
    )

    # Baseline configuration (original 2 layers, 8 experts)
    baseline_config = MixtralConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=1024,
        num_layers=2,
        num_experts=8,
        num_experts_per_token=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        max_seq_length=2048,
        enable_chronovisor=True,
    )

    # Stable basin parameters (transfer hypothesis)
    eta = 0.015
    pressure_scale = 0.5

    # Test with 3 seeds
    seeds = [42, 12345, 67890]

    print("ARCHITECTURE COMPARISON:")
    print("-" * 70)
    print(f"                    Baseline        Scaled")
    print("-" * 70)
    print(f"Layers              {baseline_config.num_layers:8d}        {scaled_config.num_layers:6d}")
    print(f"Experts/layer       {baseline_config.num_experts:8d}        {scaled_config.num_experts:6d}")
    print(f"Total MoE params    ~{baseline_config.num_layers * baseline_config.num_experts:6d}        ~{scaled_config.num_layers * scaled_config.num_experts:6d}")
    print(f"Hidden dim          {baseline_config.hidden_dim:8d}        {scaled_config.hidden_dim:6d}")
    print("-" * 70)
    print()

    print(f"P×T Configuration:")
    print(f"  η (structural T learning rate): {eta}")
    print(f"  Pressure scale: {pressure_scale}")
    print(f"  Seeds: {seeds}")
    print()

    # Run frozen baseline for scaled model
    print("=" * 70)
    print("STEP 1: Frozen Baseline (4-layer, 16-expert, no Chronovisor)")
    print("=" * 70)

    frozen_config = MixtralConfig(
        vocab_size=scaled_config.vocab_size,
        hidden_dim=scaled_config.hidden_dim,
        intermediate_dim=scaled_config.intermediate_dim,
        num_layers=scaled_config.num_layers,
        num_experts=scaled_config.num_experts,
        num_experts_per_token=scaled_config.num_experts_per_token,
        num_attention_heads=scaled_config.num_attention_heads,
        num_key_value_heads=scaled_config.num_key_value_heads,
        head_dim=scaled_config.head_dim,
        max_seq_length=scaled_config.max_seq_length,
        enable_chronovisor=False,
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

    print(f"\nFrozen baseline (4-layer, 16-expert):")
    print(f"  Final loss: {frozen_result['final_loss']:.6f}")
    print(f"  Final separation: {frozen_result['final_sep']:.6f}")
    print()

    # Run Chronovisor tests on scaled model
    print("=" * 70)
    print("STEP 2: Chronovisor with Stable Basin (η=0.015, P=0.5)")
    print("=" * 70)

    results = []
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] Seed {seed}:")
        result = run_training(
            scaled_config,
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
        print(f"  Final separation: {result['final_sep']:.6f} (Δ: {delta_sep:+.2f}%)")
        print(f"  T̄ variance: {result['T_bar_variance']:.6f}")

        # Check Pareto-better
        is_pareto = delta_loss < 0 and delta_sep > 0
        print(f"  Pareto-better: {'✓' if is_pareto else '✗'}")

    # Summary statistics
    print()
    print("=" * 70)
    print("SCALING RESULTS SUMMARY")
    print("=" * 70)

    delta_losses = [r['delta_loss'] for r in results]
    delta_seps = [r['delta_sep'] for r in results]
    T_bar_vars = [r['T_bar_variance'] for r in results]

    pareto_count = sum(1 for r in results if r['delta_loss'] < 0 and r['delta_sep'] > 0)
    robustness = pareto_count / len(results) * 100

    print(f"\n4-Layer, 16-Expert Results:")
    print(f"  Robustness: {pareto_count}/{len(results)} seeds ({robustness:.0f}%)")
    print(f"  Δ Loss: {np.mean(delta_losses):+.2f}% ± {np.std(delta_losses):.2f}%")
    print(f"  Δ Sep: {np.mean(delta_seps):+.2f}% ± {np.std(delta_seps):.2f}%")
    print(f"  T̄ variance: {np.mean(T_bar_vars):.6f} ± {np.std(T_bar_vars):.6f}")

    # Compare to baseline (2-layer, 8-expert validated results)
    print()
    print("COMPARISON TO BASELINE (2-Layer, 8-Expert):")
    print("-" * 70)
    print("Metric              2L/8E Baseline    4L/16E Scaled    Transfer")
    print("-" * 70)

    # Expected from 2-layer validation
    baseline_expected = {
        'robustness': 100.0,
        'delta_loss': -0.4,
        'delta_sep': 6.9,
        'T_bar_var': 0.002,
    }

    def transfer_status(baseline_val, scaled_val, metric_type='robustness'):
        """Determine if scaling transferred successfully."""
        if metric_type == 'robustness':
            if scaled_val >= 67.0:
                return "✓ GOOD"
            elif scaled_val >= 33.0:
                return "~ PARTIAL"
            else:
                return "✗ POOR"
        elif metric_type == 'delta_loss':
            # Should be negative (improvement)
            if scaled_val < 0 and abs(scaled_val - baseline_val) <= 2.0:
                return "✓ GOOD"
            elif scaled_val < 0:
                return "~ PARTIAL"
            else:
                return "✗ POOR"
        elif metric_type == 'delta_sep':
            # Should be positive (improvement)
            if scaled_val > 0 and abs(scaled_val - baseline_val) <= 5.0:
                return "✓ GOOD"
            elif scaled_val > 0:
                return "~ PARTIAL"
            else:
                return "✗ POOR"
        else:  # T_bar_var
            # Should be > 0 (geology active)
            if scaled_val > 0.001:
                return "✓ GOOD"
            elif scaled_val > 0.0001:
                return "~ PARTIAL"
            else:
                return "✗ POOR"

    rob_status = transfer_status(baseline_expected['robustness'], robustness, 'robustness')
    loss_status = transfer_status(baseline_expected['delta_loss'], np.mean(delta_losses), 'delta_loss')
    sep_status = transfer_status(baseline_expected['delta_sep'], np.mean(delta_seps), 'delta_sep')
    tbar_status = transfer_status(baseline_expected['T_bar_var'], np.mean(T_bar_vars), 'T_bar_var')

    print(f"Robustness         {baseline_expected['robustness']:6.0f}%          {robustness:6.0f}%       {rob_status}")
    print(f"Δ Loss             {baseline_expected['delta_loss']:+6.2f}%          {np.mean(delta_losses):+6.2f}%       {loss_status}")
    print(f"Δ Sep              {baseline_expected['delta_sep']:+6.2f}%          {np.mean(delta_seps):+6.2f}%       {sep_status}")
    print(f"T̄ var             {baseline_expected['T_bar_var']:7.4f}         {np.mean(T_bar_vars):7.4f}      {tbar_status}")
    print("-" * 70)

    # Overall verdict
    transfer_success = (
        robustness >= 67.0 and
        np.mean(delta_losses) < 0 and
        np.mean(delta_seps) > 0 and
        np.mean(T_bar_vars) > 0.001
    )

    print()
    if transfer_success:
        print("✓ SCALING SUCCESS")
        print("  Stable basin (η=0.015, P=0.5) transfers to 4-layer, 16-expert architecture.")
        print("  P×T coupling demonstrates architectural scaling robustness.")
    else:
        transfer_partial = robustness >= 33.0
        if transfer_partial:
            print("~ PARTIAL TRANSFER")
            print("  Some positive effects observed, but may require parameter tuning.")
            print("  Suggest sweep: η ∈ {0.01, 0.015, 0.02}, P ∈ {0.3, 0.5, 0.7}")
        else:
            print("✗ TRANSFER FAILED")
            print("  Stable basin did not transfer to scaled architecture.")
            print("  Requires investigation or different approach.")

    print()
    print("=" * 70)

    # Save results
    output_dir = Path(__file__).parent.parent / "scaling_test_results"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "4layer_16expert_summary.txt"
    with open(output_file, 'w') as f:
        f.write(f"Architectural Scaling Test: 4-Layer, 16-Expert Mixtral\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(f"Configuration: η={eta}, P={pressure_scale}\n")
        f.write(f"Architecture: 4 layers, 16 experts (vs baseline 2 layers, 8 experts)\n")
        f.write(f"Seeds: {seeds}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Robustness: {pareto_count}/{len(results)} ({robustness:.0f}%)\n")
        f.write(f"  Δ Loss: {np.mean(delta_losses):+.2f}% ± {np.std(delta_losses):.2f}%\n")
        f.write(f"  Δ Sep: {np.mean(delta_seps):+.2f}% ± {np.std(delta_seps):.2f}%\n")
        f.write(f"  T̄ variance: {np.mean(T_bar_vars):.6f} ± {np.std(T_bar_vars):.6f}\n\n")
        f.write(f"Transfer Status: {'SUCCESS' if transfer_success else ('PARTIAL' if transfer_partial else 'FAILED')}\n")

        if transfer_success:
            f.write(f"\nConclusion: Stable basin transfers successfully to scaled architecture.\n")
            f.write(f"P×T coupling demonstrates robustness across architectural scales.\n")

    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
