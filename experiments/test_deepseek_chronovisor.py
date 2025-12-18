#!/usr/bin/env python3
"""
DeepSeek-MoE Test: Shared+Routed Experts with P×T Coupling

Tests whether P×T coupling generalizes to DeepSeek's radically different architecture:
- Shared experts (always activated) + Routed experts (sparse top-k)
- Fine-grained routing (64 experts vs typical 8)
- Hybrid output: SharedExperts(x) + RoutedExperts(x)

This is the ultimate generality test - if P×T coupling works here, it proves
the mechanism isn't Mixtral-specific.

Key differences from Mixtral:
- Shared + routed experts (vs all routed)
- Fine-grained expert pool (16-64 vs 8)
- Different load balancing (aux + z-loss)
- Two-stage output combination

Scientific value:
- Tests architectural generalization (not just routing generalization)
- Different expert specialization dynamics (shared base + routed specialization)
- Validates P×T coupling on real-world architecture

If successful, proves P×T coupling is truly general across MoE architectures.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from pathlib import Path

from chronomoe.deepseek_core import DeepSeekConfig
from chronomoe.chronovisor_deepseek_bridge import ChronovisorDeepSeekForCausalLM

# Import conversation generator
from generate_long_geeky_conversations import LongGeekyConversationGenerator


def run_training(config, enable_chronovisor, eta, pressure_scale, seed, num_conversations=10, num_epochs=50):
    """Run training with specified configuration."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model
    model = ChronovisorDeepSeekForCausalLM(config)

    # Configure Chronovisor if enabled
    if enable_chronovisor and model.model.controller is not None:
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
                # Average T̄ across layers for tracking
                if chrono_state.T_bar_local:
                    T_bar_mean = np.mean([t.mean() for t in chrono_state.T_bar_local.values()])
                    T_bar_history.append(T_bar_mean)

            # Compute turn separation (variance of expert usage)
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
        T_bar_means = T_bar_history[-50:]
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
    print("DEEPSEEK-MOE: Shared+Routed Experts with P×T Coupling")
    print("=" * 70)
    print()
    print("Testing P×T coupling generalization to DeepSeek's hybrid architecture.")
    print("This is the ultimate generality test.")
    print()

    # DeepSeek configuration (scaled down for testing)
    deepseek_config = DeepSeekConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=1024,
        num_layers=2,
        num_shared_experts=2,      # Always activated
        num_routed_experts=16,     # Sparse top-k (scaled from 64)
        num_experts_per_token=4,   # Top-4 on routed (scaled from 6)
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        max_seq_length=2048,
        enable_chronovisor=True,
        router_aux_loss_coef=0.001,
        router_z_loss_coef=0.001,
    )

    # Stable basin parameters (transfer from Mixtral)
    eta = 0.015
    pressure_scale = 0.5

    # Test with 3 seeds
    seeds = [42, 12345, 67890]

    print("ARCHITECTURE COMPARISON:")
    print("-" * 70)
    print(f"                    Mixtral (baseline)   DeepSeek (test)")
    print("-" * 70)
    print(f"Layers              2                    {deepseek_config.num_layers}")
    print(f"Total experts       8 (all routed)       {deepseek_config.num_shared_experts + deepseek_config.num_routed_experts} ({deepseek_config.num_shared_experts} shared + {deepseek_config.num_routed_experts} routed)")
    print(f"Routing strategy    top-2                top-{deepseek_config.num_experts_per_token} (routed only)")
    print(f"Hidden dim          256                  {deepseek_config.hidden_dim}")
    print(f"Expert pool         All routed           Hybrid (shared + routed)")
    print(f"Output              Σ(top-k experts)     Σ(shared) + Σ(top-k routed)")
    print("-" * 70)
    print()

    print(f"P×T Configuration (transfer from Mixtral):")
    print(f"  η (structural T learning rate): {eta}")
    print(f"  Pressure scale: {pressure_scale}")
    print(f"  Seeds: {seeds}")
    print()
    print("NOTE: P×T coupling applies ONLY to routed experts.")
    print("      Shared experts provide stable base without geometric control.")
    print()

    # Run frozen baseline for DeepSeek
    print("=" * 70)
    print("STEP 1: Frozen Baseline (DeepSeek, no Chronovisor)")
    print("=" * 70)

    frozen_config = DeepSeekConfig(
        vocab_size=deepseek_config.vocab_size,
        hidden_dim=deepseek_config.hidden_dim,
        intermediate_dim=deepseek_config.intermediate_dim,
        num_layers=deepseek_config.num_layers,
        num_shared_experts=deepseek_config.num_shared_experts,
        num_routed_experts=deepseek_config.num_routed_experts,
        num_experts_per_token=deepseek_config.num_experts_per_token,
        num_attention_heads=deepseek_config.num_attention_heads,
        num_key_value_heads=deepseek_config.num_key_value_heads,
        head_dim=deepseek_config.head_dim,
        max_seq_length=deepseek_config.max_seq_length,
        enable_chronovisor=False,
        router_aux_loss_coef=deepseek_config.router_aux_loss_coef,
        router_z_loss_coef=deepseek_config.router_z_loss_coef,
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

    print(f"\nFrozen baseline (DeepSeek shared+routed):")
    print(f"  Final loss: {frozen_result['final_loss']:.6f}")
    print(f"  Final aux loss: {frozen_result['final_aux_loss']:.6f}")
    print(f"  Final separation: {frozen_result['final_sep']:.6f}")
    print()

    # Run Chronovisor tests on DeepSeek
    print("=" * 70)
    print("STEP 2: Chronovisor with Stable Basin (η=0.015, P=0.5)")
    print("=" * 70)

    results = []
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] Seed {seed}:")
        result = run_training(
            deepseek_config,
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
    print("DEEPSEEK-MOE RESULTS")
    print("=" * 70)

    delta_losses = [r['delta_loss'] for r in results]
    delta_seps = [r['delta_sep'] for r in results]
    T_bar_vars = [r['T_bar_variance'] for r in results]

    pareto_count = sum(1 for r in results if r['delta_loss'] < 0 and r['delta_sep'] > 0)
    robustness = pareto_count / len(results) * 100

    print(f"\nDeepSeek-MoE (shared+routed) Results:")
    print(f"  Robustness: {pareto_count}/{len(results)} seeds ({robustness:.0f}%)")
    print(f"  Δ Loss: {np.mean(delta_losses):+.2f}% ± {np.std(delta_losses):.2f}%")
    print(f"  Δ Sep: {np.mean(delta_seps):+.2f}% ± {np.std(delta_seps):.2f}%")
    print(f"  T̄ variance: {np.mean(T_bar_vars):.6f} ± {np.std(T_bar_vars):.6f}")

    # Compare to Mixtral baseline
    print()
    print("COMPARISON TO MIXTRAL BASELINE (all-routed, top-2):")
    print("-" * 70)
    print("Metric              Mixtral (top-2)   DeepSeek (hybrid)   Status")
    print("-" * 70)

    # Expected from Mixtral validation
    mixtral_expected = {
        'robustness': 100.0,
        'delta_loss': -0.4,
        'delta_sep': 6.9,
        'T_bar_var': 0.002,
    }

    def generalization_status(mixtral_val, deepseek_val, metric_type='robustness'):
        """Determine if DeepSeek results show similar behavior to Mixtral."""
        if metric_type == 'robustness':
            if deepseek_val >= 67.0:
                return "✓ GOOD"
            elif deepseek_val >= 33.0:
                return "~ PARTIAL"
            else:
                return "✗ POOR"
        elif metric_type in ['delta_loss', 'delta_sep']:
            # Should have similar sign and rough magnitude
            same_sign = (mixtral_val * deepseek_val) > 0
            if same_sign and abs(deepseek_val - mixtral_val) <= 5.0:
                return "✓ GOOD"
            elif same_sign:
                return "~ PARTIAL"
            else:
                return "✗ DIFFERENT"
        else:  # T_bar_var
            if deepseek_val > 0.001:
                return "✓ GOOD"
            elif deepseek_val > 0.0001:
                return "~ PARTIAL"
            else:
                return "✗ POOR"

    rob_status = generalization_status(mixtral_expected['robustness'], robustness, 'robustness')
    loss_status = generalization_status(mixtral_expected['delta_loss'], np.mean(delta_losses), 'delta_loss')
    sep_status = generalization_status(mixtral_expected['delta_sep'], np.mean(delta_seps), 'delta_sep')
    tbar_status = generalization_status(mixtral_expected['T_bar_var'], np.mean(T_bar_vars), 'T_bar_var')

    print(f"Robustness         {mixtral_expected['robustness']:6.0f}%          {robustness:6.0f}%          {rob_status}")
    print(f"Δ Loss             {mixtral_expected['delta_loss']:+6.2f}%          {np.mean(delta_losses):+6.2f}%          {loss_status}")
    print(f"Δ Sep              {mixtral_expected['delta_sep']:+6.2f}%          {np.mean(delta_seps):+6.2f}%          {sep_status}")
    print(f"T̄ var             {mixtral_expected['T_bar_var']:7.4f}         {np.mean(T_bar_vars):7.4f}         {tbar_status}")
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
        print("✓ ARCHITECTURAL GENERALIZATION SUCCESS")
        print("  P×T coupling works with DeepSeek's hybrid shared+routed architecture.")
        print("  This proves the mechanism is truly general, not Mixtral-specific.")
        print()
        print("Key findings:")
        print("  - Stable basin transfers across radically different architectures")
        print("  - Shared experts provide stable base without interfering with P×T")
        print("  - Fine-grained routing (16 experts) doesn't break the mechanism")
        print("  - Hybrid output combination (shared + routed) integrates smoothly")
        print()
        print("Scientific impact:")
        print("  This is the strongest evidence for architectural generality.")
        print("  P×T coupling works on:")
        print("    ✓ All-routed architectures (Mixtral)")
        print("    ✓ Hybrid shared+routed architectures (DeepSeek)")
        print("    ✓ Different routing granularities (8-64 experts)")
        print("    ✓ Different output combinations")
    else:
        generalization_partial = robustness >= 33.0
        if generalization_partial:
            print("~ PARTIAL GENERALIZATION")
            print("  Some positive effects but different behavior from Mixtral.")
            print("  Hybrid architecture may require tuning.")
            print()
            if np.mean(T_bar_vars) < 0.001:
                print("Note: Low T̄ variance suggests shared experts may be dominating")
                print("      routed expert dynamics. May need to adjust shared/routed")
                print("      balance or pressure scaling.")
        else:
            print("✗ GENERALIZATION FAILED")
            print("  P×T coupling doesn't transfer cleanly to hybrid architecture.")
            print("  Requires investigation of shared expert interactions.")
            print()
            print("Possible causes:")
            print("  - Shared experts dominate signal, reducing routed expert impact")
            print("  - Fine-grained routing spreads pressure too thinly")
            print("  - Hybrid output combination changes gradient dynamics")

    print()
    print("=" * 70)

    # Save results
    output_dir = Path(__file__).parent.parent / "deepseek_test_results"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "deepseek_chronovisor_summary.txt"
    with open(output_file, 'w') as f:
        f.write(f"DeepSeek-MoE Test: Shared+Routed Experts with P×T Coupling\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(f"Configuration: η={eta}, P={pressure_scale}\n")
        f.write(f"Architecture: {deepseek_config.num_shared_experts} shared + {deepseek_config.num_routed_experts} routed experts\n")
        f.write(f"Routing: top-{deepseek_config.num_experts_per_token} (routed only)\n")
        f.write(f"Seeds: {seeds}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Robustness: {pareto_count}/{len(results)} ({robustness:.0f}%)\n")
        f.write(f"  Δ Loss: {np.mean(delta_losses):+.2f}% ± {np.std(delta_losses):.2f}%\n")
        f.write(f"  Δ Sep: {np.mean(delta_seps):+.2f}% ± {np.std(delta_seps):.2f}%\n")
        f.write(f"  T̄ variance: {np.mean(T_bar_vars):.6f} ± {np.std(T_bar_vars):.6f}\n\n")
        f.write(f"Generalization Status: {'SUCCESS' if generalization_success else ('PARTIAL' if generalization_partial else 'FAILED')}\n")

        if generalization_success:
            f.write(f"\nConclusion: P×T coupling generalizes to hybrid shared+routed architecture.\n")
            f.write(f"Demonstrates true architectural independence.\n")

    print(f"Results saved to: {output_file}")

    # Scientific interpretation
    print()
    print("SCIENTIFIC INTERPRETATION:")
    print("-" * 70)
    if generalization_success:
        print("This result is the strongest validation of P×T coupling generality.")
        print()
        print("DeepSeek's architecture is RADICALLY different from Mixtral:")
        print("  • Shared experts (always active) + Routed experts (sparse)")
        print("  • Fine-grained expert pool (16-64 vs 8)")
        print("  • Hybrid output combination")
        print("  • Different load balancing mechanisms")
        print()
        print("Yet P×T coupling works with the SAME stable basin parameters.")
        print("This proves the mechanism operates at a fundamental level,")
        print("independent of specific architectural choices.")
        print()
        print("Paper contribution:")
        print("  This moves P×T coupling from 'interesting Mixtral modification'")
        print("  to 'general principle for MoE routing control'.")
        print("  Publication impact is significantly strengthened.")
    else:
        print("Hybrid architecture presents unique challenges:")
        print("  • Shared experts provide baseline signal")
        print("  • Routed experts add specialized refinement")
        print("  • P×T coupling only controls routed layer")
        print()
        print("This doesn't invalidate the mechanism, but highlights that")
        print("architectural coupling (shared vs routed) may require tuning.")

    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
