#!/usr/bin/env python3
"""
Noise Injection Test: Is the d=2 manifold an attractor?

Tests whether the 2D routing manifold is:
1. An ATTRACTOR (actively maintained, FNN stays 0% under noise)
2. STABLE (passive but robust, FNN degrades gracefully)
3. FRAGILE (exists only under clean conditions, FNN shatters)

Method:
- Inject Gaussian noise into router logits: logits = logits + σ * N(0,1)
- Sweep σ from 0.0 to 2.0
- Capture routing trajectories
- Run FNN analysis
- Plot FNN vs noise scale

Expected outcomes:
- Attractor: FNN ≈ 0% up to σ=1.0 (dynamics return to manifold)
- Stable: FNN degrades linearly with σ (pushed off but gracefully)
- Fragile: FNN >> 0% at σ=0.1 (immediately broken)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional

# We'll modify DeepSeek router to inject noise
from chronomoe.deepseek_core import DeepSeekConfig, DeepSeekRouter
from chronomoe.chronovisor_deepseek_bridge import ChronovisorDeepSeekForCausalLM
from generate_long_geeky_conversations import LongGeekyConversationGenerator


class NoisyDeepSeekRouter(DeepSeekRouter):
    """DeepSeek router with controllable noise injection."""

    def __init__(self, config: DeepSeekConfig, noise_scale: float = 0.0):
        super().__init__(config)
        self.noise_scale = noise_scale

    def forward(self, hidden_states, pressure_bias=None):
        """Forward with noise injection."""
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Compute router logits
        router_logits = self.gate(hidden_states)

        # INJECT NOISE HERE
        if self.noise_scale > 0.0:
            noise = self.noise_scale * torch.randn_like(router_logits)
            router_logits = router_logits + noise

        # Apply pressure bias if provided
        if pressure_bias is not None:
            router_logits = router_logits + pressure_bias.unsqueeze(0).unsqueeze(0)

        # Continue as normal
        router_probs = F.softmax(router_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(
            router_probs, self.num_experts_per_token, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        aux_loss = self._compute_aux_loss(router_logits, router_probs)

        return routing_weights, selected_experts, router_probs, aux_loss


def patch_model_with_noisy_router(model, noise_scale):
    """Replace all routers in model with noisy versions."""
    for layer in model.model.layers:
        # Replace router in MoE layer
        old_router = layer.moe.router
        new_router = NoisyDeepSeekRouter(model.config, noise_scale=noise_scale)

        # Copy weights
        new_router.gate.weight.data = old_router.gate.weight.data.clone()
        new_router.aux_loss_coef = old_router.aux_loss_coef
        new_router.z_loss_coef = old_router.z_loss_coef

        # Replace
        layer.moe.router = new_router


def capture_with_noise(noise_scale, num_conversations=10, num_epochs=20, sample_every=5):
    """Capture routing trajectory with specified noise scale."""
    print(f"\n{'=' * 70}")
    print(f"NOISE SCALE: σ = {noise_scale}")
    print(f"{'=' * 70}")

    # Create model
    config = DeepSeekConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=1024,
        num_layers=2,
        num_shared_experts=2,
        num_routed_experts=64,
        num_experts_per_token=6,
        enable_chronovisor=True,
    )

    torch.manual_seed(42)  # Same seed for fair comparison
    model = ChronovisorDeepSeekForCausalLM(config)

    # Patch with noisy router
    patch_model_with_noisy_router(model, noise_scale)

    print(f"  Patched router with noise_scale={noise_scale}")

    # Generate conversations
    gen = LongGeekyConversationGenerator(vocab_size=config.vocab_size)
    dataset = gen.generate_dataset(num_conversations=num_conversations)
    conversations = [seq['input_ids'].tolist() for seq in dataset['sequences']]

    # Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Capture
    entropy_history = []
    call_counter = 0

    print(f"  Training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        for conv_tokens in conversations:
            input_ids = torch.tensor(conv_tokens[:-1]).unsqueeze(0)
            target_ids = torch.tensor(conv_tokens[1:]).unsqueeze(0)

            # Forward
            logits, chrono_state, aux_loss = model(input_ids, update_chronovisor=True)

            # Loss
            lm_loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, config.vocab_size),
                target_ids[:, 1:].reshape(-1),
                ignore_index=-100
            )
            total_loss = lm_loss + aux_loss

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # CAPTURE
            if chrono_state is not None and chrono_state.routing_entropy:
                entropy = chrono_state.routing_entropy.get(0, 0.0)
                call_counter += 1
                if call_counter % sample_every == 0:
                    entropy_history.append(entropy)

    print(f"  Captured {len(entropy_history)} samples")
    print(f"  Mean entropy: {np.mean(entropy_history):.4f}")
    print(f"  Std entropy: {np.std(entropy_history):.4f}")

    return np.array(entropy_history)


def analyze_noise_robustness(captures, noise_scales):
    """Run FULL FNN dimension sweep for each noise condition."""
    # Use the proper TakensAnalyzer from our other script
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from analyze_deepseek_takens import TakensAnalyzer

    print(f"\n{'=' * 70}")
    print("FNN ANALYSIS ACROSS NOISE SCALES")
    print(f"{'=' * 70}")
    print()
    print("Running FULL dimension sweep for each noise level.")
    print("This tests: does optimal_d change with noise, or stay at 2?")
    print()

    results = {}

    for noise_scale, data in zip(noise_scales, captures):
        print(f"σ = {noise_scale}:")

        # Run full FNN analysis
        analyzer = TakensAnalyzer(data, delay=1)
        optimal_d, fnn_curve = analyzer.find_optimal_dimension(max_dim=15, rtol=15.0, atol=2.0)

        fnn_at_optimal = fnn_curve[optimal_d - 1] if optimal_d <= len(fnn_curve) else fnn_curve[-1]

        results[noise_scale] = {
            'optimal_d': optimal_d,
            'fnn_curve': fnn_curve,
            'fnn_at_optimal': fnn_at_optimal,
            'fnn_at_d2': fnn_curve[1] if len(fnn_curve) > 1 else 100.0,  # FNN at d=2
        }

        print(f"  Optimal dimension: d = {optimal_d}")
        print(f"  FNN at d={optimal_d}: {fnn_at_optimal:.2f}%")
        print(f"  FNN at d=2: {results[noise_scale]['fnn_at_d2']:.2f}%")
        print()

    return results


def create_diagnostic_plot(noise_scales, fnn_results, captures):
    """Create diagnostic plot showing optimal dimension and FNN vs noise scale."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    # Plot 1: Optimal Dimension vs Noise Scale
    ax1 = axes[0]
    optimal_dims = [fnn_results[scale]['optimal_d'] for scale in noise_scales]
    fnn_at_d2 = [fnn_results[scale]['fnn_at_d2'] for scale in noise_scales]

    ax1.plot(noise_scales, optimal_dims, marker='o', linewidth=2, markersize=10, color='blue', label='Optimal d')
    ax1.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='d=2 (baseline)')
    ax1.set_xlabel('Noise Scale (σ)', fontsize=14)
    ax1.set_ylabel('Optimal Embedding Dimension', fontsize=14)
    ax1.set_title('Attractor Dimension vs Noise: Does d change?', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, max(optimal_dims) + 2])

    # Plot 2: FNN at d=2 vs Noise Scale
    ax2 = axes[1]
    ax2.plot(noise_scales, fnn_at_d2, marker='s', linewidth=2, markersize=10, color='green', label='FNN at d=2')
    ax2.axhline(y=5.0, color='red', linestyle='--', alpha=0.5, label='5% threshold')
    ax2.set_xlabel('Noise Scale (σ)', fontsize=14)
    ax2.set_ylabel('FNN at d=2 (%)', fontsize=14)
    ax2.set_title('d=2 Robustness: FNN at Fixed Dimension', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-5, 105])

    # Add interpretation zones
    ax2.axhspan(0, 5, alpha=0.1, color='green', label='Attractor zone')
    ax2.axhspan(5, 20, alpha=0.1, color='yellow', label='Degrading')
    ax2.axhspan(20, 100, alpha=0.1, color='red', label='Broken')

    # Plot 3: Entropy time series for each noise level
    ax3 = axes[2]
    colors = plt.cm.viridis(np.linspace(0, 1, len(noise_scales)))

    for noise_scale, data, color in zip(noise_scales, captures, colors):
        steps = np.arange(len(data))
        ax3.plot(steps, data, alpha=0.7, linewidth=1.5,
                label=f'σ={noise_scale}', color=color)

    ax3.set_xlabel('Training Step', fontsize=14)
    ax3.set_ylabel('Routing Entropy', fontsize=14)
    ax3.set_title('Routing Trajectories Under Noise', fontsize=16, fontweight='bold')
    ax3.legend(fontsize=10, loc='best')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_dir = Path(__file__).parent.parent / "takens_data"
    output_file = output_dir / "noise_injection_test.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Diagnostic plot saved: {output_file}")


def main():
    print("=" * 70)
    print("NOISE INJECTION TEST: Is the d=2 manifold an attractor?")
    print("=" * 70)
    print()
    print("Testing whether the 2D routing manifold is:")
    print("  1. ATTRACTOR (FNN stays ~0% under noise)")
    print("  2. STABLE (FNN degrades gracefully)")
    print("  3. FRAGILE (FNN shatters immediately)")
    print()

    # Noise scales to test
    noise_scales = [0.0, 0.1, 0.5, 1.0, 2.0]

    print(f"Noise scales: {noise_scales}")
    print(f"Configuration: 64 routed experts, top-6 routing, 20 epochs")
    print()

    # Capture trajectories for each noise scale
    captures = []
    for noise_scale in noise_scales:
        data = capture_with_noise(
            noise_scale=noise_scale,
            num_conversations=10,
            num_epochs=20,
            sample_every=5
        )
        captures.append(data)

        # Save
        output_dir = Path(__file__).parent.parent / "takens_data"
        output_dir.mkdir(exist_ok=True)
        filename = f"noise_scale_{noise_scale:.1f}_routing.npy"
        np.save(output_dir / filename, data)

    # Analyze
    fnn_results = analyze_noise_robustness(captures, noise_scales)

    # Plot
    create_diagnostic_plot(noise_scales, fnn_results, captures)

    # Interpret
    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Check dimensional stability
    optimal_dims = [fnn_results[scale]['optimal_d'] for scale in noise_scales]
    d_stable = all(d == 2 for d in optimal_dims[:3])  # Check first 3 noise levels

    # Check FNN at d=2
    fnn_at_d2_low = fnn_results[0.5]['fnn_at_d2']  # σ=0.5
    fnn_at_d2_high = fnn_results[1.0]['fnn_at_d2']  # σ=1.0

    print()
    print("DIMENSIONAL STABILITY:")
    for scale in noise_scales:
        d = fnn_results[scale]['optimal_d']
        fnn = fnn_results[scale]['fnn_at_d2']
        print(f"  σ={scale:.1f}: optimal_d={d}, FNN(d=2)={fnn:.2f}%")
    print()

    if d_stable and fnn_at_d2_low < 5.0 and fnn_at_d2_high < 10.0:
        print()
        print("✓ DIMENSIONAL ATTRACTOR CONFIRMED")
        print()
        print("The manifold remains 2-dimensional under noise:")
        print(f"  σ=0.5: optimal_d={fnn_results[0.5]['optimal_d']}, FNN(d=2)={fnn_at_d2_low:.2f}%")
        print(f"  σ=1.0: optimal_d={fnn_results[1.0]['optimal_d']}, FNN(d=2)={fnn_at_d2_high:.2f}%")
        print()
        print("Interpretation:")
        print("  → The 2D manifold is a FIXED POINT of the routing dynamics")
        print("  → Noise doesn't push system to higher dimensions")
        print("  → System actively returns to d=2 surface when perturbed")
        print("  → This is not just where the system sits—it's where it returns")
        print()
        print("Implication:")
        print("  The routing system doesn't just happen to live on 2D.")
        print("  It WANTS to. The optimization carved a 2D valley and the")
        print("  dynamics actively maintain it, even under perturbation.")

    elif not d_stable:
        print()
        print("⚠ DIMENSIONAL SHIFT")
        print()
        print("Noise pushes system to higher dimensions:")
        for scale in noise_scales:
            print(f"  σ={scale:.1f}: optimal_d={fnn_results[scale]['optimal_d']}")
        print()
        print("Interpretation:")
        print("  → Noise adds complexity, increases attractor dimension")
        print("  → d=2 is not stable under perturbation")
        print("  → May indicate d=2 is artifact of clean conditions")

    else:
        print()
        print("~ ROBUST BUT DEGRADED")
        print()
        print("The manifold stays 2D but FNN degrades:")
        print(f"  σ=0.5: FNN(d=2)={fnn_at_d2_low:.2f}%")
        print(f"  σ=1.0: FNN(d=2)={fnn_at_d2_high:.2f}%")
        print()
        print("Interpretation:")
        print("  → Dimension is stable (stays d=2)")
        print("  → But noise corrupts the geometry")
        print("  → Manifold exists but becomes noisier under perturbation")

    print()
    print("=" * 70)

    # Save summary
    output_dir = Path(__file__).parent.parent / "takens_data"
    summary_file = output_dir / "noise_injection_summary.txt"

    with open(summary_file, 'w') as f:
        f.write("Noise Injection Test Results\n")
        f.write("=" * 70 + "\n\n")
        f.write("Noise Scale | FNN (%) | False Neighbors\n")
        f.write("-" * 50 + "\n")
        for scale in noise_scales:
            res = fnn_results[scale]
            f.write(f"{scale:11.1f} | {res['fnn']:7.2f} | {res['false_count']}/{res['n_points']}\n")

    print(f"\nResults saved to: {summary_file}")
    print()


if __name__ == '__main__':
    main()
