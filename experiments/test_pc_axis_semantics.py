#!/usr/bin/env python3
"""
PC Axis Semantics Test: What do PC1 and PC2 actually represent?

Method: Controlled router-only perturbations
- Persistence penalty (penalize expert switching)
- Switching reward (reward expert switching)
- Router temperature (softmax sharpness)
- Prior bias (specialization toward expert subset)

For each intervention, measure projection onto PC1/PC2 and test:
- Does the knob selectively control one axis?
- Does the response match the hypothesis?

PC1 hypothesis: Persistence, momentum, coherence
PC2 hypothesis: Adaptability, exploration, margin

Pass/fail: Monotonic change in target axis while other stays stable.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from chronomoe.deepseek_core import DeepSeekConfig, DeepSeekRouter, DeepSeekSparseMoELayer
from chronomoe.chronovisor_deepseek_bridge import ChronovisorDeepSeekForCausalLM
from generate_long_geeky_conversations import LongGeekyConversationGenerator


class PerturbedDeepSeekRouter(DeepSeekRouter):
    """DeepSeek router with controlled perturbations."""

    def __init__(
        self,
        config: DeepSeekConfig,
        persistence_penalty: float = 0.0,
        switching_reward: float = 0.0,
        router_temperature: float = 1.0,
        prior_bias_strength: float = 0.0,
        prior_bias_vector: Optional[torch.Tensor] = None,
    ):
        super().__init__(config)
        self.persistence_penalty = persistence_penalty
        self.switching_reward = switching_reward
        self.router_temperature = router_temperature
        self.prior_bias_strength = prior_bias_strength
        self.prior_bias_vector = prior_bias_vector

        # Track previous expert selection for persistence/switching
        self.prev_selected_experts = None

    def forward(self, hidden_states, pressure_bias=None):
        """Forward with controlled perturbations."""
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Compute router logits
        router_logits = self.gate(hidden_states)

        # Apply prior bias if specified
        if self.prior_bias_strength > 0.0 and self.prior_bias_vector is not None:
            router_logits = router_logits + self.prior_bias_strength * self.prior_bias_vector.unsqueeze(0).unsqueeze(0)

        # Apply pressure bias if provided
        if pressure_bias is not None:
            router_logits = router_logits + pressure_bias.unsqueeze(0).unsqueeze(0)

        # Apply router temperature
        router_logits = router_logits / self.router_temperature

        # Softmax and routing
        router_probs = F.softmax(router_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(
            router_probs, self.num_experts_per_token, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # Compute persistence/switching penalties
        aux_loss = self._compute_aux_loss(router_logits, router_probs)

        if self.prev_selected_experts is not None:
            # Compute switching metric (fraction of experts that changed)
            # selected_experts: [batch, seq, top_k]
            # Compare with prev at seq-1
            if seq_len > 1:
                prev_experts = selected_experts[:, :-1, :]  # [batch, seq-1, top_k]
                curr_experts = selected_experts[:, 1:, :]   # [batch, seq-1, top_k]

                # Fraction of experts that switched
                switched = (prev_experts != curr_experts).float().mean()

                if self.persistence_penalty > 0.0:
                    # Penalize switching
                    aux_loss = aux_loss + self.persistence_penalty * switched

                if self.switching_reward > 0.0:
                    # Reward switching (penalize staying same)
                    aux_loss = aux_loss + self.switching_reward * (1.0 - switched)

        self.prev_selected_experts = selected_experts.detach()

        return routing_weights, selected_experts, router_probs, aux_loss


def patch_model_with_perturbed_router(model, **router_kwargs):
    """Replace all routers with perturbed versions."""
    for layer in model.model.layers:
        old_router = layer.moe.router
        new_router = PerturbedDeepSeekRouter(model.config, **router_kwargs)

        # Copy weights
        new_router.gate.weight.data = old_router.gate.weight.data.clone()
        new_router.aux_loss_coef = old_router.aux_loss_coef
        new_router.z_loss_coef = old_router.z_loss_coef

        layer.moe.router = new_router


def capture_with_perturbation(
    knob_name: str,
    knob_value: float,
    num_conversations: int = 10,
    num_epochs: int = 20,
    sample_every: int = 5,
    prior_bias_vector: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Capture routing trajectory with specified perturbation."""

    # Create model with chronovisor enabled but pressure_scale=0 (statistics without coupling)
    config = DeepSeekConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=1024,
        num_layers=2,
        num_shared_experts=2,
        num_routed_experts=64,
        num_experts_per_token=6,
        enable_chronovisor=True,  # Enable for statistics collection
    )

    torch.manual_seed(42)
    model = ChronovisorDeepSeekForCausalLM(config)

    # Set pressure_scale=0 so we collect stats without applying P×T coupling
    if hasattr(model, 'controller') and model.controller is not None:
        for lens in model.controller.lenses.values():
            lens.pressure_scale = 0.0

    # Prepare router kwargs
    router_kwargs = {
        'persistence_penalty': 0.0,
        'switching_reward': 0.0,
        'router_temperature': 1.0,
        'prior_bias_strength': 0.0,
        'prior_bias_vector': None,
    }

    # Set the specific knob
    router_kwargs[knob_name] = knob_value

    if knob_name == 'prior_bias_strength' and prior_bias_vector is not None:
        router_kwargs['prior_bias_vector'] = torch.from_numpy(prior_bias_vector).float()

    # Patch routers
    patch_model_with_perturbed_router(model, **router_kwargs)

    # Generate conversations
    gen = LongGeekyConversationGenerator(vocab_size=config.vocab_size)
    dataset = gen.generate_dataset(num_conversations=num_conversations)
    conversations = [seq['input_ids'].tolist() for seq in dataset['sequences']]

    # Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Capture
    entropy_history = []
    call_counter = 0

    for epoch in range(num_epochs):
        for conv_tokens in conversations:
            input_ids = torch.tensor(conv_tokens[:-1]).unsqueeze(0)
            target_ids = torch.tensor(conv_tokens[1:]).unsqueeze(0)

            # Forward (chronovisor enabled for stats but pressure_scale=0)
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

            # CAPTURE routing entropy from chronovisor state
            if chrono_state is not None and chrono_state.routing_entropy:
                entropy = chrono_state.routing_entropy.get(0, 0.0)  # Layer 0
                call_counter += 1
                if call_counter % sample_every == 0:
                    entropy_history.append(entropy)

    return np.array(entropy_history)


def compute_pc_projection(trajectory: np.ndarray, pc1: np.ndarray, pc2: np.ndarray, delay: int = 1) -> Dict:
    """
    Project trajectory onto PC1 and PC2 axes and compute metrics.

    Returns:
        - mean_proj_pc1, mean_proj_pc2: mean absolute projection
        - var_proj_pc1, var_proj_pc2: variance of projection
        - curvature: mean turning rate
        - return_time: (placeholder for now)
    """
    # Create delay embedding
    N = len(trajectory)
    embedded = []
    for i in range(N - delay):
        point = [trajectory[i], trajectory[i + delay]]
        embedded.append(point)

    X = np.array(embedded)
    X_scaled = StandardScaler().fit_transform(X)

    # Project onto PC axes
    proj_pc1 = X_scaled @ pc1
    proj_pc2 = X_scaled @ pc2

    # Compute metrics
    metrics = {
        'mean_proj_pc1': np.mean(np.abs(proj_pc1)),
        'mean_proj_pc2': np.mean(np.abs(proj_pc2)),
        'var_proj_pc1': np.var(proj_pc1),
        'var_proj_pc2': np.var(proj_pc2),
        'curvature': compute_curvature(X_scaled),
        'return_time': 0.0,  # TODO: implement impulse-response test
    }

    return metrics, proj_pc1, proj_pc2


def compute_curvature(trajectory: np.ndarray) -> float:
    """Compute mean turning rate (curvature) of trajectory."""
    if len(trajectory) < 3:
        return 0.0

    # Compute velocity vectors
    velocities = np.diff(trajectory, axis=0)

    # Compute turning angles
    angles = []
    for i in range(len(velocities) - 1):
        v1 = velocities[i]
        v2 = velocities[i + 1]

        # Avoid zero vectors
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            continue

        # Angle between consecutive velocity vectors
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        angles.append(angle)

    return np.mean(angles) if angles else 0.0


def main():
    print("=" * 70)
    print("PC AXIS SEMANTICS TEST: Router Perturbation Suite")
    print("=" * 70)
    print()
    print("Testing what PC1 and PC2 represent through controlled interventions.")
    print()
    print("Hypothesis:")
    print("  PC1 = Persistence, momentum, coherence")
    print("  PC2 = Adaptability, exploration, margin")
    print()
    print("Method: Four router-only perturbations")
    print("  1. persistence_penalty - penalize expert switching")
    print("  2. switching_reward - reward expert switching")
    print("  3. router_temperature - softmax temperature")
    print("  4. prior_bias_strength - specialization bias")
    print()

    # First, get reference PC axes from baseline
    print("=" * 70)
    print("STEP 1: Establish Reference PC Axes (Baseline)")
    print("=" * 70)
    print()

    baseline_trajectory = capture_with_perturbation(
        knob_name='router_temperature',  # Dummy, will be 1.0
        knob_value=1.0,
        num_conversations=10,
        num_epochs=20,
        sample_every=5,
    )

    print(f"Baseline trajectory: {len(baseline_trajectory)} samples")
    print(f"Mean entropy: {np.mean(baseline_trajectory):.4f}")
    print(f"Std entropy: {np.std(baseline_trajectory):.4f}")
    print()

    # Compute reference PC axes
    delay = 1
    N = len(baseline_trajectory)
    embedded = []
    for i in range(N - delay):
        point = [baseline_trajectory[i], baseline_trajectory[i + delay]]
        embedded.append(point)

    X_baseline = np.array(embedded)
    X_baseline_scaled = StandardScaler().fit_transform(X_baseline)

    pca_baseline = PCA(n_components=2)
    pca_baseline.fit(X_baseline_scaled)

    pc1_ref = pca_baseline.components_[0]
    pc2_ref = pca_baseline.components_[1]
    var_explained = pca_baseline.explained_variance_ratio_

    print(f"Reference PC1: [{pc1_ref[0]:+.4f}, {pc1_ref[1]:+.4f}]")
    print(f"Reference PC2: [{pc2_ref[0]:+.4f}, {pc2_ref[1]:+.4f}]")
    print(f"Variance explained: PC1={var_explained[0]*100:.1f}%, PC2={var_explained[1]*100:.1f}%")
    print()

    # Now run perturbation suite
    print("=" * 70)
    print("STEP 2: Perturbation Suite")
    print("=" * 70)
    print()

    # Define knob sweeps
    knob_configs = [
        {
            'name': 'persistence_penalty',
            'values': [0.0, 0.01, 0.05, 0.1, 0.5],
            'expected': 'PC1↑ (more persistence), PC2↓ (less exploration)',
        },
        {
            'name': 'switching_reward',
            'values': [0.0, 0.01, 0.05, 0.1, 0.5],
            'expected': 'PC2↑ (more exploration), PC1→ (momentum preserved)',
        },
        {
            'name': 'router_temperature',
            'values': [0.5, 0.75, 1.0, 1.5, 2.0],
            'expected': 'PC2↑ (flatter routing = more exploration)',
        },
        {
            'name': 'prior_bias_strength',
            'values': [0.0, 0.1, 0.5, 1.0, 2.0],
            'expected': 'PC1 shift (specialization changes mean position)',
        },
    ]

    # Create prior bias vector (favor first 16 experts)
    prior_bias = np.zeros(64)
    prior_bias[:16] = 1.0

    results = {}

    for knob_config in knob_configs:
        knob_name = knob_config['name']
        knob_values = knob_config['values']
        expected = knob_config['expected']

        print(f"\n{'='*70}")
        print(f"KNOB: {knob_name}")
        print(f"Expected: {expected}")
        print(f"{'='*70}\n")

        knob_results = []

        for knob_value in knob_values:
            print(f"  {knob_name} = {knob_value}...", end=' ', flush=True)

            trajectory = capture_with_perturbation(
                knob_name=knob_name,
                knob_value=knob_value,
                num_conversations=10,
                num_epochs=20,
                sample_every=5,
                prior_bias_vector=prior_bias if knob_name == 'prior_bias_strength' else None,
            )

            # Project onto reference PC axes
            metrics, proj_pc1, proj_pc2 = compute_pc_projection(
                trajectory, pc1_ref, pc2_ref, delay=1
            )

            knob_results.append({
                'value': knob_value,
                'metrics': metrics,
                'trajectory': trajectory,
                'proj_pc1': proj_pc1,
                'proj_pc2': proj_pc2,
            })

            print(f"mean_PC1={metrics['mean_proj_pc1']:.3f}, var_PC1={metrics['var_proj_pc1']:.3f}, "
                  f"mean_PC2={metrics['mean_proj_pc2']:.3f}, var_PC2={metrics['var_proj_pc2']:.3f}")

        results[knob_name] = knob_results

        # Check for monotonicity
        print()
        print(f"  Monotonicity check:")

        values = [r['value'] for r in knob_results]
        var_pc1_series = [r['metrics']['var_proj_pc1'] for r in knob_results]
        var_pc2_series = [r['metrics']['var_proj_pc2'] for r in knob_results]

        # Check if PC1 variance changes monotonically
        pc1_diffs = np.diff(var_pc1_series)
        pc2_diffs = np.diff(var_pc2_series)

        pc1_monotonic = np.all(pc1_diffs >= 0) or np.all(pc1_diffs <= 0)
        pc2_monotonic = np.all(pc2_diffs >= 0) or np.all(pc2_diffs <= 0)

        print(f"    PC1 variance: {var_pc1_series[0]:.3f} → {var_pc1_series[-1]:.3f} "
              f"({'monotonic' if pc1_monotonic else 'non-monotonic'})")
        print(f"    PC2 variance: {var_pc2_series[0]:.3f} → {var_pc2_series[-1]:.3f} "
              f"({'monotonic' if pc2_monotonic else 'non-monotonic'})")
        print()

    # Create summary plot
    create_summary_plot(results, knob_configs)

    # Save results
    output_dir = Path(__file__).parent.parent / "takens_data"
    output_dir.mkdir(exist_ok=True)

    summary_file = output_dir / "pc_axis_semantics_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("PC Axis Semantics Test Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Reference PC1: [{pc1_ref[0]:+.4f}, {pc1_ref[1]:+.4f}]\n")
        f.write(f"Reference PC2: [{pc2_ref[0]:+.4f}, {pc2_ref[1]:+.4f}]\n\n")

        for knob_config in knob_configs:
            knob_name = knob_config['name']
            knob_results = results[knob_name]

            f.write(f"\n{knob_name}:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Value':<10} {'mean_PC1':<12} {'var_PC1':<12} {'mean_PC2':<12} {'var_PC2':<12}\n")

            for r in knob_results:
                m = r['metrics']
                f.write(f"{r['value']:<10.2f} {m['mean_proj_pc1']:<12.4f} {m['var_proj_pc1']:<12.4f} "
                       f"{m['mean_proj_pc2']:<12.4f} {m['var_proj_pc2']:<12.4f}\n")

    print(f"\nResults saved to: {summary_file}")


def create_summary_plot(results: Dict, knob_configs: List):
    """Create summary visualization of all perturbation results."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, knob_config in enumerate(knob_configs):
        ax = axes[idx]
        knob_name = knob_config['name']
        knob_results = results[knob_name]

        values = [r['value'] for r in knob_results]
        var_pc1 = [r['metrics']['var_proj_pc1'] for r in knob_results]
        var_pc2 = [r['metrics']['var_proj_pc2'] for r in knob_results]

        ax.plot(values, var_pc1, marker='o', linewidth=2, markersize=8, label='var(PC1)', color='red')
        ax.plot(values, var_pc2, marker='s', linewidth=2, markersize=8, label='var(PC2)', color='blue')

        ax.set_xlabel(knob_name, fontsize=12)
        ax.set_ylabel('Projection Variance', fontsize=12)
        ax.set_title(f'{knob_name}\n{knob_config["expected"]}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = Path(__file__).parent.parent / "takens_data"
    output_file = output_dir / "pc_axis_semantics.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Summary plot saved: {output_file}")


if __name__ == '__main__':
    main()
