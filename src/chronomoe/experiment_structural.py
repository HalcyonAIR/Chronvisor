"""
Structural Temperature Experiment: Landscape Formation.

Tracks how the structural temperature (geology) evolves over time,
creating valleys (stable expert regions) and ridges (unstable regions).

Key metrics:
- Structural temperature variance (landscape differentiation)
- Number of valleys and ridges formed
- Correlation between structural_T and reliability
- Mobility index (how much routing changes over time)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from chronomoe.moe import MoE
from chronomoe.router import Router
from chronomoe.bridge import ChronoMoEBridge, RoutingStats
from chronomoe.knob import MetaKnob
from chronomoe.experiment import SyntheticTask


@dataclass
class LandscapeMetrics:
    """Metrics tracking landscape formation."""

    step: int
    structural_T_variance: float
    structural_T_mean: float
    structural_T_min: float
    structural_T_max: float
    num_valleys: int
    num_ridges: int
    landscape_formed: bool
    routing_entropy: float
    loss: float


@dataclass
class LandscapeExperimentConfig:
    """Configuration for landscape formation experiment."""

    n_experts: int = 8
    input_dim: int = 64
    n_clusters: int = 4
    n_tokens_per_batch: int = 32
    n_steps: int = 500  # Long run to see landscape formation

    # Structural temperature parameters
    eta_structural_T: float = 0.02  # Slow geological evolution

    # Temperature parameters
    base_temperature: float = 1.0
    beta_R: float = 0.5
    beta_drift: float = 0.3

    seed: int = 42


def compute_entropy(distribution: np.ndarray) -> float:
    """Compute Shannon entropy."""
    p = np.clip(distribution, 1e-10, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def run_landscape_experiment(
    config: Optional[LandscapeExperimentConfig] = None,
) -> dict:
    """
    Run landscape formation experiment.

    Tracks how structural temperature evolves over time, forming
    valleys (stable regions) and ridges (unstable regions).

    Returns:
        Dictionary with metrics history and final analysis.
    """
    if config is None:
        config = LandscapeExperimentConfig()

    rng = np.random.default_rng(config.seed)

    print("=" * 70)
    print("Landscape Formation Experiment: Geological Temperature Evolution")
    print("=" * 70)
    print(f"Experts: {config.n_experts}, Steps: {config.n_steps}")
    print(f"η_structural_T: {config.eta_structural_T} (geological timescale)")
    print()

    # Initialize components
    moe = MoE(
        n_experts=config.n_experts,
        input_dim=config.input_dim,
        output_dim=config.input_dim,
        seed=config.seed,
    )

    router = Router(
        input_dim=config.input_dim,
        n_experts=config.n_experts,
        seed=config.seed,
    )

    bridge = ChronoMoEBridge.create(
        n_experts=config.n_experts,
        base_temperature=config.base_temperature,
        beta_R=config.beta_R,
        beta_drift=config.beta_drift,
        eta_structural_T=config.eta_structural_T,
        seed=config.seed,
    )

    task = SyntheticTask.create(
        n_clusters=config.n_clusters,
        n_experts=config.n_experts,
        input_dim=config.input_dim,
        seed=config.seed,
    )

    # Tracking
    metrics_history = []
    structural_T_snapshots = []

    print("Running landscape formation...")

    for step in range(config.n_steps):
        # Generate batch
        tokens, cluster_ids = task.sample_batch(config.n_tokens_per_batch, rng=rng)

        # Get temperature field (this updates structural_T)
        temp_field = bridge.get_temperature_field()

        # Get pressure bias
        pressure_bias = bridge.get_pressure_bias()
        router.inject_pressure(pressure_bias.combined)

        # Route with temperature warping
        router.reset_log()
        gate_weights = router.forward_with_temperature(
            tokens,
            temp_field.effective_temperatures,
            pressure_scale=1.0,
        )

        # Compute loss
        expert_indices = np.argmax(gate_weights, axis=1)
        loss, _ = task.compute_loss(cluster_ids, expert_indices)

        # Feed routing stats to Chronovisor
        stats = router.get_routing_stats()
        routing_stats = RoutingStats(
            expert_usage=stats["usage"],
            mean_gate_weights=stats["mean_gates"],
            batch_loss=loss,
        )
        bridge.feed_routing_stats(routing_stats, num_chronovisor_ticks=10)

        # Get structural temperature diagnostics
        struct_diag = bridge.get_structural_temperature_diagnostics()

        # Compute routing entropy
        distribution = stats["distribution"]
        routing_entropy = compute_entropy(distribution) if len(distribution) > 0 else 0.0

        # Record metrics
        metrics = LandscapeMetrics(
            step=step,
            structural_T_variance=struct_diag["variance"],
            structural_T_mean=struct_diag["mean"],
            structural_T_min=float(np.min(struct_diag["structural_T"])),
            structural_T_max=float(np.max(struct_diag["structural_T"])),
            num_valleys=len(struct_diag["valleys"]),
            num_ridges=len(struct_diag["ridges"]),
            landscape_formed=struct_diag["landscape_formed"],
            routing_entropy=routing_entropy,
            loss=loss,
        )
        metrics_history.append(metrics)

        # Snapshot structural_T periodically
        if step % 100 == 0:
            structural_T_snapshots.append({
                "step": step,
                "structural_T": struct_diag["structural_T"].copy(),
            })

        # Progress
        if step % 100 == 0:
            print(f"  Step {step}: variance={metrics.structural_T_variance:.4f}, "
                  f"valleys={metrics.num_valleys}, ridges={metrics.num_ridges}")

    # Final analysis
    print()
    print("=" * 70)
    print("Landscape Formation Analysis")
    print("=" * 70)

    final_struct = bridge.get_structural_temperature_diagnostics()

    print(f"\nFinal Structural Temperature:")
    print(f"  Mean: {final_struct['mean']:.4f}")
    print(f"  Std:  {final_struct['std']:.4f}")
    print(f"  Min:  {np.min(final_struct['structural_T']):.4f}")
    print(f"  Max:  {np.max(final_struct['structural_T']):.4f}")

    print(f"\nLandscape Features:")
    print(f"  Variance: {final_struct['variance']:.6f}")
    print(f"  Normalized entropy: {final_struct['entropy']:.4f}")
    print(f"  Landscape formed: {final_struct['landscape_formed']}")
    print(f"  Valleys (stable experts): {final_struct['valleys']}")
    print(f"  Ridges (unstable experts): {final_struct['ridges']}")

    # Per-expert structural temperature
    print(f"\nPer-Expert Structural Temperature:")
    for i, t in enumerate(final_struct['structural_T']):
        status = "valley" if i in final_struct['valleys'] else (
            "ridge" if i in final_struct['ridges'] else "neutral")
        print(f"  Expert {i}: T̄={t:.4f} ({status})")

    # Evolution analysis
    initial_variance = metrics_history[0].structural_T_variance
    final_variance = metrics_history[-1].structural_T_variance
    variance_growth = final_variance - initial_variance

    print(f"\nEvolution:")
    print(f"  Initial variance: {initial_variance:.6f}")
    print(f"  Final variance: {final_variance:.6f}")
    print(f"  Variance growth: {variance_growth:+.6f}")

    # Correlation between structural_T and reliability
    reliabilities = []
    for expert in bridge.controller.experts[:config.n_experts]:
        reliabilities.append(expert.s)
    reliabilities = np.array(reliabilities)

    if np.std(reliabilities) > 0 and np.std(final_struct['structural_T']) > 0:
        correlation = float(np.corrcoef(
            final_struct['structural_T'],
            reliabilities
        )[0, 1])
    else:
        correlation = 0.0

    print(f"\nCorrelations:")
    print(f"  Structural T ↔ Reliability: {correlation:.4f}")

    if correlation > 0.3:
        print("  → Positive: Reliable experts cooled (became valleys)")
    elif correlation < -0.3:
        print("  → Negative: Reliable experts heated (became ridges)")
    else:
        print("  → Weak correlation")

    return {
        "metrics_history": metrics_history,
        "structural_T_snapshots": structural_T_snapshots,
        "final_diagnostics": final_struct,
        "variance_growth": variance_growth,
        "reliability_correlation": correlation,
    }


if __name__ == "__main__":
    results = run_landscape_experiment()
