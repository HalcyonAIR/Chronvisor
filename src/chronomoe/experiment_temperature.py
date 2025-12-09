"""
Temperature Field Experiment: Pressure × Temperature Geometry.

Compares routing strategies:
1. Baseline: No pressure, no temperature warping
2. Pressure-only: With pressure injection but uniform temperature
3. Temperature-only: With per-expert temperature but no pressure
4. Full system: Pressure + Temperature (2-field routing environment)

The key insight: Pressure is the force field, Temperature is the permeability.
Together they create a dynamic energy landscape for routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from chronomoe.moe import MoE
from chronomoe.router import Router
from chronomoe.bridge import ChronoMoEBridge, RoutingStats
from chronomoe.alignment import AlignmentMatrix
from chronomoe.knob import MetaKnob
from chronomoe.experiment import SyntheticTask


@dataclass
class TemperatureExperimentMetrics:
    """Metrics from a temperature experiment run."""

    strategy: str
    final_loss: float
    mean_loss: float
    loss_history: list

    # Routing metrics
    routing_entropy: float
    expert_usage_gini: float

    # Temperature metrics (if applicable)
    mean_temperature: Optional[float] = None
    temperature_variance: Optional[float] = None

    # Pressure metrics (if applicable)
    mean_pressure_magnitude: Optional[float] = None


@dataclass
class TemperatureExperimentConfig:
    """Configuration for temperature experiment."""

    n_experts: int = 8
    input_dim: int = 64
    n_clusters: int = 4
    n_tokens_per_batch: int = 32
    n_steps: int = 100

    # Pressure parameters
    alpha_T: float = 0.3
    alpha_P: float = 0.2
    alpha_C: float = 0.1

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


def compute_gini(values: np.ndarray) -> float:
    """Compute Gini coefficient."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumsum = np.cumsum(sorted_vals)
    if cumsum[-1] == 0:
        return 0.0
    return float(
        (2 * np.sum((np.arange(1, n + 1) * sorted_vals)) - (n + 1) * cumsum[-1])
        / (n * cumsum[-1] + 1e-10)
    )


def run_strategy(
    config: TemperatureExperimentConfig,
    strategy: str,
) -> TemperatureExperimentMetrics:
    """
    Run experiment with a specific routing strategy.

    Strategies:
    - baseline: No pressure, no temperature warping
    - pressure_only: Pressure injection with uniform temperature
    - temperature_only: Per-expert temperature without pressure
    - full: Pressure + Temperature (2-field system)
    """
    rng = np.random.default_rng(config.seed)

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
        alpha_T=config.alpha_T,
        alpha_P=config.alpha_P,
        alpha_C=config.alpha_C,
        base_temperature=config.base_temperature,
        beta_R=config.beta_R,
        beta_drift=config.beta_drift,
        seed=config.seed,
    )

    task = SyntheticTask.create(
        n_clusters=config.n_clusters,
        n_experts=config.n_experts,
        input_dim=config.input_dim,
        seed=config.seed,
    )

    # Tracking
    loss_history = []
    temperature_history = []
    pressure_history = []

    for step in range(config.n_steps):
        # Generate batch
        tokens, cluster_ids = task.sample_batch(config.n_tokens_per_batch, rng=rng)

        # Get routing stats (first pass to feed Chronovisor)
        router.reset_log()
        _ = router.forward(tokens, top_k=2)
        stats = router.get_routing_stats()

        # Feed stats to Chronovisor
        routing_stats = RoutingStats(
            expert_usage=stats["usage"],
            mean_gate_weights=stats["mean_gates"],
            batch_loss=loss_history[-1] if loss_history else 0.5,
        )
        bridge.feed_routing_stats(routing_stats, num_chronovisor_ticks=10)

        # Route based on strategy
        router.reset_log()

        if strategy == "baseline":
            # Pure softmax routing
            gate_weights = router.forward(tokens, top_k=2)

        elif strategy == "pressure_only":
            # Pressure but uniform temperature
            pressure_bias = bridge.get_pressure_bias()
            router.inject_pressure(pressure_bias.combined)
            pressure_history.append(np.abs(pressure_bias.combined).mean())
            gate_weights = router.forward(tokens, top_k=2)

        elif strategy == "temperature_only":
            # Temperature but no pressure
            temp_field = bridge.get_temperature_field()
            temperature_history.append(temp_field.temperatures.copy())
            gate_weights = router.forward_with_temperature(
                tokens,
                temp_field.temperatures,
                pressure_scale=0.0,  # No pressure
            )

        elif strategy == "full":
            # Full 2-field system
            pressure_bias = bridge.get_pressure_bias()
            router.inject_pressure(pressure_bias.combined)
            pressure_history.append(np.abs(pressure_bias.combined).mean())

            temp_field = bridge.get_temperature_field()
            temperature_history.append(temp_field.temperatures.copy())

            gate_weights = router.forward_with_temperature(
                tokens,
                temp_field.temperatures,
                pressure_scale=1.0,
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Compute loss
        expert_indices = np.argmax(gate_weights, axis=1)
        loss, _ = task.compute_loss(cluster_ids, expert_indices)
        loss_history.append(loss)

    # Compute final metrics
    final_stats = router.get_routing_stats()
    distribution = final_stats["distribution"]
    usage = final_stats["usage"]

    routing_entropy = compute_entropy(distribution) if len(distribution) > 0 else 0.0
    expert_gini = compute_gini(usage) if len(usage) > 0 and usage.sum() > 0 else 0.0

    # Optional metrics
    mean_temp = None
    temp_var = None
    mean_pressure = None

    if temperature_history:
        all_temps = np.array(temperature_history)
        mean_temp = float(all_temps.mean())
        temp_var = float(all_temps.var())

    if pressure_history:
        mean_pressure = float(np.mean(pressure_history))

    return TemperatureExperimentMetrics(
        strategy=strategy,
        final_loss=loss_history[-1],
        mean_loss=float(np.mean(loss_history)),
        loss_history=loss_history,
        routing_entropy=routing_entropy,
        expert_usage_gini=expert_gini,
        mean_temperature=mean_temp,
        temperature_variance=temp_var,
        mean_pressure_magnitude=mean_pressure,
    )


def run_temperature_experiment(
    config: Optional[TemperatureExperimentConfig] = None,
) -> dict:
    """
    Run full temperature experiment comparing all strategies.

    Returns:
        Dictionary with results for each strategy.
    """
    if config is None:
        config = TemperatureExperimentConfig()

    print("=" * 70)
    print("Temperature Field Experiment: Pressure × Temperature Geometry")
    print("=" * 70)
    print(f"Experts: {config.n_experts}, Steps: {config.n_steps}")
    print(f"Pressure: α_T={config.alpha_T}, α_P={config.alpha_P}, α_C={config.alpha_C}")
    print(f"Temperature: base={config.base_temperature}, β_R={config.beta_R}, β_drift={config.beta_drift}")
    print()

    strategies = ["baseline", "pressure_only", "temperature_only", "full"]
    results = {}

    for strategy in strategies:
        print(f"Running: {strategy}...")
        results[strategy] = run_strategy(config, strategy)
        print(f"  Mean loss: {results[strategy].mean_loss:.4f}")
        print(f"  Entropy: {results[strategy].routing_entropy:.4f}")

    # Summary
    print()
    print("=" * 70)
    print("Results Summary")
    print("=" * 70)

    baseline_loss = results["baseline"].mean_loss

    for name, metrics in results.items():
        improvement = (baseline_loss - metrics.mean_loss) / baseline_loss * 100
        print(f"\n{name}:")
        print(f"  Loss: {metrics.mean_loss:.4f} ({improvement:+.1f}% vs baseline)")
        print(f"  Routing entropy: {metrics.routing_entropy:.4f}")
        print(f"  Expert Gini: {metrics.expert_usage_gini:.3f}")
        if metrics.mean_temperature is not None:
            print(f"  Mean temperature: {metrics.mean_temperature:.3f}")
            print(f"  Temperature variance: {metrics.temperature_variance:.4f}")
        if metrics.mean_pressure_magnitude is not None:
            print(f"  Mean |pressure|: {metrics.mean_pressure_magnitude:.4f}")

    # Find best strategy
    best = min(results.items(), key=lambda x: x[1].mean_loss)
    print()
    print(f"Best strategy: {best[0]} (loss={best[1].mean_loss:.4f})")

    # Analysis
    print()
    print("=" * 70)
    print("Analysis")
    print("=" * 70)

    pressure_effect = results["pressure_only"].mean_loss - results["baseline"].mean_loss
    temp_effect = results["temperature_only"].mean_loss - results["baseline"].mean_loss
    full_effect = results["full"].mean_loss - results["baseline"].mean_loss
    synergy = full_effect - (pressure_effect + temp_effect)

    print(f"\nPressure effect: {pressure_effect:+.4f}")
    print(f"Temperature effect: {temp_effect:+.4f}")
    print(f"Full system effect: {full_effect:+.4f}")
    print(f"Synergy (interaction): {synergy:+.4f}")

    if synergy < -0.01:
        print("\n✓ Positive synergy: Pressure and Temperature work better together")
    elif synergy > 0.01:
        print("\n✗ Negative synergy: Pressure and Temperature interfere")
    else:
        print("\n≈ Neutral: Effects are approximately additive")

    return results


if __name__ == "__main__":
    results = run_temperature_experiment()
