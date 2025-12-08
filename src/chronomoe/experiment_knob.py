"""
Meta-Knob Experiment: Demonstrating LLM-style control over ChronoMoE.

This experiment shows how the meta-knob κ can modulate the pressure/alignment
system to improve routing performance. We compare:
    1. Baseline: κ = 0 (neutral)
    2. Rule-based: RuleBasedKnobController adapts κ based on state
    3. Fixed explore: κ = +0.5 (more pressure/exploration)
    4. Fixed exploit: κ = -0.5 (less pressure/exploitation)

Key observation: The rule-based controller should outperform fixed strategies
by adapting to the current system state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from chronomoe.moe import MoE
from chronomoe.router import Router
from chronomoe.bridge import ChronoMoEBridge, RoutingStats
from chronomoe.alignment import AlignmentMatrix, StructuralAligner
from chronomoe.knob import MetaKnob, KnobState, RuleBasedKnobController, KnobDecision
from chronomoe.experiment import SyntheticTask


@dataclass
class KnobExperimentMetrics:
    """Metrics from a knob experiment run."""

    # Core metrics
    final_loss: float
    loss_history: list
    mean_loss: float

    # Knob metrics
    kappa_history: list
    kappa_mean: float
    kappa_std: float

    # Routing metrics
    routing_entropy_history: list
    expert_usage_gini: float

    # Alignment metrics
    alignment_entropy_history: list
    drift_correlation: float

    # Strategy identifier
    strategy: str


@dataclass
class KnobExperimentConfig:
    """Configuration for knob experiment."""

    n_experts: int = 8
    n_chrono: int = 8
    input_dim: int = 64

    n_clusters: int = 4
    n_tokens_per_batch: int = 32
    n_steps: int = 100

    # Pressure parameters
    alpha_T: float = 0.3
    alpha_P: float = 0.2
    alpha_C: float = 0.1

    # Alignment parameters
    eta_A: float = 0.1

    seed: Optional[int] = 42


def compute_entropy(distribution: np.ndarray) -> float:
    """Compute entropy of a probability distribution."""
    p = np.clip(distribution, 1e-10, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def compute_gini(values: np.ndarray) -> float:
    """Compute Gini coefficient (0 = equal, 1 = concentrated)."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumsum = np.cumsum(sorted_vals)
    return float((2 * np.sum((np.arange(1, n + 1) * sorted_vals)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1] + 1e-10))


def run_single_strategy(
    config: KnobExperimentConfig,
    strategy: str,
    fixed_kappa: Optional[float] = None,
) -> KnobExperimentMetrics:
    """
    Run experiment with a specific knob strategy.

    Args:
        config: Experiment configuration.
        strategy: One of "neutral", "rule_based", "fixed_explore", "fixed_exploit".
        fixed_kappa: Fixed κ value for fixed strategies.

    Returns:
        KnobExperimentMetrics with results.
    """
    rng = np.random.default_rng(config.seed)

    # Initialize components
    moe = MoE(
        n_experts=config.n_experts,
        input_dim=config.input_dim,
        output_dim=config.input_dim,  # Same as input for this experiment
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
        seed=config.seed,
    )

    alignment = AlignmentMatrix.create(
        n_chrono=config.n_chrono,
        n_moe=config.n_experts,
        eta_A=config.eta_A,
    )

    aligner = StructuralAligner(alignment)

    task = SyntheticTask.create(
        n_clusters=config.n_clusters,
        n_experts=config.n_experts,
        input_dim=config.input_dim,
        seed=config.seed,
    )

    knob = MetaKnob()
    controller = RuleBasedKnobController() if strategy == "rule_based" else None

    # Tracking
    loss_history = []
    kappa_history = []
    routing_entropy_history = []
    alignment_entropy_history = []

    # Simulated Chronovisor state
    chrono_confidence = rng.uniform(0.3, 0.7, config.n_chrono)

    for step in range(config.n_steps):
        # Generate batch
        tokens, cluster_ids = task.sample_batch(config.n_tokens_per_batch, rng=rng)

        # Determine κ based on strategy
        if strategy == "neutral":
            kappa = 0.0
        elif strategy == "rule_based":
            # Build state for controller
            state = KnobState(
                loss=loss_history[-1] if loss_history else 1.0,
                loss_trend=loss_history[-1] - loss_history[-2] if len(loss_history) >= 2 else 0.0,
                routing_entropy=routing_entropy_history[-1] if routing_entropy_history else 1.5,
                alignment_entropy=alignment_entropy_history[-1] if alignment_entropy_history else 1.0,
                drift_correlation=0.5,  # Simplified
                coherence_R=0.7,  # Simplified
                population=config.n_experts,
            )
            decision = controller.decide(state)
            kappa = decision.kappa
        else:
            kappa = fixed_kappa

        kappa_history.append(kappa)
        factors = knob.set_kappa(kappa)

        # Get routing stats and compute pressure
        router.reset_log()

        # First pass to get stats
        _ = router.forward(tokens, top_k=2)
        stats = router.get_routing_stats()

        # Feed stats to Chronovisor and run ticks
        routing_stats = RoutingStats(
            expert_usage=stats["usage"],
            mean_gate_weights=stats["mean_gates"],
            batch_loss=loss_history[-1] if loss_history else 0.5,
        )
        bridge.feed_routing_stats(routing_stats, num_chronovisor_ticks=10)

        # Get pressure bias from Chronovisor
        pressure_bias = bridge.get_pressure_bias()
        pressure = pressure_bias.combined

        # Apply alignment transformation
        # Note: pressure from bridge is n_experts, alignment expects n_chrono
        # For now, use pressure directly (assumes n_chrono == n_experts)
        aligned_pressure = alignment.apply_pressure(pressure)
        router.inject_pressure(aligned_pressure)

        # Forward with knob modulation
        router.reset_log()
        gate_weights = router.forward_with_knob(
            tokens,
            pressure_scale=factors.pressure_scale,
            explore_bias=factors.explore_bias,
            top_k=2,
        )

        # Compute loss
        expert_indices = np.argmax(gate_weights, axis=1)
        loss, _ = task.compute_loss(cluster_ids, expert_indices)
        loss_history.append(loss)

        # Track metrics
        distribution = stats["distribution"]
        if len(distribution) > 0:
            routing_entropy = compute_entropy(distribution)
        else:
            routing_entropy = 0.0
        routing_entropy_history.append(routing_entropy)

        align_entropy = alignment.alignment_entropy()
        alignment_entropy_history.append(align_entropy)

        # Update alignment with knob modulation
        moe_specialization = rng.uniform(0.2, 0.8, config.n_experts)
        alignment.update_with_knob(
            chrono_confidence=chrono_confidence,
            moe_specialization=moe_specialization,
            alignment_lr_mul=factors.alignment_lr_mul,
            current_tick=step,
        )

        # Slowly evolve Chronovisor confidence (simulated drift)
        chrono_confidence = 0.95 * chrono_confidence + 0.05 * rng.uniform(0.3, 0.7, config.n_chrono)

    # Compute final metrics
    final_loss = loss_history[-1]
    mean_loss = float(np.mean(loss_history))
    kappa_mean = float(np.mean(kappa_history))
    kappa_std = float(np.std(kappa_history))

    final_stats = router.get_routing_stats()
    usage = final_stats["usage"]
    if len(usage) > 0 and usage.sum() > 0:
        expert_usage_gini = compute_gini(usage)
    else:
        expert_usage_gini = 0.0

    # Compute drift correlation
    if len(loss_history) >= 10:
        recent_loss = np.array(loss_history[-10:])
        recent_entropy = np.array(alignment_entropy_history[-10:])
        if np.std(recent_loss) > 0 and np.std(recent_entropy) > 0:
            drift_corr = float(np.corrcoef(recent_loss, recent_entropy)[0, 1])
        else:
            drift_corr = 0.0
    else:
        drift_corr = 0.0

    return KnobExperimentMetrics(
        final_loss=final_loss,
        loss_history=loss_history,
        mean_loss=mean_loss,
        kappa_history=kappa_history,
        kappa_mean=kappa_mean,
        kappa_std=kappa_std,
        routing_entropy_history=routing_entropy_history,
        expert_usage_gini=expert_usage_gini,
        alignment_entropy_history=alignment_entropy_history,
        drift_correlation=drift_corr,
        strategy=strategy,
    )


def run_knob_experiment(
    config: Optional[KnobExperimentConfig] = None,
) -> dict:
    """
    Run full knob experiment comparing all strategies.

    Returns:
        Dictionary with results for each strategy.
    """
    if config is None:
        config = KnobExperimentConfig()

    print("=" * 60)
    print("Meta-Knob Experiment: Comparing Control Strategies")
    print("=" * 60)
    print(f"Experts: {config.n_experts}, Steps: {config.n_steps}")
    print(f"Pressure weights: α_T={config.alpha_T}, α_P={config.alpha_P}, α_C={config.alpha_C}")
    print()

    results = {}

    # Strategy 1: Neutral (κ = 0)
    print("Running: Neutral (κ = 0)...")
    results["neutral"] = run_single_strategy(config, "neutral", fixed_kappa=0.0)
    print(f"  Mean loss: {results['neutral'].mean_loss:.4f}")

    # Strategy 2: Rule-based controller
    print("Running: Rule-based controller...")
    results["rule_based"] = run_single_strategy(config, "rule_based")
    print(f"  Mean loss: {results['rule_based'].mean_loss:.4f}")
    print(f"  κ range: [{min(results['rule_based'].kappa_history):.2f}, {max(results['rule_based'].kappa_history):.2f}]")

    # Strategy 3: Fixed explore (κ = +0.5)
    print("Running: Fixed explore (κ = +0.5)...")
    results["fixed_explore"] = run_single_strategy(config, "fixed_explore", fixed_kappa=0.5)
    print(f"  Mean loss: {results['fixed_explore'].mean_loss:.4f}")

    # Strategy 4: Fixed exploit (κ = -0.5)
    print("Running: Fixed exploit (κ = -0.5)...")
    results["fixed_exploit"] = run_single_strategy(config, "fixed_exploit", fixed_kappa=-0.5)
    print(f"  Mean loss: {results['fixed_exploit'].mean_loss:.4f}")

    # Summary
    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)

    baseline_loss = results["neutral"].mean_loss

    for name, metrics in results.items():
        improvement = (baseline_loss - metrics.mean_loss) / baseline_loss * 100
        print(f"{name:20s}: loss={metrics.mean_loss:.4f} ({improvement:+.1f}% vs neutral)")
        print(f"                      κ_mean={metrics.kappa_mean:+.3f}, κ_std={metrics.kappa_std:.3f}")
        print(f"                      gini={metrics.expert_usage_gini:.3f}, entropy={metrics.routing_entropy_history[-1]:.3f}")

    # Find best strategy
    best = min(results.items(), key=lambda x: x[1].mean_loss)
    print()
    print(f"Best strategy: {best[0]} (loss={best[1].mean_loss:.4f})")

    return results


if __name__ == "__main__":
    results = run_knob_experiment()
