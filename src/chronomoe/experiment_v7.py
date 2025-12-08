"""
V7 Experiment: Structural Alignment Between Chronovisor and MoE Experts.

This module extends the V1 experiment to track how the alignment matrix A
evolves from identity to a meaningful mapping as both systems co-specialize.

Key metrics:
- Alignment entropy over time
- Drift correlation (Chronovisor clusters → MoE usage patterns)
- KL divergence of routing with learned A vs identity A
- Specialization stability over time
- Whether Chronovisor bifurcations predict MoE expert splitting
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from chronomoe.moe import MoE
from chronomoe.router import Router, RoutingLog
from chronomoe.bridge import ChronoMoEBridge, RoutingStats, PressureBias
from chronomoe.alignment import AlignmentMatrix, StructuralAligner, AlignmentEvent
from chronomoe.experiment import (
    SyntheticTask,
    ExperimentMetrics,
    ExperimentResult,
)


@dataclass
class V7Metrics:
    """
    Metrics specific to V7 structural alignment experiments.
    """

    @staticmethod
    def alignment_drift_correlation(
        A_history: list[np.ndarray],
        usage_history: list[np.ndarray],
    ) -> float:
        """
        Compute correlation between alignment evolution and usage patterns.

        Higher correlation = alignment is tracking actual MoE usage.
        """
        if len(A_history) < 2 or len(usage_history) < 2:
            return 0.0

        # Use the last few snapshots
        n = min(len(A_history), len(usage_history), 10)

        # Compute alignment-weighted expected usage vs actual usage
        correlations = []
        for i in range(-n, 0):
            A = A_history[i]
            usage = usage_history[i]

            # For each Chronovisor expert, compute expected MoE usage from alignment
            expected = A.sum(axis=0)  # Sum of alignments pointing to each MoE
            expected = expected / expected.sum() if expected.sum() > 0 else expected

            # Actual normalized usage
            actual = usage / usage.sum() if usage.sum() > 0 else usage

            # Correlation between expected and actual
            if len(expected) == len(actual) and len(expected) > 0:
                corr = np.corrcoef(expected, actual)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        return float(np.mean(correlations)) if correlations else 0.0

    @staticmethod
    def specialization_stability(
        specialization_history: list[float],
        window: int = 10,
    ) -> float:
        """
        Measure how stable specialization is over time.

        Lower variance = more stable specialization.
        Returns 1 - normalized_variance (higher = more stable).
        """
        if len(specialization_history) < window:
            return 0.0

        recent = specialization_history[-window:]
        variance = np.var(recent)
        mean = np.mean(recent)

        if mean == 0:
            return 1.0

        # Coefficient of variation (normalized variance)
        cv = math.sqrt(variance) / mean

        # Convert to stability (1 = perfectly stable)
        return max(0.0, 1.0 - cv)

    @staticmethod
    def alignment_concentration(A: np.ndarray) -> float:
        """
        Measure how concentrated the alignment is.

        Higher = more specialized mapping (few strong alignments).
        Lower = more diffuse mapping (many weak alignments).
        """
        # Use Gini coefficient-like measure
        A_flat = A.flatten()
        A_sorted = np.sort(A_flat)
        n = len(A_sorted)
        if n == 0 or A_sorted.sum() == 0:
            return 0.0

        # Lorenz curve area
        cumsum = np.cumsum(A_sorted)
        lorenz = cumsum / cumsum[-1]
        gini = 1 - 2 * np.mean(lorenz)

        return float(gini)


@dataclass
class AlignmentSnapshot:
    """Snapshot of alignment state at a point in time."""

    tick: int
    A: np.ndarray
    entropy: float
    concentration: float
    dominant_mapping: np.ndarray
    events: list[AlignmentEvent]


@dataclass
class V7ExperimentResult:
    """
    Results from a V7 experiment with alignment tracking.
    """

    name: str
    n_batches: int
    n_tokens: int

    # Base metrics (from V1)
    expert_distribution: np.ndarray
    entropy: float
    mean_loss: float
    loss_history: list[float]
    specialization_score: float

    # V7 alignment metrics
    alignment_history: list[AlignmentSnapshot]
    final_alignment: np.ndarray
    alignment_entropy_history: list[float]
    alignment_concentration_history: list[float]

    # Drift correlation
    drift_correlation: float

    # Specialization stability
    specialization_stability: float

    # Structural events
    total_absorb_events: int
    total_decay_events: int

    # Final state
    final_R: float
    final_alignment_entropy: float
    final_alignment_concentration: float


@dataclass
class V7ComparisonResult:
    """
    Comparison between identity alignment and learned alignment.
    """

    identity: V7ExperimentResult
    learned: V7ExperimentResult

    # Distribution shift
    kl_identity_to_learned: float
    kl_learned_to_identity: float

    # Loss comparison
    loss_delta: float
    loss_percent_change: float

    # Alignment evolution
    alignment_evolved: bool  # Did A change significantly from identity?
    alignment_entropy_delta: float
    alignment_concentration_delta: float

    # Specialization comparison
    specialization_delta: float
    stability_delta: float


@dataclass
class V7Experiment:
    """
    V7 Experiment driver with structural alignment.

    Runs experiments comparing:
    1. Identity alignment (fixed 1:1 mapping)
    2. Learned alignment (A evolves via confidence × specialization)
    """

    task: SyntheticTask
    moe: MoE
    router: Router
    bridge: ChronoMoEBridge
    aligner: StructuralAligner

    n_batches: int = 100
    batch_size: int = 32
    chronovisor_ticks_per_batch: int = 20
    alignment_update_frequency: int = 5  # Update A every N batches

    seed: int = 42

    @classmethod
    def create(
        cls,
        n_experts: int = 8,
        n_clusters: int = 4,
        input_dim: int = 64,
        output_dim: int = 64,
        n_batches: int = 100,
        batch_size: int = 32,
        chronovisor_ticks_per_batch: int = 20,
        alignment_update_frequency: int = 5,
        eta_A: float = 0.05,
        alpha_T: float = 0.3,
        alpha_P: float = 0.2,
        alpha_C: float = 0.1,
        seed: int = 42,
    ) -> V7Experiment:
        """Factory method to create V7 experiment with all components."""
        # Create synthetic task
        task = SyntheticTask.create(
            n_clusters=n_clusters,
            n_experts=n_experts,
            input_dim=input_dim,
            output_dim=output_dim,
            seed=seed,
        )

        # Create MoE
        moe = MoE(
            n_experts=n_experts,
            input_dim=input_dim,
            output_dim=output_dim,
            top_k=2,
            cluster_affinities=task.get_expert_affinities(),
            seed=seed,
        )

        # Create router
        router = Router(
            input_dim=input_dim,
            n_experts=n_experts,
            seed=seed,
        )

        # Create bridge
        bridge = ChronoMoEBridge.create(
            n_experts=n_experts,
            alpha_T=alpha_T,
            alpha_P=alpha_P,
            alpha_C=alpha_C,
            seed=seed,
        )

        # Create structural aligner
        aligner = StructuralAligner.create(
            n_chrono=n_experts,
            n_moe=n_experts,
            eta_A=eta_A,
            seed=seed,
        )

        return cls(
            task=task,
            moe=moe,
            router=router,
            bridge=bridge,
            aligner=aligner,
            n_batches=n_batches,
            batch_size=batch_size,
            chronovisor_ticks_per_batch=chronovisor_ticks_per_batch,
            alignment_update_frequency=alignment_update_frequency,
            seed=seed,
        )

    def run_identity_baseline(self) -> V7ExperimentResult:
        """
        Run experiment with fixed identity alignment (1:1 mapping).

        This is the baseline - no alignment learning.
        """
        rng = np.random.default_rng(self.seed)

        # Reset everything
        self.router.clear_pressure()
        self.router.reset_log()
        self.bridge.reset(self.seed)
        self.aligner.reset(self.seed)

        loss_history = []
        specialization_history = []
        alignment_history = []
        alignment_entropy_history = []
        alignment_concentration_history = []
        usage_history = []
        all_cluster_ids = []

        for batch_idx in range(self.n_batches):
            # Sample batch
            inputs, cluster_ids = self.task.sample_batch(self.batch_size, rng=rng)
            all_cluster_ids.extend(cluster_ids)

            # Get Chronovisor pressure (identity alignment - no transformation)
            chrono_pressure = self.bridge.get_pressure_bias()
            # Apply identity mapping (direct injection)
            self.router.inject_pressure(chrono_pressure.combined)

            # Route
            token_indices = np.arange(
                batch_idx * self.batch_size,
                (batch_idx + 1) * self.batch_size,
            )
            gate_weights = self.router.forward(inputs, token_indices=token_indices)

            # MoE forward
            moe_output = self.moe.forward(inputs, gate_weights)

            # Compute loss
            top1_experts = moe_output.chosen_experts[:, -1]
            batch_loss, _ = self.task.compute_loss(cluster_ids, top1_experts)
            loss_history.append(batch_loss)

            # Track usage
            usage = self.router.log.get_expert_usage()
            usage_history.append(usage.copy())

            # Feed to Chronovisor
            stats = RoutingStats(
                expert_usage=usage,
                mean_gate_weights=self.router.log.get_mean_gate_weights(),
                batch_loss=batch_loss,
            )
            self.bridge.feed_routing_stats(stats, self.chronovisor_ticks_per_batch)

            # Track specialization
            if len(all_cluster_ids) >= self.batch_size:
                spec, _ = ExperimentMetrics.specialization_score(
                    self.router.log,
                    np.array(all_cluster_ids[-self.batch_size * 10:]),
                    self.task.n_clusters,
                    self.task.n_experts,
                )
                specialization_history.append(spec)

            # Track alignment (stays at identity)
            alignment_entropy_history.append(self.aligner.alignment.alignment_entropy())
            alignment_concentration_history.append(
                V7Metrics.alignment_concentration(self.aligner.alignment.A)
            )

            # Snapshot alignment periodically
            if batch_idx % 10 == 0:
                snapshot = AlignmentSnapshot(
                    tick=batch_idx,
                    A=self.aligner.alignment.A.copy(),
                    entropy=self.aligner.alignment.alignment_entropy(),
                    concentration=V7Metrics.alignment_concentration(self.aligner.alignment.A),
                    dominant_mapping=self.aligner.alignment.dominant_mapping(),
                    events=[],
                )
                alignment_history.append(snapshot)

        # Compute final metrics
        distribution = self.router.log.get_expert_distribution()
        final_spec, _ = ExperimentMetrics.specialization_score(
            self.router.log,
            np.array(all_cluster_ids),
            self.task.n_clusters,
            self.task.n_experts,
        )

        drift_corr = V7Metrics.alignment_drift_correlation(
            [self.aligner.alignment.A] * len(usage_history),  # Identity doesn't change
            usage_history,
        )

        spec_stability = V7Metrics.specialization_stability(specialization_history)

        return V7ExperimentResult(
            name="identity",
            n_batches=self.n_batches,
            n_tokens=self.n_batches * self.batch_size,
            expert_distribution=distribution,
            entropy=ExperimentMetrics.entropy(distribution),
            mean_loss=float(np.mean(loss_history)),
            loss_history=loss_history,
            specialization_score=final_spec,
            alignment_history=alignment_history,
            final_alignment=self.aligner.alignment.A.copy(),
            alignment_entropy_history=alignment_entropy_history,
            alignment_concentration_history=alignment_concentration_history,
            drift_correlation=drift_corr,
            specialization_stability=spec_stability,
            total_absorb_events=0,
            total_decay_events=0,
            final_R=self.bridge.R_history[-1] if self.bridge.R_history else 0.0,
            final_alignment_entropy=self.aligner.alignment.alignment_entropy(),
            final_alignment_concentration=V7Metrics.alignment_concentration(
                self.aligner.alignment.A
            ),
        )

    def run_learned_alignment(self) -> V7ExperimentResult:
        """
        Run experiment with learned alignment (A evolves over time).

        This is V7 - alignment learns from confidence × specialization.
        """
        rng = np.random.default_rng(self.seed)

        # Reset everything
        self.router.clear_pressure()
        self.router.reset_log()
        self.bridge.reset(self.seed)
        self.aligner.reset(self.seed)

        loss_history = []
        specialization_history = []
        alignment_history = []
        alignment_entropy_history = []
        alignment_concentration_history = []
        usage_history = []
        all_cluster_ids = []
        A_history_raw = []

        for batch_idx in range(self.n_batches):
            # Sample batch
            inputs, cluster_ids = self.task.sample_batch(self.batch_size, rng=rng)
            all_cluster_ids.extend(cluster_ids)

            # Get Chronovisor pressure
            chrono_pressure = self.bridge.get_pressure_bias()

            # APPLY LEARNED ALIGNMENT: b_j = Σ_i A_ij · b_i
            aligned_pressure = self.aligner.compute_aligned_pressure(chrono_pressure.combined)
            self.router.inject_pressure(aligned_pressure)

            # Route
            token_indices = np.arange(
                batch_idx * self.batch_size,
                (batch_idx + 1) * self.batch_size,
            )
            gate_weights = self.router.forward(inputs, token_indices=token_indices)

            # MoE forward
            moe_output = self.moe.forward(inputs, gate_weights)

            # Compute loss
            top1_experts = moe_output.chosen_experts[:, -1]
            batch_loss, _ = self.task.compute_loss(cluster_ids, top1_experts)
            loss_history.append(batch_loss)

            # Track usage
            usage = self.router.log.get_expert_usage()
            usage_history.append(usage.copy())

            # Feed to Chronovisor
            stats = RoutingStats(
                expert_usage=usage,
                mean_gate_weights=self.router.log.get_mean_gate_weights(),
                batch_loss=batch_loss,
            )
            self.bridge.feed_routing_stats(stats, self.chronovisor_ticks_per_batch)

            # UPDATE ALIGNMENT every N batches
            if batch_idx > 0 and batch_idx % self.alignment_update_frequency == 0:
                # Compute Chronovisor confidence (from reliability)
                chrono_states = self.bridge.get_expert_states()
                chrono_confidence = np.array([
                    max(0.01, s["effective_reliability"] + 1.0)  # Shift to positive
                    for s in chrono_states
                ])

                # Compute MoE specialization (from usage patterns)
                recent_usage = usage_history[-self.alignment_update_frequency:]
                avg_usage = np.mean(recent_usage, axis=0)
                moe_specialization = avg_usage / (avg_usage.sum() + 1e-10)

                # Update alignment matrix
                self.aligner.update_alignment(
                    chrono_confidence=chrono_confidence,
                    moe_specialization=moe_specialization,
                    current_tick=batch_idx,
                )

                A_history_raw.append(self.aligner.alignment.A.copy())

            # Track specialization
            if len(all_cluster_ids) >= self.batch_size:
                spec, _ = ExperimentMetrics.specialization_score(
                    self.router.log,
                    np.array(all_cluster_ids[-self.batch_size * 10:]),
                    self.task.n_clusters,
                    self.task.n_experts,
                )
                specialization_history.append(spec)

            # Track alignment metrics
            alignment_entropy_history.append(self.aligner.alignment.alignment_entropy())
            alignment_concentration_history.append(
                V7Metrics.alignment_concentration(self.aligner.alignment.A)
            )

            # Snapshot alignment periodically
            if batch_idx % 10 == 0:
                snapshot = AlignmentSnapshot(
                    tick=batch_idx,
                    A=self.aligner.alignment.A.copy(),
                    entropy=self.aligner.alignment.alignment_entropy(),
                    concentration=V7Metrics.alignment_concentration(self.aligner.alignment.A),
                    dominant_mapping=self.aligner.alignment.dominant_mapping(),
                    events=list(self.aligner.alignment.events),
                )
                alignment_history.append(snapshot)

        # Compute final metrics
        distribution = self.router.log.get_expert_distribution()
        final_spec, _ = ExperimentMetrics.specialization_score(
            self.router.log,
            np.array(all_cluster_ids),
            self.task.n_clusters,
            self.task.n_experts,
        )

        drift_corr = V7Metrics.alignment_drift_correlation(A_history_raw, usage_history)
        spec_stability = V7Metrics.specialization_stability(specialization_history)

        # Count events
        absorb_events = sum(
            1 for e in self.aligner.alignment.events if e.event_type == "absorb"
        )
        decay_events = sum(
            1 for e in self.aligner.alignment.events if e.event_type == "decay"
        )

        return V7ExperimentResult(
            name="learned",
            n_batches=self.n_batches,
            n_tokens=self.n_batches * self.batch_size,
            expert_distribution=distribution,
            entropy=ExperimentMetrics.entropy(distribution),
            mean_loss=float(np.mean(loss_history)),
            loss_history=loss_history,
            specialization_score=final_spec,
            alignment_history=alignment_history,
            final_alignment=self.aligner.alignment.A.copy(),
            alignment_entropy_history=alignment_entropy_history,
            alignment_concentration_history=alignment_concentration_history,
            drift_correlation=drift_corr,
            specialization_stability=spec_stability,
            total_absorb_events=absorb_events,
            total_decay_events=decay_events,
            final_R=self.bridge.R_history[-1] if self.bridge.R_history else 0.0,
            final_alignment_entropy=self.aligner.alignment.alignment_entropy(),
            final_alignment_concentration=V7Metrics.alignment_concentration(
                self.aligner.alignment.A
            ),
        )

    def run_comparison(self) -> V7ComparisonResult:
        """Run both experiments and compare."""
        print("Running identity baseline...")
        identity = self.run_identity_baseline()

        print("Running learned alignment experiment...")
        learned = self.run_learned_alignment()

        # Compute comparison metrics
        kl_il = ExperimentMetrics.kl_divergence(
            identity.expert_distribution,
            learned.expert_distribution,
        )
        kl_li = ExperimentMetrics.kl_divergence(
            learned.expert_distribution,
            identity.expert_distribution,
        )

        loss_delta = learned.mean_loss - identity.mean_loss
        loss_percent = 0.0
        if identity.mean_loss > 0:
            loss_percent = loss_delta / identity.mean_loss * 100

        # Check if alignment evolved
        A_identity = np.eye(self.task.n_experts)
        A_diff = np.linalg.norm(learned.final_alignment - A_identity)
        alignment_evolved = bool(A_diff > 0.5)  # Significant change from identity

        return V7ComparisonResult(
            identity=identity,
            learned=learned,
            kl_identity_to_learned=kl_il,
            kl_learned_to_identity=kl_li,
            loss_delta=loss_delta,
            loss_percent_change=loss_percent,
            alignment_evolved=alignment_evolved,
            alignment_entropy_delta=(
                learned.final_alignment_entropy - identity.final_alignment_entropy
            ),
            alignment_concentration_delta=(
                learned.final_alignment_concentration - identity.final_alignment_concentration
            ),
            specialization_delta=learned.specialization_score - identity.specialization_score,
            stability_delta=learned.specialization_stability - identity.specialization_stability,
        )

    def print_comparison(self, result: V7ComparisonResult) -> None:
        """Print formatted V7 comparison results."""
        print()
        print("=" * 90)
        print("CHRONOMOE V7: STRUCTURAL ALIGNMENT EXPERIMENT")
        print("=" * 90)
        print()

        print(f"Configuration:")
        print(f"  Experts: {self.task.n_experts}")
        print(f"  Clusters: {self.task.n_clusters}")
        print(f"  Batches: {self.n_batches}")
        print(f"  Alignment update frequency: every {self.alignment_update_frequency} batches")
        print(f"  Alignment learning rate: {self.aligner.alignment.eta_A}")
        print()

        print("-" * 90)
        print("ROUTING DISTRIBUTION COMPARISON")
        print("-" * 90)
        print()
        print(f"{'Expert':<10} {'Identity':>12} {'Learned':>12} {'Delta':>12}")
        print("-" * 46)
        for i in range(self.task.n_experts):
            ident = result.identity.expert_distribution[i]
            learn = result.learned.expert_distribution[i]
            delta = learn - ident
            sign = "+" if delta >= 0 else ""
            print(f"E{i:<9} {ident:>12.4f} {learn:>12.4f} {sign}{delta:>11.4f}")
        print()

        print("-" * 90)
        print("LOSS AND SPECIALIZATION")
        print("-" * 90)
        print()
        print(f"Mean loss (identity):       {result.identity.mean_loss:.4f}")
        print(f"Mean loss (learned):        {result.learned.mean_loss:.4f}")
        print(f"Loss delta:                 {result.loss_delta:+.4f} ({result.loss_percent_change:+.1f}%)")
        print()
        print(f"Specialization (identity):  {result.identity.specialization_score:.4f}")
        print(f"Specialization (learned):   {result.learned.specialization_score:.4f}")
        print(f"Specialization delta:       {result.specialization_delta:+.4f}")
        print()
        print(f"Stability (identity):       {result.identity.specialization_stability:.4f}")
        print(f"Stability (learned):        {result.learned.specialization_stability:.4f}")
        print(f"Stability delta:            {result.stability_delta:+.4f}")
        print()

        print("-" * 90)
        print("ALIGNMENT MATRIX EVOLUTION")
        print("-" * 90)
        print()
        print(f"Alignment evolved from identity: {'YES' if result.alignment_evolved else 'NO'}")
        print()
        print(f"Alignment entropy (identity):    {result.identity.final_alignment_entropy:.4f}")
        print(f"Alignment entropy (learned):     {result.learned.final_alignment_entropy:.4f}")
        print(f"Entropy delta:                   {result.alignment_entropy_delta:+.4f}")
        print()
        print(f"Alignment concentration (identity): {result.identity.final_alignment_concentration:.4f}")
        print(f"Alignment concentration (learned):  {result.learned.final_alignment_concentration:.4f}")
        print(f"Concentration delta:                {result.alignment_concentration_delta:+.4f}")
        print()
        print(f"Drift correlation (learned):     {result.learned.drift_correlation:.4f}")
        print()
        print(f"Structural events:")
        print(f"  Absorb events: {result.learned.total_absorb_events}")
        print(f"  Decay events:  {result.learned.total_decay_events}")
        print()

        print("-" * 90)
        print("FINAL ALIGNMENT MATRIX (LEARNED)")
        print("-" * 90)
        print()
        A = result.learned.final_alignment
        print("        ", end="")
        for j in range(A.shape[1]):
            print(f"MoE{j:>2} ", end="")
        print()
        print("        " + "-" * (A.shape[1] * 6))
        for i in range(A.shape[0]):
            print(f"Chr{i:>2} | ", end="")
            for j in range(A.shape[1]):
                val = A[i, j]
                if val > 0.5:
                    print(f"\033[92m{val:.2f}\033[0m ", end="")  # Green for high
                elif val > 0.2:
                    print(f"\033[93m{val:.2f}\033[0m ", end="")  # Yellow for medium
                else:
                    print(f"{val:.2f} ", end="")  # Normal for low
            print()
        print()

        print("-" * 90)
        print("DOMINANT MAPPINGS")
        print("-" * 90)
        print()
        dominant = result.learned.alignment_history[-1].dominant_mapping if result.learned.alignment_history else np.arange(A.shape[0])
        print("Chronovisor -> MoE (dominant):")
        for i, moe_idx in enumerate(dominant):
            strength = A[i, moe_idx]
            print(f"  Chr{i} -> MoE{moe_idx} (strength: {strength:.3f})")
        print()

        print("=" * 90)
        print("INTERPRETATION")
        print("=" * 90)
        print()

        if result.alignment_evolved:
            print("* ALIGNMENT EVOLVED: The matrix departed significantly from identity")
            print("  This indicates Chronovisor-MoE co-specialization is occurring")
        else:
            print("* ALIGNMENT STABLE: The matrix stayed close to identity")
            print("  This may indicate the 1:1 mapping was already optimal")

        if result.alignment_entropy_delta < -0.1:
            print("* ALIGNMENT SHARPENED: Entropy decreased, mapping became more specialized")
        elif result.alignment_entropy_delta > 0.1:
            print("* ALIGNMENT DIFFUSED: Entropy increased, mapping became more distributed")

        if result.loss_delta < -0.01:
            print(f"* LOSS IMPROVED: Learned alignment reduced loss by {-result.loss_percent_change:.1f}%")
        elif result.loss_delta > 0.01:
            print(f"* LOSS WORSENED: Learned alignment increased loss by {result.loss_percent_change:.1f}%")
        else:
            print("* LOSS UNCHANGED: Alignment learning had minimal effect on loss")

        if result.specialization_delta > 0.05:
            print("* SPECIALIZATION INCREASED: Experts became more focused")
        elif result.specialization_delta < -0.05:
            print("* SPECIALIZATION DECREASED: Experts became more generalist")

        if result.learned.drift_correlation > 0.5:
            print("* HIGH DRIFT CORRELATION: Alignment is tracking actual MoE usage patterns")

        print()


def run_v7_experiment(
    n_experts: int = 8,
    n_clusters: int = 4,
    n_batches: int = 100,
    batch_size: int = 32,
    eta_A: float = 0.05,
    alignment_update_frequency: int = 5,
    seed: int = 42,
) -> V7ComparisonResult:
    """
    Convenience function to run a V7 experiment.

    Args:
        n_experts: Number of experts.
        n_clusters: Number of token clusters.
        n_batches: Number of batches.
        batch_size: Tokens per batch.
        eta_A: Alignment learning rate.
        alignment_update_frequency: Update A every N batches.
        seed: Random seed.

    Returns:
        V7ComparisonResult with all metrics.
    """
    experiment = V7Experiment.create(
        n_experts=n_experts,
        n_clusters=n_clusters,
        n_batches=n_batches,
        batch_size=batch_size,
        eta_A=eta_A,
        alignment_update_frequency=alignment_update_frequency,
        seed=seed,
    )

    result = experiment.run_comparison()
    experiment.print_comparison(result)

    return result


if __name__ == "__main__":
    run_v7_experiment()
