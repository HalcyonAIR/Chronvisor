"""
Experiment driver for ChronoMoE.

Runs baseline vs pressure experiments to measure how slow global pressures
reshape MoE routing trees over time.

Key metrics:
- Expert usage distribution (KL divergence before/after)
- Routing entropy
- Transition matrices
- Specialization (cluster-expert correlation)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from chronomoe.moe import MoE, Expert, MoEOutput
from chronomoe.router import Router, RoutingLog, RoutingDecision
from chronomoe.bridge import ChronoMoEBridge, RoutingStats, PressureBias


@dataclass
class SyntheticTask:
    """
    Synthetic task with clustered tokens.

    Each token belongs to a cluster (A, B, C, ...).
    Each expert has an affinity for certain clusters.
    Ground-truth loss = how well the chosen expert matches the token's cluster.
    """

    n_clusters: int
    n_experts: int
    input_dim: int
    output_dim: int

    # Cluster centers in input space
    cluster_centers: np.ndarray = field(repr=False)

    # Expert-cluster affinity matrix (n_experts x n_clusters)
    affinity_matrix: np.ndarray = field(repr=False)

    # Ground truth: best expert for each cluster
    ground_truth_experts: np.ndarray = field(repr=False)

    @classmethod
    def create(
        cls,
        n_clusters: int = 4,
        n_experts: int = 8,
        input_dim: int = 64,
        output_dim: int = 64,
        seed: int = 42,
    ) -> SyntheticTask:
        """
        Create a synthetic task with random cluster-expert affinities.

        Args:
            n_clusters: Number of token clusters.
            n_experts: Number of MoE experts.
            input_dim: Input dimension.
            output_dim: Output dimension.
            seed: Random seed.
        """
        rng = np.random.default_rng(seed)

        # Random cluster centers
        cluster_centers = rng.normal(0, 1, (n_clusters, input_dim))

        # Random affinity matrix (each expert has preferences)
        affinity_matrix = rng.random((n_experts, n_clusters))

        # Normalize so each expert's affinities sum to 1
        affinity_matrix = affinity_matrix / affinity_matrix.sum(axis=1, keepdims=True)

        # Make some experts strongly prefer certain clusters
        # (sparsify the affinity matrix)
        for i in range(n_experts):
            # Pick 1-2 clusters this expert is "good" at
            n_specialties = rng.integers(1, 3)
            top_clusters = np.argsort(affinity_matrix[i])[-n_specialties:]
            mask = np.zeros(n_clusters)
            mask[top_clusters] = 1.0
            affinity_matrix[i] = affinity_matrix[i] * mask
            # Re-normalize
            if affinity_matrix[i].sum() > 0:
                affinity_matrix[i] = affinity_matrix[i] / affinity_matrix[i].sum()
            else:
                affinity_matrix[i] = np.ones(n_clusters) / n_clusters

        # Ground truth: best expert for each cluster
        ground_truth_experts = np.argmax(affinity_matrix, axis=0)

        return cls(
            n_clusters=n_clusters,
            n_experts=n_experts,
            input_dim=input_dim,
            output_dim=output_dim,
            cluster_centers=cluster_centers,
            affinity_matrix=affinity_matrix,
            ground_truth_experts=ground_truth_experts,
        )

    def sample_batch(
        self,
        batch_size: int,
        noise_std: float = 0.3,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample a batch of tokens from clusters.

        Args:
            batch_size: Number of tokens.
            noise_std: Noise added to cluster centers.
            rng: Random generator.

        Returns:
            (inputs, cluster_ids): Token inputs and their cluster assignments.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Sample cluster assignments uniformly
        cluster_ids = rng.integers(0, self.n_clusters, batch_size)

        # Generate inputs from cluster centers with noise
        inputs = np.zeros((batch_size, self.input_dim))
        for i in range(batch_size):
            cluster = cluster_ids[i]
            inputs[i] = self.cluster_centers[cluster] + rng.normal(0, noise_std, self.input_dim)

        return inputs, cluster_ids

    def compute_loss(
        self,
        cluster_ids: np.ndarray,
        chosen_experts: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """
        Compute synthetic loss based on expert-cluster matching.

        Args:
            cluster_ids: Cluster ID for each token.
            chosen_experts: Chosen expert (top-1) for each token.

        Returns:
            (batch_loss, per_token_loss): Overall and per-token loss.
        """
        batch_size = len(cluster_ids)
        per_token_loss = np.zeros(batch_size)

        for i in range(batch_size):
            cluster = cluster_ids[i]
            expert = chosen_experts[i]

            # Loss = 1 - affinity of chosen expert for this cluster
            affinity = self.affinity_matrix[expert, cluster]
            per_token_loss[i] = 1.0 - affinity

        batch_loss = per_token_loss.mean()
        return batch_loss, per_token_loss

    def get_expert_affinities(self) -> dict[int, dict[int, float]]:
        """
        Get affinity dict for MoE expert creation.

        Returns:
            {expert_id: {cluster_id: affinity}}
        """
        affinities = {}
        for i in range(self.n_experts):
            affinities[i] = {}
            for j in range(self.n_clusters):
                affinities[i][j] = float(self.affinity_matrix[i, j])
        return affinities


@dataclass
class ExperimentMetrics:
    """
    Metrics for analyzing routing changes.

    Computes:
    - KL divergence between distributions
    - Entropy of routing
    - Transition matrix analysis
    - Specialization scores
    """

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
        """
        Compute KL(p || q).

        Args:
            p: First distribution.
            q: Second distribution.
            epsilon: Small value to avoid log(0).

        Returns:
            KL divergence.
        """
        p = np.clip(p, epsilon, 1.0)
        q = np.clip(q, epsilon, 1.0)

        # Normalize
        p = p / p.sum()
        q = q / q.sum()

        return float(np.sum(p * np.log(p / q)))

    @staticmethod
    def entropy(p: np.ndarray, epsilon: float = 1e-10) -> float:
        """
        Compute Shannon entropy of distribution.

        Args:
            p: Probability distribution.
            epsilon: Small value to avoid log(0).

        Returns:
            Entropy in nats.
        """
        p = np.clip(p, epsilon, 1.0)
        p = p / p.sum()
        return float(-np.sum(p * np.log(p)))

    @staticmethod
    def max_entropy(n: int) -> float:
        """Maximum entropy for n categories (uniform distribution)."""
        return math.log(n)

    @staticmethod
    def normalized_entropy(p: np.ndarray) -> float:
        """Entropy normalized to [0, 1] range."""
        n = len(p)
        if n <= 1:
            return 0.0
        return ExperimentMetrics.entropy(p) / ExperimentMetrics.max_entropy(n)

    @staticmethod
    def transition_entropy(matrix: np.ndarray) -> float:
        """
        Compute average entropy of transition matrix rows.

        Higher = more random transitions.
        Lower = more deterministic routing patterns.
        """
        if matrix.size == 0:
            return 0.0

        entropies = []
        for row in matrix:
            if row.sum() > 0:
                entropies.append(ExperimentMetrics.entropy(row))

        return float(np.mean(entropies)) if entropies else 0.0

    @staticmethod
    def specialization_score(
        routing_log: RoutingLog,
        cluster_ids: np.ndarray,
        n_clusters: int,
        n_experts: int,
    ) -> tuple[float, np.ndarray]:
        """
        Compute how specialized experts are for clusters.

        Returns:
            (overall_score, per_expert_scores)

        Higher score = experts consistently handle specific clusters.
        """
        # Build cluster-expert co-occurrence matrix
        cooccurrence = np.zeros((n_experts, n_clusters))

        for decision, cluster in zip(routing_log.decisions, cluster_ids):
            top_expert = decision.top_k_experts[-1]  # Top-1
            cooccurrence[top_expert, cluster] += 1

        # Normalize rows (expert -> cluster distribution)
        row_sums = cooccurrence.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        expert_cluster_dist = cooccurrence / row_sums

        # Specialization = 1 - normalized_entropy for each expert
        per_expert_scores = np.zeros(n_experts)
        for i in range(n_experts):
            if row_sums[i, 0] > 0:  # Expert was used
                per_expert_scores[i] = 1.0 - ExperimentMetrics.normalized_entropy(
                    expert_cluster_dist[i]
                )

        overall_score = float(per_expert_scores.mean())
        return overall_score, per_expert_scores


@dataclass
class ExperimentResult:
    """
    Results from a single experiment run (baseline or pressure).
    """

    name: str  # "baseline" or "pressure"
    n_batches: int
    n_tokens: int

    # Distribution metrics
    expert_distribution: np.ndarray
    entropy: float
    normalized_entropy: float

    # Loss metrics
    mean_loss: float
    loss_history: list[float]

    # Transition metrics
    transition_matrix: np.ndarray
    transition_entropy: float

    # Specialization
    specialization_score: float
    per_expert_specialization: np.ndarray

    # Pressure-specific (None for baseline)
    final_R: Optional[float] = None
    avg_pressure_magnitude: Optional[float] = None


@dataclass
class ComparisonResult:
    """
    Comparison between baseline and pressure experiments.
    """

    baseline: ExperimentResult
    pressure: ExperimentResult

    # Distribution shift
    kl_baseline_to_pressure: float
    kl_pressure_to_baseline: float
    symmetric_kl: float

    # Entropy change
    entropy_delta: float  # pressure - baseline
    normalized_entropy_delta: float

    # Loss improvement
    loss_delta: float  # pressure - baseline (negative = improvement)
    loss_percent_change: float

    # Specialization change
    specialization_delta: float

    # Transition change
    transition_entropy_delta: float


@dataclass
class Experiment:
    """
    Main experiment driver.

    Runs baseline vs pressure experiments on synthetic task.
    """

    task: SyntheticTask
    moe: MoE
    router: Router
    bridge: ChronoMoEBridge

    n_batches: int = 50
    batch_size: int = 32
    chronovisor_ticks_per_batch: int = 20

    seed: int = 42

    @classmethod
    def create(
        cls,
        n_experts: int = 8,
        n_clusters: int = 4,
        input_dim: int = 64,
        output_dim: int = 64,
        n_batches: int = 50,
        batch_size: int = 32,
        chronovisor_ticks_per_batch: int = 20,
        alpha_T: float = 0.3,
        alpha_P: float = 0.2,
        alpha_C: float = 0.1,
        seed: int = 42,
    ) -> Experiment:
        """
        Factory method to create experiment with all components.
        """
        # Create synthetic task
        task = SyntheticTask.create(
            n_clusters=n_clusters,
            n_experts=n_experts,
            input_dim=input_dim,
            output_dim=output_dim,
            seed=seed,
        )

        # Create MoE with expert affinities
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

        return cls(
            task=task,
            moe=moe,
            router=router,
            bridge=bridge,
            n_batches=n_batches,
            batch_size=batch_size,
            chronovisor_ticks_per_batch=chronovisor_ticks_per_batch,
            seed=seed,
        )

    def run_baseline(self) -> ExperimentResult:
        """
        Run baseline experiment without pressure.

        Returns:
            ExperimentResult with metrics.
        """
        rng = np.random.default_rng(self.seed)

        # Reset router
        self.router.clear_pressure()
        self.router.reset_log()

        loss_history = []
        all_cluster_ids = []

        for batch_idx in range(self.n_batches):
            # Sample batch
            inputs, cluster_ids = self.task.sample_batch(self.batch_size, rng=rng)
            all_cluster_ids.extend(cluster_ids)

            # Route (no pressure)
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

        # Compute metrics
        distribution = self.router.log.get_expert_distribution()
        transition_matrix = self.router.log.get_transition_matrix()

        specialization, per_expert_spec = ExperimentMetrics.specialization_score(
            self.router.log,
            np.array(all_cluster_ids),
            self.task.n_clusters,
            self.task.n_experts,
        )

        return ExperimentResult(
            name="baseline",
            n_batches=self.n_batches,
            n_tokens=self.n_batches * self.batch_size,
            expert_distribution=distribution,
            entropy=ExperimentMetrics.entropy(distribution),
            normalized_entropy=ExperimentMetrics.normalized_entropy(distribution),
            mean_loss=float(np.mean(loss_history)),
            loss_history=loss_history,
            transition_matrix=transition_matrix,
            transition_entropy=ExperimentMetrics.transition_entropy(transition_matrix),
            specialization_score=specialization,
            per_expert_specialization=per_expert_spec,
        )

    def run_pressure(self) -> ExperimentResult:
        """
        Run experiment with Chronovisor pressure.

        Returns:
            ExperimentResult with metrics.
        """
        rng = np.random.default_rng(self.seed)

        # Reset bridge and router
        self.bridge.reset(self.seed)
        self.router.reset_log()

        loss_history = []
        all_cluster_ids = []
        pressure_magnitudes = []

        for batch_idx in range(self.n_batches):
            # Sample batch
            inputs, cluster_ids = self.task.sample_batch(self.batch_size, rng=rng)
            all_cluster_ids.extend(cluster_ids)

            # Get pressure bias from Chronovisor
            pressure_bias = self.bridge.get_pressure_bias()
            self.router.inject_pressure(pressure_bias.combined)
            pressure_magnitudes.append(np.abs(pressure_bias.combined).mean())

            # Route (with pressure)
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

            # Feed routing stats to Chronovisor
            stats = RoutingStats(
                expert_usage=self.router.log.get_expert_usage(),
                mean_gate_weights=self.router.log.get_mean_gate_weights(),
                batch_loss=batch_loss,
            )
            self.bridge.feed_routing_stats(stats, self.chronovisor_ticks_per_batch)

        # Compute metrics
        distribution = self.router.log.get_expert_distribution()
        transition_matrix = self.router.log.get_transition_matrix()

        specialization, per_expert_spec = ExperimentMetrics.specialization_score(
            self.router.log,
            np.array(all_cluster_ids),
            self.task.n_clusters,
            self.task.n_experts,
        )

        final_R = self.bridge.R_history[-1] if self.bridge.R_history else 0.0

        return ExperimentResult(
            name="pressure",
            n_batches=self.n_batches,
            n_tokens=self.n_batches * self.batch_size,
            expert_distribution=distribution,
            entropy=ExperimentMetrics.entropy(distribution),
            normalized_entropy=ExperimentMetrics.normalized_entropy(distribution),
            mean_loss=float(np.mean(loss_history)),
            loss_history=loss_history,
            transition_matrix=transition_matrix,
            transition_entropy=ExperimentMetrics.transition_entropy(transition_matrix),
            specialization_score=specialization,
            per_expert_specialization=per_expert_spec,
            final_R=final_R,
            avg_pressure_magnitude=float(np.mean(pressure_magnitudes)),
        )

    def run_comparison(self) -> ComparisonResult:
        """
        Run both experiments and compare.

        Returns:
            ComparisonResult with all metrics.
        """
        print("Running baseline experiment...")
        baseline = self.run_baseline()

        print("Running pressure experiment...")
        pressure = self.run_pressure()

        # Compute comparison metrics
        kl_bp = ExperimentMetrics.kl_divergence(
            baseline.expert_distribution,
            pressure.expert_distribution,
        )
        kl_pb = ExperimentMetrics.kl_divergence(
            pressure.expert_distribution,
            baseline.expert_distribution,
        )

        loss_percent = 0.0
        if baseline.mean_loss > 0:
            loss_percent = (pressure.mean_loss - baseline.mean_loss) / baseline.mean_loss * 100

        return ComparisonResult(
            baseline=baseline,
            pressure=pressure,
            kl_baseline_to_pressure=kl_bp,
            kl_pressure_to_baseline=kl_pb,
            symmetric_kl=(kl_bp + kl_pb) / 2,
            entropy_delta=pressure.entropy - baseline.entropy,
            normalized_entropy_delta=pressure.normalized_entropy - baseline.normalized_entropy,
            loss_delta=pressure.mean_loss - baseline.mean_loss,
            loss_percent_change=loss_percent,
            specialization_delta=pressure.specialization_score - baseline.specialization_score,
            transition_entropy_delta=pressure.transition_entropy - baseline.transition_entropy,
        )

    def print_comparison(self, result: ComparisonResult) -> None:
        """
        Print formatted comparison results.
        """
        print()
        print("=" * 80)
        print("CHRONOMOE V1: BASELINE VS PRESSURE COMPARISON")
        print("=" * 80)
        print()

        print(f"Configuration:")
        print(f"  Experts: {self.task.n_experts}")
        print(f"  Clusters: {self.task.n_clusters}")
        print(f"  Batches: {self.n_batches}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Chronovisor ticks/batch: {self.chronovisor_ticks_per_batch}")
        print(f"  Pressure weights: T={self.bridge.alpha_T}, P={self.bridge.alpha_P}, C={self.bridge.alpha_C}")
        print()

        print("-" * 80)
        print("EXPERT DISTRIBUTION")
        print("-" * 80)
        print()
        print(f"{'Expert':<10} {'Baseline':>12} {'Pressure':>12} {'Delta':>12}")
        print("-" * 46)
        for i in range(self.task.n_experts):
            b = result.baseline.expert_distribution[i]
            p = result.pressure.expert_distribution[i]
            delta = p - b
            sign = "+" if delta >= 0 else ""
            print(f"E{i:<9} {b:>12.4f} {p:>12.4f} {sign}{delta:>11.4f}")
        print()

        print("-" * 80)
        print("DISTRIBUTION METRICS")
        print("-" * 80)
        print()
        print(f"KL(baseline || pressure): {result.kl_baseline_to_pressure:.4f}")
        print(f"KL(pressure || baseline): {result.kl_pressure_to_baseline:.4f}")
        print(f"Symmetric KL:             {result.symmetric_kl:.4f}")
        print()
        print(f"Entropy (baseline):       {result.baseline.entropy:.4f}")
        print(f"Entropy (pressure):       {result.pressure.entropy:.4f}")
        print(f"Entropy delta:            {result.entropy_delta:+.4f}")
        print()

        print("-" * 80)
        print("LOSS METRICS")
        print("-" * 80)
        print()
        print(f"Mean loss (baseline):     {result.baseline.mean_loss:.4f}")
        print(f"Mean loss (pressure):     {result.pressure.mean_loss:.4f}")
        print(f"Loss delta:               {result.loss_delta:+.4f}")
        print(f"Loss change:              {result.loss_percent_change:+.1f}%")
        print()

        print("-" * 80)
        print("SPECIALIZATION METRICS")
        print("-" * 80)
        print()
        print(f"Specialization (baseline): {result.baseline.specialization_score:.4f}")
        print(f"Specialization (pressure): {result.pressure.specialization_score:.4f}")
        print(f"Specialization delta:      {result.specialization_delta:+.4f}")
        print()

        print("-" * 80)
        print("TRANSITION METRICS")
        print("-" * 80)
        print()
        print(f"Transition entropy (baseline): {result.baseline.transition_entropy:.4f}")
        print(f"Transition entropy (pressure): {result.pressure.transition_entropy:.4f}")
        print(f"Transition entropy delta:      {result.transition_entropy_delta:+.4f}")
        print()

        if result.pressure.final_R is not None:
            print("-" * 80)
            print("CHRONOVISOR STATE")
            print("-" * 80)
            print()
            print(f"Final coherence R:         {result.pressure.final_R:.4f}")
            print(f"Avg pressure magnitude:    {result.pressure.avg_pressure_magnitude:.4f}")
            print()

        print("=" * 80)
        print("INTERPRETATION")
        print("=" * 80)
        print()

        # Interpret results
        if result.symmetric_kl > 0.01:
            print("* Distribution shift detected: pressure changed routing patterns")
        else:
            print("* Minimal distribution shift: pressure had little effect on routing")

        if result.loss_delta < -0.01:
            print(f"* Loss IMPROVED by {-result.loss_percent_change:.1f}%: pressure helped")
        elif result.loss_delta > 0.01:
            print(f"* Loss WORSENED by {result.loss_percent_change:.1f}%: pressure hurt")
        else:
            print("* Loss unchanged: pressure was neutral")

        if result.specialization_delta > 0.01:
            print("* Specialization increased: experts became more focused")
        elif result.specialization_delta < -0.01:
            print("* Specialization decreased: experts became more generalist")
        else:
            print("* Specialization unchanged")

        if result.entropy_delta < -0.05:
            print("* Entropy decreased: routing became more concentrated")
        elif result.entropy_delta > 0.05:
            print("* Entropy increased: routing became more distributed")

        print()


def run_experiment(
    n_experts: int = 8,
    n_clusters: int = 4,
    n_batches: int = 50,
    batch_size: int = 32,
    alpha_T: float = 0.3,
    alpha_P: float = 0.2,
    alpha_C: float = 0.1,
    seed: int = 42,
) -> ComparisonResult:
    """
    Convenience function to run a complete experiment.

    Args:
        n_experts: Number of MoE experts.
        n_clusters: Number of token clusters.
        n_batches: Number of batches to process.
        batch_size: Tokens per batch.
        alpha_T: Trust pressure weight.
        alpha_P: Lens pressure weight.
        alpha_C: Cultural pressure weight.
        seed: Random seed.

    Returns:
        ComparisonResult with all metrics.
    """
    experiment = Experiment.create(
        n_experts=n_experts,
        n_clusters=n_clusters,
        n_batches=n_batches,
        batch_size=batch_size,
        alpha_T=alpha_T,
        alpha_P=alpha_P,
        alpha_C=alpha_C,
        seed=seed,
    )

    result = experiment.run_comparison()
    experiment.print_comparison(result)

    return result


if __name__ == "__main__":
    run_experiment()
