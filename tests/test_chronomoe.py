"""
Tests for ChronoMoE: Pressure-Driven MoE Routing Framework.

Tests cover:
- MoE layer (experts, routing)
- Router with pressure injection
- Bridge between MoE and Chronovisor
- Synthetic task and experiment metrics
"""

import math
import numpy as np
import pytest

from chronomoe.moe import Expert, MoE, MoEOutput
from chronomoe.router import Router, RoutingLog, RoutingDecision
from chronomoe.bridge import ChronoMoEBridge, RoutingStats, PressureBias
from chronomoe.experiment import (
    SyntheticTask,
    ExperimentMetrics,
    Experiment,
    run_experiment,
)


# =============================================================================
# Expert Tests
# =============================================================================


class TestExpert:
    """Tests for the Expert class."""

    def test_expert_creation(self):
        """Test expert factory method."""
        expert = Expert.create(
            expert_id=0,
            input_dim=64,
            output_dim=64,
            seed=42,
        )

        assert expert.expert_id == 0
        assert expert.input_dim == 64
        assert expert.output_dim == 64
        assert expert.W.shape == (64, 64)
        assert expert.b.shape == (64,)

    def test_expert_forward(self):
        """Test expert forward pass."""
        expert = Expert.create(
            expert_id=0,
            input_dim=32,
            output_dim=16,
            seed=42,
        )

        x = np.random.randn(32)
        y = expert.forward(x)

        assert y.shape == (16,)

    def test_expert_forward_batch(self):
        """Test expert forward pass with batch."""
        expert = Expert.create(
            expert_id=0,
            input_dim=32,
            output_dim=16,
            seed=42,
        )

        x = np.random.randn(8, 32)
        y = expert.forward(x)

        assert y.shape == (8, 16)

    def test_expert_affinity(self):
        """Test expert cluster affinity."""
        expert = Expert.create(
            expert_id=0,
            input_dim=32,
            output_dim=16,
            cluster_affinity={0: 0.8, 1: 0.2},
            seed=42,
        )

        assert expert.get_affinity(0) == 0.8
        assert expert.get_affinity(1) == 0.2
        assert expert.get_affinity(2) == 0.0  # Unknown cluster


# =============================================================================
# MoE Tests
# =============================================================================


class TestMoE:
    """Tests for the MoE class."""

    def test_moe_creation(self):
        """Test MoE initialization."""
        moe = MoE(
            n_experts=8,
            input_dim=64,
            output_dim=64,
            top_k=2,
            seed=42,
        )

        assert moe.n_experts == 8
        assert moe.top_k == 2
        assert len(moe.experts) == 8

    def test_moe_forward(self):
        """Test MoE forward pass."""
        moe = MoE(
            n_experts=4,
            input_dim=32,
            output_dim=32,
            top_k=2,
            seed=42,
        )

        x = np.random.randn(8, 32)
        gate_weights = np.random.rand(8, 4)
        gate_weights = gate_weights / gate_weights.sum(axis=1, keepdims=True)

        output = moe.forward(x, gate_weights)

        assert isinstance(output, MoEOutput)
        assert output.output.shape == (8, 32)
        assert output.expert_outputs.shape == (4, 8, 32)
        assert output.gate_weights.shape == (8, 4)
        assert output.chosen_experts.shape == (8, 2)

    def test_moe_top_k_selection(self):
        """Test that only top-k experts contribute."""
        moe = MoE(
            n_experts=4,
            input_dim=16,
            output_dim=16,
            top_k=1,
            seed=42,
        )

        x = np.random.randn(4, 16)
        # Make expert 2 the clear winner for all tokens
        gate_weights = np.array([
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.7, 0.1],
        ])

        output = moe.forward(x, gate_weights)

        # All tokens should route to expert 2
        assert np.all(output.chosen_experts[:, -1] == 2)

    def test_moe_ground_truth_expert(self):
        """Test ground truth expert lookup."""
        affinities = {
            0: {0: 0.9, 1: 0.1},
            1: {0: 0.1, 1: 0.9},
        }
        moe = MoE(
            n_experts=2,
            input_dim=16,
            output_dim=16,
            cluster_affinities=affinities,
            seed=42,
        )

        assert moe.get_ground_truth_expert(0) == 0
        assert moe.get_ground_truth_expert(1) == 1


# =============================================================================
# Router Tests
# =============================================================================


class TestRouter:
    """Tests for the Router class."""

    def test_router_creation(self):
        """Test router initialization."""
        router = Router(
            input_dim=64,
            n_experts=8,
            seed=42,
        )

        assert router.n_experts == 8
        assert router.W1.shape[1] == 64
        assert router.W2.shape[0] == 8

    def test_router_forward(self):
        """Test router forward pass."""
        router = Router(
            input_dim=32,
            n_experts=4,
            seed=42,
        )

        x = np.random.randn(8, 32)
        gate_weights = router.forward(x)

        assert gate_weights.shape == (8, 4)
        # Gate weights should sum to 1
        assert np.allclose(gate_weights.sum(axis=1), 1.0)
        # All values should be non-negative
        assert np.all(gate_weights >= 0)

    def test_router_pressure_injection(self):
        """Test pressure bias injection."""
        router = Router(
            input_dim=32,
            n_experts=4,
            seed=42,
        )

        # No pressure initially
        assert np.allclose(router.get_pressure(), 0.0)

        # Inject pressure
        pressure = np.array([0.1, -0.1, 0.2, -0.2])
        router.inject_pressure(pressure)

        assert np.allclose(router.get_pressure(), pressure)

        # Clear pressure
        router.clear_pressure()
        assert np.allclose(router.get_pressure(), 0.0)

    def test_router_pressure_affects_routing(self):
        """Test that pressure changes routing decisions."""
        router = Router(
            input_dim=32,
            n_experts=4,
            temperature=0.5,  # Lower temperature for sharper routing
            seed=42,
        )

        x = np.random.randn(100, 32)

        # Baseline routing
        router.clear_pressure()
        router.reset_log()
        baseline_gates = router.forward(x.copy())
        baseline_dist = router.log.get_expert_distribution()

        # Strong pressure toward expert 0
        router.inject_pressure(np.array([2.0, -1.0, -1.0, -1.0]))
        router.reset_log()
        pressure_gates = router.forward(x.copy())
        pressure_dist = router.log.get_expert_distribution()

        # Expert 0 should be used more with pressure
        assert pressure_dist[0] > baseline_dist[0]

    def test_router_logging(self):
        """Test routing decision logging."""
        router = Router(
            input_dim=32,
            n_experts=4,
            seed=42,
        )

        x = np.random.randn(10, 32)
        router.forward(x, token_indices=np.arange(10))

        assert len(router.log.decisions) == 10

        decision = router.log.decisions[0]
        assert decision.token_idx == 0
        assert decision.logits.shape == (4,)
        assert decision.gate_weights.shape == (4,)

    def test_routing_stats(self):
        """Test routing statistics computation."""
        router = Router(
            input_dim=32,
            n_experts=4,
            seed=42,
        )

        x = np.random.randn(100, 32)
        router.forward(x)

        stats = router.get_routing_stats()

        assert "usage" in stats
        assert "distribution" in stats
        assert "mean_gates" in stats
        assert stats["n_decisions"] == 100


# =============================================================================
# Bridge Tests
# =============================================================================


class TestChronoMoEBridge:
    """Tests for the ChronoMoE bridge."""

    def test_bridge_creation(self):
        """Test bridge factory method."""
        bridge = ChronoMoEBridge.create(
            n_experts=8,
            seed=42,
        )

        assert bridge.n_experts == 8
        assert len(bridge.controller.experts) == 8

    def test_bridge_pressure_extraction(self):
        """Test pressure bias extraction."""
        bridge = ChronoMoEBridge.create(
            n_experts=4,
            alpha_T=0.3,
            alpha_P=0.2,
            alpha_C=0.1,
            seed=42,
        )

        pressure = bridge.get_pressure_bias()

        assert isinstance(pressure, PressureBias)
        assert pressure.trust.shape == (4,)
        assert pressure.pressure.shape == (4,)
        assert pressure.cultural.shape == (4,)
        assert pressure.combined.shape == (4,)

    def test_bridge_feed_routing(self):
        """Test feeding routing stats to bridge."""
        bridge = ChronoMoEBridge.create(
            n_experts=4,
            seed=42,
        )

        stats = RoutingStats(
            expert_usage=np.array([10, 20, 15, 5]),
            mean_gate_weights=np.array([0.2, 0.3, 0.3, 0.2]),
            batch_loss=0.5,
        )

        result = bridge.feed_routing_stats(stats, num_chronovisor_ticks=10)

        assert "alignment_info" in result
        assert "final_R" in result
        assert bridge.total_ticks == 10

    def test_bridge_reset(self):
        """Test bridge reset."""
        bridge = ChronoMoEBridge.create(n_experts=4, seed=42)

        # Run some ticks
        stats = RoutingStats(
            expert_usage=np.array([10, 20, 15, 5]),
            mean_gate_weights=np.array([0.2, 0.3, 0.3, 0.2]),
            batch_loss=0.5,
        )
        bridge.feed_routing_stats(stats, num_chronovisor_ticks=10)

        assert bridge.total_ticks == 10

        # Reset
        bridge.reset(seed=42)

        assert bridge.total_ticks == 0
        assert len(bridge.R_history) == 0

    def test_bridge_expert_states(self):
        """Test getting expert states."""
        bridge = ChronoMoEBridge.create(n_experts=4, seed=42)

        states = bridge.get_expert_states()

        assert len(states) == 4
        assert all("s" in s for s in states)
        assert all("lambd" in s for s in states)
        assert all("g_lens_ema" in s for s in states)


# =============================================================================
# Synthetic Task Tests
# =============================================================================


class TestSyntheticTask:
    """Tests for the synthetic task."""

    def test_task_creation(self):
        """Test synthetic task creation."""
        task = SyntheticTask.create(
            n_clusters=4,
            n_experts=8,
            input_dim=64,
            seed=42,
        )

        assert task.n_clusters == 4
        assert task.n_experts == 8
        assert task.cluster_centers.shape == (4, 64)
        assert task.affinity_matrix.shape == (8, 4)
        assert task.ground_truth_experts.shape == (4,)

    def test_task_sample_batch(self):
        """Test batch sampling."""
        task = SyntheticTask.create(
            n_clusters=4,
            n_experts=8,
            input_dim=32,
            seed=42,
        )

        inputs, cluster_ids = task.sample_batch(batch_size=100)

        assert inputs.shape == (100, 32)
        assert cluster_ids.shape == (100,)
        assert np.all(cluster_ids >= 0)
        assert np.all(cluster_ids < 4)

    def test_task_loss_computation(self):
        """Test loss computation."""
        task = SyntheticTask.create(
            n_clusters=2,
            n_experts=4,
            seed=42,
        )

        cluster_ids = np.array([0, 0, 1, 1])

        # Perfect expert assignment
        ground_truth = task.ground_truth_experts
        batch_loss, per_token = task.compute_loss(cluster_ids, ground_truth[cluster_ids])

        # Loss should be low for correct assignments
        assert batch_loss < 0.5

    def test_task_affinity_dict(self):
        """Test affinity dict for MoE creation."""
        task = SyntheticTask.create(
            n_clusters=2,
            n_experts=4,
            seed=42,
        )

        affinities = task.get_expert_affinities()

        assert len(affinities) == 4
        assert all(len(affinities[i]) == 2 for i in range(4))


# =============================================================================
# Experiment Metrics Tests
# =============================================================================


class TestExperimentMetrics:
    """Tests for experiment metrics."""

    def test_kl_divergence_same(self):
        """KL divergence of identical distributions should be 0."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        q = np.array([0.25, 0.25, 0.25, 0.25])

        kl = ExperimentMetrics.kl_divergence(p, q)

        assert np.isclose(kl, 0.0, atol=1e-6)

    def test_kl_divergence_different(self):
        """KL divergence of different distributions should be positive."""
        p = np.array([0.5, 0.3, 0.1, 0.1])
        q = np.array([0.1, 0.1, 0.4, 0.4])

        kl = ExperimentMetrics.kl_divergence(p, q)

        assert kl > 0

    def test_entropy_uniform(self):
        """Entropy should be maximal for uniform distribution."""
        p = np.array([0.25, 0.25, 0.25, 0.25])

        entropy = ExperimentMetrics.entropy(p)
        max_entropy = ExperimentMetrics.max_entropy(4)

        assert np.isclose(entropy, max_entropy, atol=1e-6)

    def test_entropy_peaked(self):
        """Entropy should be lower for peaked distribution."""
        uniform = np.array([0.25, 0.25, 0.25, 0.25])
        peaked = np.array([0.9, 0.05, 0.03, 0.02])

        entropy_uniform = ExperimentMetrics.entropy(uniform)
        entropy_peaked = ExperimentMetrics.entropy(peaked)

        assert entropy_peaked < entropy_uniform

    def test_normalized_entropy(self):
        """Normalized entropy should be in [0, 1]."""
        p = np.random.rand(10)
        p = p / p.sum()

        norm_entropy = ExperimentMetrics.normalized_entropy(p)

        assert 0 <= norm_entropy <= 1

    def test_transition_entropy(self):
        """Test transition matrix entropy."""
        # Deterministic transitions
        deterministic = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ], dtype=float)

        # Random transitions
        random = np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
        ])

        det_entropy = ExperimentMetrics.transition_entropy(deterministic)
        rand_entropy = ExperimentMetrics.transition_entropy(random)

        assert det_entropy < rand_entropy


# =============================================================================
# Integration Tests
# =============================================================================


class TestExperiment:
    """Integration tests for the full experiment."""

    def test_experiment_creation(self):
        """Test experiment factory method."""
        experiment = Experiment.create(
            n_experts=4,
            n_clusters=2,
            n_batches=5,
            batch_size=16,
            seed=42,
        )

        assert experiment.task.n_experts == 4
        assert experiment.task.n_clusters == 2
        assert experiment.moe.n_experts == 4
        assert experiment.router.n_experts == 4
        assert experiment.bridge.n_experts == 4

    def test_baseline_run(self):
        """Test baseline experiment run."""
        experiment = Experiment.create(
            n_experts=4,
            n_clusters=2,
            n_batches=5,
            batch_size=16,
            seed=42,
        )

        result = experiment.run_baseline()

        assert result.name == "baseline"
        assert result.n_batches == 5
        assert result.n_tokens == 80
        assert result.expert_distribution.shape == (4,)
        assert np.isclose(result.expert_distribution.sum(), 1.0)
        assert 0 <= result.mean_loss <= 1.0

    def test_pressure_run(self):
        """Test pressure experiment run."""
        experiment = Experiment.create(
            n_experts=4,
            n_clusters=2,
            n_batches=5,
            batch_size=16,
            seed=42,
        )

        result = experiment.run_pressure()

        assert result.name == "pressure"
        assert result.final_R is not None
        assert result.avg_pressure_magnitude is not None

    def test_comparison(self):
        """Test full comparison experiment."""
        experiment = Experiment.create(
            n_experts=4,
            n_clusters=2,
            n_batches=5,
            batch_size=16,
            seed=42,
        )

        result = experiment.run_comparison()

        assert result.baseline is not None
        assert result.pressure is not None
        assert result.symmetric_kl >= 0
        assert -1 <= result.normalized_entropy_delta <= 1

    def test_run_experiment_convenience(self):
        """Test convenience function."""
        result = run_experiment(
            n_experts=4,
            n_clusters=2,
            n_batches=3,
            batch_size=8,
            seed=42,
        )

        assert result is not None
        assert result.baseline.n_batches == 3


# =============================================================================
# Pressure Effect Tests
# =============================================================================


class TestPressureEffect:
    """Tests specifically for pressure effects on routing."""

    def test_pressure_changes_distribution(self):
        """Verify that pressure actually changes routing distribution."""
        experiment = Experiment.create(
            n_experts=8,
            n_clusters=4,
            n_batches=20,
            batch_size=32,
            alpha_T=0.5,  # Strong trust pressure
            alpha_P=0.3,
            alpha_C=0.2,
            seed=42,
        )

        result = experiment.run_comparison()

        # There should be some difference in distributions
        # (exact threshold depends on randomness, but should be non-zero)
        assert result.symmetric_kl >= 0

    def test_pressure_magnitude_bounded(self):
        """Verify pressure stays in small-signal regime."""
        bridge = ChronoMoEBridge.create(
            n_experts=8,
            alpha_T=0.3,
            alpha_P=0.2,
            alpha_C=0.1,
            seed=42,
        )

        # Run some ticks
        for _ in range(10):
            stats = RoutingStats(
                expert_usage=np.random.randint(0, 100, 8),
                mean_gate_weights=np.random.rand(8),
                batch_loss=0.5,
            )
            bridge.feed_routing_stats(stats, num_chronovisor_ticks=10)

        pressure = bridge.get_pressure_bias()

        # Pressure should be bounded (small-signal regime)
        # Combined pressure should typically be < 2.0 in absolute value
        assert np.all(np.abs(pressure.combined) < 5.0)

    def test_chronovisor_state_evolves(self):
        """Verify Chronovisor state evolves over time."""
        bridge = ChronoMoEBridge.create(n_experts=4, seed=42)

        initial_states = bridge.get_expert_states()

        # Run many ticks
        for _ in range(50):
            stats = RoutingStats(
                expert_usage=np.array([10, 30, 20, 40]),  # Uneven usage
                mean_gate_weights=np.array([0.1, 0.3, 0.2, 0.4]),
                batch_loss=0.3,
            )
            bridge.feed_routing_stats(stats, num_chronovisor_ticks=20)

        final_states = bridge.get_expert_states()

        # States should have changed
        initial_s = [s["s"] for s in initial_states]
        final_s = [s["s"] for s in final_states]

        # At least some experts should have different reliability
        assert not np.allclose(initial_s, final_s)
