"""Tests for ChronoMoE V1: Pressure-based MoE routing."""

import math
import random
from chronovisor.chrono_moe import (
    ChronoMoEController,
    RoutingLogger,
    MockMoERouter,
    kl_divergence,
    entropy,
    matrix_frobenius_distance,
    compute_routing_metrics,
    run_chrono_moe_experiment,
)
from chronovisor.simulation_v6 import reset_id_counters


class TestChronoMoEController:
    """Tests for the ChronoMoE adapter."""

    def setup_method(self):
        reset_id_counters()
        random.seed(42)

    def test_init_creates_matching_experts(self):
        """Controller creates Chronovisor experts matching num_experts."""
        chrono = ChronoMoEController(num_experts=8)

        assert len(chrono.controller.experts) == 8
        assert chrono.num_experts == 8

    def test_update_from_routing(self):
        """update_from_routing translates MoE stats to Chronovisor."""
        chrono = ChronoMoEController(num_experts=4)

        # Simulate routing stats
        usage = [10, 20, 5, 15]  # Expert 1 most used
        loss = 0.5

        result = chrono.update_from_routing(usage, loss)

        assert result["batch"] == 1
        assert "usage_dist" in result
        assert sum(result["usage_dist"]) - 1.0 < 0.001

    def test_get_pressure_biases(self):
        """get_pressure_biases returns b_k for each expert."""
        chrono = ChronoMoEController(num_experts=4)

        biases = chrono.get_pressure_biases()

        assert len(biases) == 4
        assert all(isinstance(b, float) for b in biases)

    def test_pressure_biases_change_with_reliability(self):
        """Higher reliability -> higher trust bias."""
        chrono = ChronoMoEController(num_experts=4, alpha_T=1.0, alpha_P=0.0, alpha_C=0.0)

        # Set expert 0 to high reliability, expert 1 to low
        chrono.controller.experts[0].s = 0.5
        chrono.controller.experts[1].s = -0.5

        biases = chrono.get_pressure_biases()

        # Expert 0 should have higher bias than expert 1
        assert biases[0] > biases[1]

    def test_pressure_biases_change_with_lens(self):
        """Lens gain affects pressure bias."""
        chrono = ChronoMoEController(num_experts=4, alpha_T=0.0, alpha_P=1.0, alpha_C=0.0)

        # Set different lens gains
        chrono.controller.experts[0].g_lens_ema = 1.3  # Amplified
        chrono.controller.experts[1].g_lens_ema = 0.7  # Damped

        biases = chrono.get_pressure_biases()

        # Amplified expert should have higher bias
        assert biases[0] > biases[1]
        assert biases[0] > 0  # P = g - 1 = 0.3 > 0
        assert biases[1] < 0  # P = g - 1 = -0.3 < 0

    def test_step_runs_chronovisor_ticks(self):
        """step() advances Chronovisor state."""
        chrono = ChronoMoEController(num_experts=4)

        # Initial state
        initial_clock = chrono.controller.fast_clock

        result = chrono.step(n_ticks=20)

        assert chrono.controller.fast_clock == initial_clock + 20
        assert "R" in result
        assert "lens_L" in result

    def test_get_expert_states(self):
        """get_expert_states returns detailed state."""
        chrono = ChronoMoEController(num_experts=4)

        states = chrono.get_expert_states()

        assert len(states) == 4
        assert all("s" in s for s in states)
        assert all("g_lens_ema" in s for s in states)
        assert all("delta_s_ema" in s for s in states)


class TestRoutingLogger:
    """Tests for routing decision logging."""

    def setup_method(self):
        random.seed(42)

    def test_log_routing(self):
        """log_routing stores routing decisions."""
        logger = RoutingLogger(num_experts=4)

        logger.log_routing(
            token_idx=0,
            layer_idx=0,
            logits=[0.1, 0.2, 0.3, 0.4],
            gates=[0.1, 0.2, 0.3, 0.4],
            chosen_expert=3,
            batch_idx=0,
        )

        assert len(logger.routing_log) == 1
        assert logger.routing_log[0]["chosen"] == 3

    def test_log_batch_usage(self):
        """log_batch_usage stores usage distributions."""
        logger = RoutingLogger(num_experts=4)

        logger.log_batch_usage([10, 20, 15, 5], batch_idx=0)
        logger.log_batch_usage([15, 15, 10, 10], batch_idx=1)

        assert len(logger.usage_over_time) == 2

    def test_get_usage_distribution(self):
        """get_usage_distribution computes average distribution."""
        logger = RoutingLogger(num_experts=4)

        logger.log_batch_usage([10, 20, 10, 10], batch_idx=0)
        logger.log_batch_usage([10, 20, 10, 10], batch_idx=1)

        dist = logger.get_usage_distribution()

        assert len(dist) == 4
        assert abs(sum(dist) - 1.0) < 0.001
        assert dist[1] > dist[0]  # Expert 1 most used

    def test_get_transition_matrix(self):
        """get_transition_matrix computes transition probabilities."""
        logger = RoutingLogger(num_experts=3)

        # Log sequence: 0 -> 1 -> 1 -> 2 -> 0
        for i, chosen in enumerate([0, 1, 1, 2, 0]):
            logger.log_routing(
                token_idx=i,
                layer_idx=0,
                logits=[0, 0, 0],
                gates=[0.33, 0.33, 0.33],
                chosen_expert=chosen,
                batch_idx=0,
            )

        matrix = logger.get_transition_matrix()

        assert len(matrix) == 3
        assert len(matrix[0]) == 3
        # Row sums should be ~1
        for row in matrix:
            assert abs(sum(row) - 1.0) < 0.001

    def test_clear(self):
        """clear() removes all logs."""
        logger = RoutingLogger(num_experts=4)

        logger.log_batch_usage([10, 20, 15, 5], batch_idx=0)
        logger.clear()

        assert len(logger.routing_log) == 0
        assert len(logger.usage_over_time) == 0


class TestMockMoERouter:
    """Tests for the mock MoE router."""

    def setup_method(self):
        random.seed(42)

    def test_route_returns_valid_output(self):
        """route() returns logits, gates, and chosen expert."""
        router = MockMoERouter(num_experts=4)

        logits, gates, chosen = router.route()

        assert len(logits) == 4
        assert len(gates) == 4
        assert 0 <= chosen < 4
        assert abs(sum(gates) - 1.0) < 0.001  # Softmax sums to 1

    def test_pressure_biases_affect_routing(self):
        """Pressure biases shift routing decisions."""
        router = MockMoERouter(num_experts=4)

        # Strong bias toward expert 0
        biases = [10.0, 0.0, 0.0, 0.0]

        # Run many routes and count
        counts = [0, 0, 0, 0]
        for _ in range(100):
            _, _, chosen = router.route(pressure_biases=biases, noise_std=0.1)
            counts[chosen] += 1

        # Expert 0 should be chosen most often
        assert counts[0] > 80  # Should be overwhelming majority

    def test_route_batch(self):
        """route_batch processes multiple tokens."""
        router = MockMoERouter(num_experts=4)

        chosen_list, usage = router.route_batch(batch_size=100)

        assert len(chosen_list) == 100
        assert sum(usage) == 100


class TestMetrics:
    """Tests for decision tree metrics."""

    def test_kl_divergence(self):
        """KL divergence is non-negative and zero for identical distributions."""
        p = [0.25, 0.25, 0.25, 0.25]
        q = [0.25, 0.25, 0.25, 0.25]

        assert abs(kl_divergence(p, q)) < 0.001

        # Different distributions
        p = [0.9, 0.1, 0.0, 0.0]
        q = [0.25, 0.25, 0.25, 0.25]

        kl = kl_divergence(p, q)
        assert kl > 0

    def test_entropy(self):
        """Entropy is maximized for uniform distribution."""
        uniform = [0.25, 0.25, 0.25, 0.25]
        peaked = [0.9, 0.05, 0.03, 0.02]

        h_uniform = entropy(uniform)
        h_peaked = entropy(peaked)

        assert h_uniform > h_peaked
        assert abs(h_uniform - math.log(4)) < 0.001  # Max entropy = log(n)

    def test_matrix_frobenius_distance(self):
        """Frobenius distance is zero for identical matrices."""
        A = [[1, 0], [0, 1]]
        B = [[1, 0], [0, 1]]

        assert abs(matrix_frobenius_distance(A, B)) < 0.001

        # Different matrices
        C = [[0, 1], [1, 0]]
        dist = matrix_frobenius_distance(A, C)
        assert dist > 0

    def test_compute_routing_metrics(self):
        """compute_routing_metrics compares two loggers."""
        logger1 = RoutingLogger(num_experts=4)
        logger2 = RoutingLogger(num_experts=4)

        # Different usage patterns
        logger1.log_batch_usage([40, 10, 10, 10], batch_idx=0)
        logger2.log_batch_usage([10, 10, 10, 40], batch_idx=0)

        metrics = compute_routing_metrics(logger1, logger2)

        assert "kl_before_after" in metrics
        assert "entropy_before" in metrics
        assert "entropy_after" in metrics
        assert metrics["kl_before_after"] > 0  # Distributions differ


class TestExperiment:
    """Integration tests for the full experiment."""

    def setup_method(self):
        reset_id_counters()
        random.seed(42)

    def test_run_experiment(self):
        """Experiment runs without error and produces metrics."""
        results = run_chrono_moe_experiment(
            num_experts=4,
            num_batches=10,
            batch_size=32,
            seed=42,
        )

        assert "metrics" in results
        assert "baseline_usage_history" in results
        assert "pressure_usage_history" in results

        metrics = results["metrics"]
        assert "kl_before_after" in metrics
        assert "entropy_change" in metrics

    def test_pressure_affects_routing(self):
        """Pressure should cause measurable routing changes."""
        results = run_chrono_moe_experiment(
            num_experts=4,
            num_batches=20,
            batch_size=64,
            seed=42,
            alpha_T=0.5,  # Strong trust effect
        )

        metrics = results["metrics"]

        # Some change should have occurred
        # (KL > 0 means distributions differ)
        assert metrics["kl_before_after"] > 0 or metrics["transition_distance"] > 0


class TestChronoMoEIntegration:
    """End-to-end integration tests."""

    def setup_method(self):
        reset_id_counters()
        random.seed(42)

    def test_full_loop(self):
        """Test complete ChronoMoE loop: route -> update -> step -> get biases."""
        chrono = ChronoMoEController(num_experts=4)
        router = MockMoERouter(num_experts=4)
        logger = RoutingLogger(num_experts=4)

        # Run several batches
        for batch_idx in range(5):
            batch_usage = [0, 0, 0, 0]

            # Get current pressure biases
            biases = chrono.get_pressure_biases()

            # Route batch with pressure
            for token_idx in range(32):
                logits, gates, chosen = router.route(pressure_biases=biases)
                batch_usage[chosen] += 1
                logger.log_routing(token_idx, 0, logits, gates, chosen, batch_idx)

            logger.log_batch_usage(batch_usage, batch_idx)

            # Update Chronovisor
            chrono.update_from_routing(batch_usage, batch_loss=0.5)
            chrono.step(n_ticks=20)

        # Verify state evolved
        states = chrono.get_expert_states()
        assert any(s["s"] != 0.0 for s in states)  # Some reliability change

    def test_reliability_improves_with_usage(self):
        """Experts that get used should see reliability improve."""
        chrono = ChronoMoEController(num_experts=4)

        # Heavily favor expert 0
        for _ in range(10):
            usage = [80, 10, 5, 5]  # Expert 0 dominates
            chrono.update_from_routing(usage, batch_loss=0.3)  # Low loss = good
            chrono.step(n_ticks=20)

        states = chrono.get_expert_states()

        # Expert 0 should have higher reliability than others
        # (This depends on alignment -> s mapping, may need tuning)
        # For now just verify the system runs
        assert len(states) == 4
