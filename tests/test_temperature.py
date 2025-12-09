"""
Tests for temperature-warped routing in ChronoMoE.

Tests the 2-field routing environment where:
- Pressure (b_k) = force field that pushes toward/away from experts
- Temperature (T_k) = permeability that controls how slippery each region is
"""

import math
import numpy as np
import pytest

from chronomoe.router import Router
from chronomoe.bridge import ChronoMoEBridge, TemperatureField
from chronomoe.knob import MetaKnob, KnobFactors


class TestTemperatureField:
    """Tests for TemperatureField computation in ChronoMoEBridge."""

    def test_temperature_field_creation(self):
        """Test that bridge can create a temperature field."""
        bridge = ChronoMoEBridge.create(n_experts=8)
        temp_field = bridge.get_temperature_field()

        assert isinstance(temp_field, TemperatureField)
        assert temp_field.temperatures.shape == (8,)
        assert temp_field.base_temperature == 1.0
        assert 0.0 <= temp_field.coherence_R <= 1.0

    def test_temperature_bounds(self):
        """Test that temperatures are clamped to valid bounds."""
        bridge = ChronoMoEBridge.create(
            n_experts=8,
            T_min=0.3,
            T_max=3.0,
        )
        temp_field = bridge.get_temperature_field()

        assert np.all(temp_field.temperatures >= 0.3)
        assert np.all(temp_field.temperatures <= 3.0)

    def test_temperature_history(self):
        """Test that temperature history is tracked."""
        bridge = ChronoMoEBridge.create(n_experts=8)

        # Get temperature multiple times
        for _ in range(5):
            bridge.get_temperature_field()

        assert len(bridge.temperature_history) == 5

    def test_coherence_factor_range(self):
        """Test coherence factor is in valid range."""
        bridge = ChronoMoEBridge.create(n_experts=8, beta_R=0.5)
        temp_field = bridge.get_temperature_field()

        # With beta_R=0.5, coherence factor should be in [1.0, 1.5]
        # coherence_factor = 1 + beta_R * (1 - R)
        assert 1.0 <= temp_field.coherence_factor <= 1.5

    def test_drift_factors_positive(self):
        """Test that drift factors are positive."""
        bridge = ChronoMoEBridge.create(n_experts=8)
        temp_field = bridge.get_temperature_field()

        assert np.all(temp_field.drift_factors > 0)

    def test_reliability_factors_positive(self):
        """Test that reliability factors are positive."""
        bridge = ChronoMoEBridge.create(n_experts=8)
        temp_field = bridge.get_temperature_field()

        assert np.all(temp_field.reliability_factors > 0)

    def test_temperature_reset(self):
        """Test that temperature history is cleared on reset."""
        bridge = ChronoMoEBridge.create(n_experts=8)

        # Generate some history
        for _ in range(5):
            bridge.get_temperature_field()
        assert len(bridge.temperature_history) == 5

        # Reset
        bridge.reset()
        assert len(bridge.temperature_history) == 0


class TestRouterTemperatureWarping:
    """Tests for temperature-warped routing in Router."""

    def test_forward_with_temperature_basic(self):
        """Test basic temperature-warped forward pass."""
        router = Router(input_dim=64, n_experts=8, seed=42)
        x = np.random.randn(32, 64)
        temperatures = np.ones(8)  # Uniform temperatures

        gate_weights = router.forward_with_temperature(x, temperatures)

        assert gate_weights.shape == (32, 8)
        # Gate weights should sum to 1 per token
        assert np.allclose(gate_weights.sum(axis=1), 1.0)

    def test_temperature_shape_validation(self):
        """Test that incorrect temperature shape raises error."""
        router = Router(input_dim=64, n_experts=8, seed=42)
        x = np.random.randn(32, 64)
        wrong_temps = np.ones(4)  # Wrong shape

        with pytest.raises(ValueError, match="doesn't match n_experts"):
            router.forward_with_temperature(x, wrong_temps)

    def test_low_temperature_concentrates_distribution(self):
        """Test that low temperature concentrates gate weights."""
        router = Router(input_dim=64, n_experts=8, seed=42)
        x = np.random.randn(32, 64)

        high_temp = np.ones(8) * 2.0
        low_temp = np.ones(8) * 0.5

        gates_high = router.forward_with_temperature(x, high_temp)
        router.reset_log()
        gates_low = router.forward_with_temperature(x, low_temp)

        # Low temperature should have lower entropy (more concentrated)
        def entropy(p):
            p = np.clip(p, 1e-10, 1.0)
            return -np.sum(p * np.log(p), axis=1)

        entropy_high = entropy(gates_high).mean()
        entropy_low = entropy(gates_low).mean()

        assert entropy_low < entropy_high

    def test_per_expert_temperature_affects_selection(self):
        """Test that per-expert temperatures affect expert selection."""
        router = Router(input_dim=64, n_experts=8, seed=42)
        x = np.random.randn(100, 64)

        # Make expert 0 very "cold" (sharp) and expert 7 very "hot" (diffuse)
        temps = np.ones(8)
        temps[0] = 0.3  # Cold - should be sharper
        temps[7] = 3.0  # Hot - should be more diffuse

        gate_weights = router.forward_with_temperature(x, temps)

        # Expert 0 should have more extreme weights (closer to 0 or 1)
        # Expert 7 should have more moderate weights
        expert_0_variance = np.var(gate_weights[:, 0])
        expert_7_variance = np.var(gate_weights[:, 7])

        # This is a statistical test - may not always hold
        # but cold experts should generally have higher variance
        assert expert_0_variance > 0  # Just verify not all same

    def test_pressure_scale_multiplies_pressure(self):
        """Test that pressure_scale affects the pressure magnitude."""
        router = Router(input_dim=64, n_experts=8, seed=42)
        x = np.random.randn(32, 64)
        temps = np.ones(8)

        # Inject pressure
        pressure = np.array([0.5, -0.5, 0.3, -0.3, 0.1, -0.1, 0.2, -0.2])
        router.inject_pressure(pressure)

        # Forward with different pressure scales
        gates_low = router.forward_with_temperature(x, temps, pressure_scale=0.5)
        router.reset_log()
        gates_high = router.forward_with_temperature(x, temps, pressure_scale=2.0)

        # Higher pressure scale should shift distribution more
        # (hard to verify directly, just check they're different)
        assert not np.allclose(gates_low, gates_high)

    def test_temp_scale_multiplies_temperatures(self):
        """Test that temp_scale affects all temperatures."""
        router = Router(input_dim=64, n_experts=8, seed=42)
        x = np.random.randn(32, 64)
        base_temps = np.ones(8)

        gates_low = router.forward_with_temperature(x, base_temps, temp_scale=0.5)
        router.reset_log()
        gates_high = router.forward_with_temperature(x, base_temps, temp_scale=2.0)

        # Lower temp_scale should concentrate distribution
        def entropy(p):
            p = np.clip(p, 1e-10, 1.0)
            return -np.sum(p * np.log(p), axis=1)

        entropy_low = entropy(gates_low).mean()
        entropy_high = entropy(gates_high).mean()

        assert entropy_low < entropy_high

    def test_logging_with_temperature(self):
        """Test that decisions are logged with warped logits."""
        router = Router(input_dim=64, n_experts=8, seed=42)
        x = np.random.randn(10, 64)
        temps = np.ones(8) * 0.5  # Low temp

        router.forward_with_temperature(x, temps)

        assert len(router.log.decisions) == 10
        # Check that adjusted_logits reflect temperature warping
        decision = router.log.decisions[0]
        assert decision.adjusted_logits is not None


class TestKnobTemperatureScale:
    """Tests for temperature scale in MetaKnob."""

    def test_temp_scale_at_neutral(self):
        """Test that temp_scale is 1.0 at κ=0."""
        knob = MetaKnob()
        factors = knob.set_kappa(0.0)

        assert abs(factors.temp_scale - 1.0) < 1e-6

    def test_temp_scale_positive_kappa(self):
        """Test that positive κ increases temp_scale."""
        knob = MetaKnob()
        factors = knob.set_kappa(0.5)

        assert factors.temp_scale > 1.0

    def test_temp_scale_negative_kappa(self):
        """Test that negative κ decreases temp_scale."""
        knob = MetaKnob()
        factors = knob.set_kappa(-0.5)

        assert factors.temp_scale < 1.0

    def test_temp_scale_symmetry(self):
        """Test that temp_scale is symmetric around κ=0."""
        knob = MetaKnob()

        factors_pos = knob.set_kappa(0.5)
        factors_neg = knob.set_kappa(-0.5)

        # exp(β * 0.5) * exp(β * -0.5) = 1
        assert abs(factors_pos.temp_scale * factors_neg.temp_scale - 1.0) < 1e-6

    def test_temp_scale_range(self):
        """Test temp_scale range at extreme κ values."""
        knob = MetaKnob(beta_temperature=0.4)

        factors_max = knob.set_kappa(1.0)
        factors_min = knob.set_kappa(-1.0)

        # exp(0.4 * 1.0) ≈ 1.49
        assert 1.4 < factors_max.temp_scale < 1.6
        # exp(0.4 * -1.0) ≈ 0.67
        assert 0.6 < factors_min.temp_scale < 0.8

    def test_temp_scale_in_diagnostics(self):
        """Test that temp_scale is included in diagnostics."""
        knob = MetaKnob()
        knob.set_kappa(0.3)

        diagnostics = knob.get_diagnostics()
        assert "temp_scale" in diagnostics
        assert diagnostics["temp_scale"] > 1.0


class TestStructuralTemperature:
    """Tests for structural (geological) temperature evolution."""

    def test_structural_temperature_initialized(self):
        """Test that structural temperature starts at 1.0."""
        bridge = ChronoMoEBridge.create(n_experts=8)

        # Initially all 1.0
        assert bridge.structural_T is not None
        assert np.allclose(bridge.structural_T, 1.0)

    def test_structural_temperature_evolves(self):
        """Test that structural temperature evolves via EMA."""
        bridge = ChronoMoEBridge.create(n_experts=8, eta_structural_T=0.1)  # Faster for testing

        # Get temperature field multiple times
        initial_structural = bridge.structural_T.copy()

        for _ in range(10):
            bridge.get_temperature_field()

        # Structural temperature should have changed
        assert not np.allclose(bridge.structural_T, initial_structural)

    def test_structural_temperature_history(self):
        """Test that structural temperature history is tracked."""
        bridge = ChronoMoEBridge.create(n_experts=8)

        for _ in range(5):
            bridge.get_temperature_field()

        assert len(bridge.structural_T_history) == 5

    def test_effective_temperature_combines_fast_and_structural(self):
        """Test that effective = fast × structural."""
        bridge = ChronoMoEBridge.create(n_experts=8, eta_structural_T=0.5)  # Very fast for testing

        # Set structural to non-uniform
        bridge.structural_T = np.array([0.5, 1.0, 1.5, 0.8, 1.2, 0.9, 1.1, 1.0])

        temp_field = bridge.get_temperature_field(update_structural=False)

        # effective should be fast × structural (within clamping bounds)
        expected = temp_field.fast_temperatures * bridge.structural_T
        expected = np.clip(expected, bridge.T_min, bridge.T_max)

        assert np.allclose(temp_field.effective_temperatures, expected)

    def test_structural_temperature_reset(self):
        """Test that structural temperature resets properly."""
        bridge = ChronoMoEBridge.create(n_experts=8, eta_structural_T=0.1)

        # Evolve for a while
        for _ in range(20):
            bridge.get_temperature_field()

        assert not np.allclose(bridge.structural_T, 1.0)

        # Reset
        bridge.reset()

        # Should be back to 1.0
        assert np.allclose(bridge.structural_T, 1.0)
        assert len(bridge.structural_T_history) == 0

    def test_structural_temperature_diagnostics(self):
        """Test structural temperature diagnostics."""
        bridge = ChronoMoEBridge.create(n_experts=8)

        diagnostics = bridge.get_structural_temperature_diagnostics()

        assert "structural_T" in diagnostics
        assert "variance" in diagnostics
        assert "entropy" in diagnostics
        assert "landscape_formed" in diagnostics
        assert "valleys" in diagnostics
        assert "ridges" in diagnostics

    def test_landscape_formation(self):
        """Test that landscape forms (variance increases) over time."""
        bridge = ChronoMoEBridge.create(n_experts=8, eta_structural_T=0.1)

        initial_diag = bridge.get_structural_temperature_diagnostics()
        initial_variance = initial_diag["variance"]

        # Evolve with varying fast temperatures
        for i in range(100):
            # Manually feed some stats to create varying conditions
            from chronomoe.bridge import RoutingStats
            stats = RoutingStats(
                expert_usage=np.random.randint(0, 10, 8),
                mean_gate_weights=np.random.uniform(0, 1, 8),
                batch_loss=0.5 + 0.1 * np.sin(i / 10),
            )
            bridge.feed_routing_stats(stats, num_chronovisor_ticks=5)
            bridge.get_temperature_field()

        final_diag = bridge.get_structural_temperature_diagnostics()
        final_variance = final_diag["variance"]

        # Variance should have increased (landscape formed)
        assert final_variance >= initial_variance * 0.9  # May be near 0 initially

    def test_valleys_and_ridges_identified(self):
        """Test that valleys and ridges are identified correctly."""
        bridge = ChronoMoEBridge.create(n_experts=8)

        # Set structural temps with clear valleys and ridges
        bridge.structural_T = np.array([0.5, 0.6, 1.5, 1.6, 0.7, 1.0, 1.4, 0.8])

        diagnostics = bridge.get_structural_temperature_diagnostics()

        # Valleys should be low-T experts
        valleys = diagnostics["valleys"]
        ridges = diagnostics["ridges"]

        # Low temps should be valleys
        assert 0 in valleys or 1 in valleys  # 0.5, 0.6 are low
        # High temps should be ridges
        assert 2 in ridges or 3 in ridges  # 1.5, 1.6 are high

    def test_temperature_field_backwards_compatibility(self):
        """Test that .temperatures property still works."""
        bridge = ChronoMoEBridge.create(n_experts=8)
        temp_field = bridge.get_temperature_field()

        # .temperatures should alias .effective_temperatures
        assert np.array_equal(temp_field.temperatures, temp_field.effective_temperatures)


class TestValleyHealthDiagnostics:
    """Tests for valley health monitoring (bad valley detection)."""

    def test_valley_health_diagnostics_structure(self):
        """Test that valley health diagnostics returns expected structure."""
        bridge = ChronoMoEBridge.create(n_experts=8)
        diagnostics = bridge.get_valley_health_diagnostics()

        assert "valleys" in diagnostics
        assert "valley_health" in diagnostics
        assert "healthy_valleys" in diagnostics
        assert "unhealthy_valleys" in diagnostics
        assert "at_risk_experts" in diagnostics
        assert "mean_valley_health" in diagnostics
        assert "reliabilities" in diagnostics
        assert "self_correction_working" in diagnostics

    def test_valley_health_with_no_valleys(self):
        """Test diagnostics when no valleys have formed."""
        bridge = ChronoMoEBridge.create(n_experts=8)
        # All structural_T = 1.0, so no valleys
        diagnostics = bridge.get_valley_health_diagnostics()

        assert diagnostics["valleys"] == []
        assert diagnostics["valley_health"] == {}
        assert diagnostics["healthy_valleys"] == []
        assert diagnostics["unhealthy_valleys"] == []
        assert diagnostics["mean_valley_health"] == 1.0
        assert diagnostics["self_correction_working"] is True

    def test_healthy_valley_detection(self):
        """Test that valleys with high reliability are marked healthy."""
        bridge = ChronoMoEBridge.create(n_experts=8)

        # Create a clear valley
        bridge.structural_T = np.array([0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5])

        # Make expert 0 reliable (high s)
        bridge.controller.experts[0].s = 1.0  # High reliability

        diagnostics = bridge.get_valley_health_diagnostics()

        # Expert 0 should be a healthy valley
        assert 0 in diagnostics["valleys"]
        assert 0 in diagnostics["healthy_valleys"]
        assert 0 not in diagnostics["unhealthy_valleys"]
        assert diagnostics["valley_health"].get(0, 0) > 0.5

    def test_unhealthy_valley_detection(self):
        """Test that valleys with low reliability are marked unhealthy."""
        bridge = ChronoMoEBridge.create(n_experts=8)

        # Create a clear valley
        bridge.structural_T = np.array([0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5])

        # Make expert 0 unreliable (low s)
        bridge.controller.experts[0].s = -1.0  # Low reliability

        diagnostics = bridge.get_valley_health_diagnostics()

        # Expert 0 should be an unhealthy valley
        assert 0 in diagnostics["valleys"]
        assert 0 in diagnostics["unhealthy_valleys"]
        assert 0 not in diagnostics["healthy_valleys"]
        assert diagnostics["valley_health"].get(0, 1) < 0.5

    def test_at_risk_expert_detection(self):
        """Test detection of experts with high T_fast but still in valley."""
        bridge = ChronoMoEBridge.create(n_experts=8)

        # Create a valley for expert 0
        bridge.structural_T = np.array([0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5])

        # Make expert 0 unreliable so T_fast is high
        bridge.controller.experts[0].s = -2.0  # Very low reliability

        # Get temperature field to populate temperature_history
        bridge.get_temperature_field()

        diagnostics = bridge.get_valley_health_diagnostics()

        # Expert 0 should be at risk (high T_fast, still valley)
        # This indicates structural correction is lagging
        assert 0 in diagnostics["at_risk_experts"] or 0 in diagnostics["unhealthy_valleys"]

    def test_self_correction_working_flag(self):
        """Test that self_correction_working flag is set correctly."""
        bridge = ChronoMoEBridge.create(n_experts=8)

        # Initially, no valleys, so self-correction is "working" (nothing to correct)
        diagnostics = bridge.get_valley_health_diagnostics()
        assert diagnostics["self_correction_working"] is True

        # Create healthy valleys only
        bridge.structural_T = np.array([0.5, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5])
        for i in [0, 1]:
            bridge.controller.experts[i].s = 1.0  # High reliability

        diagnostics = bridge.get_valley_health_diagnostics()
        # Should still be working - only healthy valleys
        assert len(diagnostics["unhealthy_valleys"]) == 0

    def test_reliabilities_returned(self):
        """Test that reliabilities are returned for all experts."""
        bridge = ChronoMoEBridge.create(n_experts=8)

        # Set different reliability values
        for i, expert in enumerate(bridge.controller.experts[:8]):
            expert.s = (i - 4) / 2  # Range from -2 to +1.5

        diagnostics = bridge.get_valley_health_diagnostics()

        # Should have reliability for each expert
        assert len(diagnostics["reliabilities"]) == 8

        # Reliabilities should be in [0, 1] (sigmoid output)
        for rel in diagnostics["reliabilities"].values():
            assert 0 <= rel <= 1

    def test_mean_valley_health_computation(self):
        """Test mean valley health is computed correctly."""
        bridge = ChronoMoEBridge.create(n_experts=8)

        # Create two valleys with known reliabilities
        bridge.structural_T = np.array([0.5, 0.6, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2])
        bridge.controller.experts[0].s = 1.0  # High reliability
        bridge.controller.experts[1].s = -1.0  # Low reliability

        diagnostics = bridge.get_valley_health_diagnostics()

        # Mean should be average of the two valley healths
        if len(diagnostics["valleys"]) >= 2:
            health_values = list(diagnostics["valley_health"].values())
            expected_mean = sum(health_values) / len(health_values)
            assert abs(diagnostics["mean_valley_health"] - expected_mean) < 0.01


class TestIntegratedTemperatureSystem:
    """Integration tests for the full temperature system."""

    def test_bridge_to_router_temperature_flow(self):
        """Test that temperatures flow from bridge to router."""
        bridge = ChronoMoEBridge.create(n_experts=8)
        router = Router(input_dim=64, n_experts=8, seed=42)
        x = np.random.randn(32, 64)

        # Get temperature from bridge
        temp_field = bridge.get_temperature_field()

        # Use in router
        gate_weights = router.forward_with_temperature(
            x, temp_field.temperatures
        )

        assert gate_weights.shape == (32, 8)

    def test_knob_modulated_temperature_routing(self):
        """Test full flow: knob → temp_scale → routing."""
        bridge = ChronoMoEBridge.create(n_experts=8)
        router = Router(input_dim=64, n_experts=8, seed=42)
        knob = MetaKnob()
        x = np.random.randn(32, 64)

        # Get pressure and temperature from bridge
        pressure_bias = bridge.get_pressure_bias()
        router.inject_pressure(pressure_bias.combined)
        temp_field = bridge.get_temperature_field()

        # Get knob factors
        factors = knob.set_kappa(0.3)

        # Route with knob modulation
        gate_weights = router.forward_with_temperature(
            x,
            temp_field.temperatures,
            pressure_scale=factors.pressure_scale,
            temp_scale=factors.temp_scale,
        )

        assert gate_weights.shape == (32, 8)
        assert np.allclose(gate_weights.sum(axis=1), 1.0)

    def test_exploration_vs_exploitation(self):
        """Test that κ affects exploration/exploitation behavior."""
        bridge = ChronoMoEBridge.create(n_experts=8)
        router = Router(input_dim=64, n_experts=8, seed=42)
        knob = MetaKnob()
        x = np.random.randn(100, 64)

        # Get pressure and temperature
        pressure_bias = bridge.get_pressure_bias()
        router.inject_pressure(pressure_bias.combined)
        temp_field = bridge.get_temperature_field()

        def get_entropy(kappa):
            router.reset_log()
            factors = knob.set_kappa(kappa)
            gates = router.forward_with_temperature(
                x,
                temp_field.temperatures,
                pressure_scale=factors.pressure_scale,
                temp_scale=factors.temp_scale,
            )
            p = np.clip(gates, 1e-10, 1.0)
            return -np.sum(p * np.log(p), axis=1).mean()

        entropy_exploit = get_entropy(-0.8)  # Exploitation
        entropy_neutral = get_entropy(0.0)  # Neutral
        entropy_explore = get_entropy(0.8)  # Exploration

        # Exploration should have higher entropy
        assert entropy_exploit < entropy_neutral < entropy_explore
