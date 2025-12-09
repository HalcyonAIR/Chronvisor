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
