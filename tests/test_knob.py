"""
Tests for LLM-Controlled Meta-Knob.

Tests cover:
- MetaKnob κ → factor conversions
- Factor bounds and behavior
- KnobState summary generation
- RuleBasedKnobController heuristics
- Router integration with knob
- Alignment integration with knob
"""

import math
import numpy as np
import pytest

from chronomoe.knob import (
    MetaKnob,
    KnobFactors,
    KnobState,
    KnobDecision,
    RuleBasedKnobController,
    LLMKnobInterface,
)
from chronomoe.router import Router
from chronomoe.alignment import AlignmentMatrix


# =============================================================================
# MetaKnob Tests
# =============================================================================


class TestMetaKnob:
    """Tests for the MetaKnob class."""

    def test_creation(self):
        """Test MetaKnob creation."""
        knob = MetaKnob()

        assert knob.current_kappa == 0.0
        assert knob.beta_pressure == 0.5
        assert knob.beta_explore == 0.3
        assert knob.beta_alignment == 0.7

    def test_neutral_kappa(self):
        """Test that κ = 0 gives neutral factors."""
        knob = MetaKnob()
        factors = knob.set_kappa(0.0)

        assert factors.kappa == 0.0
        assert factors.pressure_scale == pytest.approx(1.0)
        assert factors.explore_bias == pytest.approx(1.0)
        assert factors.alignment_lr_mul == pytest.approx(1.0)

    def test_positive_kappa(self):
        """Test that κ > 0 increases factors."""
        knob = MetaKnob()
        factors = knob.set_kappa(1.0)

        assert factors.kappa == 1.0
        assert factors.pressure_scale > 1.0
        assert factors.explore_bias > 1.0
        assert factors.alignment_lr_mul > 1.0

    def test_negative_kappa(self):
        """Test that κ < 0 decreases factors."""
        knob = MetaKnob()
        factors = knob.set_kappa(-1.0)

        assert factors.kappa == -1.0
        assert factors.pressure_scale < 1.0
        assert factors.explore_bias < 1.0
        assert factors.alignment_lr_mul < 1.0

    def test_kappa_clamping(self):
        """Test that κ is clamped to [-1, +1]."""
        knob = MetaKnob()

        factors_high = knob.set_kappa(5.0)
        assert factors_high.kappa == 1.0

        factors_low = knob.set_kappa(-5.0)
        assert factors_low.kappa == -1.0

    def test_factor_symmetry(self):
        """Test that +κ and -κ are symmetric around 1.0."""
        knob = MetaKnob()

        factors_pos = knob.set_kappa(0.5)
        factors_neg = knob.set_kappa(-0.5)

        # Product should be ~1.0 for each factor
        assert factors_pos.pressure_scale * factors_neg.pressure_scale == pytest.approx(1.0)
        assert factors_pos.explore_bias * factors_neg.explore_bias == pytest.approx(1.0)
        assert factors_pos.alignment_lr_mul * factors_neg.alignment_lr_mul == pytest.approx(1.0)

    def test_history_tracking(self):
        """Test that κ history is tracked."""
        knob = MetaKnob()

        knob.set_kappa(0.1)
        knob.set_kappa(0.2)
        knob.set_kappa(-0.3)

        assert len(knob.kappa_history) == 3
        assert knob.kappa_history == [0.1, 0.2, -0.3]

    def test_intent_tracking(self):
        """Test that intent is tracked."""
        knob = MetaKnob()

        knob.set_kappa(0.5, intent="explore")
        assert knob.current_intent == "explore"

        factors = knob.get_factors()
        assert factors.intent == "explore"

    def test_reset(self):
        """Test knob reset."""
        knob = MetaKnob()

        knob.set_kappa(0.5, intent="test")
        knob.reset()

        assert knob.current_kappa == 0.0
        assert knob.current_intent == ""
        assert len(knob.kappa_history) == 0

    def test_smoothing(self):
        """Test κ smoothing (EMA)."""
        knob = MetaKnob(use_smoothing=True, eta_smooth=0.5)

        knob.set_kappa(1.0)
        # With eta=0.5, kappa should be 0.5 * 0 + 0.5 * 1.0 = 0.5
        assert knob.current_kappa == pytest.approx(0.5)

        knob.set_kappa(1.0)
        # kappa should be 0.5 * 0.5 + 0.5 * 1.0 = 0.75
        assert knob.current_kappa == pytest.approx(0.75)

    def test_diagnostics(self):
        """Test diagnostic output."""
        knob = MetaKnob()
        knob.set_kappa(0.3, intent="test")

        diag = knob.get_diagnostics()

        assert "kappa" in diag
        assert "intent" in diag
        assert "pressure_scale" in diag
        assert "explore_bias" in diag
        assert "alignment_lr_mul" in diag
        assert diag["kappa"] == 0.3

    def test_apply_decision(self):
        """Test applying a KnobDecision."""
        knob = MetaKnob()

        decision = KnobDecision(kappa=0.4, intent="explore", reasoning="test")
        factors = knob.apply_decision(decision)

        assert knob.current_kappa == 0.4
        assert factors.kappa == 0.4
        assert len(knob.decision_history) == 1


# =============================================================================
# KnobFactors Tests
# =============================================================================


class TestKnobFactors:
    """Tests for KnobFactors calculation."""

    def test_factor_ranges(self):
        """Test that factors stay in reasonable ranges."""
        knob = MetaKnob()

        for kappa in np.linspace(-1, 1, 21):
            factors = knob.set_kappa(kappa)

            # With default betas, factors should be in reasonable ranges
            assert 0.5 <= factors.pressure_scale <= 2.0
            assert 0.7 <= factors.explore_bias <= 1.4
            assert 0.4 <= factors.alignment_lr_mul <= 2.5

    def test_custom_betas(self):
        """Test custom beta parameters."""
        knob = MetaKnob(beta_pressure=1.0, beta_explore=0.5, beta_alignment=0.5)

        factors = knob.set_kappa(1.0)

        assert factors.pressure_scale == pytest.approx(math.e)  # e^1
        assert factors.explore_bias == pytest.approx(math.sqrt(math.e))  # e^0.5


# =============================================================================
# KnobState Tests
# =============================================================================


class TestKnobState:
    """Tests for KnobState summary."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        state = KnobState(
            loss=0.74,
            loss_trend=-0.01,
            routing_entropy=0.69,
            alignment_entropy=2.02,
            drift_correlation=0.99,
            coherence_R=0.45,
            population=8,
        )

        d = state.to_dict()

        assert d["loss"] == 0.74
        assert d["loss_trend"] == -0.01
        assert d["population"] == 8

    def test_to_prompt(self):
        """Test prompt generation."""
        state = KnobState(
            loss=0.74,
            loss_trend=-0.01,
            routing_entropy=0.69,
            alignment_entropy=2.02,
            drift_correlation=0.99,
            coherence_R=0.45,
            population=8,
        )

        prompt = state.to_prompt()

        assert "loss" in prompt.lower()
        assert "0.74" in prompt
        assert "entropy" in prompt.lower()

    def test_optional_fields(self):
        """Test optional fields in state."""
        state = KnobState(
            loss=0.5,
            loss_trend=0.0,
            routing_entropy=1.0,
            alignment_entropy=1.5,
            drift_correlation=0.8,
            coherence_R=0.5,
            population=8,
            specialization=0.6,
            mode="outside_box",
        )

        d = state.to_dict()
        assert d["specialization"] == 0.6
        assert d["mode"] == "outside_box"


# =============================================================================
# RuleBasedKnobController Tests
# =============================================================================


class TestRuleBasedKnobController:
    """Tests for rule-based controller."""

    def test_neutral_state(self):
        """Test decision on neutral state."""
        controller = RuleBasedKnobController()

        state = KnobState(
            loss=0.5,
            loss_trend=0.0,
            routing_entropy=1.5,  # At target
            alignment_entropy=1.0,
            drift_correlation=0.5,
            coherence_R=0.5,
            population=8,
        )

        decision = controller.decide(state)

        # Should be roughly neutral
        assert -0.5 <= decision.kappa <= 0.5

    def test_stuck_exploration(self):
        """Test that stuck loss triggers exploration."""
        controller = RuleBasedKnobController()

        state = KnobState(
            loss=0.7,
            loss_trend=0.0,
            routing_entropy=1.5,
            alignment_entropy=1.0,
            drift_correlation=0.5,
            coherence_R=0.5,
            population=8,
        )

        # Simulate being stuck for several steps
        for _ in range(5):
            decision = controller.decide(state)

        # After being stuck, should suggest exploration (κ > 0)
        assert decision.kappa > 0
        assert "stuck" in decision.intent or "explore" in decision.intent

    def test_improving_stabilization(self):
        """Test that improving loss triggers stabilization."""
        controller = RuleBasedKnobController()

        # First call to set prev_loss
        state1 = KnobState(
            loss=0.7,
            loss_trend=0.0,
            routing_entropy=1.5,
            alignment_entropy=1.0,
            drift_correlation=0.5,
            coherence_R=0.5,
            population=8,
        )
        controller.decide(state1)

        # Second call with improved loss
        state2 = KnobState(
            loss=0.6,  # Improved
            loss_trend=-0.1,
            routing_entropy=1.5,
            alignment_entropy=1.0,
            drift_correlation=0.5,
            coherence_R=0.5,
            population=8,
        )
        decision = controller.decide(state2)

        # Should suggest stabilization (κ < 0)
        assert decision.kappa < 0

    def test_low_coherence_exploration(self):
        """Test that low coherence triggers exploration."""
        controller = RuleBasedKnobController()

        state = KnobState(
            loss=0.5,
            loss_trend=0.0,
            routing_entropy=1.5,
            alignment_entropy=1.0,
            drift_correlation=0.5,
            coherence_R=0.1,  # Very low coherence
            population=8,
        )

        decision = controller.decide(state)

        assert decision.kappa > 0


# =============================================================================
# LLMKnobInterface Tests
# =============================================================================


class TestLLMKnobInterface:
    """Tests for LLM interface."""

    def test_format_state(self):
        """Test state formatting for LLM."""
        interface = LLMKnobInterface()

        state = KnobState(
            loss=0.5,
            loss_trend=-0.01,
            routing_entropy=1.0,
            alignment_entropy=1.5,
            drift_correlation=0.8,
            coherence_R=0.5,
            population=8,
        )

        prompt = interface.format_state(state)

        assert "loss" in prompt.lower()
        assert "kappa" in prompt.lower() or "κ" in prompt

    def test_parse_json_response(self):
        """Test parsing JSON response."""
        interface = LLMKnobInterface()

        response = '{"kappa": 0.3, "intent": "explore"}'
        decision = interface.parse_response(response)

        assert decision.kappa == 0.3
        assert decision.intent == "explore"

    def test_parse_embedded_json(self):
        """Test parsing JSON embedded in text."""
        interface = LLMKnobInterface()

        response = 'Based on the state, I recommend {"kappa": -0.2, "intent": "stabilize"} to reduce pressure.'
        decision = interface.parse_response(response)

        assert decision.kappa == -0.2
        assert decision.intent == "stabilize"

    def test_parse_fallback(self):
        """Test fallback parsing."""
        interface = LLMKnobInterface()

        response = "I think kappa should be around 0.4"
        decision = interface.parse_response(response)

        assert decision.kappa == pytest.approx(0.4)

    def test_parse_failure(self):
        """Test parse failure returns neutral."""
        interface = LLMKnobInterface()

        response = "I have no idea what to do"
        decision = interface.parse_response(response)

        assert decision.kappa == 0.0
        assert "failed" in decision.intent

    def test_system_prompt(self):
        """Test system prompt exists and contains instructions."""
        interface = LLMKnobInterface()

        system, user = interface.create_prompt(KnobState(
            loss=0.5,
            loss_trend=0.0,
            routing_entropy=1.0,
            alignment_entropy=1.0,
            drift_correlation=0.5,
            coherence_R=0.5,
            population=8,
        ))

        assert "κ" in system or "kappa" in system.lower()
        assert "-1" in system and "+1" in system


# =============================================================================
# Router Integration Tests
# =============================================================================


class TestRouterKnobIntegration:
    """Tests for router + knob integration."""

    def test_forward_with_knob_neutral(self):
        """Test knob-modulated forward with neutral κ."""
        router = Router(input_dim=32, n_experts=4, seed=42)

        x = np.random.randn(10, 32)

        # Baseline
        gate_baseline = router.forward(x.copy())

        # With knob (neutral)
        router.reset_log()
        gate_knob = router.forward_with_knob(
            x.copy(),
            pressure_scale=1.0,
            explore_bias=1.0,
        )

        # Should be identical
        assert np.allclose(gate_baseline, gate_knob)

    def test_forward_with_knob_pressure_scale(self):
        """Test that pressure_scale affects routing."""
        router = Router(input_dim=32, n_experts=4, seed=42)

        # Inject some pressure
        router.inject_pressure(np.array([1.0, -0.5, 0.0, -0.5]))

        x = np.random.randn(100, 32)

        # Low pressure scale
        router.reset_log()
        gate_low = router.forward_with_knob(x.copy(), pressure_scale=0.1)
        dist_low = router.log.get_expert_distribution()

        # High pressure scale
        router.reset_log()
        gate_high = router.forward_with_knob(x.copy(), pressure_scale=3.0)
        dist_high = router.log.get_expert_distribution()

        # High pressure should shift more toward expert 0
        assert dist_high[0] > dist_low[0]

    def test_forward_with_knob_explore_bias(self):
        """Test that explore_bias affects distribution entropy."""
        router = Router(input_dim=32, n_experts=4, seed=42)

        x = np.random.randn(100, 32)

        # Low explore bias (sharper)
        router.reset_log()
        gate_sharp = router.forward_with_knob(x.copy(), explore_bias=0.5)

        # High explore bias (softer)
        router.reset_log()
        gate_soft = router.forward_with_knob(x.copy(), explore_bias=2.0)

        # Soft should have higher entropy (more uniform)
        # Check via variance of gate weights
        var_sharp = np.var(gate_sharp, axis=1).mean()
        var_soft = np.var(gate_soft, axis=1).mean()

        # Sharper distribution has higher variance per row
        assert var_sharp > var_soft


# =============================================================================
# Alignment Integration Tests
# =============================================================================


class TestAlignmentKnobIntegration:
    """Tests for alignment + knob integration."""

    def test_update_with_knob_neutral(self):
        """Test knob-modulated update with neutral κ."""
        am = AlignmentMatrix.create(n_chrono=4, n_moe=4, eta_A=0.1, seed=42)

        chrono_conf = np.array([1.0, 2.0, 1.5, 0.5])
        moe_spec = np.array([0.2, 0.3, 0.3, 0.2])

        # Normal update
        A_before = am.A.copy()
        am.update(chrono_conf, moe_spec)
        A_normal = am.A.copy()

        # Reset and try with knob
        am.reset(seed=42)
        am.update_with_knob(chrono_conf, moe_spec, alignment_lr_mul=1.0)
        A_knob = am.A.copy()

        # Should be identical
        assert np.allclose(A_normal, A_knob)

    def test_update_with_knob_faster(self):
        """Test that alignment_lr_mul > 1 learns faster."""
        # Normal update
        am1 = AlignmentMatrix.create(n_chrono=4, n_moe=4, eta_A=0.1, seed=42)
        chrono_conf = np.array([10.0, 1.0, 1.0, 1.0])
        moe_spec = np.array([0.1, 0.8, 0.05, 0.05])

        am1.update(chrono_conf, moe_spec)
        delta_normal = np.linalg.norm(am1.A - np.eye(4))

        # Faster update
        am2 = AlignmentMatrix.create(n_chrono=4, n_moe=4, eta_A=0.1, seed=42)
        am2.update_with_knob(chrono_conf, moe_spec, alignment_lr_mul=3.0)
        delta_fast = np.linalg.norm(am2.A - np.eye(4))

        # Faster should have moved further from identity
        assert delta_fast > delta_normal

    def test_update_with_knob_frozen(self):
        """Test that alignment_lr_mul ≈ 0 freezes alignment."""
        am = AlignmentMatrix.create(n_chrono=4, n_moe=4, eta_A=0.1, seed=42)

        A_before = am.A.copy()

        chrono_conf = np.array([10.0, 1.0, 1.0, 1.0])
        moe_spec = np.array([0.1, 0.8, 0.05, 0.05])

        result = am.update_with_knob(chrono_conf, moe_spec, alignment_lr_mul=0.0)

        # Should be frozen
        assert result["frozen"] is True
        assert np.allclose(am.A, A_before)

    def test_update_with_knob_returns_effective_eta(self):
        """Test that update returns effective learning rate."""
        am = AlignmentMatrix.create(n_chrono=4, n_moe=4, eta_A=0.1, seed=42)

        chrono_conf = np.array([1.0, 1.0, 1.0, 1.0])
        moe_spec = np.array([0.25, 0.25, 0.25, 0.25])

        result = am.update_with_knob(chrono_conf, moe_spec, alignment_lr_mul=2.0)

        assert result["effective_eta"] == pytest.approx(0.2)
        assert result["frozen"] is False
