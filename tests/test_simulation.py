"""Tests for the toy simulation components."""

import random
from chronovisor.simulation import (
    LensState,
    ToyExpert,
    compute_ensemble_coherence,
)
from chronovisor.expert_harness import ExpertHarness


class TestLensState:
    """Tests for LensState class."""

    def test_default_initialisation(self):
        """LensState defaults to [0.0, 0.0]."""
        lens = LensState()
        assert lens.vector == [0.0, 0.0]

    def test_custom_initialisation(self):
        """LensState accepts custom initial vector."""
        lens = LensState(initial=[1.5, -0.5])
        assert lens.vector == [1.5, -0.5]

    def test_update_adds_delta(self):
        """update() adds delta to the vector."""
        lens = LensState(initial=[1.0, 2.0])
        lens.update([0.5, -0.5])
        assert lens.vector == [1.5, 1.5]

    def test_update_multiple_times(self):
        """Multiple updates accumulate."""
        lens = LensState()
        lens.update([0.1, 0.1])
        lens.update([0.1, 0.1])
        lens.update([0.1, 0.1])
        assert abs(lens.vector[0] - 0.3) < 1e-10
        assert abs(lens.vector[1] - 0.3) < 1e-10

    def test_repr(self):
        """LensState has readable repr."""
        lens = LensState(initial=[0.123, -0.456])
        repr_str = repr(lens)
        assert "Lens" in repr_str
        assert "+0.123" in repr_str
        assert "-0.456" in repr_str


class TestToyExpert:
    """Tests for ToyExpert class."""

    def test_is_subclass_of_harness(self):
        """ToyExpert is a subclass of ExpertHarness."""
        assert issubclass(ToyExpert, ExpertHarness)

    def test_default_sensitivity(self):
        """ToyExpert defaults to sensitivity [1.0, 1.0]."""
        expert = ToyExpert()
        assert expert.sensitivity == [1.0, 1.0]

    def test_custom_sensitivity(self):
        """ToyExpert accepts custom sensitivity."""
        expert = ToyExpert(sensitivity=[0.5, 1.5])
        assert expert.sensitivity == [0.5, 1.5]

    def test_sense_returns_required_keys(self):
        """sense() returns dict with all required keys."""
        expert = ToyExpert()
        lens = LensState()
        result = expert.sense(lens)

        assert "gain" in result
        assert "tilt" in result
        assert "stability" in result
        assert "out_of_tolerance" in result

    def test_sense_with_none_falls_back_to_parent(self):
        """sense(None) returns parent's fixed values."""
        expert = ToyExpert()
        result = expert.sense(None)

        assert result["gain"] == 1.0
        assert result["tilt"] == 0.0
        assert result["stability"] == 1.0
        assert result["out_of_tolerance"] is False

    def test_sense_responds_to_lens_state(self):
        """sense() produces different values for different lens states."""
        random.seed(42)
        expert = ToyExpert(sensitivity=[1.0, 1.0])

        lens_zero = LensState(initial=[0.0, 0.0])
        lens_shifted = LensState(initial=[2.0, 2.0])

        result_zero = expert.sense(lens_zero)
        result_shifted = expert.sense(lens_shifted)

        # Gain should be higher when lens[0] is positive
        # (accounting for noise, the difference should be clear)
        assert result_shifted["gain"] > result_zero["gain"] + 0.1

    def test_sense_stability_decreases_with_stretch(self):
        """Stability decreases as lens vector grows."""
        random.seed(42)
        expert = ToyExpert(sensitivity=[1.0, 1.0])

        lens_small = LensState(initial=[0.0, 0.0])
        lens_large = LensState(initial=[3.0, 3.0])

        result_small = expert.sense(lens_small)
        result_large = expert.sense(lens_large)

        assert result_large["stability"] < result_small["stability"]

    def test_out_of_tolerance_when_stability_low(self):
        """out_of_tolerance is True when stability drops below 0.5."""
        random.seed(42)
        expert = ToyExpert(sensitivity=[1.0, 1.0])

        # Large lens vector should cause low stability
        lens = LensState(initial=[5.0, 5.0])
        result = expert.sense(lens)

        # Stability should be very low, triggering OOT
        assert result["stability"] < 0.5 or result["out_of_tolerance"]


class TestEnsembleCoherence:
    """Tests for compute_ensemble_coherence function."""

    def test_empty_list_returns_zero(self):
        """Empty expert list returns 0.0 coherence."""
        assert compute_ensemble_coherence([]) == 0.0

    def test_single_expert_returns_zero(self):
        """Single expert returns 0.0 (can't compute variance)."""
        signals = [{"gain": 1.0, "stability": 1.0}]
        assert compute_ensemble_coherence(signals) == 0.0

    def test_identical_signals_high_coherence(self):
        """Identical signals produce high coherence (zero variance)."""
        signals = [
            {"gain": 1.0, "stability": 0.9},
            {"gain": 1.0, "stability": 0.9},
            {"gain": 1.0, "stability": 0.9},
        ]
        coherence = compute_ensemble_coherence(signals)
        assert coherence == 0.0  # Zero variance = zero negative

    def test_divergent_signals_low_coherence(self):
        """Divergent signals produce low (negative) coherence."""
        signals = [
            {"gain": 0.5, "stability": 0.2},
            {"gain": 1.0, "stability": 0.5},
            {"gain": 1.5, "stability": 0.8},
        ]
        coherence = compute_ensemble_coherence(signals)
        assert coherence < 0.0  # High variance = negative coherence

    def test_coherence_is_deterministic(self):
        """Same inputs produce same coherence."""
        signals = [
            {"gain": 0.8, "stability": 0.7},
            {"gain": 1.2, "stability": 0.9},
        ]
        c1 = compute_ensemble_coherence(signals)
        c2 = compute_ensemble_coherence(signals)
        assert c1 == c2
