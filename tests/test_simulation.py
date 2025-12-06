"""Tests for the toy simulation components."""

import math
import random
from chronovisor.simulation import (
    # V2 components
    TraceFieldLens,
    KuramotoExpert,
    GatingController,
    compute_kuramoto_R,
    sigmoid,
    # V1 compatibility
    compute_ensemble_coherence,
)
from chronovisor.expert_harness import ExpertHarness


# =============================================================================
# Tests for TraceFieldLens (V2)
# =============================================================================

class TestTraceFieldLens:
    """Tests for the emergent trace field lens."""

    def test_default_initialisation(self):
        """TraceFieldLens starts with zero trace."""
        lens = TraceFieldLens()
        assert lens.trace == [0.0, 0.0]
        assert lens.dim == 2

    def test_gamma_default(self):
        """Default gamma is 0.9."""
        lens = TraceFieldLens()
        assert lens.gamma == 0.9

    def test_set_gamma_clamps(self):
        """set_gamma clamps to [0, 1]."""
        lens = TraceFieldLens()
        lens.set_gamma(1.5)
        assert lens.gamma == 1.0
        lens.set_gamma(-0.5)
        assert lens.gamma == 0.0

    def test_magnitude_zero_initially(self):
        """Magnitude is zero when trace is zero."""
        lens = TraceFieldLens()
        assert lens.magnitude == 0.0

    def test_vector_normalised(self):
        """vector property returns normalised L(t)."""
        lens = TraceFieldLens()
        lens.trace = [3.0, 4.0]
        L = lens.vector
        norm = math.sqrt(L[0]**2 + L[1]**2)
        assert abs(norm - 1.0) < 0.01  # Should be unit vector (approx due to epsilon)

    def test_accumulate_with_zero_gamma(self):
        """With gamma=0, old trace is forgotten."""
        lens = TraceFieldLens()
        lens.trace = [10.0, 10.0]
        lens.set_gamma(0.0)
        lens.accumulate([[1.0, 2.0]])
        assert lens.trace == [1.0, 2.0]

    def test_accumulate_with_full_gamma(self):
        """With gamma=1, old trace is preserved."""
        lens = TraceFieldLens()
        lens.trace = [1.0, 1.0]
        lens.set_gamma(1.0)
        lens.accumulate([[0.5, 0.5]])
        assert lens.trace == [1.5, 1.5]

    def test_accumulate_multiple_traces(self):
        """Multiple traces are summed."""
        lens = TraceFieldLens()
        lens.set_gamma(0.0)
        lens.accumulate([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        assert lens.trace == [1.5, 1.5]


# =============================================================================
# Tests for KuramotoExpert (V2)
# =============================================================================

class TestKuramotoExpert:
    """Tests for experts with Kuramoto phase dynamics."""

    def test_is_subclass_of_harness(self):
        """KuramotoExpert is a subclass of ExpertHarness."""
        assert issubclass(KuramotoExpert, ExpertHarness)

    def test_phase_initialised_randomly(self):
        """Phase is initialised in [0, 2π)."""
        random.seed(42)
        expert = KuramotoExpert()
        assert 0 <= expert.phase < 2 * math.pi

    def test_tick_phase_evolves(self):
        """tick_phase() evolves the phase."""
        random.seed(42)
        expert = KuramotoExpert(omega=0.1, noise_scale=0.0)
        initial_phase = expert.phase
        expert.tick_phase()
        expected = (initial_phase + 0.1) % (2 * math.pi)
        assert abs(expert.phase - expected) < 0.001

    def test_sense_returns_required_keys(self):
        """sense() returns dict with all required keys."""
        expert = KuramotoExpert()
        lens = TraceFieldLens()
        lens.trace = [1.0, 0.0]
        result = expert.sense(lens)

        assert "gain" in result
        assert "tilt" in result
        assert "stability" in result
        assert "out_of_tolerance" in result
        assert "phase" in result
        assert "trace" in result

    def test_sense_trace_is_2d_vector(self):
        """Emitted trace is a 2D vector."""
        expert = KuramotoExpert()
        lens = TraceFieldLens()
        lens.trace = [1.0, 0.0]
        result = expert.sense(lens)

        assert len(result["trace"]) == 2
        assert isinstance(result["trace"][0], float)
        assert isinstance(result["trace"][1], float)

    def test_sense_with_none_falls_back(self):
        """sense(None) returns parent's fixed values."""
        expert = KuramotoExpert()
        result = expert.sense(None)
        assert result["gain"] == 1.0
        assert result["stability"] == 1.0


# =============================================================================
# Tests for compute_kuramoto_R
# =============================================================================

class TestKuramotoR:
    """Tests for Kuramoto order parameter computation."""

    def test_empty_phases_returns_zero(self):
        """Empty phase list returns R=0."""
        R, psi = compute_kuramoto_R([])
        assert R == 0.0

    def test_single_phase_returns_one(self):
        """Single phase gives R=1 (perfect coherence)."""
        R, psi = compute_kuramoto_R([0.5])
        assert abs(R - 1.0) < 0.001

    def test_identical_phases_returns_one(self):
        """Identical phases give R=1."""
        R, psi = compute_kuramoto_R([1.0, 1.0, 1.0, 1.0])
        assert abs(R - 1.0) < 0.001

    def test_opposite_phases_returns_zero(self):
        """Opposite phases cancel out."""
        R, psi = compute_kuramoto_R([0.0, math.pi])
        assert abs(R) < 0.001

    def test_uniform_phases_returns_zero(self):
        """Uniformly distributed phases give R≈0."""
        phases = [i * 2 * math.pi / 4 for i in range(4)]
        R, psi = compute_kuramoto_R(phases)
        assert abs(R) < 0.001

    def test_R_in_unit_interval(self):
        """R is always in [0, 1]."""
        random.seed(42)
        for _ in range(20):
            phases = [random.uniform(0, 2 * math.pi) for _ in range(5)]
            R, psi = compute_kuramoto_R(phases)
            assert 0.0 <= R <= 1.0


# =============================================================================
# Tests for sigmoid
# =============================================================================

class TestSigmoid:
    """Tests for logistic sigmoid function."""

    def test_sigmoid_at_zero(self):
        """σ(0) = 0.5."""
        assert abs(sigmoid(0) - 0.5) < 0.001

    def test_sigmoid_large_positive(self):
        """σ(large) ≈ 1."""
        assert sigmoid(10) > 0.999
        assert sigmoid(100) > 0.9999

    def test_sigmoid_large_negative(self):
        """σ(-large) ≈ 0."""
        assert sigmoid(-10) < 0.001
        assert sigmoid(-100) < 0.0001

    def test_sigmoid_handles_extreme_values(self):
        """Sigmoid doesn't overflow on extreme inputs."""
        assert sigmoid(1000) == 1.0
        assert sigmoid(-1000) == 0.0


# =============================================================================
# Tests for GatingController (V2)
# =============================================================================

class TestGatingController:
    """Tests for the decay-gating controller."""

    def test_initial_clocks_zero(self):
        """Clocks start at zero."""
        ctrl = GatingController()
        assert ctrl.state.fast_clock == 0
        assert ctrl.state.micro_clock == 0
        assert ctrl.state.macro_clock == 0

    def test_compute_gamma_at_R0(self):
        """At R=R0, gamma is midpoint between min and max."""
        ctrl = GatingController(gamma_min=0.7, gamma_max=0.99, R0=0.5, beta=10.0)
        gamma = ctrl.compute_gamma(0.5)
        expected = 0.7 + (0.99 - 0.7) * 0.5  # sigmoid(0) = 0.5
        assert abs(gamma - expected) < 0.001

    def test_compute_gamma_high_R(self):
        """High R gives gamma near gamma_max."""
        ctrl = GatingController(gamma_min=0.7, gamma_max=0.99, R0=0.5, beta=10.0)
        gamma = ctrl.compute_gamma(1.0)
        assert gamma > 0.98

    def test_compute_gamma_low_R(self):
        """Low R gives gamma near gamma_min."""
        ctrl = GatingController(gamma_min=0.7, gamma_max=0.99, R0=0.5, beta=10.0)
        gamma = ctrl.compute_gamma(0.0)
        assert gamma < 0.72

    def test_tick_increments_fast_clock(self):
        """tick() increments fast clock."""
        ctrl = GatingController(micro_period=5, macro_period=20)
        experts = [KuramotoExpert() for _ in range(3)]
        lens = TraceFieldLens()

        for i in range(5):
            ctrl.tick(experts, lens)
            assert ctrl.state.fast_clock == i + 1

    def test_micro_clock_increments_at_period(self):
        """Micro clock increments at micro_period boundaries."""
        ctrl = GatingController(micro_period=5, macro_period=20)
        experts = [KuramotoExpert() for _ in range(3)]
        lens = TraceFieldLens()

        for _ in range(4):
            ctrl.tick(experts, lens)
        assert ctrl.state.micro_clock == 0

        ctrl.tick(experts, lens)  # 5th tick
        assert ctrl.state.micro_clock == 1

    def test_macro_clock_increments_at_period(self):
        """Macro clock increments at macro_period boundaries."""
        ctrl = GatingController(micro_period=5, macro_period=20)
        experts = [KuramotoExpert() for _ in range(3)]
        lens = TraceFieldLens()

        for _ in range(19):
            ctrl.tick(experts, lens)
        assert ctrl.state.macro_clock == 0

        ctrl.tick(experts, lens)  # 20th tick
        assert ctrl.state.macro_clock == 1

    def test_tick_returns_info_dict(self):
        """tick() returns dict with expected keys."""
        ctrl = GatingController()
        experts = [KuramotoExpert() for _ in range(3)]
        lens = TraceFieldLens()

        info = ctrl.tick(experts, lens)

        assert "fast" in info
        assert "micro" in info
        assert "macro" in info
        assert "R" in info
        assert "gamma" in info
        assert "signals" in info


# =============================================================================
# Tests for V1 compatibility
# =============================================================================

class TestEnsembleCoherence:
    """Tests for V1 compute_ensemble_coherence function."""

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
