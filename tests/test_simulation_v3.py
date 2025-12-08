"""Tests for Chronovisor V3: Per-Expert Temperament Simulation."""

import math
import random
from chronovisor.simulation_v3 import (
    compute_kuramoto_R_and_psi,
    sigmoid,
    alignment,
    AdaptiveExpert,
    TemperamentController,
)


# =============================================================================
# Tests for compute_kuramoto_R_and_psi
# =============================================================================

class TestKuramotoRAndPsi:
    """Tests for Kuramoto order parameter computation."""

    def test_empty_phases_returns_zero(self):
        """Empty phase list returns R=0."""
        R, psi = compute_kuramoto_R_and_psi([])
        assert R == 0.0

    def test_single_phase_returns_one(self):
        """Single phase gives R=1."""
        R, psi = compute_kuramoto_R_and_psi([0.5])
        assert abs(R - 1.0) < 0.001

    def test_identical_phases_returns_one(self):
        """Identical phases give R=1."""
        R, psi = compute_kuramoto_R_and_psi([1.0, 1.0, 1.0])
        assert abs(R - 1.0) < 0.001

    def test_opposite_phases_returns_zero(self):
        """Opposite phases cancel out."""
        R, psi = compute_kuramoto_R_and_psi([0.0, math.pi])
        assert abs(R) < 0.001

    def test_psi_is_mean_phase_for_identical(self):
        """Psi equals the common phase when all identical."""
        phi = 1.5
        R, psi = compute_kuramoto_R_and_psi([phi, phi, phi])
        assert abs(psi - phi) < 0.001

    def test_R_in_unit_interval(self):
        """R is always in [0, 1]."""
        random.seed(42)
        for _ in range(20):
            phases = [random.uniform(0, 2 * math.pi) for _ in range(5)]
            R, psi = compute_kuramoto_R_and_psi(phases)
            assert 0.0 <= R <= 1.0


# =============================================================================
# Tests for alignment
# =============================================================================

class TestAlignment:
    """Tests for per-expert alignment function."""

    def test_in_phase_returns_one(self):
        """Expert in phase with ensemble returns +1."""
        a = alignment(0.5, 0.5)
        assert abs(a - 1.0) < 0.001

    def test_anti_phase_returns_negative_one(self):
        """Expert anti-phase returns -1."""
        a = alignment(0.0, math.pi)
        assert abs(a - (-1.0)) < 0.001

    def test_orthogonal_returns_zero(self):
        """Expert orthogonal to mean phase returns 0."""
        a = alignment(0.0, math.pi / 2)
        assert abs(a) < 0.001

    def test_alignment_in_range(self):
        """Alignment is always in [-1, 1]."""
        random.seed(42)
        for _ in range(20):
            phi_k = random.uniform(0, 2 * math.pi)
            psi = random.uniform(0, 2 * math.pi)
            a = alignment(phi_k, psi)
            assert -1.0 <= a <= 1.0


# =============================================================================
# Tests for sigmoid
# =============================================================================

class TestSigmoid:
    """Tests for numerically stable sigmoid."""

    def test_sigmoid_at_zero(self):
        """σ(0) = 0.5."""
        assert abs(sigmoid(0) - 0.5) < 0.001

    def test_sigmoid_large_positive(self):
        """σ(large) ≈ 1."""
        assert sigmoid(10) > 0.999

    def test_sigmoid_large_negative(self):
        """σ(-large) ≈ 0."""
        assert sigmoid(-10) < 0.001

    def test_sigmoid_handles_extreme_values(self):
        """Sigmoid doesn't overflow on extreme inputs."""
        assert 0.0 <= sigmoid(1000) <= 1.0
        assert 0.0 <= sigmoid(-1000) <= 1.0


# =============================================================================
# Tests for AdaptiveExpert
# =============================================================================

class TestAdaptiveExpert:
    """Tests for experts with per-expert temperament dynamics."""

    def test_initial_state(self):
        """Expert initialises with given values."""
        e = AdaptiveExpert(name="Test", phi=1.0, omega=0.1)
        assert e.name == "Test"
        assert e.phi == 1.0
        assert e.omega == 0.1
        assert e.theta == 0.0
        assert e.v == 0.0
        assert e.lambd == 0.05

    def test_tick_fast_updates_phase(self):
        """tick_fast updates phase by omega (plus noise)."""
        random.seed(42)
        e = AdaptiveExpert(name="Test", phi=0.0, omega=0.1)
        e.tick_fast(psi=0.0, noise_phi_std=0.0, noise_v_std=0.0)
        assert abs(e.phi - 0.1) < 0.001

    def test_tick_fast_returns_required_keys(self):
        """tick_fast returns dict with all required keys."""
        e = AdaptiveExpert(name="Test", phi=0.0, omega=0.1)
        result = e.tick_fast(psi=0.0)

        assert "name" in result
        assert "phi" in result
        assert "align" in result
        assert "tilt" in result
        assert "velocity" in result
        assert "gain" in result
        assert "stability" in result

    def test_tick_fast_alignment_affects_velocity(self):
        """Positive alignment accelerates velocity."""
        random.seed(42)
        e = AdaptiveExpert(name="Test", phi=0.0, omega=0.0, lambd=0.1)
        e.tick_fast(psi=0.0, noise_phi_std=0.0, noise_v_std=0.0)
        # Alignment is cos(0-0) = 1, so v += lambd * 1 = 0.1
        assert e.v > 0

    def test_tick_fast_accumulates_alignment_stats(self):
        """tick_fast accumulates alignment for micro/macro updates."""
        e = AdaptiveExpert(name="Test", phi=0.0, omega=0.0)
        e.tick_fast(psi=0.0)
        assert e.micro_align_count == 1
        assert e.macro_align_count == 1

    def test_do_micro_update_increases_lambda_for_positive_alignment(self):
        """Micro update increases lambda when alignment is positive."""
        e = AdaptiveExpert(name="Test", phi=0.0, omega=0.0, lambd=0.05)
        # Simulate positive alignment
        e.micro_align_sum = 0.8
        e.micro_align_count = 1

        old_lambda = e.lambd
        e.do_micro_update(lambda_min=0.01, lambda_max=0.2, eta_lambda=0.5)

        assert e.lambd > old_lambda

    def test_do_micro_update_decreases_lambda_for_negative_alignment(self):
        """Micro update decreases lambda when alignment is negative."""
        e = AdaptiveExpert(name="Test", phi=0.0, omega=0.0, lambd=0.15)
        # Simulate negative alignment
        e.micro_align_sum = -0.5
        e.micro_align_count = 1

        old_lambda = e.lambd
        e.do_micro_update(lambda_min=0.01, lambda_max=0.2, eta_lambda=0.5)

        # Negative alignment clips to 0, so target is lambda_min
        assert e.lambd < old_lambda

    def test_do_micro_update_resets_counters(self):
        """Micro update resets alignment counters."""
        e = AdaptiveExpert(name="Test", phi=0.0, omega=0.0)
        e.micro_align_sum = 1.0
        e.micro_align_count = 5
        e.do_micro_update()

        assert e.micro_align_sum == 0.0
        assert e.micro_align_count == 0

    def test_do_macro_update_updates_reliability(self):
        """Macro update adjusts reliability score s."""
        e = AdaptiveExpert(name="Test", phi=0.0, omega=0.0, s=0.0)
        e.macro_align_sum = 0.5
        e.macro_align_count = 1

        e.do_macro_update(eta_s=0.5)

        assert e.s > 0.0

    def test_do_macro_update_returns_weight(self):
        """Macro update returns trust weight."""
        e = AdaptiveExpert(name="Test", phi=0.0, omega=0.0, s=0.0)
        e.macro_align_sum = 0.5
        e.macro_align_count = 1

        w = e.do_macro_update()

        assert 0.0 <= w <= 1.0


# =============================================================================
# Tests for TemperamentController
# =============================================================================

class TestTemperamentController:
    """Tests for the temperament controller."""

    def setup_method(self):
        """Create test experts and controller."""
        random.seed(42)
        self.experts = [
            AdaptiveExpert(name="A", phi=0.0, omega=0.1),
            AdaptiveExpert(name="B", phi=1.0, omega=0.1),
            AdaptiveExpert(name="C", phi=2.0, omega=0.1),
        ]
        self.controller = TemperamentController(
            experts=self.experts,
            micro_period=5,
            macro_period=2,
        )

    def test_initial_clocks_zero(self):
        """Clocks start at zero."""
        assert self.controller.fast_clock == 0
        assert self.controller.micro_clock == 0
        assert self.controller.macro_clock == 0

    def test_tick_increments_fast_clock(self):
        """tick() increments fast clock."""
        for i in range(5):
            self.controller.tick()
            assert self.controller.fast_clock == i + 1

    def test_micro_clock_increments_at_period(self):
        """Micro clock increments at micro_period boundaries."""
        for _ in range(4):
            self.controller.tick()
        assert self.controller.micro_clock == 0

        self.controller.tick()  # 5th tick
        assert self.controller.micro_clock == 1

    def test_macro_clock_increments_at_period(self):
        """Macro clock increments at macro_period micro boundaries."""
        # micro_period=5, macro_period=2 (in micro ticks)
        # So macro triggers at 10 fast ticks
        for _ in range(9):
            self.controller.tick()
        assert self.controller.macro_clock == 0

        self.controller.tick()  # 10th tick
        assert self.controller.macro_clock == 1

    def test_tick_returns_info_dict(self):
        """tick() returns dict with expected keys."""
        info = self.controller.tick()

        assert "fast_clock" in info
        assert "micro_clock" in info
        assert "macro_clock" in info
        assert "R" in info
        assert "expert_signals" in info
        assert "micro_event" in info
        assert "macro_event" in info

    def test_tick_computes_kuramoto_R(self):
        """tick() computes Kuramoto R from expert phases."""
        info = self.controller.tick()
        assert 0.0 <= info["R"] <= 1.0

    def test_micro_event_true_at_boundary(self):
        """micro_event is True at micro boundaries."""
        for _ in range(4):
            info = self.controller.tick()
            assert info["micro_event"] is False

        info = self.controller.tick()  # 5th tick
        assert info["micro_event"] is True

    def test_macro_event_returns_weights(self):
        """macro_event includes trust weights."""
        # Run until macro event
        for _ in range(10):
            info = self.controller.tick()

        assert info["macro_event"] is True
        assert info["weights"] is not None
        assert len(info["weights"]) == 3  # Three experts
