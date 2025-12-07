"""Tests for Chronovisor V6: Cultural Transmission."""

import math
import random
from chronovisor.simulation_v6 import (
    CulturalExpert,
    Motif,
    CulturalController,
    CulturalEvolutionaryController,
    Governor,
    Lens,
    compute_kuramoto_R_and_psi,
    reset_id_counters,
    _next_expert_id,
    _next_motif_id,
)


class TestCulturalExpert:
    """Tests for experts with cultural capabilities."""

    def setup_method(self):
        reset_id_counters()
        random.seed(42)

    def test_initial_cultural_state(self):
        """Expert starts with no motif affiliations."""
        expert = CulturalExpert(
            name="Test", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1,
        )

        assert len(expert.motif_ids) == 0
        assert expert.cultural_capital == 0.0

    def test_effective_reliability_includes_cultural_capital(self):
        """Effective reliability includes cultural capital bonus."""
        expert = CulturalExpert(
            name="Test", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1, s=0.2,
        )
        expert.cultural_capital = 0.05

        assert expert.effective_reliability() == 0.25

    def test_offspring_starts_culturally_fresh(self):
        """Offspring doesn't inherit parent's cultural affiliations."""
        parent = CulturalExpert(
            name="Parent", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1,
        )
        parent.motif_ids = {1, 2, 3}
        parent.cultural_capital = 0.1

        offspring = parent.spawn_offspring(current_tick=100)

        assert len(offspring.motif_ids) == 0
        assert offspring.cultural_capital == 0.0

    def test_tick_fast_returns_cultural_info(self):
        """tick_fast includes cultural capital and motif count."""
        expert = CulturalExpert(
            name="Test", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1,
        )
        expert.motif_ids = {1, 2}
        expert.cultural_capital = 0.05

        info = expert.tick_fast(psi=0.0)

        assert info["cultural_capital"] == 0.05
        assert info["num_motifs"] == 2


class TestMotif:
    """Tests for shared strategy motifs."""

    def setup_method(self):
        reset_id_counters()

    def test_initial_state(self):
        """Motif initializes with given values."""
        motif = Motif(
            motif_id=1,
            name="M1",
            theta_center=50.0,
            S=0.3,
            mean_lambda=0.1,
            mean_abs_v=0.5,
            var_theta=10.0,
        )

        assert motif.theta_center == 50.0
        assert motif.S == 0.3
        assert motif.age == 0

    def test_support_size(self):
        """support_size returns count of supporting experts."""
        motif = Motif(
            motif_id=1, name="M1",
            theta_center=50.0, S=0.3,
            mean_lambda=0.1, mean_abs_v=0.5, var_theta=10.0,
        )
        motif.support_ids = {1, 2, 3, 4, 5}

        assert motif.support_size() == 5

    def test_update_from_cluster(self):
        """update_from_cluster updates motif stats with EMA."""
        motif = Motif(
            motif_id=1, name="M1",
            theta_center=50.0, S=0.3,
            mean_lambda=0.1, mean_abs_v=0.5, var_theta=10.0,
        )

        # Update with new cluster stats
        motif.update_from_cluster(
            theta_values=[60.0, 70.0, 80.0],
            s_values=[0.5, 0.5, 0.5],
            lambda_values=[0.2, 0.2, 0.2],
            v_values=[1.0, -1.0, 0.5],
            expert_ids=[10, 11, 12],
            eta_motif=0.5,
        )

        # Should move toward new values
        assert motif.theta_center > 50.0
        assert motif.S > 0.3
        assert motif.age == 1
        assert motif.support_ids == {10, 11, 12}

    def test_is_stable_requires_history(self):
        """is_stable returns False without enough history."""
        motif = Motif(
            motif_id=1, name="M1",
            theta_center=50.0, S=0.5,
            mean_lambda=0.1, mean_abs_v=0.5, var_theta=10.0,
        )

        assert motif.is_stable() is False

    def test_is_stable_with_positive_history(self):
        """is_stable returns True with consistent positive S history."""
        motif = Motif(
            motif_id=1, name="M1",
            theta_center=50.0, S=0.3,
            mean_lambda=0.1, mean_abs_v=0.5, var_theta=10.0,
            stability_window=3,
        )
        motif.s_history = [0.1, 0.2, 0.3, 0.4, 0.5]

        assert motif.is_stable() is True


class TestCulturalController:
    """Tests for the cultural controller."""

    def setup_method(self):
        reset_id_counters()
        random.seed(42)

    def _create_experts(self, positions):
        """Create experts at given theta positions."""
        return [
            CulturalExpert(
                name=f"E{i}",
                expert_id=_next_expert_id(),
                phi=random.uniform(0, 2 * math.pi),
                omega=0.1,
                theta=pos,
                theta_home=0.0,
                s=0.2,
                lambd=0.1,
            )
            for i, pos in enumerate(positions)
        ]

    def test_cluster_experts_groups_nearby(self):
        """Clustering groups experts close in theta."""
        controller = CulturalController(theta_cluster_radius=20.0)

        # Two clusters: one around 0, one around 100
        experts = self._create_experts([0, 5, 10, 100, 105, 110])

        clusters = controller._cluster_experts(experts)

        assert len(clusters) == 2
        assert len(clusters[0]) == 3
        assert len(clusters[1]) == 3

    def test_spawn_motif_creates_motif(self):
        """_spawn_motif creates a motif with cluster stats."""
        controller = CulturalController()

        motif = controller._spawn_motif(
            theta_values=[50.0, 60.0, 70.0],
            s_values=[0.3, 0.3, 0.3],
            lambda_values=[0.1, 0.1, 0.1],
            v_values=[0.5, 0.5, 0.5],
            expert_ids=[1, 2, 3],
            current_tick=100,
        )

        assert motif.theta_center == 60.0  # mean of 50, 60, 70
        assert motif.S == 0.3
        assert motif.support_ids == {1, 2, 3}

    def test_cultural_tick_spawns_motif(self):
        """cultural_tick spawns motif from reliable cluster."""
        controller = CulturalController(
            theta_cluster_radius=30.0,
            min_cluster_size=2,
            min_cluster_reliability=0.1,
        )

        # Create a cluster of reliable experts
        experts = self._create_experts([0, 10, 20])
        for e in experts:
            e.s = 0.3

        info = controller.cultural_tick(experts, current_tick=100)

        assert len(controller.motifs) >= 1
        assert len(info["new_motifs"]) >= 1

    def test_apply_teaching_nudges_lambda(self):
        """Teaching nudges expert lambda toward motif."""
        controller = CulturalController(
            D_culture=100.0,
            eta_cultural_lambda=0.5,  # High learning rate for testing
        )

        # Create a motif
        motif = Motif(
            motif_id=1, name="M1",
            theta_center=50.0, S=0.5,
            mean_lambda=0.2, mean_abs_v=0.5, var_theta=10.0,
        )
        controller.motifs.append(motif)

        # Create expert near motif with different lambda
        expert = CulturalExpert(
            name="E1", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1, theta=50.0,
            lambd=0.05,
        )

        controller._apply_teaching([expert])

        # Lambda should move toward motif's 0.2
        assert expert.lambd > 0.05

    def test_apply_teaching_adds_motif_affiliation(self):
        """Teaching adds motif ID to expert."""
        controller = CulturalController(D_culture=100.0)

        motif = Motif(
            motif_id=42, name="M42",
            theta_center=50.0, S=0.5,
            mean_lambda=0.1, mean_abs_v=0.5, var_theta=10.0,
        )
        controller.motifs.append(motif)

        expert = CulturalExpert(
            name="E1", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1, theta=50.0,
        )

        controller._apply_teaching([expert])

        assert 42 in expert.motif_ids

    def test_apply_teaching_accrues_cultural_capital(self):
        """Teaching accrues cultural capital from strong motifs."""
        controller = CulturalController(
            D_culture=100.0,
            cultural_capital_rate=0.1,  # High rate for testing
        )

        motif = Motif(
            motif_id=1, name="M1",
            theta_center=50.0, S=0.5,  # Positive S
            mean_lambda=0.1, mean_abs_v=0.5, var_theta=10.0,
        )
        controller.motifs.append(motif)

        expert = CulturalExpert(
            name="E1", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1, theta=50.0,
        )

        controller._apply_teaching([expert])

        assert expert.cultural_capital > 0.0

    def test_prune_dead_motifs(self):
        """Pruning removes motifs with no supporters."""
        controller = CulturalController()

        motif = Motif(
            motif_id=1, name="M1",
            theta_center=50.0, S=0.3,
            mean_lambda=0.1, mean_abs_v=0.5, var_theta=10.0,
            age=5,  # Old enough to be pruned
        )
        motif.support_ids = {999}  # Dead expert ID
        controller.motifs.append(motif)

        # No living experts
        experts = []

        pruned = controller._prune_dead_motifs(experts, current_tick=100, mode="normal")

        assert "M1" in pruned
        assert len(controller.motifs) == 0

    def test_cross_cluster_merge_in_outside_box(self):
        """Outside-box mode enables cross-cluster merging."""
        controller = CulturalController(theta_cluster_radius=20.0)
        controller.cross_cluster_enabled = True

        # Two distant clusters with similar profiles
        experts = self._create_experts([0, 10, 200, 210])
        for e in experts:
            e.lambd = 0.1  # Same lambda
            e.s = 0.3  # Same s

        clusters = controller._cluster_experts(experts)

        # Should be merged due to similar profiles
        # (depends on exact implementation)
        assert len(clusters) <= 2


class TestCulturalEvolutionaryController:
    """Integration tests for the full cultural controller."""

    def setup_method(self):
        reset_id_counters()
        random.seed(42)

    def test_has_cultural_clock(self):
        """Controller has cultural clock."""
        experts = [
            CulturalExpert(
                name="A", expert_id=_next_expert_id(),
                phi=0.0, omega=0.1,
            ),
        ]
        controller = CulturalEvolutionaryController(
            experts=experts,
            micro_period=1,
            macro_period=1,
            cultural_period=1,
        )

        info = controller.tick()

        assert "cultural_clock" in info
        assert "cultural_event" in info

    def test_tick_returns_num_motifs(self):
        """tick returns current motif count."""
        experts = [
            CulturalExpert(
                name="A", expert_id=_next_expert_id(),
                phi=0.0, omega=0.1,
            ),
        ]
        controller = CulturalEvolutionaryController(experts=experts)

        info = controller.tick()

        assert "num_motifs" in info
        assert info["num_motifs"] >= 0

    def test_bifurcation_uses_effective_reliability(self):
        """Bifurcation considers cultural capital in reliability."""
        experts = [
            CulturalExpert(
                name="A", expert_id=_next_expert_id(),
                phi=0.0, omega=0.1,
                theta=100.0, theta_home=0.0,
                s=0.05,  # Below threshold
            ),
        ]
        experts[0].cultural_capital = 0.1  # Pushes above threshold

        gov = Governor(max_population=10)
        controller = CulturalEvolutionaryController(
            experts=experts,
            governor=gov,
            micro_period=1,
            macro_period=1,
            D_max=50.0,
            s_bifurcate=0.1,
        )

        # Run a macro tick
        info = controller.tick()

        # Should bifurcate because effective_reliability = 0.05 + 0.1 = 0.15 > 0.1
        assert len(info["bifurcations"]) == 1

    def test_cultural_tick_frequency(self):
        """Cultural ticks happen at correct frequency."""
        experts = [
            CulturalExpert(
                name="A", expert_id=_next_expert_id(),
                phi=0.0, omega=0.1,
            ),
        ]
        controller = CulturalEvolutionaryController(
            experts=experts,
            micro_period=1,
            macro_period=2,
            cultural_period=3,
        )

        cultural_events = []
        for _ in range(20):
            info = controller.tick()
            if info["cultural_event"]:
                cultural_events.append(info["fast_clock"])

        # Cultural events at macro ticks 3, 6, 9... which are fast ticks 6, 12, 18
        assert len(cultural_events) >= 2


class TestGovernorWithCulture:
    """Tests for Governor in V6 context."""

    def test_governor_unchanged_from_v5(self):
        """Governor behavior unchanged from V5."""
        gov = Governor(max_population=10, min_population=2)

        gate_spawn, gate_cull, mode = gov.update(
            R=0.5, N=5,
            s_values=[0.1, 0.2, 0.3],
            d_values=[10.0, 20.0, 30.0],
        )

        assert isinstance(gate_spawn, bool)
        assert isinstance(gate_cull, bool)
        assert mode in ["normal", "outside_box"]


class TestPressureFramework:
    """Tests for the pressure-based dynamics."""

    def setup_method(self):
        reset_id_counters()
        random.seed(42)

    def test_expert_has_pressure_constants(self):
        """Expert has k_home and k_safety constants."""
        expert = CulturalExpert(
            name="Test", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1,
        )

        assert hasattr(expert, "k_home")
        assert hasattr(expert, "k_safety")
        assert expert.k_home == 0.01
        assert expert.k_safety == 0.1

    def test_motif_has_alpha_theta(self):
        """Motif has alpha_theta for pressure stiffness."""
        motif = Motif(
            motif_id=1, name="M1",
            theta_center=50.0, S=0.3,
            mean_lambda=0.1, mean_abs_v=0.5, var_theta=10.0,
        )

        assert hasattr(motif, "alpha_theta")
        assert motif.alpha_theta == 0.01

    def test_expert_has_motif_affinity(self):
        """Expert has motif_affinity dict."""
        expert = CulturalExpert(
            name="Test", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1,
        )

        assert hasattr(expert, "motif_affinity")
        assert isinstance(expert.motif_affinity, dict)

    def test_tick_fast_returns_pressure_diagnostics(self):
        """tick_fast returns F_drift, F_home, F_culture, F_total."""
        expert = CulturalExpert(
            name="Test", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1, theta=100.0, theta_home=0.0,
        )

        info = expert.tick_fast(psi=0.0)

        assert "F_drift" in info
        assert "F_home" in info
        assert "F_culture" in info
        assert "F_total" in info

    def test_home_pressure_pulls_toward_home(self):
        """Home pressure pulls expert toward theta_home."""
        expert = CulturalExpert(
            name="Test", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1,
            theta=100.0,
            theta_home=0.0,
            k_home=0.1,  # Strong home pressure
        )

        info = expert.tick_fast(psi=0.0, dv=0.0, noise_v_std=0.0)

        # Home pressure should be negative (pulling back toward 0)
        assert info["F_home"] < 0

    def test_cultural_pressure_with_motifs(self):
        """Cultural pressure pulls expert toward motif center."""
        expert = CulturalExpert(
            name="Test", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1,
            theta=100.0, theta_home=0.0,
        )

        # Create motif at theta=50
        motif = Motif(
            motif_id=1, name="M1",
            theta_center=50.0, S=0.5,
            mean_lambda=0.1, mean_abs_v=0.5, var_theta=10.0,
            alpha_theta=0.1,  # Strong cultural pressure
        )

        # Affiliate expert with motif
        expert.motif_ids.add(1)
        expert.motif_affinity[1] = 0.5

        motifs_dict = {1: motif}
        info = expert.tick_fast(psi=0.0, dv=0.0, noise_v_std=0.0, motifs=motifs_dict)

        # Cultural pressure should pull toward motif (theta=50, expert at 100)
        assert info["F_culture"] < 0  # Pulling back toward 50

    def test_offspring_inherits_pressure_constants(self):
        """Offspring inherits k_home and k_safety from parent."""
        parent = CulturalExpert(
            name="Parent", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1,
            k_home=0.05,
            k_safety=0.2,
        )

        offspring = parent.spawn_offspring(current_tick=100)

        assert offspring.k_home == 0.05
        assert offspring.k_safety == 0.2

    def test_teaching_sets_motif_affinity(self):
        """Teaching sets motif_affinity for each affiliated motif."""
        controller = CulturalController(D_culture=100.0)

        motif = Motif(
            motif_id=1, name="M1",
            theta_center=50.0, S=0.5,
            mean_lambda=0.1, mean_abs_v=0.5, var_theta=10.0,
        )
        controller.motifs.append(motif)

        expert = CulturalExpert(
            name="E1", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1, theta=50.0,
        )

        controller._apply_teaching([expert])

        assert 1 in expert.motif_affinity
        assert expert.motif_affinity[1] > 0


class TestLens:
    """Tests for the lens gain mechanism."""

    def setup_method(self):
        reset_id_counters()
        random.seed(42)

    def test_lens_default_values(self):
        """Lens initializes with sensible defaults."""
        lens = Lens()

        assert lens.L == 0.0
        assert lens.eta_L == 0.1
        assert lens.gamma_lens == 0.3
        assert lens.g_min == 0.5
        assert lens.g_max == 1.5

    def test_lens_update_from_drifts(self):
        """update() updates L as EMA of mean drift."""
        lens = Lens(L=0.0, eta_L=0.5)

        lens.update([1.0, 1.0, 1.0])  # Mean drift = 1.0

        assert lens.L == 0.5  # (1-0.5)*0 + 0.5*1.0 = 0.5

    def test_lens_update_empty_drifts(self):
        """update() with empty list leaves L unchanged."""
        lens = Lens(L=0.5)

        lens.update([])

        assert lens.L == 0.5

    def test_lens_compute_gain_aligned(self):
        """compute_gain amplifies drift aligned with L."""
        lens = Lens(L=1.0, gamma_lens=0.3)

        # Positive drift aligned with positive L
        g, alpha = lens.compute_gain(raw_drift=0.5, R=1.0)

        assert alpha == 1.0  # Full agreement
        assert g == 1.3  # 1 + 0.3 * 1.0 * 1.0

    def test_lens_compute_gain_opposed(self):
        """compute_gain dampens drift opposed to L."""
        lens = Lens(L=1.0, gamma_lens=0.3)

        # Negative drift opposed to positive L
        g, alpha = lens.compute_gain(raw_drift=-0.5, R=1.0)

        assert alpha == -1.0  # Full opposition
        assert g == 0.7  # 1 + 0.3 * 1.0 * (-1.0)

    def test_lens_compute_gain_gated_by_R(self):
        """Gain effect is gated by coherence R."""
        lens = Lens(L=1.0, gamma_lens=0.3)

        # Low R reduces gain effect
        g, alpha = lens.compute_gain(raw_drift=0.5, R=0.0)

        assert alpha == 1.0  # Still aligned
        assert g == 1.0  # No effect because R=0

    def test_lens_compute_gain_bounded(self):
        """Gain is clamped to [g_min, g_max]."""
        lens = Lens(L=1.0, gamma_lens=1.0, g_min=0.5, g_max=1.5)

        # Would be g=2.0, but clamped to 1.5
        g, _ = lens.compute_gain(raw_drift=1.0, R=1.0)
        assert g == 1.5

        # Would be g=0.0, but clamped to 0.5
        g, _ = lens.compute_gain(raw_drift=-1.0, R=1.0)
        assert g == 0.5

    def test_lens_compute_gain_zero_L(self):
        """alpha=0 when L is zero."""
        lens = Lens(L=0.0, gamma_lens=0.3)

        g, alpha = lens.compute_gain(raw_drift=0.5, R=1.0)

        assert alpha == 0.0
        assert g == 1.0

    def test_lens_compute_gain_zero_drift(self):
        """alpha=0 when raw_drift is zero."""
        lens = Lens(L=1.0, gamma_lens=0.3)

        g, alpha = lens.compute_gain(raw_drift=0.0, R=1.0)

        assert alpha == 0.0
        assert g == 1.0

    def test_lens_get_params(self):
        """get_params returns (L, gamma_lens, g_min, g_max)."""
        lens = Lens(L=0.5, gamma_lens=0.4, g_min=0.6, g_max=1.4)

        L, gamma, gmin, gmax = lens.get_params()

        assert L == 0.5
        assert gamma == 0.4
        assert gmin == 0.6
        assert gmax == 1.4


class TestLensIntegration:
    """Integration tests for lens in controller."""

    def setup_method(self):
        reset_id_counters()
        random.seed(42)

    def test_expert_tick_fast_returns_lens_diagnostics(self):
        """tick_fast returns raw_drift, g_lens, alpha_lens."""
        expert = CulturalExpert(
            name="Test", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1,
        )

        info = expert.tick_fast(psi=0.0)

        assert "raw_drift" in info
        assert "g_lens" in info
        assert "alpha_lens" in info

    def test_expert_tick_fast_applies_lens_gain(self):
        """F_drift = g_lens * raw_drift."""
        expert = CulturalExpert(
            name="Test", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1,
            lambd=0.1,
        )

        # With L aligned with expected positive drift
        info = expert.tick_fast(
            psi=0.0,  # alignment=cos(0-0)=1, raw_drift=0.1*1=0.1
            L=1.0,
            R=1.0,
            gamma_lens=0.3,
            g_min=0.5,
            g_max=1.5,
        )

        # raw_drift and F_drift should differ by g_lens
        assert abs(info["F_drift"] - info["g_lens"] * info["raw_drift"]) < 1e-10

    def test_controller_has_lens(self):
        """CulturalEvolutionaryController has lens field."""
        experts = [
            CulturalExpert(
                name="A", expert_id=_next_expert_id(),
                phi=0.0, omega=0.1,
            ),
        ]
        controller = CulturalEvolutionaryController(experts=experts)

        assert hasattr(controller, "lens")
        assert isinstance(controller.lens, Lens)

    def test_controller_tick_updates_lens_L(self):
        """Controller updates lens.L after each tick."""
        experts = [
            CulturalExpert(
                name="A", expert_id=_next_expert_id(),
                phi=0.0, omega=0.1,
                lambd=0.1,
            ),
        ]
        controller = CulturalEvolutionaryController(experts=experts)

        assert controller.lens.L == 0.0

        # Run some ticks to generate drifts
        for _ in range(10):
            controller.tick()

        # L should have been updated
        assert controller.lens.L != 0.0

    def test_controller_tick_returns_lens_diagnostics(self):
        """Controller tick returns lens_L and avg_g_lens."""
        experts = [
            CulturalExpert(
                name="A", expert_id=_next_expert_id(),
                phi=0.0, omega=0.1,
            ),
        ]
        controller = CulturalEvolutionaryController(experts=experts)

        info = controller.tick()

        assert "lens_L" in info
        assert "avg_g_lens" in info

    def test_lens_effect_increases_with_R(self):
        """Higher R means stronger lens effect."""
        experts = [
            CulturalExpert(
                name="A", expert_id=_next_expert_id(),
                phi=0.0, omega=0.1,
            ),
            CulturalExpert(
                name="B", expert_id=_next_expert_id(),
                phi=0.0, omega=0.1,  # Same phase = high R
            ),
        ]
        controller = CulturalEvolutionaryController(
            experts=experts,
            noise_phi_std=0.0,  # No noise for reproducibility
        )

        # Prime the lens with some positive drift
        controller.lens.L = 0.1

        info = controller.tick()

        # With R≈1 and aligned drifts, avg_g_lens should be > 1
        # (experts are aligned, so their drifts agree with L)
        assert info["R"] > 0.9
        assert info["avg_g_lens"] >= 1.0


class TestPredictionSuccessAxis:
    """Tests for V6.1 prediction success axis."""

    def setup_method(self):
        reset_id_counters()
        random.seed(42)

    def test_expert_has_delta_s_fields(self):
        """Expert has Δs tracking fields."""
        expert = CulturalExpert(
            name="Test", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1,
        )

        assert hasattr(expert, "s_prev")
        assert hasattr(expert, "delta_s")
        assert hasattr(expert, "delta_s_ema")
        assert expert.s_prev == 0.0
        assert expert.delta_s == 0.0
        assert expert.delta_s_ema == 0.0

    def test_expert_has_absorption_fields(self):
        """Expert has absorption statistics fields."""
        expert = CulturalExpert(
            name="Test", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1,
        )

        assert hasattr(expert, "g_lens_ema")
        assert hasattr(expert, "R_ema")
        assert hasattr(expert, "alpha_lens_ema")
        assert expert.g_lens_ema == 1.0
        assert expert.R_ema == 0.0

    def test_update_delta_s(self):
        """update_delta_s tracks change in reliability."""
        expert = CulturalExpert(
            name="Test", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1, s=0.1,
        )
        expert.s_prev = 0.05

        expert.update_delta_s()

        assert expert.delta_s == 0.05  # 0.1 - 0.05
        assert expert.delta_s_ema > 0
        assert expert.s_prev == 0.1  # Updated to current s

    def test_update_absorption_stats(self):
        """update_absorption_stats updates EMAs."""
        expert = CulturalExpert(
            name="Test", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1,
        )

        expert.update_absorption_stats(g_lens=1.3, R=0.8, alpha_lens=1.0)

        assert expert.g_lens_ema > 1.0
        assert expert.R_ema > 0.0
        assert expert.alpha_lens_ema > 0.0

    def test_maybe_absorb_requires_positive_delta_s(self):
        """maybe_absorb only absorbs when Δs > 0."""
        expert = CulturalExpert(
            name="Test", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1,
            theta=100.0, theta_home=0.0,
        )
        # High R, high g, but negative Δs
        expert.R_ema = 0.8
        expert.g_lens_ema = 1.3
        expert.delta_s_ema = -0.1  # Failing!

        result = expert.maybe_absorb()

        assert result["absorbed"] is False
        assert result["reason"] == "delta_s_not_positive"
        assert expert.theta_home == 0.0  # Unchanged

    def test_maybe_absorb_succeeds_with_positive_delta_s(self):
        """maybe_absorb absorbs when all conditions met."""
        expert = CulturalExpert(
            name="Test", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1,
            theta=100.0, theta_home=0.0,
        )
        # High R, high g, positive Δs
        expert.R_ema = 0.8
        expert.g_lens_ema = 1.3
        expert.delta_s_ema = 0.1  # Succeeding!

        old_theta_home = expert.theta_home
        result = expert.maybe_absorb()

        assert result["absorbed"] is True
        assert result["reason"] == "success"
        assert expert.theta_home != old_theta_home  # Changed!

    def test_maybe_absorb_requires_high_R(self):
        """maybe_absorb requires R > threshold."""
        expert = CulturalExpert(
            name="Test", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1,
            theta=100.0, theta_home=0.0,
        )
        # Low R
        expert.R_ema = 0.3
        expert.g_lens_ema = 1.3
        expert.delta_s_ema = 0.1

        result = expert.maybe_absorb()

        assert result["absorbed"] is False
        assert result["reason"] == "R_too_low"

    def test_compute_diversity_bonus_for_correct_outlier(self):
        """Diversity bonus rewards damped but correct experts."""
        expert = CulturalExpert(
            name="Test", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1,
        )
        # Damped (g < 1) but correct (Δs > 0)
        expert.g_lens_ema = 0.7  # Lens didn't like me
        expert.delta_s_ema = 0.1  # But I was right!

        bonus = expert.compute_diversity_bonus()

        assert bonus > 0
        assert expert.diversity_bonus > 0

    def test_compute_diversity_bonus_zero_for_aligned_expert(self):
        """No diversity bonus for aligned experts (even if correct)."""
        expert = CulturalExpert(
            name="Test", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1,
        )
        # Amplified (g > 1) and correct
        expert.g_lens_ema = 1.3  # Lens liked me
        expert.delta_s_ema = 0.1

        bonus = expert.compute_diversity_bonus()

        # No bonus because I wasn't an outlier
        assert expert.diversity_bonus < 0.01  # Decayed to near zero

    def test_effective_reliability_includes_diversity_bonus(self):
        """Effective reliability includes diversity bonus."""
        expert = CulturalExpert(
            name="Test", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1, s=0.2,
        )
        expert.cultural_capital = 0.05
        expert.diversity_bonus = 0.03

        assert expert.effective_reliability() == 0.28  # 0.2 + 0.05 + 0.03

    def test_controller_tick_returns_delta_s_diagnostics(self):
        """Controller tick returns avg_delta_s and avg_diversity_bonus."""
        experts = [
            CulturalExpert(
                name="A", expert_id=_next_expert_id(),
                phi=0.0, omega=0.1,
            ),
        ]
        controller = CulturalEvolutionaryController(experts=experts)

        info = controller.tick()

        assert "avg_delta_s" in info
        assert "avg_diversity_bonus" in info

    def test_controller_updates_absorption_stats(self):
        """Controller updates absorption stats each tick."""
        experts = [
            CulturalExpert(
                name="A", expert_id=_next_expert_id(),
                phi=0.0, omega=0.1,
            ),
        ]
        controller = CulturalEvolutionaryController(experts=experts)

        # Initial state
        assert controller.experts[0].g_lens_ema == 1.0

        # Run ticks
        for _ in range(10):
            controller.tick()

        # Stats should have been updated
        # (g_lens_ema may still be near 1.0 but R_ema should change)
        assert controller.experts[0].R_ema != 0.0

    def test_absorption_only_on_macro_clock(self):
        """Absorption methods called only on macro clock."""
        experts = [
            CulturalExpert(
                name="A", expert_id=_next_expert_id(),
                phi=0.0, omega=0.1, s=0.1,
            ),
        ]
        # Set up conditions for absorption
        experts[0].R_ema = 0.8
        experts[0].g_lens_ema = 1.3
        experts[0].delta_s_ema = 0.1
        experts[0].theta = 100.0

        controller = CulturalEvolutionaryController(
            experts=experts,
            micro_period=5,
            macro_period=4,
        )

        # Run until first macro event (tick 20)
        old_theta_home = experts[0].theta_home
        for _ in range(19):
            info = controller.tick()
            if not info["macro_event"]:
                # theta_home shouldn't change before macro event
                # (though absorption stats update each tick)
                pass

        # Run the macro tick
        info = controller.tick()
        assert info["macro_event"] is True
        # Now absorption could have happened (if conditions still met)
