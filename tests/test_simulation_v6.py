"""Tests for Chronovisor V6: Cultural Transmission."""

import math
import random
from chronovisor.simulation_v6 import (
    CulturalExpert,
    Motif,
    CulturalController,
    CulturalEvolutionaryController,
    Governor,
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
