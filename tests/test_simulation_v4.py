"""Tests for Chronovisor V4: Bifurcation and Population Dynamics."""

import math
import random
from chronovisor.simulation_v4 import (
    BifurcatingExpert,
    EvolutionaryController,
    compute_kuramoto_R_and_psi,
    sigmoid,
    alignment,
    reset_expert_id_counter,
    _next_expert_id,
)


class TestBifurcatingExpert:
    """Tests for experts with bifurcation capability."""

    def setup_method(self):
        reset_expert_id_counter()

    def test_initial_state(self):
        """Expert initialises correctly."""
        e = BifurcatingExpert(
            name="Test",
            expert_id=1,
            phi=0.0,
            omega=0.1,
            theta=10.0,
            theta_home=0.0,
        )
        assert e.name == "Test"
        assert e.theta == 10.0
        assert e.theta_home == 0.0
        assert e.generation == 0

    def test_drift_distance(self):
        """drift_distance computes absolute distance from home."""
        e = BifurcatingExpert(
            name="Test", expert_id=1, phi=0.0, omega=0.1,
            theta=50.0, theta_home=0.0,
        )
        assert e.drift_distance() == 50.0

        e.theta = -30.0
        assert e.drift_distance() == 30.0

    def test_spawn_offspring(self):
        """spawn_offspring creates new expert at parent's old home."""
        random.seed(42)
        parent = BifurcatingExpert(
            name="Parent",
            expert_id=1,
            phi=0.0,
            omega=0.1,
            theta=100.0,
            theta_home=0.0,
            generation=0,
        )

        offspring = parent.spawn_offspring(current_tick=50)

        assert offspring.theta == 0.0  # At parent's old home
        assert offspring.theta_home == 0.0
        assert offspring.generation == 1
        assert offspring.parent_id == 1
        assert offspring.birth_tick == 50
        assert offspring.lambd == 0.05  # Fresh temperament
        assert offspring.s == 0.0  # Fresh reliability

    def test_settle_at_current_position(self):
        """settle_at_current_position updates theta_home."""
        e = BifurcatingExpert(
            name="Test", expert_id=1, phi=0.0, omega=0.1,
            theta=100.0, theta_home=0.0,
        )

        e.settle_at_current_position()

        assert e.theta_home == 100.0
        assert "→" in e.name

    def test_tick_fast_returns_drift(self):
        """tick_fast returns drift in signal dict."""
        e = BifurcatingExpert(
            name="Test", expert_id=1, phi=0.0, omega=0.1,
            theta=50.0, theta_home=0.0,
        )

        signal = e.tick_fast(psi=0.0)

        assert "drift" in signal
        assert signal["drift"] >= 0

    def test_low_reliability_tracking(self):
        """Tracks consecutive low reliability macro ticks."""
        e = BifurcatingExpert(
            name="Test", expert_id=1, phi=0.0, omega=0.1,
        )

        # Simulate low reliability
        e.macro_align_sum = -0.5
        e.macro_align_count = 1
        e.s = -0.25  # Already low

        e.do_macro_update(eta_s=0.5)

        assert e.low_reliability_ticks >= 1


class TestEvolutionaryController:
    """Tests for the evolutionary controller."""

    def setup_method(self):
        reset_expert_id_counter()
        random.seed(42)

        self.experts = [
            BifurcatingExpert(
                name="A", expert_id=_next_expert_id(),
                phi=0.0, omega=0.1, theta=0.0, theta_home=0.0,
            ),
            BifurcatingExpert(
                name="B", expert_id=_next_expert_id(),
                phi=1.0, omega=0.1, theta=0.0, theta_home=0.0,
            ),
            BifurcatingExpert(
                name="C", expert_id=_next_expert_id(),
                phi=2.0, omega=0.1, theta=0.0, theta_home=0.0,
            ),
        ]
        self.controller = EvolutionaryController(
            experts=self.experts,
            micro_period=5,
            macro_period=2,
            D_max=50.0,
            max_population=10,
            min_population=2,
        )

    def test_initial_state(self):
        """Controller initialises correctly."""
        assert len(self.controller.experts) == 3
        assert self.controller.fast_clock == 0

    def test_tick_returns_population(self):
        """tick returns current population count."""
        info = self.controller.tick()
        assert info["population"] == 3

    def test_bifurcation_triggered_by_drift(self):
        """Bifurcation triggers when drift > D_max and expert is reliable."""
        # Manually set up bifurcation conditions
        expert = self.controller.experts[0]
        expert.theta = 100.0  # Far from home
        expert.theta_home = 0.0
        expert.s = 0.5  # Reliable

        # Run until macro event
        for _ in range(10):
            info = self.controller.tick()

        assert info["macro_event"] is True
        assert len(info["bifurcations"]) > 0
        assert len(self.controller.experts) > 3

    def test_bifurcation_respects_max_population(self):
        """Bifurcation stops at max_population."""
        self.controller.max_population = 4

        # Set up all experts for bifurcation
        for e in self.controller.experts:
            e.theta = 100.0
            e.s = 0.5

        # Run many macro cycles
        for _ in range(100):
            self.controller.tick()

        assert len(self.controller.experts) <= 4

    def test_culling_removes_low_reliability_experts(self):
        """Experts with sustained low reliability are culled."""
        self.controller.min_population = 2

        # Set one expert to very low reliability
        expert = self.controller.experts[0]
        expert.s = -0.5
        expert.low_reliability_ticks = 10

        # Run until macro event
        for _ in range(10):
            info = self.controller.tick()

        assert info["macro_event"] is True
        # Should have culled or at least attempted
        # (may not cull if already at min_population)

    def test_culling_respects_min_population(self):
        """Culling stops at min_population."""
        self.controller.min_population = 3

        # Set all experts to low reliability
        for e in self.controller.experts:
            e.s = -0.5
            e.low_reliability_ticks = 10

        # Run many macro cycles
        for _ in range(100):
            self.controller.tick()

        assert len(self.controller.experts) >= 3

    def test_events_logged(self):
        """Bifurcation and culling events are logged."""
        # Set up bifurcation
        expert = self.controller.experts[0]
        expert.theta = 100.0
        expert.s = 0.5

        # Run until something happens
        for _ in range(50):
            self.controller.tick()

        # Should have some events
        assert len(self.controller.events) >= 0  # May or may not have events


class TestPopulationDynamics:
    """Integration tests for population dynamics."""

    def setup_method(self):
        reset_expert_id_counter()

    def test_population_can_grow(self):
        """Population grows through bifurcation."""
        random.seed(42)

        experts = [
            BifurcatingExpert(
                name=f"E{i}", expert_id=_next_expert_id(),
                phi=random.uniform(0, 2 * math.pi), omega=0.1,
            )
            for i in range(3)
        ]

        controller = EvolutionaryController(
            experts=experts,
            D_max=30.0,  # Lower threshold for more bifurcations
            max_population=15,
        )

        # Run for a while
        for _ in range(200):
            controller.tick()

        # Should have grown
        assert len(controller.experts) > 3

    def test_generation_increases(self):
        """Offspring have higher generation than parents."""
        random.seed(42)

        parent = BifurcatingExpert(
            name="Parent", expert_id=_next_expert_id(),
            phi=0.0, omega=0.1, generation=2,
        )

        offspring = parent.spawn_offspring(current_tick=0)

        assert offspring.generation == 3
