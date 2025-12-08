"""Tests for the Controller class."""

from chronovisor.controller import Controller


class MockExpert:
    """Minimal expert stub with sense() method."""

    def sense(self, lens_state):
        return {"stub": True}


class MockLens:
    """Minimal lens stub."""

    pass


def test_controller_init():
    """Controller initialises with correct clock values."""
    ctrl = Controller(micro_period=4, macro_period=16)

    assert ctrl.micro_period == 4
    assert ctrl.macro_period == 16
    assert ctrl.fast_clock == 0
    assert ctrl.micro_clock == 0
    assert ctrl.macro_clock == 0


def test_fast_clock_increments_every_tick():
    """Fast clock increments on every tick."""
    ctrl = Controller(micro_period=4, macro_period=16)
    experts = [MockExpert()]
    lens = MockLens()

    for i in range(5):
        ctrl.tick(experts, lens)
        assert ctrl.fast_clock == i + 1


def test_micro_clock_increments_at_period():
    """Micro clock increments every micro_period ticks."""
    ctrl = Controller(micro_period=4, macro_period=16)
    experts = [MockExpert()]
    lens = MockLens()

    # Tick 3 times - micro should still be 0
    for _ in range(3):
        ctrl.tick(experts, lens)
    assert ctrl.micro_clock == 0

    # Tick once more (4th tick) - micro should be 1
    ctrl.tick(experts, lens)
    assert ctrl.micro_clock == 1

    # Tick 4 more times (8th tick) - micro should be 2
    for _ in range(4):
        ctrl.tick(experts, lens)
    assert ctrl.micro_clock == 2


def test_macro_clock_increments_at_period():
    """Macro clock increments every macro_period ticks."""
    ctrl = Controller(micro_period=4, macro_period=16)
    experts = [MockExpert()]
    lens = MockLens()

    # Tick 15 times - macro should still be 0
    for _ in range(15):
        ctrl.tick(experts, lens)
    assert ctrl.macro_clock == 0

    # Tick once more (16th tick) - macro should be 1
    ctrl.tick(experts, lens)
    assert ctrl.macro_clock == 1


def test_compute_delta_coherence_returns_placeholder():
    """compute_delta_coherence returns fixed 0.0."""
    ctrl = Controller(micro_period=4, macro_period=16)

    result = ctrl.compute_delta_coherence([])
    assert result == 0.0

    result = ctrl.compute_delta_coherence([{"a": 1}, {"b": 2}])
    assert result == 0.0


def test_tick_collects_expert_signals():
    """tick() calls sense() on each expert."""
    ctrl = Controller(micro_period=4, macro_period=16)

    call_count = 0

    class CountingExpert:
        def sense(self, lens_state):
            nonlocal call_count
            call_count += 1
            return {}

    experts = [CountingExpert(), CountingExpert(), CountingExpert()]
    lens = MockLens()

    ctrl.tick(experts, lens)
    assert call_count == 3

    ctrl.tick(experts, lens)
    assert call_count == 6
