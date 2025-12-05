"""Tests for the ExpertHarness interface."""

from chronovisor.expert_harness import ExpertHarness
from chronovisor.expert_stub import ExpertStub


class TestExpertHarness:
    """Tests for ExpertHarness base class."""

    def test_init_without_name(self):
        """ExpertHarness can be created without a name."""
        harness = ExpertHarness()
        assert harness.name is None

    def test_init_with_name(self):
        """ExpertHarness stores the provided name."""
        harness = ExpertHarness(name="test_expert")
        assert harness.name == "test_expert"

    def test_sense_returns_dict_with_required_keys(self):
        """sense() returns a dict with exactly the required keys."""
        harness = ExpertHarness()
        result = harness.sense(lens_state=None)

        assert isinstance(result, dict)
        assert "gain" in result
        assert "tilt" in result
        assert "stability" in result
        assert "out_of_tolerance" in result

    def test_sense_returns_fixed_placeholder_values(self):
        """sense() returns the fixed placeholder values."""
        harness = ExpertHarness()
        result = harness.sense(lens_state=None)

        assert result["gain"] == 1.0
        assert result["tilt"] == 0.0
        assert result["stability"] == 1.0
        assert result["out_of_tolerance"] is False

    def test_sense_accepts_none_lens_state(self):
        """sense() does not raise when lens_state is None."""
        harness = ExpertHarness()
        result = harness.sense(lens_state=None)
        assert result is not None

    def test_in_tolerance_returns_true(self):
        """in_tolerance() always returns True."""
        harness = ExpertHarness()
        assert harness.in_tolerance() is True


class TestExpertStub:
    """Tests for ExpertStub subclass."""

    def test_stub_is_subclass_of_harness(self):
        """ExpertStub is a subclass of ExpertHarness."""
        assert issubclass(ExpertStub, ExpertHarness)

    def test_stub_init_without_name(self):
        """ExpertStub can be created without a name."""
        stub = ExpertStub()
        assert stub.name is None

    def test_stub_init_with_name(self):
        """ExpertStub stores the provided name."""
        stub = ExpertStub(name="stub_1")
        assert stub.name == "stub_1"

    def test_stub_sense_returns_same_as_harness(self):
        """ExpertStub.sense() behaves exactly as ExpertHarness."""
        stub = ExpertStub()
        result = stub.sense(lens_state=None)

        assert result["gain"] == 1.0
        assert result["tilt"] == 0.0
        assert result["stability"] == 1.0
        assert result["out_of_tolerance"] is False

    def test_stub_in_tolerance_returns_true(self):
        """ExpertStub.in_tolerance() behaves exactly as ExpertHarness."""
        stub = ExpertStub()
        assert stub.in_tolerance() is True
