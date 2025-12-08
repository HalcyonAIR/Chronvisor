"""Basic tests to verify package structure."""

import chronovisor


def test_version():
    """Verify package version is accessible."""
    assert chronovisor.__version__ == "0.1.0"


def test_imports():
    """Verify all modules are importable."""
    from chronovisor import controller
    from chronovisor import lens
    from chronovisor import expert_harness
    from chronovisor import expert_stub

    assert controller is not None
    assert lens is not None
    assert expert_harness is not None
    assert expert_stub is not None
