"""
Expert Stub: Toy experts for initial testing.

Simulated experts for validating the control loop before
integrating with real model components.
"""

from chronovisor.expert_harness import ExpertHarness


class ExpertStub(ExpertHarness):
    """
    Stub expert for testing.

    Inherits all behaviour from ExpertHarness without modification.
    """

    def __init__(self, name: str = None) -> None:
        """
        Initialise the stub expert.

        Args:
            name: Optional name for this expert.
        """
        super().__init__(name=name)
