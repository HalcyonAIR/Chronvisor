"""
Expert Harness: Interface that experts expose to the controller.

Experts become passive sensors. They:
- Read through the shared lens surface
- Expose: local gain, tilt, stability score, out-of-tolerance flags
- Operate under whatever geometry the controller provides
"""


class ExpertHarness:
    """
    Base interface for experts.

    Experts are passive sensors with a narrow, predictable interface.
    The controller has complete authority over interpretation.
    """

    def __init__(self, name: str = None) -> None:
        """
        Initialise the expert harness.

        Args:
            name: Optional name for this expert.
        """
        self.name = name

    def sense(self, lens_state) -> dict:
        """
        Return sensor readings as a fixed dictionary.

        Args:
            lens_state: Current lens state (unused in base implementation).

        Returns:
            Dictionary with keys: gain, tilt, stability, out_of_tolerance.
        """
        return {
            "gain": 1.0,
            "tilt": 0.0,
            "stability": 1.0,
            "out_of_tolerance": False,
        }

    def in_tolerance(self) -> bool:
        """
        Check if expert is within tolerance.

        Returns:
            Always True in base implementation.
        """
        return True
