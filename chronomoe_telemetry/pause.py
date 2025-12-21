"""
Pause: Candidate_done logic.

Determines when to propose a pause based on stillness signals.
This is a proposal, not a decision. Humans own the final judgment.
"""

from typing import Tuple
from .stillness import StillnessFlags


class PauseDetector:
    """
    Proposes pause points based on stillness signals.

    Important: This does not decide when execution should stop.
    It only proposes candidate pause points for human review.
    """

    def __init__(
        self,
        require_fast: bool = True,
        require_medium: bool = True,
        require_slow: bool = False,
    ):
        """
        Args:
            require_fast: Must fast stillness be triggered?
            require_medium: Must medium stillness be triggered?
            require_slow: Must slow stillness be triggered?
        """
        self.require_fast = require_fast
        self.require_medium = require_medium
        self.require_slow = require_slow

    def should_pause(self, stillness_flags: StillnessFlags) -> Tuple[bool, str]:
        """
        Determine if a pause should be proposed.

        Args:
            stillness_flags: Current stillness state

        Returns:
            (should_pause, reason)
        """
        # Check required conditions
        conditions_met = []
        conditions_failed = []

        if self.require_fast:
            if stillness_flags.fast:
                conditions_met.append('fast')
            else:
                conditions_failed.append('fast')

        if self.require_medium:
            if stillness_flags.medium:
                conditions_met.append('medium')
            else:
                conditions_failed.append('medium')

        if self.require_slow:
            if stillness_flags.slow:
                conditions_met.append('slow')
            else:
                conditions_failed.append('slow')

        # All required conditions must be met
        if len(conditions_failed) == 0:
            reason = "candidate_stillness"
            if len(conditions_met) == 3:
                reason = "candidate_stillness_all"
            return (True, reason)
        else:
            return (False, "not_still")

    def configure(
        self,
        require_fast: bool = None,
        require_medium: bool = None,
        require_slow: bool = None,
    ):
        """
        Reconfigure pause requirements.

        Can only be called between sessions, not during execution.
        """
        if require_fast is not None:
            self.require_fast = require_fast
        if require_medium is not None:
            self.require_medium = require_medium
        if require_slow is not None:
            self.require_slow = require_slow


def should_pause_default(stillness_flags: StillnessFlags) -> Tuple[bool, str]:
    """
    Default pause logic: fast AND medium.

    This is a sensible default for most cases.

    Args:
        stillness_flags: Current stillness state

    Returns:
        (should_pause, reason)
    """
    if stillness_flags.fast and stillness_flags.medium:
        return (True, "candidate_stillness")
    return (False, "not_still")
