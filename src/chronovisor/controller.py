"""
Controller: Clock management, delta-coherence tracking, lens update logic.

The controller is the sole decision-maker. It:
- Runs three clocks at different timescales (fast, micro, macro)
- Measures delta-coherence between micro-turns
- Decides when lens adjustments are permitted
- Modulates magnitude and locality of geometric shifts
"""


class Controller:
    """
    Manages the timing rhythm for the Chronovisor system.

    Three clocks operate at different timescales:
    - fast_clock: increments every tick
    - micro_clock: increments every micro_period ticks
    - macro_clock: increments every macro_period ticks
    """

    def __init__(self, micro_period: int, macro_period: int) -> None:
        """
        Initialise the controller with clock periods.

        Args:
            micro_period: Number of fast ticks per micro tick.
            macro_period: Number of fast ticks per macro tick.
        """
        self.micro_period = micro_period
        self.macro_period = macro_period

        self.fast_clock = 0
        self.micro_clock = 0
        self.macro_clock = 0

    def compute_delta_coherence(self, expert_states: list) -> float:
        """
        Placeholder for delta-coherence computation.

        Args:
            expert_states: List of expert sensor outputs.

        Returns:
            Fixed float value (no logic implemented yet).
        """
        return 0.0

    def tick(self, experts: list, lens) -> None:
        """
        Advance the controller by one tick.

        Increments clocks, collects expert signals, and logs state.
        Does not modify the lens or implement coherence behaviour.

        Args:
            experts: List of expert objects (must have a sense() method).
            lens: Lens object (unused in this version).
        """
        # Increment fast clock
        self.fast_clock += 1

        # Update micro clock
        if self.fast_clock % self.micro_period == 0:
            self.micro_clock += 1

        # Update macro clock
        if self.fast_clock % self.macro_period == 0:
            self.macro_clock += 1

        # Collect expert signals
        expert_signals = [expert.sense(lens) for expert in experts]

        # Compute delta coherence (placeholder)
        delta = self.compute_delta_coherence(expert_signals)

        # Log current state
        print(
            f"fast={self.fast_clock} "
            f"micro={self.micro_clock} "
            f"macro={self.macro_clock} "
            f"Δ={delta} "
            f"(no lens update yet)"
        )
