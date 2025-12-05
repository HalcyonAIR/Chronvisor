"""
Controller: Clock management, delta-coherence tracking, lens update logic.

The controller is the sole decision-maker. It:
- Runs three clocks at different timescales (fast, micro, macro)
- Measures delta-coherence between micro-turns
- Decides when lens adjustments are permitted
- Modulates magnitude and locality of geometric shifts
"""
