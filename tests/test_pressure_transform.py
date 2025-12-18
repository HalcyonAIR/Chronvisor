import numpy as np

from chronomoe.bridge import ChronoMoEBridge


def test_pressure_bias_can_push_up_and_down():
    bridge = ChronoMoEBridge.create(n_experts=2)

    # Force expert reliabilities to opposing extremes
    bridge.controller.experts[0].s = 5.0  # Very reliable
    bridge.controller.experts[1].s = -5.0  # Very unreliable

    pressure = bridge.get_pressure_bias()

    # Trust term should now produce opposite-signed pushes
    assert pressure.combined[0] > 0.0
    assert pressure.combined[1] < 0.0

    # Magnitudes should flip if we swap reliabilities
    bridge.controller.experts[0].s = -5.0
    bridge.controller.experts[1].s = 5.0

    flipped = bridge.get_pressure_bias()
    assert flipped.combined[0] < 0.0
    assert flipped.combined[1] > 0.0
