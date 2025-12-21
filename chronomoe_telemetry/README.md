# ChronoMoE Telemetry

> **This module does not decide when the system is done.**
> **It only proposes pause points and exposes the evidence.**

Non-invasive observability layer for ChronoMoE dynamics. Provides pause-point detection, stillness signals, and calibration artifacts for the "silent moment" detection experiment.

## Design Principles

1. **Read-only observation** — Never modifies model state
2. **Session-local** — No persistence across runs
3. **Human-auditable** — Every measurement has clear interpretation
4. **Proxy-capable** — Can run without real ChronoMoE for testing
5. **Separation of concerns** — Telemetry depends on ChronoMoE, never the reverse

## Architecture

```
chronomoe_telemetry/
├── observer.py    # Read-only state access (ChronoMoE or proxy)
├── metrics.py     # Pure measurement functions (PC1/PC2, entropy, etc.)
├── stillness.py   # Multi-timescale stillness detection
├── pause.py       # Candidate pause proposals
├── schema.py      # JSON format + glossary
└── recorder.py    # Main orchestrator
```

## Three-Layer Validation

The telemetry system supports a rigorous calibration protocol:

1. **System belief** — "I think I'm done"
   - Stillness flags (fast/medium/slow)
   - PC velocities near zero
   - Routing entropy collapse

2. **Execution phenomenology** — "From inside, this felt complete/premature/stuck"
   - Subjective but structured
   - Text-based, retrospective
   - Slow-clock authoritative

3. **Reality check** — "What happens if we continue with no new input?"
   - Does continuation add structure?
   - Or does it add filler/repetition?
   - Ground truth proxy

Together these produce a **confusion matrix**, not opinions.

## Usage

### With Real ChronoMoE

```python
from chronomoe_telemetry import (
    ChronoMoEObserver,
    TelemetryRecorder,
    StillnessThresholds,
)

# Create recorder
observer = ChronoMoEObserver()
recorder = TelemetryRecorder(
    session_id="experiment_001",
    observer=observer,
    stillness_thresholds=StillnessThresholds(
        fast=0.01,
        medium=0.02,
        slow=0.005,
    ),
)

# During inference loop
for step in range(max_steps):
    _, chrono_state, _ = model(input_ids, update_chronovisor=True)

    # Capture telemetry
    snapshot = recorder.capture_snapshot(chrono_state=chrono_state)

    # Check if pause should be proposed
    should_pause, reason = recorder.should_pause()

    if should_pause:
        print(snapshot.to_json())
        # Pause for human review
        break
```

### Proxy Mode (No ChronoMoE Required)

```python
from chronomoe_telemetry import ProxyObserver, TelemetryRecorder

# Create proxy observer with synthetic data
observer = ProxyObserver(num_experts=64, num_layers=2)

recorder = TelemetryRecorder(
    session_id="proxy_test",
    observer=observer,
)

# Simulate dynamics
for step in range(30):
    observer.step(noise_scale=0.01)

    snapshot = recorder.capture_snapshot(chrono_state=None)
    should_pause, reason = recorder.should_pause()

    if should_pause:
        print(f"Pause proposed: {reason}")
```

## Telemetry Snapshot Format

```json
{
  "session_id": "experiment_001",
  "pause_index": 1,
  "pause_reason": "candidate_stillness",

  "pc1": 1.87,
  "pc2": 0.14,

  "d_pc1": 0.03,
  "d_pc2": -0.01,

  "routing_entropy": 1.21,
  "expert_switch_rate": 0.18,

  "stillness_flags": {
    "fast": true,
    "medium": true,
    "slow": false
  },

  "gloss": {
    "pc1": "Momentum / continuation strength ...",
    "pc2": "Exploration margin ...",
    ...
  }
}
```

The `gloss` field provides human-readable interpretations of each measurement.

## Calibration Protocol

For phenomenology calibration experiments:

1. **Capture snapshots** at candidate pause points
2. **Observer provides labels**:
   ```json
   {
     "snapshot_id": 1,
     "felt_state": "complete | premature | stuck",
     "confidence": 0.8,
     "notes": "Explanation..."
   }
   ```
3. **Run resume test**: Continue inference without new input
4. **Compare all three**:
   - System belief (stillness flags)
   - Phenomenology (felt state)
   - Reality (did structure continue?)
5. **Build confusion matrix**:
   - True stillness: system=done, felt=complete, resume=filler ✓
   - False positive: system=done, felt=premature, resume=structure ✗
   - False negative: system=not done, felt=complete ✗

6. **Tune thresholds** between sessions (never mid-run)

## Testing

Three test scripts demonstrate different aspects:

```bash
# Proxy mode (no ChronoMoE required)
python experiments/test_telemetry_proxy.py

# Modular with real ChronoMoE
PYTHONPATH=src:. python experiments/test_telemetry_modular.py

# Original single-file version (deprecated)
PYTHONPATH=src python experiments/test_telemetry_calibration.py
```

## Key Constraints

1. **Thresholds are priors** — Stillness detection uses engineered thresholds, not learned parameters
2. **No self-modification** — The model cannot adjust its own stopping criteria
3. **Pause = proposal** — The system proposes pauses, humans decide
4. **Session-local** — No state persists across runs
5. **Observable** — Every measurement is legible and interpretable

## What This Is Not

- ❌ Not autonomous stopping criteria
- ❌ Not self-reflection or introspection
- ❌ Not consciousness detection
- ❌ Not persistent agent goals

## What This Is

- ✓ Observable dynamics
- ✓ Pause-point proposals
- ✓ Calibration infrastructure
- ✓ Human-in-the-loop measurement

## Connection to "Silent Moment" Hypothesis

This module implements the technical infrastructure for testing:

> **"User input should enter only at dynamically detected phase boundaries, not continuously throughout execution."**

The hypothesis is that meaningful execution phases have natural completion points (stillness) that can be detected from routing dynamics, and these points are the correct handoff for new user input.

Telemetry allows us to:
- Detect these candidate points (stillness flags)
- Test whether they're real (resume test)
- Calibrate detection (confusion matrix)
- Tune thresholds (between sessions)

All without introducing autonomous behavior or persistent goals.

## References

See `experiments/test_path_wear.py` for the original path wear experiment that motivated this design.

See `TECHNICAL_REPORT_D2_MANIFOLD.md` for background on the d=2 routing manifold and PC1/PC2 interpretation.
