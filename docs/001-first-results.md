# Chronovisor: First Results

**Date:** Phase 1 prototype complete
**Status:** Toy simulation operational

---

## What We Built

The first functional Chronovisor prototype consists of four components:

### 1. Controller (`controller.py`)

Three-clock timing system:
- **fast_clock**: increments every tick
- **micro_clock**: increments every `micro_period` ticks (default: 5)
- **macro_clock**: increments every `macro_period` ticks (default: 20)

The controller collects expert signals each tick and computes Δcoherence — the change in ensemble agreement since the last measurement.

### 2. ExpertHarness (`expert_harness.py`)

Defines the passive sensor interface:
```python
{
    "gain": float,        # amplification factor
    "tilt": float,        # directional bias
    "stability": float,   # how settled the expert feels
    "out_of_tolerance": bool  # stability < 0.5
}
```

Experts don't self-regulate. They only sense.

### 3. LensState (`simulation.py`)

Placeholder geometry — a 2D vector that experts "read through":
```python
lens.vector = [x, y]
lens.update([dx, dy])  # adds delta
```

No constraints, no bounds. Just a surface that can drift.

### 4. ToyExpert (`simulation.py`)

Experts with "personalities" (sensitivity vectors):
- **Alpha**: `[1.0, 0.5]` — more responsive to x-axis
- **Beta**: `[0.5, 1.0]` — more responsive to y-axis
- **Gamma**: `[0.8, 0.8]` — balanced
- **Delta**: `[1.2, 0.3]` — strongly x-biased

Each expert's readings depend on lens state + small noise:
- `gain = 1.0 + 0.1 * sensitivity[0] * lens[0] + noise`
- `stability` decreases as lens "stretches" the expert

---

## First Simulation Run

**Configuration:**
- 4 experts (Alpha, Beta, Gamma, Delta)
- 100 ticks
- micro_period=5, macro_period=20
- Lens drifts every 7 ticks (magnitude=0.1)
- Seed: 42 (reproducible)

**Sample output:**
```
t=  5 | Lens([+0.000, +0.000]) | Δ=-0.0041 | stab=1.017 gain=1.002 [micro]
t=  7 | Lens([+0.062, -0.061]) | Δ=+0.0052 | stab=0.983 gain=0.991 ~drift~
t= 20 | Lens([+0.071, +0.009]) | Δ=+0.0085 | stab=0.977 gain=1.000 [MACRO]
t= 35 | Lens([+0.149, +0.202]) | Δ=-0.0035 | stab=0.940 gain=1.014 ~drift~ [micro]
t= 91 | Lens([+0.052, +0.040]) | Δ=+0.0058 | stab=0.994 gain=1.003 ~drift~
```

**Final state:**
```
Final lens state: Lens([+0.025, -0.060])
Final clocks: fast=100, micro=20, macro=5

Final expert readings:
  Alpha: gain=0.959, tilt=0.010, stability=0.963, oot=False
  Beta: gain=0.990, tilt=-0.037, stability=0.957, oot=False
  Gamma: gain=0.994, tilt=-0.015, stability=1.006, oot=False
  Delta: gain=1.012, tilt=-0.008, stability=1.005, oot=False
```

---

## Observations

### 1. Δcoherence responds to drift

When the lens drifts, Δcoherence fluctuates. This is expected — experts with different sensitivities see the same geometry differently, so variance changes.

Example: At t=28, a drift caused Δ=+0.0121 (coherence improved). At t=35, another drift caused Δ=-0.0035 (coherence degraded slightly).

### 2. Stability tracks lens stretch

Average stability drops when the lens vector grows:
- t=5 (lens near origin): stab=1.017
- t=35 (lens=[+0.149, +0.202]): stab=0.940

When the lens drifts back toward origin (t=91), stability recovers to 0.994.

### 3. No out-of-tolerance events in this run

With drift_magnitude=0.1, the lens never stretched far enough to trigger OOT (stability < 0.5). This suggests the system is operating in a "comfortable" regime.

### 4. Multi-clock rhythm is visible

The `[micro]` and `[MACRO]` markers show the timing hierarchy. Every 5th tick is a micro boundary; every 20th is a macro boundary. This will matter when we add decision logic — the controller will only consider lens updates at specific clock boundaries.

---

## What This Validates

1. **The control loop runs.** Controller ticks, experts sense, coherence computes.
2. **Experts respond to geometry.** Different personalities create diverse signals.
3. **Δcoherence is a meaningful signal.** It moves in response to environmental change.
4. **The timing system works.** Three clocks at different rates, clearly visible in output.

---

## Next Steps

1. **Add lens update logic** — Controller should adjust lens geometry based on Δcoherence trajectory
2. **Implement decision gates** — Updates only permitted at micro/macro boundaries
3. **Add damping/momentum** — Prevent overreaction to transient coherence changes
4. **Test edge cases** — Force OOT conditions, observe recovery behaviour
5. **Visualisation** — Plot coherence and lens trajectory over time

---

## Running the Simulation

```bash
# From repo root
PYTHONPATH=src python -m chronovisor.simulation

# Or with make (after make init)
make test  # runs all 37 tests
```

---

*This document records Phase 1 results for the Chronovisor prototype.*
