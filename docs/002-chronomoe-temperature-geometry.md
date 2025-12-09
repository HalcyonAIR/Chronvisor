# ChronoMoE: Temperature × Pressure Routing Geometry

**Date:** Structural temperature implementation complete
**Status:** 2-field routing geometry operational

---

## Overview

ChronoMoE extends standard MoE routing with a **2-field geometry** that gives routing decisions both direction and texture:

- **Pressure (b_k)**: Force field — pushes routing toward/away from experts
- **Temperature (T_k)**: Permeability field — controls how "sticky" each expert region is

The combined effect:

```
logits'_k = (logits_k + b_k) / T_k
p_k = softmax_k(logits'_k)
```

This creates anisotropic routing where each expert has different "terrain" — some are easy to commit to (valleys), others are slippery (ridges).

---

## The Physical Metaphor

Think of routing as a marble rolling across a landscape:

| Component | Physical Analog | Effect on Routing |
|-----------|-----------------|-------------------|
| Logits | Marble's initial position | Where it starts |
| Pressure b_k | Wind | Pushes toward/away from experts |
| Temperature T_k | Terrain softness | How easily marble moves through region |
| Structural T̄_k | Geology (erosion) | Long-term terrain memory |

**Low temperature** = hard ground, marble commits easily (exploitation)
**High temperature** = soft sand, marble slips through (exploration)

---

## Why Not Just Noisy Top-K?

Standard noisy gating adds random noise to break ties:

```
Noisy top-k:  logits + ε(random)  →  no memory, flat geometry
```

ChronoMoE's 2-field system is fundamentally different:

```
ChronoMoE:    logits + b_k         →  directional (pressure)
              ─────────────        →  anisotropic (temperature)
               T_fast × T̄_k       →  with memory (structural)
```

Key differences:
1. **Pressure is directional**, derived from Chronovisor signals (trust, lens, culture)
2. **Temperature is per-expert**, not uniform noise
3. **Structural temperature has memory** — the landscape learns where routing works

---

## The Two Temperature Fields

### Fast Temperature (T_fast)

Instantaneous terrain permeability, computed each tick from Chronovisor state:

```python
T_k^fast = T_base × coherence_factor × drift_k × reliability_k

where:
  coherence_factor = 1 + α × (1 - R)     # Higher when incoherent
  drift_k = 1 + β × |drift_k|            # Higher for drifting experts
  reliability_k = 1 + γ × (1 - rel_k)    # Higher for unreliable experts
```

Fast temperature responds to current conditions:
- **High R (coherent)** → lower T → sharper routing (exploit)
- **High drift** → higher T → softer routing (explore)
- **Low reliability** → higher T → hedge bets

### Structural Temperature (T̄_k)

Slow-evolving "geology" that remembers where routing historically works:

```
T̄_k(t+1) = (1 - η_T) × T̄_k(t) + η_T × T_k^fast(t)
```

With η_T ≈ 0.01 (very slow), this creates:

- **Valleys**: Experts that consistently have low T_fast → structural T erodes down → easier to route to
- **Ridges**: Experts that consistently have high T_fast → structural T builds up → harder to route to

### Effective Temperature

What routing actually uses:

```
T_k^effective = T_k^fast × T̄_k
```

This multiplicative combination means:
- Even if current conditions are good (low T_fast), a ridge expert (high T̄) stays slippery
- Even if current conditions are uncertain (high T_fast), a valley expert (low T̄) remains accessible

---

## Mathematical Details

### Full Routing Equation

```
p_k = softmax_k((logits_k + pressure_scale × b_k) / (temp_scale × T_k^effective))
```

Where:
- `pressure_scale` and `temp_scale` are meta-knob controlled
- `b_k` comes from ChronoMoEBridge (trust, lens, culture signals)
- `T_k^effective = T_k^fast × T̄_k`

### Temperature Field Computation

```python
def get_temperature_field(self) -> TemperatureField:
    # Coherence factor: explore more when incoherent
    coherence_R = self.coherence_estimator.R
    coherence_factor = 1.0 + self.alpha_coherence * (1.0 - coherence_R)

    # Per-expert drift factor
    drift_factors = 1.0 + self.beta_drift * np.abs(self.expert_drift)

    # Per-expert reliability factor
    reliability_factors = 1.0 + self.gamma_reliability * (1.0 - self.expert_reliability)

    # Fast temperature
    fast_temperatures = (
        self.T_base * coherence_factor * drift_factors * reliability_factors
    )

    # Update structural temperature (EMA)
    self.structural_T = (
        (1 - self.eta_structural_T) * self.structural_T
        + self.eta_structural_T * fast_temperatures
    )

    # Effective = fast × structural
    effective_temperatures = fast_temperatures * self.structural_T

    return TemperatureField(
        fast_temperatures=fast_temperatures,
        structural_temperatures=self.structural_T.copy(),
        effective_temperatures=np.clip(effective_temperatures, T_min, T_max),
        ...
    )
```

### Structural Temperature EMA Dynamics

The EMA update has interesting convergence properties:

- **Steady state**: If T_fast is constant at T*, then T̄ → T* as t → ∞
- **Time constant**: τ = 1/η_T ≈ 100 steps for η_T = 0.01
- **Differentiation**: Variance in T̄ grows as experts experience different conditions

After sufficient time:
- Valleys form around consistently reliable experts
- Ridges form around consistently unstable experts
- The landscape "remembers" which experts to trust

---

## Landscape Formation

### Detecting Valleys and Ridges

```python
def get_structural_temperature_diagnostics(self) -> dict:
    mean_T = np.mean(self.structural_T)
    std_T = np.std(self.structural_T)

    # Valleys: significantly below mean (stable, reliable)
    valleys = np.where(self.structural_T < mean_T - 0.5 * std_T)[0]

    # Ridges: significantly above mean (unstable, unreliable)
    ridges = np.where(self.structural_T > mean_T + 0.5 * std_T)[0]

    # Landscape formed when variance exceeds threshold
    landscape_formed = np.var(self.structural_T) > 0.01

    return {
        "structural_T": self.structural_T.copy(),
        "variance": np.var(self.structural_T),
        "valleys": valleys.tolist(),
        "ridges": ridges.tolist(),
        "landscape_formed": landscape_formed,
    }
```

### Example Results

After 500 steps with η_T = 0.02:

```
Landscape Formation:
  Variance growth: 0 → 0.0064
  Valleys (stable): [0, 2, 5, 7]
  Ridges (unstable): [1, 3, 4]
  Temperature range: 1.51 (valley) to 1.72 (ridge)
```

The 1.72/1.51 ≈ 1.14 ratio means valley experts get ~14% sharper routing than ridge experts, even with identical instantaneous conditions.

---

## Meta-Knob Integration

The meta-knob κ ∈ [-1, +1] modulates temperature via `temp_scale`:

```python
temp_scale = exp(β_temperature × κ)

# With β_temperature = 0.4:
κ = -1  →  temp_scale ≈ 0.67  (sharper, exploit)
κ =  0  →  temp_scale = 1.0   (neutral)
κ = +1  →  temp_scale ≈ 1.49  (softer, explore)
```

This allows the LLM controller to globally adjust terrain stiffness:
- **Negative κ**: Harden all terrain → commit more strongly
- **Positive κ**: Soften all terrain → explore more freely

The structural temperature provides the **local** variation; temp_scale provides **global** control.

---

## Running the Experiments

### Temperature Field Comparison

```bash
PYTHONPATH=src python src/chronomoe/experiment_temperature.py
```

Compares:
- Baseline (no pressure, no temperature)
- Pressure-only
- Temperature-only
- Full system (pressure + temperature)

### Landscape Formation

```bash
PYTHONPATH=src python src/chronomoe/experiment_structural.py
```

Tracks geological evolution over 500 steps:
- Structural temperature variance growth
- Valley/ridge formation
- Correlation with expert reliability

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `T_base` | 1.0 | Base temperature |
| `T_min` | 0.1 | Minimum effective temperature |
| `T_max` | 10.0 | Maximum effective temperature |
| `alpha_coherence` | 0.5 | Coherence → temperature sensitivity |
| `beta_drift` | 0.3 | Drift → temperature sensitivity |
| `gamma_reliability` | 0.4 | Reliability → temperature sensitivity |
| `eta_structural_T` | 0.01 | Structural temperature EMA rate |
| `beta_temperature` | 0.4 | Meta-knob → temp_scale sensitivity |

---

## Key Insights

1. **Temperature creates anisotropic routing** — each expert has different permeability, unlike uniform noise

2. **Structural temperature is learned routing prior** — encodes "where has routing historically worked" into geometry

3. **Two timescales complement each other**:
   - Fast temperature responds to current conditions
   - Structural temperature captures long-term patterns

4. **Pressure provides direction, temperature provides texture**:
   - Pressure says "tilt this way"
   - Temperature says "how committed should we be?"

5. **The landscape self-organizes** — valleys form around reliable experts without explicit supervision

---

## Tests

All temperature functionality is covered in `tests/test_temperature.py`:

```bash
PYTHONPATH=src python -m pytest tests/test_temperature.py -v
# 32 tests covering:
#   - Temperature field creation and bounds
#   - Router temperature warping
#   - Meta-knob temp_scale
#   - Structural temperature evolution
#   - Landscape formation
#   - Backwards compatibility
```

---

*This document describes the 2-field routing geometry for ChronoMoE.*
