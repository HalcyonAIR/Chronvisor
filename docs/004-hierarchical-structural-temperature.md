# Hierarchical Structural Temperature

**Status:** Implemented and validated
**Date:** 2025-12-10

---

## Overview

Implemented **hierarchical structural temperature** in response to Halcyon AI feedback, replacing the original per-layer design with a two-level system that captures both global and local geological patterns.

## The Problem

**Initial implementation:** Per-layer structural temperature
- Each layer had its own `T̄[layer]`
- Layers formed independent geological topologies
- No cross-layer coordination
- Missing global system-wide patterns

**Halcyon AI observation:**
> "Each layer forms its own geological topology. That may actually be great — but if you intended global structural temperature shared across all layers, that's not what was implemented."

## The Solution: Hierarchical Structural Temperature

Two-level hierarchy:
1. **T̄_global**: System-wide geological pattern (shared across all layers)
2. **T̄_local[layer]**: Layer-specific geological pattern (per-layer nuance)
3. **T̄_hierarchical = T̄_global × T̄_local**: Combined effect

### Formula

```
T_fast[layer] = base_T * f(R, drift, reliability)

T̄_local[layer](t+1) = (1 - η_local) * T̄_local[layer](t) + η_local * T_fast[layer](t)

T̄_global(t+1) = (1 - η_global) * T̄_global(t) + η_global * mean(T_fast[all layers])

T̄_hierarchical[layer] = T̄_global × T̄_local[layer]

T_effective[layer] = T_fast[layer] × T̄_hierarchical[layer]
```

### Parameters

- **η_global = 0.005**: Very slow (captures deep system-wide patterns)
- **η_local = 0.01**: Slow (captures layer-specific patterns)
- **Timescale separation**: η_global < η_local ensures proper hierarchy

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 0: T_fast[0]                                          │
│  Layer 1: T_fast[1]                                          │
│  Layer 2: T_fast[2]                                          │
│  Layer 3: T_fast[3]                                          │
│  ...                                                         │
└────────────────────┬─────────────────────────────────────────┘
                     │ Average
                     ↓
┌──────────────────────────────────────────────────────────────┐
│  T̄_global (system-wide)                                     │
│  EMA with η=0.005 (very slow)                                │
│  Shared across all layers                                    │
└────────────────────┬─────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┬───────────┐
         ↓                       ↓           ↓
┌─────────────────┐    ┌─────────────────┐  ...
│ T̄_local[0]      │    │ T̄_local[1]      │
│ EMA η=0.01       │    │ EMA η=0.01       │
│ Layer-specific   │    │ Layer-specific   │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         ↓ ×                     ↓ ×
┌─────────────────┐    ┌─────────────────┐
│ T̄_hierarchical  │    │ T̄_hierarchical  │
│ = T̄_global ×    │    │ = T̄_global ×    │
│   T̄_local[0]    │    │   T̄_local[1]    │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         ↓ ×                     ↓ ×
┌─────────────────┐    ┌─────────────────┐
│ T_effective[0]  │    │ T_effective[1]  │
│ = T_fast[0] ×   │    │ = T_fast[1] ×   │
│   T̄_hierarchical│    │   T̄_hierarchical│
└─────────────────┘    └─────────────────┘
```

---

## What It Provides

### 1. Global Meaning
- **T̄_global** captures system-wide coherence patterns
- All layers share the same global geological baseline
- Ensures coordinated behavior across the entire model

### 2. Local Nuance
- **T̄_local[layer]** captures layer-specific specialization
- Early layers can have different patterns than late layers
- Allows diversity within global constraints

### 3. Stability
- **Two timescales** prevent rapid oscillations:
  * Global: η=0.005 (very slow, deep patterns)
  * Local: η=0.01 (slow, layer adaptations)
- Geological memory at both scales

### 4. Diversity
- Layers can diverge in local patterns
- But constrained by global multiplicative gating
- If T̄_global is low (sharp), all layers become sharp
- If T̄_local[i] is high but T̄_global is low, result is still constrained

### 5. Controllability
- **Two independent η parameters**:
  * Tune η_global for system-wide adaptation rate
  * Tune η_local for layer-specific adaptation rate
- Can freeze one while adapting the other
- Natural hierarchical control structure

---

## Multiplicative Combination

**Why T̄_hierarchical = T̄_global × T̄_local (not addition)?**

### Gating Semantics
- If T̄_global = 0.5 (system-wide sharp), all layers are constrained to ≤ 0.5
- If T̄_local[i] = 2.0 (layer wants diffuse), result is 0.5 × 2.0 = 1.0
- Global acts as a gate on local deviations

### Preserves Meaning
- T̄_global = 1.0 → no global constraint (neutral)
- T̄_local = 1.0 → no local deviation (neutral)
- T̄_hierarchical = 1.0 × 1.0 = 1.0 (neutral system)

### Cross-Layer Coordination
- If system is uncertain (T̄_global high), all layers explore
- If system is confident (T̄_global low), all layers exploit
- But layers can still have local variations within this constraint

---

## Validated Behavior

### Initial State (Tick 1)
```
T̄_global:        [1.0, 1.0, ...] (mean=1.0000)
T̄_local[0]:      [1.0, 1.0, ...] (mean=1.0000)
T̄_hierarchical:  [1.0, 1.0, ...] (mean=1.0000)
```

### After 10 Ticks (Tick 11)
```
T̄_global:        [1.002, 1.002, ...] (mean=1.0020)  ← Very slow (η=0.005)
T̄_local[0]:      [1.008, 1.008, ...] (mean=1.0081)  ← Slower (η=0.01)
T̄_hierarchical:  [1.010, 1.010, ...] (mean=1.0101)  ← Product
```

**Observations:**
- T̄_global evolves very slowly (0.2% change over 10 ticks)
- T̄_local evolves faster (0.8% change)
- T̄_hierarchical = T̄_global × T̄_local ≈ 1.002 × 1.008 ≈ 1.010 ✓
- System shows proper timescale separation

---

## Implementation Details

### Controller Initialization
```python
class ChronovisorMixtralController:
    def __init__(
        self,
        config,
        eta_structural_T_global=0.005,  # Very slow
        eta_structural_T_local=0.01,    # Slow
    ):
        # Global structural temperature (shared)
        self.structural_T_global = np.ones(config.num_experts)

        # Per-layer lenses with local structural T
        self.lenses = {
            i: MixtralLens(eta_structural_T=eta_structural_T_local)
            for i in range(config.num_layers)
        }
```

### Update on Micro Boundaries
```python
if self.fast_clock % self.micro_period == 0:
    # 1. Average fast temperature across all layers
    avg_fast_T = mean([lens.temperature_fast for lens in lenses])

    # 2. Update global structural T (very slow EMA)
    T̄_global = (1 - η_global) * T̄_global + η_global * avg_fast_T

    # 3. Update each layer
    for layer_idx, lens in lenses.items():
        # Update local structural T
        lens.structural_T = (1 - η_local) * lens.structural_T
                          + η_local * lens.temperature_fast

        # Apply hierarchical combination
        lens.structural_T_hierarchical = T̄_global × lens.structural_T
        lens.temperature_effective = lens.temperature_fast × lens.structural_T_hierarchical
```

---

## Comparison to Alternatives

### Option 1: Pure Per-Layer (Original)
```
T̄[layer] independent for each layer
```
**Pros:** Maximum layer diversity
**Cons:** No global coordination, layers can diverge wildly

### Option 2: Pure Global
```
T̄_global shared across all layers
```
**Pros:** Perfect coordination
**Cons:** No layer-specific adaptation, loses nuance

### Option 3: Additive
```
T̄ = T̄_global + T̄_local
```
**Pros:** Simple
**Cons:** Doesn't gate properly, hard to interpret semantics

### Option 4: Hierarchical (Implemented) ✓
```
T̄ = T̄_global × T̄_local
```
**Pros:** Global coordination + local nuance + gating semantics
**Cons:** Slightly more complex

---

## Future Extensions

### 1. Per-Expert Global T
- Currently: T̄_global is per-expert but averaged across layers
- Could: Track T̄_global[expert] separately for each expert ID
- Would: Allow expert-level global patterns

### 2. Layer Groups
- Currently: All layers contribute equally to T̄_global
- Could: Group layers (early/middle/late) with separate T̄_global per group
- Would: Allow stage-specific global patterns

### 3. Adaptive η
- Currently: η_global and η_local are fixed
- Could: Modulate η based on coherence or training phase
- Would: Speed up/slow down adaptation dynamically

### 4. Cross-Layer Attention
- Currently: T̄_global is simple average
- Could: Weighted average based on layer importance
- Would: Let critical layers dominate global pattern

---

## References

- **Halcyon AI Feedback**: Identified per-layer limitation, suggested hierarchical
- **Multiplicative Gating**: Common in neural architecture (e.g., LSTM gates)
- **Timescale Separation**: Principle from dynamical systems theory
- **Geological Memory**: Slow accumulation metaphor from earth sciences

---

## Testing

Run hierarchical structural temperature test:
```bash
source .venv/bin/activate
python -c "from chronomoe.chronovisor_mixtral_bridge import test_hierarchical_structural_T; test_hierarchical_structural_T()"
```

Or via the main test:
```bash
python src/chronomoe/chronovisor_mixtral_bridge.py
```

Expected to see T̄_global evolving slower than T̄_local, with proper multiplicative combination.

---

*Hierarchical structural temperature implemented 2025-12-10 per Halcyon AI recommendation*
