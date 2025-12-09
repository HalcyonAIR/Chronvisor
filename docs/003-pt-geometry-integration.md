# P×T Geometry Integration - Complete

**Status:** Integration complete and tested
**Date:** 2025-12-10

---

## Summary

Successfully integrated the full P×T (Pressure × Temperature) geometry from ChronoMoE into the Mixtral implementation, completing the gap identified by Halcyon AI.

## What Was Added

### 1. Temperature Field in Router (`mixtral_core.py`)

**Before:**
```python
router_logits = self.gate(hidden_states)
if pressure_bias is not None:
    router_logits = router_logits + pressure_bias
```

**After:**
```python
router_logits = self.gate(hidden_states)

# Apply pressure (P)
if pressure_bias is not None:
    router_logits = router_logits + pressure_bias

# Apply temperature (T)
if temperature_field is not None:
    temp_safe = torch.clamp(temperature_field, min=0.1, max=10.0)
    router_logits = router_logits / temp_safe.unsqueeze(0).unsqueeze(0)
```

**Formula:** `logits'_k = (logits_k + P_k) / T_k`

Where:
- **P_k** (Pressure): Force field pushing toward/away from experts
- **T_k** (Temperature): Permeability controlling routing sharpness

### 2. Structural Temperature (Geological Memory)

Added to `MixtralLens`:

```python
# Fast temperature (immediate)
T_fast = base_T * (1 + β_R*(1-R)) * (1 + β_drift*d_k) * (1 + β_rel*(1-s_k))

# Structural temperature (geological memory via EMA)
T̄(t+1) = (1 - η) * T̄(t) + η * T_fast(t)

# Effective temperature
T_effective = T_fast × T̄
```

Parameters:
- `eta_structural_T = 0.01` - EMA rate (slow geological memory)
- `base_temperature = 1.0` - Baseline temperature

### 3. Kuramoto Coherence (Patent-Compliant)

**Before:** Entropy-based coherence
```python
entropy = -Σ p_i log(p_i)
coherence = 1 - (entropy / log(n_experts))
```

**After:** Kuramoto order parameter R
```python
Z = (1/N) Σ exp(i*φ_k)
R = |Z|  # R ∈ [0, 1]
psi = angle(Z)
```

Where:
- **R = 1**: Perfect synchronization (high coherence)
- **R = 0**: Complete desynchronization (low coherence)
- **φ_k**: Phase of expert k

This is the patent-compliant coherence metric from Chronovisor.

### 4. Language Model Head

Added `ChronovisorMixtralForCausalLM`:

```python
class ChronovisorMixtralForCausalLM(nn.Module):
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.model = ChronovisorMixtralModel(config)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        hidden_states, chrono_state = self.model(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, chrono_state
```

Includes a simple `generate()` method for autoregressive sampling.

---

## P×T Geometry Flow

```
┌──────────────────────────────────────────────────────────────┐
│  Expert Phases (φ_k)                                         │
│  Updated based on routing activity                           │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ↓
┌──────────────────────────────────────────────────────────────┐
│  Kuramoto Coherence                                          │
│  R = |mean(exp(i*φ_k))|                                      │
│  R ∈ [0, 1], measures synchronization                        │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ↓
┌──────────────────────────────────────────────────────────────┐
│  Temperature Field (per layer)                               │
│  T_fast = base_T * f(R, drift, reliability)                  │
│  T̄ = (1-η)*T̄ + η*T_fast  (geological memory)               │
│  T_eff = T_fast × T̄                                         │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ↓
┌──────────────────────────────────────────────────────────────┐
│  Pressure Field (per layer)                                  │
│  P_k = (ideal_usage - actual_usage) * 2.0                    │
│  Encourages underutilized experts                            │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ↓
┌──────────────────────────────────────────────────────────────┐
│  Mixtral Router                                              │
│  logits'_k = (logits_k + P_k) / T_k                          │
│  High T → diffuse (explore), Low T → sharp (exploit)         │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ↓
┌──────────────────────────────────────────────────────────────┐
│  Expert Selection                                            │
│  Top-k experts selected, weighted by routing probs           │
│  Routing statistics fed back to controller                   │
└──────────────────────────────────────────────────────────────┘
```

---

## Temperature Field Components

Temperature is computed from three factors:

### 1. Coherence Factor (Global)
```python
coherence_factor = 1.0 + β_R * (1.0 - R)
```
- Low R (desynchronized) → High temperature (explore)
- High R (synchronized) → Low temperature (exploit)
- `β_R = 0.5` (default)

### 2. Drift Factor (Per-Expert)
```python
drift_factors = 1.0 + β_drift * normalized_drift_k
```
- High drift → High temperature (less trusted)
- Low drift → Low temperature (more trusted)
- `β_drift = 0.3` (default)

### 3. Reliability Factor (Per-Expert)
```python
reliability_factors = 1.0 + β_rel * (1.0 - reliability_k)
```
- Low reliability → High temperature (uncertain)
- High reliability → Low temperature (confident)
- `β_reliability = 0.2` (default)

### Final Temperature
```python
T_k = base_T * coherence_factor * drift_factors[k] * reliability_factors[k]
T_k = clip(T_k, T_min=0.3, T_max=3.0)
```

---

## Validated Behaviors

### ✓ Router P×T Application
```
Input: hidden_states (2, 16, 256)
Router logits: (2, 16, 4) experts
+ Pressure: (4,) per-expert bias
/ Temperature: (4,) per-expert scale
→ Routing weights: (2, 16, 2) top-2
```

### ✓ Kuramoto Coherence Evolution
```
Tick 1: R=0.1878, ΔR=+0.1878
Tick 2: R=0.1872, ΔR=-0.0006
Tick 3: R=0.1866, ΔR=-0.0006
...
Tick 11: R=0.1829, ΔR=-0.0004
```
System shows coherent evolution of Kuramoto R.

### ✓ Structural Temperature (EMA)
```python
# Fast temperature responds immediately to coherence changes
# Structural T̄ slowly accumulates geological memory
# Effective T combines both for stable control
```

### ✓ Language Model Forward Pass
```
Input: input_ids (2, 16) token indices
→ Embeddings: (2, 16, 256)
→ Mixtral + Chronovisor: (2, 16, 256)
→ LM head: (2, 16, 1000) logits
✓ Kuramoto R tracked throughout
```

---

## Key Design Decisions

### 1. Why T_eff = T_fast × T̄ (not addition)?
- **Multiplicative** allows both fields to gate each other
- If either is low (sharp), result is sharp
- If both are high (diffuse), result is very diffuse
- More expressive than additive combination

### 2. Why EMA for structural T?
- **Geological memory**: Slow accumulation of routing history
- **Stability**: Prevents rapid oscillations in temperature
- **η = 0.01**: Very slow, captures long-term patterns

### 3. Why Kuramoto R instead of entropy?
- **Patent compliance**: Core Chronovisor metric
- **Physical interpretation**: Phase synchronization of oscillators
- **Richer dynamics**: Captures phase relationships, not just distribution

### 4. Why separate P and T?
- **Pressure (P)**: "Where should routing go?" (directional force)
- **Temperature (T)**: "How sharp should routing be?" (permeability)
- **Orthogonal control**: Two degrees of freedom for routing modulation

---

## Files Modified

### `mixtral_core.py`
- Updated `MixtralRouter.forward()` to apply temperature division
- Added `temperature_field` buffer to `MixtralSparseMoELayer`
- Updated docstrings to document P×T geometry

### `chronovisor_mixtral_bridge.py`
- Added `compute_kuramoto_R_and_psi()` function
- Extended `MixtralLens` with temperature computation and structural T
- Replaced entropy coherence with Kuramoto R in controller
- Added `get_temperature_for_layer()` method
- Created `ChronovisorMixtralForCausalLM` language model class
- Updated forward pass to apply both P and T fields

---

## Next Steps (As Per Halcyon AI)

### Completed ✓
1. ✓ Wire temperature field into router
2. ✓ Add structural temperature (geological memory)
3. ✓ Use Kuramoto coherence (patent-compliant)
4. ✓ Add language modeling head

### Remaining (Future Work)
1. **Valley health diagnostics** - Track healthy/unhealthy expert valleys
2. **Meta-knob integration** - LLM-controllable κ for pressure/temp scaling
3. **Training infrastructure** - Dataset loading, optimization, checkpointing
4. **Evaluation framework** - Baseline comparisons, metrics

---

## Testing

Run the complete integration test:
```bash
source .venv/bin/activate
python src/chronomoe/chronovisor_mixtral_bridge.py
```

Expected output:
```
Testing ChronovisorMixtralModel (hidden states)
✓ Output shape: torch.Size([2, 16, 256])
✓ Kuramoto R: 0.1878
✓ Δ R: 0.1878

Testing ChronovisorMixtralForCausalLM (language model)
✓ Logits shape: torch.Size([2, 16, 1000])
✓ Kuramoto R: 0.2072

✓ P×T Mixtral integration complete!
  - Pressure field (P): Biases routing toward/away from experts
  - Temperature field (T): Controls routing sharpness/diffuseness
  - Structural T: Geological memory via EMA
  - Kuramoto R: Patent-compliant coherence metric
  - Language modeling: Embeddings + LM head ready for training
```

---

## References

- **ChronoMoE Bridge** (`bridge.py`): Original P×T implementation with full CulturalEvolutionaryController
- **Router** (`router.py`): `forward_with_temperature()` method pattern
- **Kuramoto Model**: Phase synchronization in coupled oscillators
- **Seven Guarantees** (`docs/005-seven-guarantees.md`): Why this architecture is stable

---

*P×T geometry integration completed 2025-12-10*
