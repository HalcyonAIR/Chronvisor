# Geology Fix: Investigation → Diagnosis → Solution → Verification

## Executive Summary

**Problem**: Structural temperature variance (T̄_var) stayed at exactly 0.000000000000000 throughout all training, despite proto-roles forming.

**Root Cause**: Temperature computation only received scalar coherence R, not per-expert drift and reliability vectors. All experts got identical temperatures.

**Fix**: Connected routing statistics to temperature computation via per-expert signals (drift, reliability).

**Verification**: ✅ Test confirms T̄_var now escapes zero and grows over time.

---

## Timeline of Investigation

### 1. Proto-Role Detection (Halcyon's Specification)

**Task**: Build turn-level usage analyzer to detect if experts learned phase-specific roles.

**Method**: Compare 7×8 expert usage matrix (UNTRAINED vs TRAINED)

**Result**:
```
L2 norm: 0.6082 (threshold: 0.5)
✅ STRONG PROTO-ROLE FORMATION
```

**Key Finding**: Expert 6 emerged as early-phase specialist (+26% Inquiry, +25% Premise, +27% Complication)

**Paradox**: Proto-roles formed despite T̄_var=0 → Temperature system was a "passenger", not driver

---

### 2. Bug Investigation

**Diagnostic Script**: `experiments/find_tbar_bug.py`

**Smoking Gun**:
```python
ALL LAYERS, ALL EXPERTS:
  T_fast       = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  structural_T = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

Temperature fields **NEVER UPDATED** during 1000 training steps!

---

### 3. Root Cause Analysis

**Location**: `src/chronomoe/chronovisor_mixtral_bridge.py:428`

**Broken Code**:
```python
# WRONG - only passes scalar coherence_R
lens.compute_temperature(coherence_R=self.coherence_R)
```

**Why This Fails**:
```python
# In MixtralLens.compute_temperature:
temperatures = np.ones(n_experts) * base_T * coherence_factor
# ↑ All experts get SAME temperature (scalar broadcast)
# → variance(temperatures) = 0
# → EMA propagates uniform values
# → structural_T stays uniform forever
```

---

### 4. The Fix (Aligned with Halcyon's Architecture)

**Halcyon's Guidance**:
> "The fix is not 'tune β', it is 'stop throwing away the vectors.'"

**Implementation** (`chronovisor_mixtral_bridge.py:425-458`):

```python
# Compute per-expert drift from usage imbalance
usage = self.expert_usage[layer_idx]
total_usage = usage.sum()
if total_usage > 0:
    usage_dist = usage / total_usage
    ideal_dist = 1.0 / self.config.num_experts
    expert_drifts = np.abs(usage_dist - ideal_dist)
else:
    expert_drifts = np.zeros(self.config.num_experts)

# Compute per-expert reliability from routing weights
if total_usage > 0:
    expert_reliabilities = usage / usage.max()  # Normalize to [0, 1]
else:
    expert_reliabilities = np.ones(self.config.num_experts)

# NOW pass per-expert signals!
lens.compute_temperature(
    coherence_R=self.coherence_R,
    expert_drifts=expert_drifts,
    expert_reliabilities=expert_reliabilities,
)
```

**Temperature Computation** (already correctly implemented in `MixtralLens`):

```python
def compute_temperature(self, coherence_R, expert_drifts=None, expert_reliabilities=None):
    # Coherence factor (global)
    coherence_factor = 1.0 + beta_R * (1.0 - coherence_R)

    # Base temperature
    temperatures = np.ones(n_experts) * base_T * coherence_factor

    # Apply per-expert drift factor
    if expert_drifts is not None:
        normalized_drifts = expert_drifts / expert_drifts.max()
        drift_factors = 1.0 + beta_drift * normalized_drifts
        temperatures *= drift_factors  # ← Per-expert differentiation!

    # Apply per-expert reliability factor
    if expert_reliabilities is not None:
        reliability_factors = 1.0 + beta_reliability * (1.0 - expert_reliabilities)
        temperatures *= reliability_factors  # ← More differentiation!

    # Update fast temperature (now varies per expert)
    self.temperature_fast = temperatures

    # EMA update for structural T
    self.structural_T = (1 - η) * self.structural_T + η * self.temperature_fast
```

---

### 5. Verification Test

**Test Script**: `experiments/test_geology_fix.py`

**Results** (200 training steps):

```
Checkpoint | Loss   | R     | T̄_var (15 decimals)        | T_fast spread
--------------------------------------------------------------------------------
Step   0   | 158.967 | 0.161 | 0.000000000000000 | 0.000000
Step  10   | 110.677 | 0.159 | 0.000000191503439 | 0.452228
Step  50   |  46.678 | 0.188 | 0.000009658897249 | 0.639481
Step 100   |  14.510 | 0.216 | 0.000061442644576 | 0.313933
Step 150   |   6.544 | 0.153 | 0.000096416869359 | 0.544145
Step 199   |   2.428 | 0.110 | 0.000149436518546 | 0.601513
```

**Final State (Layer 0)**:

```
T_fast per expert:
  Values: [1.77, 1.69, 1.75, 1.56, 1.83, 2.16, 1.56, 1.60]
  Spread: 0.60 (was 0.00)
  Std: 0.19 (was 0.00)

structural_T per expert:
  Values: [1.24, 1.21, 1.25, 1.22, 1.25, 1.30, 1.28, 1.27]
  Spread: 0.10 (was 0.00)
  Std: 0.03 (was 0.00)

T̄_hierarchical variance: 0.001730 (was 0.000000)
```

**Verdict**:
```
✅ GEOLOGY IS AWAKE!
   T̄ variance: 0.000149436518546 (was 0.0)
   T_fast std: 0.185170 (was 0.0)

   The structural temperature system is now updating!
```

---

## Key Insights

### Before Fix:
- Proto-roles formed purely through gradient descent on router weights
- Temperature system was a "cardboard cutout" (Halcyon's term)
- ChronoMoE's P×T geometry was not connected to the learning loop

### After Fix:
- Temperature system receives per-expert signals from routing statistics
- Each expert gets different temperature based on its usage pattern
- Variance propagates through EMA to structural_T
- The geology is now actually participating in the system dynamics

---

## Next Steps

### 1. Full Proto-Role Diagnostic with Fix ⏳ RUNNING
Re-run complete 1000-step training with fixed code to confirm:
- Proto-roles still form
- T̄_var grows throughout training
- Temperature system influences learning dynamics

### 2. Geological Valley Detection
With working temperature system, check if valleys form:
- Look for expert clusters with similar structural_T
- Track valley depth and ridge height over training
- Verify valleys correspond to functional roles

### 3. Pressure × Temperature Interaction
Now that T is working, analyze P×T coupling:
- Does pressure bias expert selection?
- Does temperature modulate routing sharpness?
- Are there emergent attractor basins?

### 4. Scaling Study
Test if geology scales to:
- More experts (16, 32)
- Deeper models (8, 12 layers)
- Larger datasets (50k, 100k samples)

---

## Files Modified

### Core Fix
- `src/chronomoe/chronovisor_mixtral_bridge.py` (lines 425-458)
  - Added per-expert drift and reliability computation
  - Pass vectors to lens.compute_temperature()

### Diagnostic Tools Created
- `experiments/analyze_turn_usage.py` (377 lines)
  - Turn-level expert usage analyzer with router hooks
- `experiments/proto_role_diagnostic.py` (270 lines)
  - Complete workflow: untrained → train → trained → compare
- `experiments/find_tbar_bug.py` (197 lines)
  - Inspect trained model to find temperature freeze
- `experiments/test_geology_fix.py` (156 lines)
  - Quick verification test for the fix

### Dataset Fix
- `experiments/conversational_dataset.py`
  - Now tracks actual turn boundaries for precise analysis

---

## Attribution

**Investigation Direction**: Halcyon AI's turn-level usage specification
**Root Cause Diagnosis**: Halcyon AI's architectural analysis
**Implementation**: Claude Code
**Verification**: Test-driven validation approach

---

## Status: ✅ FIX VERIFIED

The structural temperature system is now wired correctly and updating during training. Full diagnostic results pending (currently running in background).
