# Multistep Pause Controller + Clock Pressure System

## Validation Checklist (Mixtral → DeepSeek)

**Rule 0 (Non-Negotiable):**
All Phase 1–3 tests run with **learning disabled**. Clocks are observers only.
No attractor updates, no value updates, no transition learning.

---

## Phase 0 — Wiring & Invariants

✓ Controller compiles and runs with:
  * single_turn mode
  * multistep mode
  * explicit user commands parsed

✓ No background execution:
  * generation halts on pause
  * no auto-continuation without user input

✓ No second forward pass anywhere:
  * clocks consume only existing signals
  * no re-embedding, no hidden calls

✓ Logs include per-chunk:
  * pressures (fast, mid, slow)
  * weights (w_fast, w_mid, w_slow)
  * net_pressure
  * residual_intent
  * pause_reason

**Status**: ✓ PASS (verified in integration tests)

---

## Phase 1 — Controller Sanity (Model-Agnostic)

### 1.1 Deterministic Pause Behavior

✓ Fixed seed + identical prompt yields identical:
  * chunk boundaries
  * pause reasons
  * net_pressure sign

✓ Pause only occurs when:
  * `pressure_fast < threshold`
  * `net_pressure < 0`
  * `mode == multistep`

✓ No pauses mid-token

**Test**: `tests/test_phase1_1_determinism.py`
**Results**: `test_results/phase1_1_determinism.json`
**Status**: ✓ PASS

---

### 1.2 Pressure Monotonicity

Synthetic sweeps (no model text needed):

✓ Increasing router_entropy → fast_pressure strictly decreases
✓ Increasing margin → mid_pressure strictly decreases
✓ Increasing constraint_penalty → slow_pressure strictly decreases

✓ No sign flips without signal change
✓ No cross-clock leakage (fast ignores macro, slow ignores entropy)

**Test**: `tests/test_phase1_2_monotonicity.py`
**Results**: `test_results/phase1_2_monotonicity.json`
**Status**: ✓ PASS

---

## Phase 2 — Mixtral (Baseline MoE)

### 2.1 Router Entropy Profile

Run identical prompts:

☐ single_turn entropy curve recorded
☐ multistep entropy curve recorded

Expected:
☐ Late-stage entropy collapse reduced in multistep
☐ Expert monopolies reduced
☐ Fast pressure spikes earlier, not only at end

---

### 2.2 Thesis Pressure Test

Prompt: long explanatory task.

☐ pressure_mid decays smoothly across chunks
☐ residual_intent trends toward zero
☐ No large end-of-answer pressure spike
☐ No forced summary behavior

---

### 2.3 User Perturbation Steering

Procedure:
* Chunk 1: baseline explanation
* User: "more technical"
* Chunk 2
* User: "example instead"
* Chunk 3

☐ meso basin transitions occur cleanly
☐ pressure_mid remains positive but decays
☐ slow_pressure remains near zero
☐ No restart-like behavior

---

## Phase 3 — DeepSeek (Clean Geometry)

### 3.1 Pressure Smoothness Comparison

Run same prompts as Mixtral.

Compare statistics:
☐ var(pressure_mid)_DeepSeek < var(pressure_mid)_Mixtral
☐ var(net_pressure)_DeepSeek < var(net_pressure)_Mixtral
☐ fewer fast-instability pauses

---

### 3.2 Manifold Preservation

Using existing FNN / attractor tooling:

☐ basin assignments stable across chunks
☐ no increase in false nearest neighbors
☐ drift magnitude bounded

Expected:
☐ multistep does NOT increase dimensionality

---

### 3.3 Novelty Without Veto

Prompt class: novel but valid reasoning.

☐ slow_pressure near zero
☐ veto rate low
☐ margin low + pressure_mid high correlation present

Contrast with invalid novelty:
☐ slow_pressure strongly negative
☐ veto fires only with high confidence

---

## Phase 4 — Failure Injection

### 4.1 Fast Pressure Abuse

Artificially inflate router_entropy.

☐ immediate pause
☐ no steering
☐ no slow-clock activation

Fail if:
* trajectory changes
* content degrades instead of pausing

---

### 4.2 Identity Violation

Inject constraint penalties.

☐ slow_pressure dominates
☐ veto fires only with confidence
☐ pause precedes any unsafe continuation

---

## Phase 5 — Logging & Plots (Paper-Grade)

☐ pressure_fast / mid / slow vs chunk index
☐ net_pressure vs time
☐ residual_intent decay curves
☐ router_entropy: single vs multistep
☐ Mixtral vs DeepSeek pressure variance

Table:
☐ pause_reason distribution
☐ veto rate (good vs bad novelty)
☐ avg chunks per response

---

## Exit Criteria (Go / No-Go)

**GO if:**
☐ pauses are explainable by pressure balance
☐ fast pressure never steers
☐ multistep removes thesis-per-turn behavior
☐ DeepSeek shows smoother dynamics than Mixtral

**NO-GO if:**
☐ pressure oscillates without signal change
☐ fast clock dominates direction
☐ multistep increases instability or dimensionality

---

**If all GO conditions pass:**
Enable learning updates and repeat Phases 2–3.

This checklist defines correctness, not performance.
