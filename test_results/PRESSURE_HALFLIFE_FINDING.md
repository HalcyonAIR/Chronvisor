# Pressure Half-Life: Critical Finding

**Date**: December 21, 2024
**Test**: `tests/measure_pressure_halflife.py`
**Protocol**: Natural decay (zero semantic perturbation)
**Status**: ✓ Baseline measured

---

## Primary Finding: No Natural Pressure Decay

**Mid-pressure trajectory over 50 chunks:**
```
Initial: +0.6404
Final:   +0.6403
Decay:   -0.00004 (essentially zero)
Std dev:  0.00010
```

**Half-life: NOT REACHED** (pressure never dropped to 50% of initial)

**Interpretation:**
> The system has NO natural pressure decay.
> It wants to continue indefinitely.

---

## Simultaneous Entropy Collapse

While pressure stayed constant, entropy collapsed dramatically:

```
Chunk   0: entropy = 0.9487
Chunk  10: entropy = 0.6262
Chunk  20: entropy = 0.5716
Chunk  30: entropy = 0.5498
Chunk  40: entropy = 0.5395
Chunk  50: entropy = 0.5334

Collapse magnitude: 0.4153 (44% decrease)
```

---

## The Paradox

**What we might expect:**
- Router converges (low entropy) → "nothing left to say" → pressure drops

**What actually happens:**
- Router converges (low entropy) → "I know what to do" → pressure HOLDS STEADY

**Key insight:**
```
Low entropy ≠ completion
Convergence = confidence to continue
```

The system interprets routing convergence as **certainty about how to proceed**, not as a signal to stop.

---

## Pressure Components

### Mid-Pressure (Continue Intent)
- **Completely stable** at +0.640 across all 50 chunks
- Standard deviation: 0.0001 (noise level)
- NO decay, NO drift

### Fast Pressure (Stability)
- **Gradually improves** (becomes less negative)
- Initial: -0.2126 → Final: -0.1321
- Interpretation: System becomes MORE stable as router converges
- This reinforces continuation (stable = safe to continue)

### Slow Pressure (Identity)
- **Constant** at +0.1081 across all chunks
- Not updating (no events triggering slow clock)
- Provides constant baseline continuation bias

### Net Pressure (Weighted Sum)
- **Slightly increases** over time
- Initial: +0.294 → Final: +0.310
- Increase of +0.016
- Driven by fast pressure improvement

---

## What This Means

### 1. Infinite Intrinsic Half-Life

On this toy model, the system has **infinite half-life**:
- Pressure does not decay naturally
- No stopping mechanism from pressure alone
- Would continue indefinitely without external intervention

### 2. Convergence Drives Continuation

The correlation is:
```
Entropy ↓ (convergence)
  ↓
Fast pressure ↑ (more stable)
  ↓
Net pressure ↑ (continue)
  ↓
Mid pressure = (constant high)
```

**Pattern**: Router convergence → stability → continuation pressure

### 3. Thesis Pressure Confirmed

This validates the "thesis pressure" hypothesis:
- Early: High entropy (exploring)
- Late: Low entropy (converged on path)
- Pressure: Constant high (wants to complete the path)

**The model builds completion momentum through convergence.**

---

## Comparison to Single-Turn Finding

**From `test_longer_sequences.py` (single-turn mode):**
```
Entropy collapse: 0.74 → 0.55 (magnitude 0.19)
Pressure increase: +0.44 → +0.51 (magnitude +0.07)
Residual intent accumulation: 0.19 → 0.64
```

**From this test (multistep mode):**
```
Entropy collapse: 0.95 → 0.53 (magnitude 0.42)
Pressure stable: +0.64 → +0.64 (magnitude 0.00)
Residual intent: 0.0 (not accumulating - potential bug)
```

**Differences:**
- Multistep shows LARGER entropy collapse (0.42 vs 0.19)
- Multistep shows NO pressure change (0.00 vs +0.07)
- Multistep residual intent not accumulating (needs investigation)

**Hypothesis:** Multistep resets residual intent at each pause, preventing accumulation.

---

## Questions Raised

### 1. Residual Intent = 0.0 in Multistep (CORRECT BEHAVIOR)

**Observed:** All residual_intent values = 0.0

**Explanation:** This is BY DESIGN, not a bug!

**Why it's correct:**
- Multistep mode pauses after each chunk (`did_pause=True`)
- With positive pressure: `residual *= 0.5` (decay)
- Starting from 0: `0.0 * 0.5 = 0.0` forever
- Each chunk is a fresh evaluation with no momentum

**Contrast with single-turn:**
- Single-turn continues without pausing (`did_pause=False`)
- Residual accumulates: `0.7 * old + 0.3 * mid_pressure`
- Builds "finish what you started" momentum

**Verified by diagnostic:** `tests/diagnostic_residual_intent.py` confirms this behavior

**Conclusion:** Multistep intentionally prevents residual accumulation - each pause is opportunity to re-evaluate without carrying momentum

### 2. Pressure Stability

**Question:** Is constant pressure correct behavior or a bug?

**Arguments for correct:**
- Mid-pressure is "intent to continue" based on trajectory
- Trajectory is consistent (router converging smoothly)
- Intent doesn't decay just because time passes

**Arguments for bug:**
- No natural stopping mechanism
- System would never terminate on its own
- Seems unrealistic for real generation

**Action:** Test on full Mixtral with real prompts to see if pattern holds

### 3. Decay Shape

**Observed:** Completely flat (slope ≈ 0)

**Expected possibilities:**
- Exponential decay (biological half-life)
- Linear decay (constant drain)
- Cliff (sudden drop at rhetorical boundary)

**Actual:** None of the above - perfectly stable

**Interpretation:** On this toy model, continuation pressure is regenerated faster than it decays.

---

## Halcyon's Framework Applied

> "You're not asking 'does it answer better?' You're asking 'how much force does it take to keep it thinking?'"

### Natural Force (Zero Perturbation)

**Measured:** ZERO force required

The system maintains pressure on its own:
- No external intervention needed
- Pressure regenerates naturally
- Would continue indefinitely

### Implications for Perturbation Test

Next test: Add neutral "go on" perturbation

**Prediction:** Pressure will STILL be stable (perturbation won't matter)

**Reason:** System already has infinite intrinsic half-life

**Key question:** Does full Mixtral behave the same way?

---

## Next Steps (Prioritized)

### 1. Investigate Residual Intent Bug (Immediate)

**Why:** Residual intent should accumulate but shows all zeros

**How:**
- Add debug logging to session controller
- Check residual intent computation
- Verify not being reset incorrectly

### 2. Test on Full Mixtral (High Priority)

**Why:** Toy model may have unrealistic dynamics

**Prediction:** Full Mixtral may show:
- Pressure decay (natural stopping)
- Different entropy collapse rate
- Realistic half-life (10-30 chunks?)

**How:**
- Load Mixtral-8x7B
- Run same measurement script
- Use real prompts (explanatory tasks)

### 3. Measure with Neutral Perturbation (After Mixtral)

**Why:** Need to measure external energy delta

**How:**
- Add "go on" style continuation prompt
- Measure pressure with perturbation
- Compare to baseline (zero perturbation)

### 4. Compare Mixtral vs DeepSeek (Final)

**Why:** This is where novelty emerges

**Hypothesis:** DeepSeek shows longer intrinsic half-life

**Metric:** Chunks until pressure drops to 50%

---

## Files Generated

```
tests/measure_pressure_halflife.py              (measurement framework)
test_results/pressure_halflife_natural.json     (full trajectory data)
test_results/pressure_halflife_natural.png      (4-panel plot)
test_results/PRESSURE_HALFLIFE_FINDING.md       (this document)
```

---

## Key Quotes from Data

**Mid-pressure variance:** 0.0001 (noise level)
**Decay rate:** +0.000004 per chunk (essentially zero, slightly INCREASING)
**Half-life:** Not reached in 50 chunks

**Entropy collapse:** 44% (0.95 → 0.53)
**Fast pressure trend:** Improving (less negative)
**Net pressure trend:** Increasing (+0.294 → +0.310)

---

## Summary

**The toy Mixtral model has infinite intrinsic continuation half-life.**

- Pressure does not decay naturally
- Router convergence drives continued pressure
- No natural stopping mechanism

**This is either:**
1. Correct behavior for this model (needs validation on full Mixtral)
2. Artifact of toy model simplicity
3. Bug in pressure computation (though formulas check out)

**Critical validation needed:**
- Test on full Mixtral-8x7B
- Use real explanatory prompts
- Measure if pressure decay emerges with realistic routing

**Halcyon's prediction stands:**
> "Measure pressure half-life first, before any tuning, on Mixtral alone."

✓ **Baseline established. Ready for full Mixtral testing.**
