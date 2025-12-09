# Valley Self-Correction: Why Bad Valleys Fix Themselves

**Date:** Design analysis following structural temperature implementation
**Status:** Design rationale + monitoring implementation

---

## The Problem

Structural temperature creates "geology" — valleys (low T̄) that are easy to route to, and ridges (high T̄) that are hard to route to.

But what if a valley forms around a **bad** expert?

```
Bad Valley = {
    low structural_T (easy to route to)
    low reliability (poor performance when selected)
}
```

This seems dangerous: the router keeps falling into a hole that produces bad outputs.

---

## The Naive Solution (And Why We Don't Need It)

The obvious fix is **asymmetric erosion**: make η_T larger for unreliable experts so bad valleys fill in faster.

```python
# Asymmetric erosion (NOT implemented)
if is_valley and is_unreliable:
    eta_effective = eta_T * 5.0  # Faster erosion
else:
    eta_effective = eta_T  # Normal rate
```

But this adds complexity. Do we actually need it?

---

## The Existing Self-Correction Mechanism

The system already handles bad valleys through the **reliability → T_fast → T̄** pathway.

### Step 1: Bad Expert Gets Hot (Fast Temperature)

Fast temperature depends on reliability:

```python
reliability_factor = 1.0 + β_reliability × (1 - reliability)
T_fast = base_T × coherence_factor × drift_factor × reliability_factor
```

When an expert has low reliability:
- `reliability` is low (e.g., 0.2)
- `(1 - reliability)` is high (e.g., 0.8)
- `reliability_factor` increases (e.g., 1.0 + 0.2 × 0.8 = 1.16)
- `T_fast` increases

**Result:** Bad experts run hot. Their terrain is soft, slippery, hard to commit to.

### Step 2: Hot T_fast Feeds Into T̄ (Structural Temperature)

Structural temperature follows T_fast via EMA:

```
T̄(t+1) = (1 - η_T) × T̄(t) + η_T × T_fast(t)
```

If an expert consistently has high T_fast (because it's unreliable), then T̄ gradually rises.

**Result:** The valley fills in over time. The geological memory erodes.

### Step 3: Rising T̄ Makes Routing Avoid the Expert

Effective temperature = T_fast × T̄:

```
T_effective = T_fast × T̄
```

As T̄ rises, T_effective rises, which means:
- Softer routing (more exploration)
- Less probability mass on this expert
- Router naturally avoids the bad valley

**Result:** The bad valley becomes self-correcting.

---

## The Math: Two Levels of Protection

### Level 1: Behavioral (Instant)

Even before T̄ catches up, the bad expert is already hot:

```
Good expert:  T_fast ≈ 1.0 (reliable)
Bad expert:   T_fast ≈ 1.5 (unreliable)
```

This 50% difference immediately softens routing toward the bad expert.

### Level 2: Structural (Gradual)

Over time, T̄ follows T_fast:

```
After 100 steps:
  Good expert: T̄ → 1.0 (valley deepens)
  Bad expert:  T̄ → 1.5 (valley fills in)
```

The behavioral protection (T_fast) provides instant feedback.
The structural protection (T̄) provides permanent memory.

---

## Why This Is Sufficient

The key insight: **bad valleys are dynamically prohibited states**.

For a valley to persist, the expert must have:
- Low T_fast (which requires high reliability)
- Low T̄ (which requires sustained low T_fast)

But a bad expert has low reliability, which means:
- High T_fast (terrain is hot)
- T̄ trends upward (valley fills in)

**You cannot have a persistent bad valley because the conditions that make it bad (low reliability) also make it fill in.**

The only experts that can maintain deep valleys are those that:
1. Have low drift (stable)
2. Have high coherence alignment (in sync with ensemble)
3. Have high reliability (actually perform well)

These are exactly the experts we *want* to be easy to route to.

---

## The Timescale Question

One might ask: isn't there a dangerous lag between behavioral adaptation (instant) and structural adaptation (~100 steps)?

During that lag:
- T_fast is high (behavioral avoidance)
- T̄ is still low (structural valley persists)

But this is fine because:

1. **T_effective = T_fast × T̄**: Even with low T̄, high T_fast raises T_effective
2. **Routing uses T_effective**: The behavioral correction is immediate
3. **T̄ is just memory**: It smooths out noise, not the only signal

The structural temperature is a *memory buffer*, not the primary control signal. The primary signal is T_fast, which responds instantly.

---

## When Would We Need Asymmetric Erosion?

Asymmetric erosion would only be needed if:

1. **β_reliability is too small**: T_fast doesn't heat up enough for bad experts
2. **Reliability signal is delayed**: Bad performance takes too long to reflect in s_k
3. **Extreme valley depth**: T̄ is so low that even high T_fast × T̄ is still low

In practice:
- β_reliability = 0.2 provides meaningful differentiation
- Reliability updates every tick (no lag)
- T̄ is bounded by T_min/T_max

So the existing mechanism should be sufficient.

---

## Monitoring for Edge Cases

Rather than preemptively adding asymmetric erosion, we implement **valley health monitoring** to detect if the issue ever emerges.

### Valley Health Metric

For each expert k:

```python
is_valley = structural_T[k] < mean - 0.5 * std
reliability = sigmoid(beta_s * s[k])

if is_valley:
    valley_health[k] = reliability  # 0 = bad valley, 1 = good valley
else:
    valley_health[k] = None  # Not a valley
```

### Diagnostic Output

```python
def get_valley_health_diagnostics(self) -> dict:
    return {
        "valleys": [...],           # List of valley expert indices
        "valley_health": [...],     # Reliability of each valley
        "healthy_valleys": [...],   # Valleys with reliability > 0.5
        "unhealthy_valleys": [...], # Valleys with reliability < 0.5
        "at_risk_experts": [...],   # High T_fast but low T̄ (trending bad)
    }
```

### When to Worry

- **unhealthy_valleys > 0 for extended periods**: The self-correction isn't working
- **at_risk_experts increasing**: Valleys forming around unreliable experts

If we observe these patterns in production, we can revisit asymmetric erosion. Until then, we trust the existing mechanism.

---

## Summary

| Question | Answer |
|----------|--------|
| Can bad valleys form? | Temporarily, yes |
| Do bad valleys persist? | No — reliability → T_fast → T̄ fills them |
| Is asymmetric erosion needed? | No — existing mechanism is sufficient |
| How do we verify? | Valley health monitoring |

The design principle: **trust the feedback loop, monitor for failures**.

---

## Configuration Parameters

| Parameter | Default | Effect on Valley Self-Correction |
|-----------|---------|----------------------------------|
| `beta_reliability` | 0.2 | Higher = faster behavioral correction |
| `eta_structural_T` | 0.01 | Higher = faster structural correction |
| `T_min` | 0.3 | Lower = deeper possible valleys |
| `T_max` | 3.0 | Higher = hotter possible ridges |

If bad valleys become a problem, increase `beta_reliability` before adding asymmetric erosion.

---

*This document explains why ChronoMoE doesn't need asymmetric erosion for bad valley correction.*
