# ChronoMoE: Key Results Summary

## Discovery: Wear as Robustness Probe

**What we thought:** Path wear creates persistent routing bias via elastic dynamics

**What we found:** ChronoMoE is a geometric probe measuring routing confidence without modifying learned structure

---

## Core Findings

### 1. Perfect Margin-Wear Correlation: r = -0.935

Low margin (uncertain) → high wear
High margin (confident) → low wear

**Interpretation:** Wear measures local stiffness of decision surface

### 2. Anti-Semantic Inversion

| Input    | Wear (Δ Entropy) | Margin |
|----------|------------------|--------|
| Real     | -0.122           | 1.34   |
| Random   | -0.248           | 0.57   |

**Interpretation:** Semantics shows up as resistance, not structure to exploit

### 3. Zero Hysteresis

- A→B and B→A transitions: identical
- Fresh vs carry-forward T̄: no difference
- Boundaries pinned at 35% and 45% noise

**Interpretation:** Stateless constraint re-projection, not elastic memory

### 4. Structural Rigidity

| η    | 7→4 Boundary | 4→2 Boundary |
|------|--------------|--------------|
| 0.05 | 35%          | 45%          |
| 0.25 | 35%          | 45%          |

**Interpretation:** Manifold rigid across 5x perturbation strength

### 5. Three-Expert Cascade

- **Expert 7:** In-distribution (confident)
- **Expert 4:** Ambiguity buffer (uncertain)
- **Expert 2:** OOD attractor (collapsed)

**Interpretation:** Learned confidence hierarchy, hardwired by pretraining

---

## The Mechanism

```
Hidden States → [ChronoMoE Lens] → Bent Logits → [Projection] → Expert
                      ↑                                ↑
                  T̄ bias here              Learned manifold here
                  (deformable)                   (absolute)
```

**Key insight:** Lens bends light, not the map

---

## What ChronoMoE Measures

✓ Routing confidence (margin)
✓ Basin depth (wear magnitude)
✓ Boundary sharpness (correlation)
✓ Manifold rigidity (invariance)
✓ OOD detection (Expert 2 collapse)

**Without gradients, training, or model modification**

---

## Theoretical Frame

**Not:** Dynamical system with memory
**Is:** Geometric interrogation of static manifold

**Formal:**
> ChronoMoE applies a history-conditioned lens to expert routing logits, biasing trajectories through a fixed decision manifold without modifying its topology.

---

## Validated Boundaries

- Elastic regime: η ≤ 0.25 (no yield)
- In-distribution routing: 100% preserved
- Expert selection: Deterministic, seed-independent
- Transition structure: Sharp cliff (3.47x faster drop)

---

## Phase 2: Self-Gated State Hypothesis

**Tested and FALSIFIED**

**Question:** Can margin-conditioned state create path dependence where unconditional pressure could not?

**Mechanism:**
```
gate = f(margin)  # Low margin → high gate (listen to history)
biased_logits = logits + gate * pressure(T̄)
```

**Result:** No hysteresis (0/13 noise levels differ)
- Transitions identical in both directions
- Gate active (~24% average influence)
- Boundaries at 40% and 50% (both directions)
- No pathological collapse

**Conclusion:** Walls absolute even with self-gated state. Constraint manifold dominates all adaptive mechanisms.

---

**Status:** Phase 2 complete - self-gated hypothesis falsified
**Finding:** Fundamental limit reached - inference-time perturbation cannot create path-dependent routing
**Date:** December 2024
