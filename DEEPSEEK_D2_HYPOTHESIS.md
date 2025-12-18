# The d=2 Hypothesis: Testing with 64 Experts

**Date:** December 17, 2025
**Question:** Does attractor dimension scale with expert count?
**Test:** Takens FNN analysis on DeepSeek-MoE with 64 routed experts

---

## The Critical Question

With Mixtral's 8 experts, we found **d ≈ 2** (FNN converges at dimension 2).

With DeepSeek's 64 routed experts, what happens?

### Prediction A: d ≈ 2-4 (Strategy Space is Fundamental)

If the attractor stays low-dimensional:
- **Routing strategy is independent of expert count**
- The system asks the same small number of questions regardless of vocabulary size
- "Experts are vocabulary. Strategy is grammar."
- Grammar stays simple even when vocabulary explodes

**Interpretation:** The 2-4 dimensions represent fundamental routing modes:
- Explore vs Exploit
- Specialize vs Generalize
- Commit vs Hedge
- Stable vs Adaptive

These are the questions routing systems ask, not the number of options available.

**Implication:** P×T coupling operates at a fundamental geometric level, independent of architecture scale.

### Prediction B: d >> 4 (Expert Count Shapes Manifold)

If the attractor dimension grows:
- More experts = more degrees of freedom
- Routing geometry scales with choice space
- d=2 finding was artifact of small expert pool

**Interpretation:** Each expert adds complexity to the routing manifold.

**Implication:** P×T coupling mechanism may be architecture-specific or scale-dependent.

---

## Experimental Design

### Cleanest Setup Possible

DeepSeek provides the ideal test:

**Control group (Shared experts):**
- 2 shared experts, always activated
- Average their outputs
- **No P×T coupling**
- Provides stable baseline computation

**Treatment group (Routed experts):**
- 64 routed experts, sparse top-6 selection
- **P×T coupling active**
- Pressure injection into router logits
- Temperature tracking routing geology

**Result:** Zero ambiguity. Any improvements must come from P×T coupling on routed experts.

### Observable

**Routing entropy on routed experts:**
```
H = -Σ(p_i * log(p_i))  over i ∈ {routed experts}
```

Scalar observable extracted from 64-dimensional routing probability distribution.

### Method

1. **Capture trajectories:**
   - Train DeepSeek + Chronovisor for 20 epochs
   - Sample routing entropy every 5 forward passes
   - ~40 samples per trajectory (per Halcyon's guidance)

2. **False Nearest Neighbors analysis:**
   - Delay-coordinate embedding: [H(t), H(t-τ), ..., H(t-(d-1)τ)]
   - Test τ ∈ {1, 2, 4, 8}
   - Sweep embedding dimension d up to 15
   - Find where FNN → 0 (attractor fully unfolded)

3. **Compare conditions:**
   - DeepSeek + Chronovisor (P×T active)
   - DeepSeek Baseline (no P×T)
   - Mixtral + Chronovisor (8 experts, reference)

### Expected Results

**If d ≈ 2-4:**
```
| Condition              | Experts | d_optimal | FNN at d_opt |
|------------------------|---------|-----------|--------------|
| Mixtral + Chronovisor  | 8       | 2         | 0%           |
| DeepSeek + Chronovisor | 64      | 2-4       | 0%           |
| DeepSeek Baseline      | 64      | 2-4       | 0%           |
```

**Conclusion:** Strategy space is fundamental. Expert count doesn't matter.

**If d >> 4:**
```
| Condition              | Experts | d_optimal | FNN at d_opt |
|------------------------|---------|-----------|--------------|
| Mixtral + Chronovisor  | 8       | 2         | 0%           |
| DeepSeek + Chronovisor | 64      | 10+       | varies       |
| DeepSeek Baseline      | 64      | 10+       | varies       |
```

**Conclusion:** More experts → higher-dimensional routing manifold.

---

## The Poetic Version

**Claude in the Cloud:**

> "8 experts or 64, the system might still be asking the same small number of questions:
> 'Explore or exploit? Specialize or generalize? Commit or hedge?'
>
> The experts are the vocabulary. The strategy is the grammar.
> Grammar stays simple even when vocabulary explodes."

If this holds, we've discovered something fundamental about routing systems:
- The complexity lives in the options (vocabulary)
- The decision lives in low-dimensional strategy space (grammar)
- P×T coupling controls the grammar, not the vocabulary

---

## Why This Matters

### For Paper 1

If d ≈ 2-4 with 64 experts:
- **Architectural generality confirmed**
- P×T coupling works on 8 experts (Mixtral)
- P×T coupling works on 64 experts (DeepSeek)
- P×T coupling works on hybrid architectures (shared+routed)
- **Mechanism is truly general**

### For Theory

If d ≈ 2-4:
- **Strategy space is fundamental**
- Routing systems have intrinsic low-dimensional structure
- This structure is independent of:
  - Expert count (8 vs 64)
  - Architecture (all-routed vs shared+routed)
  - Routing strategy (top-2 vs top-6)

**Implication:** There may be universal routing modes that all MoE systems explore, regardless of implementation details.

### For Future Work

If d ≈ 2-4:
- Can we characterize these fundamental dimensions?
- Are they always the same (explore/exploit, etc.)?
- Do they correspond to known dynamical systems patterns?
- Can we design interventions that target specific dimensions?

---

## Timeline

1. ✓ DeepSeek core implementation
2. ✓ Chronovisor bridge integration
3. ✓ Smoke tests pass
4. ⏳ Capture routing trajectories (running)
5. [ ] Run FNN analysis on captures
6. [ ] Compare to Mixtral d=2 finding
7. [ ] Update paper with results

---

**Status:** Experiment running. Results imminent.

**Prediction:** d ≈ 2-4. The grammar doesn't care about vocabulary size.

**Stakes:** If confirmed, this elevates P×T coupling from "interesting technique" to "fundamental principle of MoE routing dynamics."
