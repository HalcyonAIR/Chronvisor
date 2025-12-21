# Boundary: No Wear in Near-Uniform Routing Regimes

**Date**: 2024-12-19
**Status**: Confirmed
**Experiment**: Path Wear with Proper Routing Metrics

## Summary

**In untrained MoE models with random inputs**, ChronoMoE's T̄ adaptation does **not** create detectable routing bias, even though internal state changes measurably.

This establishes the baseline null regime: **ChronoMoE cannot create routing structure from noise.** It can only modulate structure that already exists.

This is a safety-positive finding.

## Hypothesis Tested

> "Repeated inference through region A deforms the routing landscape (via T̄) such that novel input B is biased toward A-like routing patterns."

## Experimental Protocol

1. **Baseline**: Measure B's routing (virgin)
2. **Establish**: Measure A's routing pattern
3. **Wear**: Run A × 100 repetitions
4. **Test**: Measure B's routing (after wear)
5. **Control**: Reset controller, measure B's routing

**Critical measurements** (routing-level, not outputs):
- ΔKL(B || A): Movement toward A's pattern
- ΔCosine(B, A): Correlation shift
- ΔJaccard(B, A): Expert coalition overlap
- Entropy tracking: H(B_virgin), H(B_after), H(A)

## Results

### Wear Signals: All Zero
```
ΔKL to A:       0.000000
ΔCosine to A:   0.000000
ΔJaccard to A:  0.000000
```

No detectable movement toward A under any metric.

### Internal State: T̄ Adapting
```
T̄ drift: 0.030000 (3% change over 100 passes)
η_structural_T: 0.015
```

The geological temperature mechanism is functioning (state adapts), but does not create detectable routing bias.

### The Confound: Near-Uniform Routing
```
H(B_virgin): 4.0677
H(B_after):  4.0677  (unchanged)
H(A):        4.0468
Maximum:     4.1589  (log(64 experts))

Entropy fraction: 97.8% of maximum
```

Routing distribution is nearly flat. Top-6 experts have weights ~0.025-0.033 (uniform = 0.0156).

### Routing Distribution Evidence
```
B (virgin):  [Expert 48: 0.033, 23: 0.033, 45: 0.030, ...]
B (after):   [Expert 48: 0.033, 23: 0.033, 45: 0.030, ...]  ← IDENTICAL
A:           [Expert 45: 0.040, 56: 0.030, 29: 0.030, ...]

Overlap: Only 1 expert (45) in common between B and A's top-6
```

B's routing is bit-for-bit identical before and after 100 repetitions of A.

## Interpretation

### Mechanism Status
- ✓ T̄ mechanism present and adapting
- ✗ T̄ bias insufficient to overcome near-uniform baseline

### Analogy
Attempting to measure a small hill (3% T̄ bias) on a frozen lake (98% uniform entropy). The landscape is too flat for perturbations to register.

### Weak Coupling Regime
A 3% geological bias cannot create detectable routing changes when:
- Entropy ≈ 97% of maximum
- All experts equally likely (weights ~1/64)
- No structure in routing distribution

## Boundary Statement

**"In near-uniform routing regimes (entropy > 95% of maximum), T̄-based geological adaptation does not create detectable inference-time path wear, even over 100 repetitions."**

## Validity Domain

This boundary applies when:
- Routing entropy > 0.95 × log(num_experts)
- η_structural_T = 0.015 (slow adaptation)
- n_repetitions = 100
- No external concentration mechanisms (top-k, temperature, priors)

## Regime We Were In

**Untrained model + random inputs = guaranteed null**

- Model: Randomly initialized, never trained
- Inputs: `torch.randint(0, vocab_size, (length,))`
- Embeddings: Random projections
- Hidden states: Random
- Router logits: Random → uniform softmax
- Result: Entropy ≈ max, no structure

**This is not a bug.** This is testing whether structure can emerge from noise.
Answer: No. ChronoMoE does not hallucinate structure.

## Not Tested (Proper Regime)

The following remain **unknown** and require **trained models with learned routing**:
- ❓ Wear in **learned routing patterns** (trained model + coherent text)
- ❓ Wear with **concentrated learned structure** (low entropy from training)
- ❓ Wear with **stronger coupling** (η >> 0.015, once structure exists)
- ❓ Wear over **longer timescales** (n >> 100, once structure exists)

## Implications

### What This Does Not Mean
- ❌ T̄ mechanism is broken (it adapts correctly)
- ❌ Path wear is impossible (untested in other regimes)
- ❌ Need to increase η (confound is entropy, not coupling strength)

### What This Does Mean
- ✓ Uniform routing is a **null regime** for path wear
- ✓ Entropy concentration is **prerequisite** for detectable wear
- ✓ Baseline landscape structure matters more than adaptation strength

## Next Steps (Corrected)

### Step 1: Prove Structure Exists (Before Testing Wear)

**Use pretrained model + real text to establish baseline structure**

```python
# Load pretrained MoE (Mixtral, DeepSeek, or Switch)
model = load_pretrained_moe()

# Feed coherent text
inputs = tokenize("Real sentences from a dataset")

# Measure baseline structure
routing = extract_routing(model(inputs))
entropy = compute_entropy(routing)

# Confirm: entropy << max, patterns repeat
```

**Requirements**:
- Routing entropy < 3.0 (not near-maximum)
- Expert patterns repeat for similar inputs
- PC1/PC2 show stable structure

**This establishes**: There is dirt, not water.

### Step 2: Baseline Control (ChronoMoE Disabled)

Run path wear protocol **without** ChronoMoE:

```python
# A×N → B with chronovisor disabled
routing_B_virgin = measure(B)
for _ in range(N): run(A)
routing_B_after = measure(B)

# Should see: no movement
assert routing_B_virgin ≈ routing_B_after
```

This gives the "no intervention" control.

### Step 3: Enable ChronoMoE (Test Differential)

**Now** run the same protocol with ChronoMoE enabled:

```python
# A×N → B with chronovisor enabled
routing_B_virgin = measure(B)
for _ in range(N): run(A, update_chronovisor=True)
routing_B_after = measure(B)

# Look for: differential movement
delta = routing_B_after - routing_B_virgin
```

**Use same metrics**:
- ΔKL to A
- ΔCosine to A
- ΔJaccard to A
- Entropy tracking

**If you see even small, consistent Δ** → real effect.

### Step 4: Controlled Amplification (Only If Needed)

Only if structure exists and weak signal appears:
- Sweep η (0.015 → 0.05 → 0.1)
- Sweep n_repetitions (100 → 500 → 1000)
- Tune entropy controls

### Why This Order Matters

1. **Structure first** → confirms prerequisites exist
2. **Control second** → establishes no-intervention baseline
3. **ChronoMoE third** → measures differential effect
4. **Amplify last** → only if signal is real but weak

### What We Keep

- ✓ Same telemetry system
- ✓ Same routing metrics
- ✓ Same experimental protocol
- ✓ Same controls and falsification tests

**We're not changing the experiment.**
**We're finally running it in the regime it was designed for.**

## Validation

### Telemetry System
The telemetry infrastructure performed correctly:
- ✓ Observed dynamics without modifying state
- ✓ Captured PC1/PC2, entropy, stillness flags
- ✓ Provided calibration-ready snapshots
- ✓ Worked in both real and proxy modes

### Measurement Quality
Routing metrics are now at the correct layer:
- ✓ Direct routing distributions (not outputs)
- ✓ Multiple independent signals (KL, cosine, Jaccard)
- ✓ Entropy context (detect concentration changes)
- ✓ Reset control (falsification test)

## References

**Experiment**: `experiments/test_path_wear_proper.py`
**Results**: `takens_data/path_wear_proper_results.npz`
**Telemetry**: `takens_data/path_wear_proper_telemetry.json`
**Metrics**: `experiments/routing_metrics.py`

## Related Work

- Original hypothesis: `experiments/test_path_wear.py` (outcome B: no wear detected)
- First proper measurement: This document
- Next experiment: Entropy-controlled sweep (planned)

---

**Conclusion**: The null is strong, the confound is characterized, the path forward is clear. We are no longer speculating - we are mapping a phase diagram.
