# ChronoMoE as Routing Robustness Probe

**Status**: Experimentally validated
**Date**: December 2024
**Key Finding**: ChronoMoE measures routing confidence and basin geometry without modifying the learned decision manifold

---

## Executive Summary

We set out to test if repeated inference creates "path wear" - persistent routing bias via geological temperature (T̄) adaptation. What we discovered instead is more fundamental:

**ChronoMoE is a geometric interrogation tool that reveals the structure of pretrained routing decisions through controlled perturbation.**

The mechanism is not elastic dynamics with memory. It's **stateless constraint satisfaction** - each forward pass re-projects perturbed logits onto a fixed, learned decision manifold.

This reframes ChronoMoE from a potential training mechanism to an **inference-time probe** of routing robustness, confidence boundaries, and basin geometry.

---

## The Experimental Arc

### Initial Hypothesis: Path Wear via Elastic Dynamics

> "Repeated inference through region A deforms the routing landscape (via T̄) such that novel input B is biased toward A-like routing patterns."

**Expected signatures:**
- Cross-wear: A×N → B shows routing drift toward A
- Hysteresis: Transition points depend on approach direction
- Wear magnitude scales with η (learning rate)
- Elastic relaxation after perturbation removal

### What We Found Instead

**1. Anti-Semantic Inversion**
- Real text: Δ entropy = -0.122 (weak wear)
- Random noise: Δ entropy = -0.248 (2x stronger wear)
- **Semantics shows up as resistance, not as structure to exploit**

**2. Perfect Margin-Wear Correlation**
- r = -0.935 between initial margin and wear magnitude
- Low margin (uncertain routing) → high wear
- High margin (confident routing) → low wear
- **Wear is a probe for routing flatness/stiffness**

**3. Zero Hysteresis**
- A→B and B→A transitions identical
- Carry-forward T̄ vs fresh T̄: no difference
- Transition points invariant (35% and 45% noise, pinned)
- **No path dependence, no elastic memory**

**4. Structural Rigidity Under Stress**
- η swept from 0.05 to 0.25 (5x range)
- Boundaries unmoved: 7→4 at 35%, 4→2 at 45%
- Expert 4 width constant at 10%
- In-distribution routing perfect (100% → Expert 7)
- **No yield point up to 500% of detection threshold**

---

## The Mechanism: Constraint Re-Projection

### What Actually Happens Each Forward Pass

```
1. Compute router logits from hidden states
2. Add T̄ bias (perturbation from geological lens)
3. Project onto learned constraint manifold ← This dominates
4. Select expert according to pretrained partition
```

**Step 3 is absolute.** The router continuously re-solves:

*"Given these logits + bias, what is the valid routing under my learned partition of feature space?"*

If the bias tries to push routing into an invalid region, the projection step **kills it instantly**.

### Why This Explains Everything

**Elasticity would predict:**
- Delayed relaxation
- Partial boundary drift
- Hysteresis loops under repeated cycling

**Constraint re-projection predicts:**
- Perfect up/down symmetry ✓ (observed)
- Immediate return to canonical attractors ✓ (observed)
- Identical transitions regardless of history ✓ (observed)
- Strong correlation with instantaneous geometry ✓ (r=-0.935)

### The Three-Expert Cascade is Not a Potential Well

Expert selection: 7 → 4 → 2 as noise increases

This is not a smooth potential landscape. It's a **piecewise decision manifold** learned during pretraining:

- **Expert 7**: "I know this" (in-distribution, high confidence)
- **Expert 4**: "I'm uncertain" (ambiguity buffer, boundary region)
- **Expert 2**: "I give up" (OOD attractor, default under uncertainty)

Expert 4 exists because the router **learned** "ambiguous region". It's a predefined buffer zone, not a dynamical attractor that softens with traversal.

---

## Key Results Summary

### 1. Semantic Structure Provides Stiffness

| Input Type | Δ Top-1 Mass | Δ Entropy | Initial Margin |
|------------|--------------|-----------|----------------|
| Real text  | +0.054       | -0.122    | 1.34           |
| Shuffled   | +0.057       | -0.125    | 1.05           |
| Random     | +0.113       | -0.248    | 0.57           |

**Interpretation:** In-distribution inputs sit in deep basins (high margin). OOD inputs probe flat regions (low margin). Pretrained structure resists perturbation.

### 2. Margin-Wear Correlation: r = -0.935

![Margin vs Wear](../takens_data/wear_as_robustness_probe.json)

Near-perfect inverse correlation between routing confidence (margin) and wear magnitude. This is the signature of a **geometric probe** measuring local curvature.

### 3. Expert Attractors Are Prompt-Specific

| Prompt | Top Expert | Consistency |
|--------|------------|-------------|
| Text A | Expert 7   | 5/5 seeds   |
| Text B | Expert 7   | 5/5 seeds   |
| Text C | Expert 7   | 5/5 seeds   |
| Random | Expert 2   | 5/5 seeds   |

**Selective carving:** Different prompts concentrate on different expert subsets (50% overlap), but OOD always collapses to Expert 2 (garbage attractor).

### 4. The Confidence Cliff

**Transition structure:**
- 0-35% noise: Expert 7 (confident)
- 40-45% noise: Expert 4 (ambiguous)
- 50-100% noise: Expert 2 (collapsed)

**Sharp transition:**
- Expert 7 mass drops 3.47x faster at cliff than average
- Cliff steepness: 1.25 margin/noise
- Single 10% noise step (30%→40%) flips expert

**Three-state cascade reveals learned confidence hierarchy.**

### 5. Zero Hysteresis, Perfect Symmetry

| Direction | Transition (7→4) | Transition (4→2) | Expert 4 Width |
|-----------|------------------|------------------|----------------|
| Up (A→B)  | 35%              | 45%              | 10%            |
| Down (B→A)| 35%              | 45%              | 10%            |

Fresh vs carry-forward T̄: identical results.

**No path dependence. Grass bends, doesn't stay bent.**

### 6. Structural Rigidity: No Yield Point

| η    | 7→4 Boundary | 4→2 Boundary | Expert 4 Width | In-dist → 7 |
|------|--------------|--------------|----------------|-------------|
| 0.05 | 35%          | 45%          | 10%            | 100%        |
| 0.10 | 35%          | 45%          | 10%            | 100%        |
| 0.15 | 35%          | 45%          | 10%            | 100%        |
| 0.20 | 35%          | 45%          | 10%            | 100%        |
| 0.25 | 35%          | 45%          | 10%            | 100%        |

**Boundaries pinned across 5x η range. Manifold is structurally stable.**

---

## Theoretical Framework

### ChronoMoE as Geometric Probe

**Not:** A dynamical system with elastic memory
**Is:** A geometric interrogation of a static decision manifold

**Formal statement:**

*ChronoMoE applies a history-conditioned lens to expert routing logits, biasing trajectories through a fixed decision manifold without modifying its topology.*

### What the Lens Can and Cannot Do

**Can:**
- Bias motion within admissible basins
- Modulate relative margins between near-tie experts
- Accumulate logit bias in pre-projection space
- Reveal local curvature (stiffness) of decision surface

**Cannot:**
- Move basin boundaries
- Create hysteresis across separatrices
- Deform the learned constraint manifold
- Override projection onto pretrained partition

**Metaphor:** You can shuffle furniture inside each room. You cannot knock down walls (unless you apply gradients during training).

### Where Does Wear Live?

In **pre-projection space only.**

```
Hidden States
    ↓
[ChronoMoE Lens] ← T̄ shapes the glass (history-conditioned)
    ↓
Bent Logits (bias applied, wear accumulates here)
    ↓
[Router Projection] ← Learned manifold is absolute
    ↓
Expert Selection (topology preserved)
```

**The lens bends light going through the map. It does not rewrite the map.**

Wear accumulates in the **optics**, not in the **walls**.

---

## Implications

### 1. ChronoMoE is an Interpretability Instrument

This reframes ChronoMoE from a training mechanism to a **probe**.

**What it measures:**
- Routing confidence (via margin)
- Basin depth (via wear magnitude)
- Boundary sharpness (via margin-wear correlation)
- Decision manifold rigidity (via boundary invariance)
- OOD detection (via Expert 2 collapse)

**Without:**
- Gradient updates
- Parameter modification
- Training-time access
- Model retraining

### 2. Inference-Time Geometry Measurement

We can interrogate routing decisions at inference time by applying controlled perturbations and measuring:

- **Stiffness:** How much does routing resist bias?
- **Confidence:** What is the margin at decision boundaries?
- **Robustness:** Do boundaries shift under pressure?
- **Uncertainty:** Where is the ambiguity buffer (Expert 4)?

This is a new capability: **geometric probing without gradients**.

### 3. The Router Has Learned Structure

The three-expert cascade (7→4→2) is not arbitrary. It reveals:

- **Expert 7:** General-purpose in-distribution handler
- **Expert 4:** Learned ambiguity buffer (boundary region)
- **Expert 2:** Garbage attractor for OOD (default under uncertainty)

This structure is **hardwired by pretraining** and resistant to inference-time perturbation.

### 4. Semantic Robustness is Real

Real text is more robust than noise:
- Higher initial margins (1.34 vs 0.57)
- Lower wear magnitude (0.12 vs 0.25)
- Stable expert selection (consistent across seeds)

The router **defends its trained patterns**. In-distribution routing sits in deep basins that resist perturbation. This is evidence of learned robustness.

---

## Comparison to Related Work

### vs. Traditional MoE Analysis

**Standard approach:** Analyze routing decisions post-hoc via logging

**ChronoMoE approach:** Apply controlled perturbations and measure response

**Advantage:** Reveals basin geometry, not just point decisions

### vs. Gradient-Based Probing

**Standard approach:** Compute gradients w.r.t routing decisions

**ChronoMoE approach:** Inference-time perturbation without gradients

**Advantage:** Works on frozen models, no backprop required

### vs. Adversarial Robustness Testing

**Standard approach:** Find inputs that maximize routing confusion

**ChronoMoE approach:** Systematically map confidence landscape

**Advantage:** Characterizes full decision manifold, not just failure modes

---

## Technical Details

### Experimental Setup

**Model:** google/switch-base-8 (pretrained T5-based MoE)
- 8 experts per layer
- Top-1 routing (sparse selection)
- Encoder-decoder architecture

**ChronoMoE Configuration:**
- η (geological learning rate): 0.015-0.25 (swept)
- n_repetitions: 100 (fixed across experiments)
- Layer: 1 (first MoE layer in encoder)

**Metrics:**
- **Margin:** top1_logit - top2_logit (stiffness proxy)
- **Entropy:** -Σ p log p (concentration measure)
- **Expert selection:** argmax(routing_probs)
- **Wear magnitude:** |Δ entropy| after repetitions

### Key Experiments

1. **η Sweep** (0.0025 → 0.25): Linear scaling, detection at η=0.05
2. **Semantic Anchoring** (real/shuffled/random): Anti-semantic inversion
3. **Expert Selection** (multi-seed, multi-prompt): Selective carving
4. **Confidence Cliff** (0-60% noise gradient): Three-state cascade
5. **Hysteresis Test** (A→B, B→A, fresh/carry): Zero path dependence
6. **Yield Point Mapping** (η=0.05→0.25): No boundary movement

### Reproducibility

All code available at: `experiments/test_*.py`

Key files:
- `test_switch_eta_sweep.py`: η scaling
- `test_semantic_anchoring.py`: Real vs noise comparison
- `analyze_expert_selection.py`: Multi-prompt analysis
- `test_confidence_cliff.py`: Transition structure
- `test_hysteresis_across_cliff.py`: Path dependence test
- `test_yield_point_mapping.py`: Boundary stress test

Results: `takens_data/*.json`

---

## Limitations and Boundary Conditions

### What We Tested

- Single model: google/switch-base-8
- Single task: encoder routing (no decoder analysis)
- Single layer: Layer 1 (first MoE)
- Fixed architecture: Top-1 routing
- Inference-time only: No gradient updates

### Known Boundaries

1. **Elastic regime only:** Never reached yield point (η ≤ 0.25)
2. **Shallow sequence analysis:** Token-level routing, not long-range dependencies
3. **Single-language domain:** English text only
4. **Frozen weights:** Pretrained model, no fine-tuning

### Unanswered Questions

- Do deeper layers show different geometry?
- Does top-k routing (k>1) change the structure?
- Can fine-tuning create more deformable manifolds?
- What about decoder routing in generation tasks?

---

## Next Steps

### Immediate Extensions

1. **Multi-layer analysis:** Do all layers have the same 7→4→2 structure?
2. **Cross-model validation:** Does Mixtral/DeepSeek show similar patterns?
3. **Task-specific geometry:** How does routing change for different domains?

### Theoretical Development

1. **Formal characterization** of constraint manifold properties
2. **Information-theoretic framing** of margin-wear relationship
3. **Connection to neural collapse** in classification

### New Hypothesis (Next Phase)

**Can self-gated state create hysteresis where pressure alone cannot?**

If T̄ influence is gated by margin (strong when uncertain, weak when confident), can we create **conditional path dependence** without breaking in-distribution routing?

Test: Does margin-gated ChronoMoE shift the confidence cliff based on approach direction?

---

## Conclusions

### What We Proved

1. ✓ **Wear is real** - entropy reduction, concentration increase (η-controllable)
2. ✓ **Wear measures stiffness** - r=-0.935 correlation with margin
3. ✓ **Mechanism is stateless** - constraint re-projection, not elastic memory
4. ✓ **Manifold is rigid** - boundaries invariant under 5x η stress
5. ✓ **Semantics = resistance** - in-distribution routing is robust

### What This Enables

**Inference-time geometry measurement:**
- Probe routing confidence without gradients
- Map decision boundaries without retraining
- Detect OOD via collapse to garbage attractor
- Measure robustness via margin-wear correlation

### The Core Insight

*ChronoMoE reveals the geometry of pretrained routing decisions by applying controlled perturbations whose effects are continuously re-projected onto the model's learned constraint manifold.*

**The lens bends light, not the map.**

And that's exactly what makes it useful.

---

## Acknowledgments

This work emerged from a series of hypothesis tests that systematically falsified our initial assumptions about path wear, leading to a deeper understanding of routing geometry.

Key experimental pivot: Recognizing that the **absence of hysteresis** is not a null result, but a fundamental property revealing the mechanism's true nature.

The anti-semantic inversion (random > real) was the first clue. The r=-0.935 correlation sealed it. The zero hysteresis proved it. The rigid boundaries under stress confirmed it.

Science done properly.

---

**Document Status:** Complete experimental characterization
**Next Phase:** Self-gated state hypothesis (margin-conditioned influence)
**Date:** December 2024
