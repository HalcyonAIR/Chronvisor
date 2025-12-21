

# Full Mixtral Testing: Pressure Half-Life Measurement

**Status**: Infrastructure ready, awaiting execution
**Created**: December 21, 2024

---

## Objective

Measure natural pressure half-life on real Mixtral-8x7B model.

**Per Halcyon's guidance:**
> "Next concrete move: measure pressure half-life first, before any tuning, on Mixtral alone. DeepSeek comes after, not as a crutch but as a contrast."

---

## Why This Matters

### Toy Model Limitations

**Toy baseline findings:**
- Mid-pressure: Constant at 0.640 (infinite half-life)
- Entropy collapse: 0.95 → 0.53 (44%)
- No natural decay mechanism

**Critical questions:**
1. Is infinite half-life correct behavior?
2. Or artifact of toy model simplicity?
3. Will full Mixtral show natural pressure decay?

### What Full Mixtral Adds

**Real routing behavior:**
- Pre-trained weights (not random initialization)
- Semantic routing decisions
- Real expert specialization

**Real semantic content:**
- Explanatory prompts (not random tokens)
- Rhetorical structure
- Natural completion points

**Expected:**
- Realistic half-life (10-30 chunks?)
- Pressure decay at rhetorical boundaries
- Cliff patterns when topic exhausted

---

## Implementation

### 1. External Mixtral Adapter

**File:** `src/chronomoe/external_mixtral_adapter.py`

**Purpose:** Wrap HuggingFace Mixtral models with Chronovisor integration

**Key features:**
```python
class ExternalMixtralAdapter:
    - Loads pre-trained Mixtral from HuggingFace
    - Extracts expert routing signals from router_logits
    - Integrates with ChronovisorMixtralController
    - Enables pressure-based multistep generation
```

**Signal extraction:**
```python
def _extract_routing_signals(outputs):
    # Get router logits from HF Mixtral
    router_logits = outputs.router_logits

    # Compute expert usage per layer
    for layer_idx, logit in enumerate(router_logits):
        probs = softmax(logit[:, -1, :])
        usage = probs.mean(dim=0)  # Average over batch

        # Compute entropy
        entropy = -sum(p * log(p) for p in usage)

        # Store in chrono_state
        expert_usage[layer_idx] = usage
        routing_entropy[layer_idx] = entropy

    # Update Chronovisor controller
    chrono_state = chronovisor.tick(expert_usage, routing_entropy)
```

**Integration points:**
- Session controller (pressure computation)
- Clock heads (temporal control)
- Telemetry (trajectory logging)

---

### 2. Measurement Script

**File:** `tests/measure_mixtral_halflife.py`

**Protocol:**
1. Load Mixtral-8x7B (with 8-bit quantization for memory)
2. Tokenize explanatory prompt
3. Generate N chunks in multistep mode
4. Extract mid-pressure trajectory
5. Compute half-life and decay shape
6. Compare to toy baseline

**Usage:**
```bash
# Basic (requires GPU with 24GB+ VRAM)
python tests/measure_mixtral_halflife.py

# With 8-bit quantization (reduces to ~12GB)
python tests/measure_mixtral_halflife.py --8bit

# Customize
python tests/measure_mixtral_halflife.py \
    --model mistralai/Mixtral-8x7B-v0.1 \
    --chunks 50 \
    --chunk-size 20 \
    --seed 42
```

**Outputs:**
- `test_results/mixtral_pressure_halflife.json` (trajectory data)
- `test_results/mixtral_vs_toy_comparison.png` (comparison plot)
- Console report with findings

---

### 3. Comparison Framework

**Metrics:**

**Half-life:**
- Chunks until pressure drops to 50% of initial
- Mixtral expected: 10-30 chunks
- Toy baseline: ∞ (never reached)

**Decay shape:**
- Linear slope (pressure/chunk)
- Exponential tau (half-life constant)
- Max single drop (cliff detection)

**Entropy evolution:**
- Collapse magnitude
- Convergence rate
- Early vs late mean

**Generated text:**
- Rhetorical structure
- Natural completion points
- Semantic coherence

---

## Resource Requirements

### Minimum (8-bit quantization)

```
GPU: 12GB VRAM (RTX 3090, A100 40GB, etc.)
RAM: 32GB
Disk: 50GB (for model download)
Time: ~30 minutes (50 chunks × 20 tokens)
```

### Recommended (float16)

```
GPU: 24GB VRAM (A100 40GB, A6000, etc.)
RAM: 64GB
Disk: 50GB
Time: ~20 minutes
```

### Cloud Alternative

```
Service: Google Colab Pro, Lambda Labs, RunPod
GPU: A100 (40GB or 80GB)
Cost: ~$1-2 per hour
Setup: Mount drive, install dependencies
```

---

## Expected Findings

### Scenario A: Natural Decay (Most Likely)

**Prediction:**
- Half-life: 15-25 chunks
- Decay shape: Exponential with cliffs at rhetorical boundaries
- Entropy: Converges as topic develops
- Pressure: Drops when explanation complete

**Interpretation:**
- Real routing shows natural completion pressure
- Toy model infinite half-life was artifact
- Rhetorical structure drives pressure decay

**Implications:**
- Validate pressure as control surface
- Ready for DeepSeek comparison
- Can measure "external energy" needed to continue

---

### Scenario B: Infinite Half-Life (Unlikely)

**Prediction:**
- Half-life: Not reached in 50 chunks
- Pressure: Constant or increasing
- Similar to toy model

**Interpretation:**
- Infinite half-life is correct for transformer MoEs
- No natural stopping mechanism
- Continuation is default behavior

**Implications:**
- Pausing is purely structural (chunk boundaries)
- Pressure measures "can continue" not "should stop"
- DeepSeek comparison focuses on variance, not half-life

---

### Scenario C: Sharp Cliffs (Possible)

**Prediction:**
- Half-life: Short (5-10 chunks)
- Decay shape: Cliff pattern (sudden drops)
- Pressure: Stable until rhetorical boundary, then drops sharply

**Interpretation:**
- Mixtral shows "section awareness"
- Pressure collapses at completion points
- Halcyon's prediction: "Mixtral should show sharper cliffs"

**Implications:**
- Confirms thesis pressure hypothesis
- Ready for DeepSeek comparison (expect smoother decay)
- External perturbation will be critical

---

## Comparison to Toy Baseline

### Toy Model Findings
```
Mid-pressure: 0.6404 → 0.6403 (constant)
Half-life: ∞ (not reached)
Entropy: 0.95 → 0.53 (44% collapse)
Decay rate: +0.000004 per chunk (zero)
```

### Expected Mixtral Findings
```
Mid-pressure: 0.65 → 0.30 (natural decay)
Half-life: 15-25 chunks (realistic)
Entropy: 0.80 → 0.45 (semantic convergence)
Decay rate: -0.01 to -0.02 per chunk (measurable)
```

### Key Differences
- Toy: Random routing, no semantics → infinite half-life
- Mixtral: Learned routing, real semantics → natural decay

---

## Next Steps After Measurement

### 1. Analyze Results

**Questions:**
- What is the measured half-life?
- Is decay exponential, linear, or cliff-like?
- Does entropy correlate with pressure decay?
- Are there rhetorical boundary effects?

**Actions:**
- Document findings
- Compare to toy baseline
- Identify patterns

---

### 2. Neutral Perturbation Test

**Protocol (per Halcyon):**
> "Add the tiniest Claude-style perturbation. A neutral 'go on' equivalent. Measure how much external energy it takes to keep mid-pressure above zero."

**Implementation:**
```python
# After natural pause
continuation_prompt = " Please continue:"
input_ids = tokenize(generated_text + continuation_prompt)

# Generate next chunk
# Measure pressure with perturbation

# Compare to baseline (zero perturbation)
delta_pressure = pressure_with_perturbation - pressure_baseline
```

**Metric:** External energy required to sustain pressure

---

### 3. DeepSeek Comparison

**After Mixtral baseline established:**

**Halcyon's hypothesis:**
> "DeepSeek should show longer intrinsic half-life. Mixtral should show sharper cliffs."

**Implementation:**
- Same protocol, different model
- Load DeepSeek-V2 or DeepSeek-V3
- Run identical measurement
- Compare half-lives and decay shapes

**Expected:**
- DeepSeek: Smoother decay, longer half-life
- Mixtral: Sharper cliffs, shorter half-life
- Variance: DeepSeek lower, Mixtral higher

**This is where novelty emerges** (not from claims, from measurements)

---

## Files

### Created
```
src/chronomoe/external_mixtral_adapter.py     (adapter implementation)
tests/measure_mixtral_halflife.py              (measurement script)
docs/007-full-mixtral-testing.md               (this document)
```

### To Be Generated
```
test_results/mixtral_pressure_halflife.json    (trajectory data)
test_results/mixtral_vs_toy_comparison.png     (comparison plot)
test_results/MIXTRAL_HALFLIFE_FINDING.md       (analysis document)
```

---

## Critical Notes

### DO NOT Tune Yet

**Temptation:** Adjust pressure weights after seeing Mixtral results

**Halcyon's guidance:**
> "Resist the urge to tune mid-pressure weights yet. Don't reward it before you've measured its natural dynamics."

**Why:**
- Tuning before measurement hides structure
- Need to see natural dynamics first
- Comparison requires unmodified baseline

**When to tune:** After measuring both Mixtral and DeepSeek

---

### Focus on Dynamics, Not Quality

**Wrong question:** "Does Mixtral generate better explanations with pressure pausing?"

**Right question:** "How much force does it take to keep Mixtral thinking?"

**Metrics:**
- Half-life (pressure decay rate)
- External energy required (perturbation delta)
- Variance (cliff sharpness)

**NOT metrics:**
- Answer quality
- Factual accuracy
- Rhetorical coherence

(Those are interesting, but not the point)

---

## Summary

**Built:** Complete infrastructure for Mixtral testing

**Ready:**
- External adapter (HuggingFace integration)
- Measurement script (pressure half-life)
- Comparison framework (vs toy baseline)

**Next:** Execute measurement on full Mixtral-8x7B

**Expected:** Natural pressure decay (unlike toy model's infinite half-life)

**Critical:** Measure before tuning, compare before claiming novelty

**Status:** ✓ Infrastructure complete. Ready for execution (requires GPU).

---

**The walls stay put. We measure how much force to keep thinking.**
