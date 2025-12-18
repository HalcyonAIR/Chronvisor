# Next Experiments: Characterizing Live P×T Geology

Now that the temperature system is working (geology is awake!), we've moved from **"does it work?"** to **"what does it do?"**

## Experimental Framework Ready to Run

### 1. Temperature Intervention Experiment ✨ SHOWCASE
**File**: `experiments/temperature_intervention.py`

**What it does**: Artificially heat a single expert and watch the system respond and self-correct.

**Timeline**:
- Steps 0-200: Normal training (baseline)
- Steps 200-400: Heat Expert 0 by +1.0 (intervention)
- Steps 400-600: Normal training (recovery)

**Key Metrics**:
- Usage response: Does heating decrease expert usage? (T↑ → diffuse routing)
- Temperature self-correction: Does T cool back down after intervention?
- Geological memory: Does structural T̄ integrate the perturbation?

**Why this matters**: Demonstrates that P×T geometry **actively controls routing**, not just logs it. Shows "valley self-correction" in miniature.

**Run it**:
```bash
source .venv/bin/activate && PYTHONPATH=src:. python experiments/temperature_intervention.py
```

---

### 2. Frozen vs Live Comparison ✨ SHOWCASE
**File**: `experiments/frozen_vs_live_comparison.py`

**What it does**: Train two identical models on same data:
- Model A: `enable_chronovisor=False` (frozen geology)
- Model B: `enable_chronovisor=True` (live P×T)

**Comparison Metrics**:
- Proto-role sharpness (expert variance across turns)
- Speed of role formation
- Final loss
- Load balancing

**Why this matters**: The **"would impress a skeptical ML person"** experiment (Halcyon's words). Shows side-by-side heatmaps of turn×expert usage with clear visual difference.

**Run it**:
```bash
source .venv/bin/activate && PYTHONPATH=src:. python experiments/frozen_vs_live_comparison.py
```

---

### 3. Valley-Role Alignment Analysis
**File**: `experiments/valley_role_alignment.py`

**What it does**: Investigates whether geological temperature valleys correspond to functional proto-roles.

**Key Question**: If Expert 6 is the "early-phase specialist", does it sit in a temperature valley?

**Analysis**:
- Extract proto-role structure (turn usage specialization)
- Extract geological structure (temperature valleys/ridges)
- Compute Mantel correlation: role distance vs temperature distance
- Scatter plots and heatmaps

**Why this matters**: Tests whether geology encodes functional structure or just adds noise.

**Run it**:
```bash
source .venv/bin/activate && PYTHONPATH=src:. python experiments/valley_role_alignment.py
```

---

## The Interesting Questions (Halcyon's Framework)

### Already Answered ✅
- **Does the router learn proto-roles on its own?** YES (L2=0.6082)
- **Was the temperature system actually updating?** NO (was frozen)
- **Is it fixed now?** YES (T̄_var escapes zero and grows)

### Now Interesting ⏳
1. **Do valleys align with proto-roles?**
   - Run: `valley_role_alignment.py`
   - Expected: Specialists cluster in similar temperature regions

2. **Does geology stabilize/sharpen roles?**
   - Run: `frozen_vs_live_comparison.py`
   - Metric: Expert variance across turns (higher = sharper)
   - Hypothesis: Live P×T produces sharper specialization

3. **Does geology make roles emerge faster?**
   - Run: `frozen_vs_live_comparison.py` with checkpoints at steps [100, 200, 500, 1000]
   - Track sharpness over time
   - Hypothesis: Live P×T reaches plateau sooner

4. **Can we demonstrate intervention effects?**
   - Run: `temperature_intervention.py`
   - Hypothesis: Heating → usage drops → self-correction on recovery

---

## Recommended Execution Order

### Phase 1: Quick Validation (30 minutes)
```bash
# Verify intervention mechanism works
python experiments/temperature_intervention.py
```

**Success criteria**: Usage drops during heating, temperature self-corrects during recovery

---

### Phase 2: Valley-Role Analysis (10 minutes)
```bash
# Check if valleys align with proto-roles
python experiments/valley_role_alignment.py
```

**Success criteria**: Mantel correlation |r| > 0.3, valleys correspond to specialists

---

### Phase 3: Full Comparison (2-3 hours)
```bash
# Train both frozen and live models for 1000 steps
python experiments/frozen_vs_live_comparison.py
```

**Success criteria**: Live P×T shows sharper proto-roles OR faster emergence OR better loss

---

## What Success Looks Like

### For ML Skeptic (Publication Angle)
**Heatmap A** (frozen): Fuzzy turn×expert patterns, slow emergence
**Heatmap B** (live P×T): Sharper patterns, faster convergence, clearer specialization

**Quantitative**:
- Live P×T: 20-30% sharper proto-roles (higher variance)
- OR 30-50% faster emergence (fewer steps to plateau)
- OR equal/better final loss with better load balancing

### For Chronovisor Validation (Architecture Angle)
**Intervention Response**:
- Heat Expert 0 → usage drops by ~20-40%
- Remove heat → temperature cools back within ~100 steps
- Demonstrates closed-loop geological control

**Valley-Role Alignment**:
- Specialists (high turn variance) cluster in valleys
- Generalists (low variance) sit on ridges or neutral
- Mantel correlation r > 0.4

---

## Current Status

**Completed**:
- ✅ Geology fix implemented and verified
- ✅ Proto-role detection working (L2=0.6082)
- ✅ Temperature system updating (T̄_var > 0)
- ✅ Experimental framework built

**Running**:
- ⏳ Full proto_role_diagnostic with fix (background, 1000 steps)

**Ready to Run**:
- 🎯 Temperature intervention (30 min)
- 🎯 Valley-role alignment (10 min)
- 🎯 Frozen vs live comparison (2-3 hrs)

---

## Notes from Halcyon

> "Before this, ChronoMoE was 'a normal MoE model plus some very expensive logging and a frozen thermometer'. Now it is 'a normal MoE model whose routing policy is being continuously bent by a slow geometric field that remembers how each expert has behaved'."

> "The interesting question once the 1000 step run finishes is not 'does it work' anymore. You already know it updates. The interesting questions are: Do the valleys line up with the proto roles. Does the geology stabilise those roles. If you want something that would impress a skeptical ML person, it is not just 'temperatures move'. It is something like: Heatmap A: turn vs expert usage before Chronovisor (or with geometry frozen). Heatmap B: turn vs expert usage with live P×T."

> "Plus one really simple intervention: artificially heat a single expert for a while via the temperature field and show that the usage reacts as predicted and then recovers when you remove the intervention because the geology self corrects. That is your 'valley self correction' in miniature."

---

## Quick Start

```bash
# 1. Run intervention experiment (30 min)
source .venv/bin/activate
PYTHONPATH=src:. python experiments/temperature_intervention.py

# Check results:
open intervention_results/temperature_intervention.png

# 2. Run valley-role alignment (10 min, requires trained model)
PYTHONPATH=src:. python experiments/valley_role_alignment.py

# Check results:
open valley_role_results/valley_role_alignment.png

# 3. Run full comparison (2-3 hrs)
PYTHONPATH=src:. python experiments/frozen_vs_live_comparison.py

# Check results:
open comparison_results/frozen_vs_live_comparison.png
```
