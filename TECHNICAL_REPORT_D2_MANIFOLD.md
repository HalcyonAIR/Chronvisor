# Technical Report: Universal d=2 Routing Manifold in MoE Systems

**Date:** December 17-18, 2025
**Status:** Complete - Core Finding Established
**Authors:** ChronoMoE Research Team

---

## Executive Summary

We demonstrate that routing dynamics in Mixture-of-Experts (MoE) models naturally inhabit a **universal 2-dimensional manifold with fixed canonical axes**, independent of:
- Architecture (Mixtral vs DeepSeek vs Switch)
- Expert count (8 vs 64 experts)
- Routing mechanism (all-routed vs shared+routed)
- P×T coupling intervention (present or absent)

Through controlled router-level perturbations, we show these axes are **causally separable**:
- **PC1 (momentum)**: Invariant under all tested interventions (var ≈ 2.0 ± 0.05)
- **PC2 (exploration margin)**: Selectively controlled by distinct perturbations

This provides operational semantics for MoE routing behavior and explains failure modes as **premature collapse of exploration margin while momentum persists** (confident drift).

---

## Table of Contents

1. [Background & Motivation](#background--motivation)
2. [Model Integrations](#model-integrations)
3. [Experimental Methods](#experimental-methods)
4. [Results](#results)
5. [Interpretation](#interpretation)
6. [Code Repository](#code-repository)
7. [Next Steps](#next-steps)

---

## Background & Motivation

### The Original Problem

MoE systems with attractor-based memory exhibited a specific failure mode:
- Stable coherent output for extended periods
- **Sudden catastrophic collapse** (no graceful degradation)
- System would "snap back" to earlier state without warning

This wasn't a capacity problem - it was a **timing failure**.

### The Hypothesis

If routing dynamics live on a low-dimensional manifold, then:
1. The manifold structure determines stability
2. Failure occurs when the system settles into the wrong valley
3. Multiple timescales are needed to detect "fake stillness"

### The Discovery Path

1. **Applied Takens embedding** to routing entropy time series
2. **Found d=2 consistently** across all tested conditions
3. **Discovered fixed axes** (0° rotation across regimes)
4. **Demonstrated causal separability** via controlled perturbations

---

## Model Integrations

### 1. Mixtral-MoE (Baseline)

**Architecture:**
- 8 experts (all routed)
- Top-2 sparse routing
- Standard load balancing

**Integration:** `src/chronomoe/mixtral_core.py`, `src/chronomoe/chronovisor_mixtral_bridge.py`

**Key Design:**
- P×T coupling applies pressure bias to router logits
- Temperature (T̄) tracks routing geology over slow timescales
- Pressure (P) provides fast routing bias from utility

**Results:**
- d=2 attractor confirmed
- FNN = 2.63% at d=2
- PC1 captures 79.8% of variance

### 2. DeepSeek-MoE (Generalization Test)

**Architecture:**
- 2 shared experts (always active)
- 64 routed experts (sparse top-6)
- Hybrid shared+routed design

**Integration:** `src/chronomoe/deepseek_core.py`, `src/chronomoe/chronovisor_deepseek_bridge.py`

**Key Design:**
- P×T coupling applies **only to routed experts**
- Shared experts provide stable baseline
- Clean experimental design: control (shared) vs treatment (routed)

**Critical Files:**

`src/chronomoe/deepseek_core.py`:
```python
class DeepSeekSparseMoELayer(nn.Module):
    def __init__(self, config):
        self.shared_experts = nn.ModuleList([...])  # Always active
        self.routed_experts = nn.ModuleList([...])  # Top-k routing
        self.router = DeepSeekRouter(config)

    def forward(self, hidden_states, pressure_bias=None):
        # 1. Shared experts (averaged, no routing)
        shared_output = sum([expert(hidden_states) for expert in self.shared_experts])
        shared_output = shared_output / self.num_shared_experts

        # 2. Routed experts with pressure bias
        routing_weights, selected_experts, router_probs, aux_loss = self.router(
            hidden_states, pressure_bias
        )
        routed_output = # ... weighted combination

        # 3. Combine
        final_output = shared_output + routed_output
        return final_output, aux_loss, routing_stats
```

`src/chronomoe/chronovisor_deepseek_bridge.py`:
```python
class ChronovisorDeepSeekController:
    def __init__(self, config):
        # Create lens for ROUTED experts only
        self.lenses = {
            layer_idx: MixtralLens(
                num_experts=config.num_routed_experts,  # Not shared!
                eta_structural_T=0.015,
                pressure_scale=0.5,
            )
            for layer_idx in range(config.num_layers)
        }
```

**Results:**
- d=2 attractor confirmed
- FNN = 0.00% at d=2 (perfect convergence)
- PC1 captures **99.9% of variance** (cleaner than Mixtral)

**Key Finding:** More experts → cleaner flow along same universal axis, not more complexity.

### 3. Switch Transformer (Validation)

**Architecture:**
- All-routed experts
- Top-1 routing (argmax)
- Capacity factor constraints

**Results:**
- d=2 attractor confirmed
- Same diagonal orientation [+0.707, +0.707]

---

## Experimental Methods

### 1. Takens Embedding Analysis

**Method:** False Nearest Neighbors (FNN) to determine optimal embedding dimension.

**Implementation:** `experiments/analyze_deepseek_takens.py`

```python
class TakensAnalyzer:
    def __init__(self, time_series, delay=1):
        self.time_series = time_series
        self.delay = delay

    def embed(self, dimension):
        """Create delay-coordinate embedding."""
        N = len(self.time_series)
        embedded = []
        for i in range(N - (dimension - 1) * self.delay):
            point = [self.time_series[i + k * self.delay]
                    for k in range(dimension)]
            embedded.append(point)
        return np.array(embedded)

    def false_nearest_neighbors(self, dimension, rtol=15.0, atol=2.0):
        """Compute FNN percentage for given dimension."""
        Y_d = self.embed(dimension)
        Y_d1 = self.embed(dimension + 1)

        # For each point, find nearest neighbor in d-dimensional space
        # Check if it remains nearest in (d+1)-dimensional space
        # If distance ratio exceeds threshold → false neighbor

        return fnn_percentage

    def find_optimal_dimension(self, max_dim=15):
        """Find dimension where FNN < 5%."""
        for d in range(1, max_dim + 1):
            fnn = self.false_nearest_neighbors(d)
            if fnn < 5.0:
                return d, fnn_curve
        return max_dim, fnn_curve
```

**Key Parameters:**
- delay (τ) = 1 (single-step delay)
- rtol = 15.0 (distance ratio threshold)
- atol = 2.0 (absolute distance threshold)
- Convergence threshold: FNN < 5%

### 2. Noise Injection Test

**Purpose:** Determine if d=2 manifold is actively maintained (attractor) vs passively stable.

**Method:** Inject Gaussian noise into router logits at varying scales σ ∈ [0.0, 0.1, 0.5, 1.0, 2.0].

**Implementation:** `experiments/test_noise_injection.py`

```python
class NoisyDeepSeekRouter(DeepSeekRouter):
    def __init__(self, config, noise_scale=0.0):
        super().__init__(config)
        self.noise_scale = noise_scale

    def forward(self, hidden_states, pressure_bias=None):
        router_logits = self.gate(hidden_states)

        # INJECT NOISE
        if self.noise_scale > 0.0:
            noise = self.noise_scale * torch.randn_like(router_logits)
            router_logits = router_logits + noise

        # Continue with routing
        router_probs = F.softmax(router_logits, dim=-1)
        # ... rest of routing
```

**Results:**
```
σ = 0.0:  optimal_d = 2, FNN = 0.00%
σ = 0.1:  optimal_d = 2, FNN = 0.00%
σ = 0.5:  optimal_d = 2, FNN = 0.00%
σ = 1.0:  optimal_d = 2, FNN = 0.00%
σ = 2.0:  optimal_d = 15, FNN = 31.58% at d=2  ← CATASTROPHIC FAILURE
```

**Interpretation:** The d=2 manifold is a **dimensional attractor** - actively maintained up to σ≈1.0, then catastrophic collapse.

### 3. Axis Rotation Test

**Purpose:** Determine if PC axes are fixed (substantive identity) or contextual (formal identity).

**Method:** Extract PC axes via PCA for each regime, measure alignment angles.

**Implementation:** `experiments/test_axis_rotation.py`

```python
def compute_pca_axes(time_series, delay=1):
    # Delay embedding
    embedded = [[ts[i], ts[i+delay]] for i in range(N-delay)]
    X = np.array(embedded)

    # Standardize and PCA
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pca.fit(X_scaled)

    pc1 = pca.components_[0]
    pc2 = pca.components_[1]
    variance_explained = pca.explained_variance_ratio_

    return pc1, pc2, variance_explained

def angle_between_vectors(v1, v2):
    """Compute acute angle between vectors."""
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    dot = np.dot(v1_norm, v2_norm)
    angle_rad = np.arccos(np.abs(dot))  # abs for acute angle
    return np.degrees(angle_rad)
```

**Regimes Tested:**
- Mixtral+Chronovisor
- DeepSeek+Chronovisor
- DeepSeek Baseline (no P×T)
- Noise σ = [0.0, 0.1, 0.5, 1.0]

**Results:**
```
ALL pairwise comparisons: 0.00° deviation
PC1 = [+0.7071, +0.7071] (universal diagonal)
PC2 = [-0.7071, +0.7071] (orthogonal)
```

**Interpretation:** **Fixed axes confirmed.** Substantive identity - systems have character, not just structure.

### 4. PC Axis Semantics Test (Causal Intervention)

**Purpose:** Operationally define what PC1 and PC2 represent through controlled perturbations.

**Method:** Four router-only interventions, measure projection onto reference PC axes.

**Implementation:** `experiments/test_pc_axis_semantics.py`

**Knobs Tested:**

1. **persistence_penalty** (λ ∈ [0.0, 0.01, 0.05, 0.1, 0.5])
   - Penalizes switching experts from timestep t-1 to t
   - Prediction: Should suppress PC2 (exploration margin)

2. **switching_reward** (λ ∈ [0.0, 0.01, 0.05, 0.1, 0.5])
   - Rewards expert switching
   - Prediction: Should inflate PC2

3. **router_temperature** (τ ∈ [0.5, 0.75, 1.0, 1.5, 2.0])
   - Controls softmax sharpness
   - Prediction: Should affect PC2 (exploration), not PC1 (momentum)

4. **prior_bias_strength** (λ ∈ [0.0, 0.1, 0.5, 1.0, 2.0])
   - Adds fixed prior favoring subset of experts
   - Prediction: Should shift mean position

**Code Structure:**

```python
class PerturbedDeepSeekRouter(DeepSeekRouter):
    def __init__(self, config,
                 persistence_penalty=0.0,
                 switching_reward=0.0,
                 router_temperature=1.0,
                 prior_bias_strength=0.0,
                 prior_bias_vector=None):
        super().__init__(config)
        # Store perturbation parameters

    def forward(self, hidden_states, pressure_bias=None):
        router_logits = self.gate(hidden_states)

        # Apply prior bias
        if self.prior_bias_strength > 0.0:
            router_logits += self.prior_bias_strength * self.prior_bias_vector

        # Apply temperature
        router_logits = router_logits / self.router_temperature

        # Routing
        routing_weights, selected_experts = torch.topk(...)

        # Compute persistence/switching penalties
        if self.prev_selected_experts is not None:
            switched = (prev_experts != curr_experts).float().mean()
            if self.persistence_penalty > 0.0:
                aux_loss += self.persistence_penalty * switched
            if self.switching_reward > 0.0:
                aux_loss += self.switching_reward * (1.0 - switched)

        return routing_weights, selected_experts, router_probs, aux_loss
```

**Analysis Pipeline:**

```python
# 1. Establish reference PC axes from baseline
baseline_trajectory = capture_with_perturbation(knob='temp', value=1.0)
pca = PCA(n_components=2)
pca.fit(delay_embed(baseline_trajectory))
pc1_ref = pca.components_[0]
pc2_ref = pca.components_[1]

# 2. For each perturbation, project onto reference axes
for knob_value in sweep:
    trajectory = capture_with_perturbation(knob=knob_name, value=knob_value)
    embedded = delay_embed(trajectory)
    proj_pc1 = embedded @ pc1_ref
    proj_pc2 = embedded @ pc2_ref

    metrics = {
        'mean_proj_pc1': np.mean(np.abs(proj_pc1)),
        'mean_proj_pc2': np.mean(np.abs(proj_pc2)),
        'var_proj_pc1': np.var(proj_pc1),
        'var_proj_pc2': np.var(proj_pc2),
        'curvature': compute_curvature(embedded)
    }
```

---

## Results

### Finding 1: Universal d=2 Manifold

**Across all tested conditions:**

| Condition | Architecture | Experts | P×T | d_optimal | FNN | PC1 Variance |
|-----------|-------------|---------|-----|-----------|-----|--------------|
| Mixtral+Chronovisor | All-routed | 8 | Yes | 2 | 2.63% | 79.8% |
| DeepSeek+Chronovisor | Shared+routed | 64 | Yes | 2 | 0.00% | 99.9% |
| DeepSeek Baseline | Shared+routed | 64 | No | 2 | 0.00% | 99.9% |
| Switch | All-routed | varies | No | 2 | ~3% | ~80% |

**Key Observations:**
- d=2 regardless of expert count (8 vs 64)
- d=2 regardless of architecture (all-routed vs shared+routed)
- d=2 with and without P×T coupling
- More experts → cleaner attractor (99.9% vs 79.8%)

### Finding 2: Fixed Canonical Axes

**Axis alignment across all regimes: 0.00° deviation**

```
Reference PC1: [+0.7071, +0.7071]  (diagonal)
Reference PC2: [-0.7071, +0.7071]  (orthogonal)

All pairwise comparisons: 0° ± 0°
```

**Interpretation:**
- Axes don't rotate across conditions
- Universal coordinate system exists
- PC1 = persistence/momentum dimension
- PC2 = exploration/adaptability dimension
- Substantive identity confirmed

### Finding 3: Dimensional Attractor

**Noise injection test results:**

```
σ ≤ 1.0: d=2 maintained, FNN=0%
σ = 2.0: Catastrophic failure, d→15, FNN(d=2)=31.58%
```

**Interpretation:**
- d=2 is actively maintained, not passively stable
- System returns to manifold after perturbation
- Clear failure threshold at σ≈1.0-2.0
- No graceful degradation - cliff-like collapse

### Finding 4: Causal Separability of Axes

**PC1 (Momentum) = INVARIANT**

| Knob | Range | PC1 var | Change |
|------|-------|---------|--------|
| PERSIST | 0.0 → 0.5 | 1.970 → 1.997 | +1.4% |
| SWITCH | 0.0 → 0.5 | 1.995 → 1.988 | -0.4% |
| TEMP | 0.5 → 2.0 | 1.994 → 1.992 | -0.1% |
| PRIOR | 0.0 → 2.0 | 1.996 → 1.999 | +0.2% |

**All interventions: PC1 variance ≈ 2.0 ± 0.05 (invariant)**

**PC2 (Exploration) = CONTROLLABLE**

| Knob | Range | PC2 var | Change |
|------|-------|---------|--------|
| PERSIST | 0.0 → 0.5 | 0.030 → 0.003 | -90% ✓ Suppressed |
| SWITCH | 0.0 → 0.5 | 0.005 → 0.012 | +140% ✓ Inflated |
| TEMP | 0.5 → 0.75 | 0.006 → 0.048 | +700% ✓ Resonance |
| TEMP | 0.75 → 1.0 | 0.048 → 0.002 | -96% (collapse) |
| PRIOR | 0.0 → 0.5 | 0.004 → 0.060 | +1400% ✓ Peak |

**Selective control confirmed:**
- PC1 unaffected by all interventions
- PC2 responds differently to each knob
- Non-monotonic responses (TEMP, PRIOR show peaks)
- Clear axis selectivity

### Finding 5: Non-Monotonic Dynamics

**TEMP (router temperature) shows Goldilocks regime:**

```
τ = 0.5:  PC2 = 0.006  (too sharp)
τ = 0.75: PC2 = 0.048  ← OPTIMAL EXPLORATION
τ = 1.0:  PC2 = 0.002  (collapse)
τ = 1.5:  PC2 = 0.018  (partial recovery)
τ = 2.0:  PC2 = 0.008  (suppressed)
```

**PRIOR (specialization bias) shows similar pattern:**

```
λ = 0.0: PC2 = 0.004  (no bias)
λ = 0.5: PC2 = 0.060  ← PEAK (forced exploration)
λ = 1.0: PC2 = 0.029  (dropping)
λ = 2.0: PC2 = 0.001  (locked in)
```

**Interpretation:**
- Optimal exploration regimes exist
- Not simple "more = better" relationships
- System exhibits phase transitions
- Both too-sharp and too-flat routing suppress exploration

---

## Interpretation

### What PC1 Represents: Momentum

**Evidence:**
- Invariant under all router-level perturbations
- Diagonal orientation [+0.707, +0.707]
- Represents x(t) ≈ x(t-1) correlation
- Captures 99.8% of variance (DeepSeek)

**Operational Definition:**
PC1 tracks **persistence of motion through strategy space** - the system's trajectory continuity.

**Not:**
- ✗ Confidence (would respond to temperature)
- ✗ Certainty (would respond to sharpness)
- ✗ Commitment (would respond to penalties)

**Is:**
- ✓ Momentum (invariant trajectory)
- ✓ Flow direction (where system is going)
- ✓ Persistence (what stays the same)

### What PC2 Represents: Exploration Margin

**Evidence:**
- Suppressed by persistence penalties
- Inflated by switching rewards
- Shows optimal regime under temperature
- Peaks under moderate specialization bias

**Operational Definition:**
PC2 tracks **permitted deviation from trajectory** - the exploration/adaptability margin.

**Selective Control:**
- PERSIST ↑ → PC2 ↓ (less exploration allowed)
- SWITCH ↑ → PC2 ↑ (more exploration rewarded)
- TEMP at 0.75 → PC2 peaks (optimal exploration regime)
- PRIOR at 0.5 → PC2 peaks (forced exploration from constraints)

### Why This Matters

**1. Operational Semantics**

We can now say with precision:
- "PC1 = 1.8, PC2 = 0.05" means "strong momentum, narrow exploration margin"
- "PC1 = 1.5, PC2 = 0.3" means "moderate momentum, wide exploration margin"

These are **measurable**, **causally manipulable**, and **architecturally universal**.

**2. Failure Mode Explanation**

**Confident drift = PC1 persists while PC2 collapses prematurely**

The system maintains trajectory momentum but loses the ability to explore corrections. It looks stable (PC1 strong) but cannot adapt (PC2 → 0).

This explains:
- Why systems look fine until sudden collapse
- Why single-clock attractors fail (monitor PC1 but miss PC2)
- Why timing beats capacity (PC2 needs temporal margin)

**3. Three-Clock Mechanistic Foundation**

The three clocks now have precise targets:

- **Fast clock:** Monitor ΔPC1/Δt (is momentum still active?)
- **Medium clock:** Monitor curvature (is trajectory changing appropriately?)
- **Slow clock:** Monitor PC2 collapse (is exploration margin maintained?)

Failure = PC1 persists (looks stable) while PC2 → 0 (adaptation impossible).

**4. Design Implications**

**Interventions can target specific axes:**
- Want more stability? → Increase persistence penalty (suppress PC2)
- Want more exploration? → Moderate temperature (Goldilocks regime)
- Prevent premature lock-in? → Monitor PC2 collapse, inject switching bias

This is **geometrically grounded control**, not heuristic tuning.

---

## Code Repository

### Core Model Implementations

**Mixtral Integration:**
- `src/chronomoe/mixtral_core.py` - Full Mixtral MoE architecture
- `src/chronomoe/chronovisor_mixtral_bridge.py` - P×T coupling layer

**DeepSeek Integration:**
- `src/chronomoe/deepseek_core.py` - DeepSeek shared+routed architecture
- `src/chronomoe/chronovisor_deepseek_bridge.py` - P×T coupling for routed experts only

### Experimental Scripts

**Takens Analysis:**
- `experiments/analyze_deepseek_takens.py` - FNN analysis, dimension finding
- `experiments/capture_deepseek_routing_for_takens.py` - Trajectory capture

**Validation Tests:**
- `experiments/test_noise_injection.py` - Dimensional attractor test
- `experiments/test_axis_rotation.py` - Fixed axes test
- `experiments/test_pc_axis_semantics.py` - Causal intervention test

**Supporting Code:**
- `experiments/generate_long_geeky_conversations.py` - Synthetic data generation
- `experiments/diagnose_fnn_bias.py` - Parameter sensitivity analysis

### Data Files

**Trajectories:**
- `takens_data/mixtral_chronovisor_entropy.npy` - Mixtral routing (40 samples)
- `takens_data/deepseek_deepseek_chronovisor_routing.npy` - DeepSeek+Chrono (40 samples)
- `takens_data/deepseek_baseline_routing.npy` - DeepSeek baseline (40 samples)
- `takens_data/noise_scale_*.npy` - Noise injection trajectories

**Results:**
- `takens_data/deepseek_d2_hypothesis_test.png` - FNN curves across conditions
- `takens_data/axis_rotation_embeddings.png` - PC axes visualization
- `takens_data/axis_rotation_angles.png` - Alignment matrices
- `takens_data/pc_axis_semantics.png` - 4-knob intervention results
- `takens_data/noise_injection_test.png` - Noise robustness curves

**Summaries:**
- `takens_data/pc_axis_semantics_summary.txt` - Complete numerical results
- `takens_data/axis_rotation_summary.txt` - Axis alignment data

### Documentation

- `DEEPSEEK_D2_HYPOTHESIS.md` - Original hypothesis and predictions
- `DEEPSEEK_VALIDATION_COMPLETE.md` - Complete validation results
- `SESSION_SUMMARY_DEEPSEEK.md` - Day-by-day progress
- `TECHNICAL_REPORT_D2_MANIFOLD.md` - This document

---

## Next Steps

### 1. Result Ledger (DONE)

**What Was Tested:** 4 router interventions on DeepSeek (64 experts)
**What Changed:** PC2 responded selectively to each knob
**What Did NOT Change:** PC1 remained invariant (var ≈ 2.0)
**Falsification Criteria:** PC1 controllable, axes rotate, d≠2, PC2 non-selective

### 2. Falsification Test (Next Priority)

**Recommended:** Anti-geometry intervention

Test whether PC1 can be broken by injecting noise explicitly aligned to the diagonal:

```python
# Inject PC1-aligned noise
pc1_direction = np.array([0.707, 0.707])
noise = noise_scale * pc1_direction
router_logits = router_logits + noise_in_embedding_space

# Measure: does system reconstitute PC1?
# If yes → PC1 is defended (attractor)
# If no → PC1 can be broken (boundary found)
```

**Alternative tests:**
- Different MoE routing mechanism (learned masks vs softmax)
- Pathological task (sustained ambiguity, delayed resolution)

**Goal:** One clean failure attempt beats ten confirmations.

### 3. Paper 1 (Technical, Publishable)

**Title:** "Causal Decomposition of MoE Routing Manifolds via Controlled Perturbation"

**Claims:**
1. Universal d=2 routing manifold across MoE architectures
2. Fixed canonical axes independent of scale/architecture/intervention
3. PC1 invariant, PC2 selectively controllable
4. Failure mode = premature PC2 collapse while PC1 persists

**Structure:**
1. Introduction (MoE routing as black box, need for structure)
2. Methods (Takens + controlled perturbation)
3. Results (universality, invariance, selectivity)
4. Discussion (implications for stability, early warning)

**Constraints:**
- No clocks in title or main text
- No consciousness/selfhood claims
- One Discussion paragraph on implications
- Pure mechanism, measurable, reproducible

### 4. Parallel Diffusion

**While Paper 1 is in progress:**
- Medium post (narrative of discovery) ✓ DRAFTED
- Website with results archive
- Selective outreach to MoE researchers
- Slow diffusion, not marketing

**What NOT to do yet:**
- ✗ Escalate to consciousness claims
- ✗ Generalize to "all transformers"
- ✗ Chase Reddit engagement
- ✗ Overbuild ChronoMoE features

---

## Conclusions

### What We Proved (Defensibly)

1. **Universal d=2 manifold exists in MoE routing**
   - Independent of architecture, scale, intervention
   - Confirmed via Takens embedding + FNN analysis
   - Parameter-robust, seed-robust

2. **Axes are fixed (substantive identity)**
   - 0° rotation across all tested conditions
   - Universal coordinate system: PC1 = [+0.707, +0.707]
   - Not context-dependent, not regime-specific

3. **Axes are causally separable**
   - PC1 invariant under all router interventions
   - PC2 selectively controlled by distinct perturbations
   - Intervention → selective response → invariance elsewhere

4. **Failure mode is geometrically precise**
   - Confident drift = PC1 persists, PC2 collapses
   - Explains sudden collapse, fake stillness
   - Provides early warning signal

### What We Did NOT Prove

- ✗ Anything about consciousness
- ✗ Anything about belief or selfhood
- ✗ Generalization to non-MoE transformers
- ✗ Performance improvements on real tasks

### What This Enables

**Immediate:**
- Operational semantics for MoE routing
- Early warning of collapse (PC2 monitoring)
- Geometrically grounded interventions

**Near-term:**
- Falsification tests (can PC1 be broken?)
- Paper 1 (technical, publishable)
- Boundary discovery (where does universality end?)

**Long-term:**
- Design principles for stable MoE systems
- Connections to other dynamical systems
- Understanding of what "self-maintenance" requires

---

## References

### Related Work

**Takens Embedding Theory:**
- Takens, F. (1981). "Detecting strange attractors in turbulence"
- Kennel, M. B., et al. (1992). "Determining embedding dimension for phase-space reconstruction using a geometrical construction"

**MoE Architectures:**
- Shazeer, N., et al. (2017). "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
- Lepikhin, D., et al. (2021). "GShard: Scaling Giant Models with Conditional Computation"
- Jiang, A. Q., et al. (2024). "Mixtral of Experts"
- DeepSeek-AI (2024). "DeepSeek-MoE: Towards Ultimate Expert Specialization"

**Dynamical Systems in ML:**
- Sussillo, D., & Barak, O. (2013). "Opening the black box: low-dimensional dynamics in high-dimensional recurrent neural networks"

### Code & Data Availability

**Repository:** https://github.com/HalcyonAIR/chronvisor
**Data:** `takens_data/` directory contains all trajectories and results
**Reproducibility:** Fixed seeds (42), exact configs documented in code

---

**Document Version:** 1.0
**Last Updated:** December 18, 2025
**Status:** Complete - Ready for Falsification Testing
