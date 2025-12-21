# Clock-Gated Mixtral Architecture

**Status**: Design complete, ready for testing
**Date**: December 2024
**Key Innovation**: Stateful temporal arbitration outside the forward pass

---

## Executive Summary

We've extended Mixtral with three stateful clock heads that arbitrate token selection based on temporal coherence at different timescales.

**The base model remains completely stateless.** Clocks sit outside the forward pass, judging outputs after generation.

This architecture respects everything Phase 1 & 2 taught us:
- Constraint manifold dominates (can't deform it)
- State cannot survive projection
- Memory must live outside routing

**Solution**: Don't fight the walls. Judge what comes out instead.

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│                    BASE MODEL (Stateless)                  │
│                                                            │
│  input_ids → embeddings → Mixtral layers → lm_head        │
│                              ↓                             │
│                      Chronovisor Controller               │
│                      (tracks coherence R,                 │
│                       updates P×T fields)                 │
│                                                            │
│                    logits [batch, seq, vocab]             │
└──────────────────────┬─────────────────────────────────────┘
                       │
                       ↓
               Sample top-k candidates
               (e.g., 5 most probable tokens)
                       │
       ┌───────────────┼───────────────┐
       ↓               ↓               ↓
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ FAST CLOCK  │ │MEDIUM CLOCK │ │ SLOW CLOCK  │
│             │ │             │ │             │
│Half-life: 5 │ │Half-life:50 │ │Half-life:500│
│Turns        │ │Turns        │ │Turns        │
│             │ │             │ │             │
│High bw      │ │Moderate bw  │ │Low bw       │
│Low inertia │ │Med inertia  │ │High inertia │
└──────┬──────┘ └──────┬──────┘ └──────┬──────┘
       │               │               │
       ↓               ↓               ↓
    score₁         score₂          score₃
       │               │               │
       └───────────────┼───────────────┘
                       ↓
            ELIMINATION TOURNAMENT
           (lowest avg score drops)
                       ↓
                 Final Token
                       ↓
            Update clock states
           (temporal persistence)
```

---

## Component Breakdown

### 1. Base Model (ChronovisorMixtralForCausalLM)

**Location**: `chronomoe/chronovisor_mixtral_bridge.py`

**What it does**:
- Token embeddings
- Mixtral decoder stack with Chronovisor P×T control
- Language modeling head
- Generates logits over vocabulary

**Key property**: **Completely stateless**

The forward pass is vanilla Mixtral. Chronovisor tracks coherence and updates P×T fields, but that state lives in the controller, not in the model weights or activations.

### 2. Clock Heads (Fast, Medium, Slow)

**Location**: `chronomoe/clock_heads.py`

**What they do**:
- Evaluate temporal coherence of candidate outputs
- Maintain compressed geometric state
- Update state based on observed outcomes
- Apply temporal decay (what makes memory fade)

**What they store** (the "1GB"):

```python
class ClockState:
    # Reference frame [d_hidden, k_basis]
    # - Learned basis for projecting interactions
    # - Defines coordinate system over behavior space

    # Attractor map [n_attractors, d_hidden]
    # - Centroids of stable basins
    # - Spreads (variance within basin)
    # - Curvatures (stiffness/resistance)

    # Transition model [n_attractors, n_attractors]
    # - P(A→B | conditions)
    # - Learned from observed transitions

    # Constraint set [list of vectors]
    # - Soft penalties on style, safety, preferences
    # - Not retrieved, just enforced

    # Value surface [d_hidden]
    # - What historically leads to good outcomes
    # - Reinforced by positive results

    # Episodic sketchpad [rolling buffer]
    # - Recent embeddings, margins, importance
    # - FIFO with importance sampling
```

**Key property**: **Memory is what refuses to disappear**

State is not facts or text. It's compressed geometry that manifests as changed judgment.

**Differences between clocks**:

| Clock  | Half-life | Attractors | Basis Dim | Question |
|--------|-----------|------------|-----------|----------|
| Fast   | 5 turns   | 64         | 128       | "Local continuity broken?" |
| Medium | 50 turns  | 32         | 64        | "Theme or deviation?" |
| Slow   | 500 turns | 16         | 32        | "Still true over time?" |

### 3. Clock-Gated Generation

**Location**: `chronomoe/clock_gated_generation.py`

**Flow for each generation step**:

1. **Base model generates logits** (stateless forward pass)
2. **Sample top-k candidates** (e.g., 5 most probable tokens)
3. **Each clock scores each candidate**:
   ```python
   for candidate in candidates:
       # Embed (context + candidate) into feature space
       x = embed_interaction(context, candidate)

       # Find nearest basin
       basin_idx, distance, stiffness = find_nearest_basin(x)

       # Compute score
       score = f(proximity, transition_prob, constraints, value)
   ```
4. **Elimination tournament**:
   ```python
   # Compute average score per candidate
   avg_scores = [mean([fast, medium, slow]) for each candidate]

   # Winner = highest average
   winner = candidates[argmax(avg_scores)]
   ```
5. **Update clock states**:
   ```python
   for clock in [fast, medium, slow]:
       clock.update(context, winner, outcome)
       clock.apply_decay()  # Temporal forgetting
   ```

**Key property**: **Arbitration, not generation**

Clocks never produce tokens. They only judge what the base model produced.

---

## How Scoring Works

Each clock computes a score ∈ [0, 1] for a candidate token:

```python
def compute_score(context, candidate) -> float:
    # 1. Embed interaction
    x = embed(context + candidate)

    # 2. Find nearest attractor basin
    basin_idx, distance, stiffness = find_nearest_basin(x)

    # 3. Basin proximity score
    proximity_score = 1 / (1 + distance)  # Closer = better

    # 4. Transition validity score
    if current_basin is not None:
        trans_prob = P(current_basin → basin_idx)
    else:
        trans_prob = 1.0  # No history, accept anything

    # 5. Constraint violation score
    penalties = sum(max(0, constraint · x) for each constraint)
    constraint_score = exp(-penalties)

    # 6. Value score
    value = value_surface · x
    value_score = sigmoid(value)

    # 7. Combine (weighted geometric mean)
    score = proximity^0.4 * trans_prob^0.3 * constraint^0.2 * value^0.1

    return clip(score, 0, 1)
```

**What this measures**:
- **Proximity**: Is this near a known stable region?
- **Transition**: Is this a valid move from where we are?
- **Constraints**: Does this violate learned preferences?
- **Value**: Has this direction worked historically?

---

## How State Updates Work

When a token is selected, each clock updates its internal geometry:

```python
def update(context, selected_token, outcome) -> None:
    # 1. Embed the interaction
    x = embed(context + selected_token)

    # 2. Find which basin this belongs to
    basin_idx, distance, stiffness = find_nearest_basin(x)

    # 3. Update transition model (if we had a previous basin)
    if current_basin is not None:
        # Increment edge weight (EMA)
        transition_matrix[current_basin, basin_idx] += decay_rate * (1 - old_prob)
        renormalize_row()

    # 4. Move attractor centroid toward this point
    # THIS IS THE "MEMORY" - basins drift toward frequent regions
    centroids[basin_idx] = (1 - decay_rate) * old_centroid + decay_rate * x

    # 5. Update attractor spread (variance)
    spreads[basin_idx] = (1 - decay_rate) * old_spread + decay_rate * distance

    # 6. Update value surface
    value_delta = outcome - 0.5  # Positive reinforces, negative penalizes
    value_surface += decay_rate * value_delta * x

    # 7. Add to episodic buffer
    recent_embeddings.append(x)
    recent_margins.append(distance)

    # 8. Update current basin
    current_basin = basin_idx

    # 9. Increment tick counter
    ticks += 1
```

**Key insight**: State evolves via **EMA with decay**.

After `half_life` ticks, an influence decays to 50%. This is what distinguishes the clocks - not how much they store, but how long it persists.

---

## How Decay Works

Periodically (every generation step), clocks apply temporal decay:

```python
def apply_decay() -> None:
    # 1. Decay attractor curvatures toward baseline
    curvatures = (1 - decay_rate*0.1) * curvatures + decay_rate*0.1 * baseline

    # 2. Decay transition matrix toward uniform
    uniform = ones(n_attractors, n_attractors) / n_attractors
    transition_matrix = (1 - decay_rate*0.01) * transition_matrix + decay_rate*0.01 * uniform

    # 3. Decay value surface (prevent unbounded growth)
    value_surface *= (1 - decay_rate*0.1)
```

**This is what makes memory fade.**

Fast clock (decay_rate=0.2) forgets quickly.
Slow clock (decay_rate=0.0014) barely changes.

**Memory is not what is stored, memory is what refuses to disappear.**

---

## Elimination Tournament

Simple and interpretable:

```python
def elimination_tournament(candidates, scores):
    # Compute average score for each candidate
    avg_scores = [
        (scores["fast"][i] + scores["medium"][i] + scores["slow"][i]) / 3
        for i in range(len(candidates))
    ]

    # Winner = highest average
    return candidates[argmax(avg_scores)]
```

**Properties**:
- No learned router
- No silent blending
- No mushy averaging
- Clear failure modes (can diagnose which clock rejected what)
- Easy to debug

**Interpretation**:
- If fast clock wins: Local continuity mattered most
- If medium clock wins: Session trajectory mattered most
- If slow clock wins: Long-term identity mattered most

---

## Where Everything Lives

```
Mixtral Model Structure:
└── ChronovisorMixtralForCausalLM
    ├── embed_tokens: Embedding(vocab_size, hidden_dim)
    ├── model: ChronovisorMixtralModel
    │   ├── layers: List[MixtralDecoderLayer]
    │   │   └── For each layer:
    │   │       ├── self_attn: GroupedQueryAttention
    │   │       └── moe: MixtralSparseMoELayer
    │   │           ├── router: MixtralRouter
    │   │           │   └── Receives P×T fields from controller
    │   │           └── experts: List[MixtralExpert]
    │   └── controller: ChronovisorMixtralController  ← Lives here
    │       ├── lenses: Dict[layer_idx → MixtralLens]
    │       ├── structural_T_global
    │       └── clocks: (fast, micro, macro)
    └── lm_head: Linear(hidden_dim, vocab_size)

Clock-Gated Extension:
└── ClockGatedMixtralForCausalLM(ChronovisorMixtralForCausalLM)
    ├── [inherits everything above]
    ├── fast_clock: FastClock      ← NEW: Sits outside forward pass
    ├── medium_clock: MediumClock  ← NEW: Judges outputs
    └── slow_clock: SlowClock      ← NEW: Updates after selection
```

**Separation of concerns**:
- **Base model**: Generation (stateless)
- **Chronovisor controller**: Coherence tracking, P×T control
- **Clock heads**: Temporal arbitration (stateful, outside forward pass)

---

## Interaction with Chronovisor

The clocks and Chronovisor controller are **complementary but independent**:

**Chronovisor Controller**:
- Operates during the forward pass
- Tracks routing statistics per layer
- Computes Kuramoto coherence (R)
- Updates P×T fields (pressure, temperature)
- Multi-scale clocks (fast, micro, macro) for control updates

**Clock Heads**:
- Operate after the forward pass
- Judge final outputs (tokens)
- Compute temporal coherence across turns
- Update attractor maps and transitions
- Half-life-based decay (fast, medium, slow)

**Both contribute to system behavior**:
- Chronovisor shapes routing (via P×T)
- Clocks filter outputs (via arbitration)

---

## Why This Respects Phase 1 & 2 Findings

**Phase 1 proved**: The router's projection step is stateless and absolute.

**Phase 2 proved**: Even self-gated state can't deform the manifold.

**This architecture**:
- ✓ Never touches the projection step
- ✓ Never tries to create path-dependent routing
- ✓ Respects that the base model is stateless
- ✓ Puts memory where it can actually persist (outside the forward pass)

**The walls stay put. We filter what comes out instead.**

---

## Example Generation Flow

```
Step 1:
  Input: "The cat sat on the"
  Base model → logits
  Top-5 candidates: [mat, floor, chair, roof, table]

  Fast clock scores:   [0.82, 0.75, 0.68, 0.45, 0.71]
  Medium clock scores: [0.79, 0.81, 0.72, 0.50, 0.68]
  Slow clock scores:   [0.88, 0.85, 0.80, 0.55, 0.82]

  Average scores:      [0.83, 0.80, 0.73, 0.50, 0.74]
  Winner: "mat" (highest average)

  Clocks update:
    - "mat" is near basin 7 (common furniture contexts)
    - Transition 3→7 incremented (from "the" to "mat")
    - Value surface reinforced in this direction

Step 2:
  Input: "The cat sat on the mat"
  Base model → logits
  Top-5 candidates: [., and, in, while, ,]

  [Repeat scoring and arbitration]
```

---

## Statistics Tracked

During generation, we track:

```python
clock_stats = {
    "fast_wins": int,      # How often fast clock vetoed runner-up
    "medium_wins": int,    # How often medium clock vetoed runner-up
    "slow_wins": int,      # How often slow clock vetoed runner-up

    "fast_scores": List[List[float]],    # All scores per step
    "medium_scores": List[List[float]],
    "slow_scores": List[List[float]],

    "decisions": List[dict],  # Which clock vetoed which candidate
}
```

This lets us diagnose:
- Which timescale matters most for this generation task
- Whether clocks are agreeing or disagreeing
- If any clock is consistently winning (might be miscalibrated)

---

## Next Steps

1. **Test on small Mixtral**:
   - Run clock-gated generation
   - Verify clocks update correctly
   - Check that state persists as expected

2. **Measure impact**:
   - Compare vanilla vs clock-gated generation
   - Does temporal arbitration improve coherence?
   - Which clock wins most often?

3. **Tune half-lives**:
   - Current: fast=5, medium=50, slow=500
   - May need adjustment based on task

4. **Add outcome signals**:
   - Currently outcome=1.0 (assume good)
   - Could use loss, perplexity, or reward
   - This would make value surface more meaningful

5. **Visualize clock state**:
   - Plot attractor basins over time
   - Show transition graphs
   - Track which basins are active

---

## Theoretical Grounding

**Halcyon's framing** (from conversation):

> "Memory is not what is stored, memory is what refuses to disappear."

**This architecture implements**:
- Clocks as integrators with decay laws
- State distinguished by temporal persistence (half-life)
- Geometry as spatialized time (attractors = persistent regions)
- Arbitration without generation (judgment, not creation)

**The key insight**:
> "You don't need smarter clocks, you need clocks that disagree about how long something has to keep being true before it matters."

Fast clock: "This broke local continuity" (sensitive, forgetful)
Medium clock: "This is off-trajectory" (balanced)
Slow clock: "This isn't who we are" (stable, persistent)

---

## Code Locations

- **Clock heads**: `src/chronomoe/clock_heads.py`
- **Clock-gated generation**: `src/chronomoe/clock_gated_generation.py`
- **Mixtral core**: `src/chronomoe/mixtral_core.py`
- **Chronovisor bridge**: `src/chronomoe/chronovisor_mixtral_bridge.py`

---

**Status**: Architecture complete, ready for testing
**Innovation**: Temporal arbitration outside the forward pass
**Next**: Run on actual Mixtral and measure impact
**Date**: December 2024
