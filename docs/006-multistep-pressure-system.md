# Multistep Generation with Pressure-Based Termination

**Status**: Design complete, implementation pending
**Date**: December 2024
**Key Innovation**: Session-level pause control via temporal pressure dynamics

---

## The Premise

### Problem Statement

Current LLM generation has a "completion pressure" problem:

**Single-turn mode forces premature closure**:
- User asks broad question
- Model generates to completion (EOS)
- Output becomes a mini-thesis
- No natural breakpoints for user steering

**This causes**:
- End-of-answer expert monopolization
- Loss of user control mid-generation
- "Restart from scratch" as only steering mechanism
- Inability to pause and redirect

### Why Existing Solutions Fail

**Auto-continue systems**: Agentic, run without user input → dangerous

**Hard token limits**: Arbitrary cutoffs that don't respect semantic boundaries

**Prompt engineering**: "Be brief" → inconsistent, unreliable

**What we need**: A system that knows when to pause based on internal state, but never continues without user permission.

### The Core Insight

**Temporal clocks already measure coherence at different timescales.**

We can use these measurements to compute **pressure** - a directional force that indicates whether the system wants to continue or stop.

**Key constraint**: Pressure can suggest pause, but **only the user can trigger continuation**.

---

## Design Principles

### 1. Non-Agentic by Construction

**Rule**: Continuation requires user input.

In multistep mode:
- Generate chunk
- Compute pressures
- Pause unconditionally
- Wait for user

No background processing. No auto-continue. System is "warm" only in state variables.

### 2. Pressure ≠ Score

**Pressure is directional force, not goodness.**

- Positive pressure: push to continue/expand
- Negative pressure: restrain/pause/damp
- Zero pressure: locally indifferent

This is force balance, not optimization.

### 3. Authority Separation by Timescale

**Fast clock**: Can interrupt (force pause), cannot steer

**Mid clock**: Carries momentum (drives continuation), primary authority in multistep

**Slow clock**: Can veto (identity violations), cannot rush

Each clock has different powers. No single clock dominates.

### 4. Explicit User Contract

Mode is **user-controlled via commands**:

```
"multistep on"  → Enable chunked generation
"multistep off" → Return to single-turn
"end loop"      → Collapse residual intent, reset state
```

No magic inference of intent. User explicitly sets the mode.

### 5. Measurable, Non-Anthropomorphic

Telemetry exposes:
- Pressure values (numeric)
- Stability metrics (numeric)
- Pause reasons (categorical)

No "I'm unsure" or "I think we should continue". Just instrumentation.

---

## Architecture

### Component Stack

```
┌─────────────────────────────────────────────────────────┐
│               SESSION CONTROLLER (NEW)                  │
│                                                         │
│  Mode: single_turn | multistep                         │
│  Residual Intent: [0, 1]                               │
│  Pause Rules: f(pressures, mode)                       │
│                                                         │
│  Commands: "multistep on/off", "end loop"              │
└────────────────┬────────────────────────────────────────┘
                 │
                 ↓
        Chunked Generation Loop
        (generate N tokens → compute pressures → pause)
                 │
         ┌───────┼───────┐
         ↓       ↓       ↓
    ┌────────────────────────────────────┐
    │    PRESSURE COMPUTATION (NEW)      │
    │                                    │
    │  Fast:  P_fast  (stability)        │
    │  Mid:   P_mid   (intent)           │
    │  Slow:  P_slow  (identity)         │
    │                                    │
    │  Weights: f(entropy, intent, conf) │
    │  Net: w_f·P_f + w_m·P_m + w_s·P_s │
    └────────────────────────────────────┘
                 │
         ┌───────┼───────┐
         ↓       ↓       ↓
    ┌─────────────────────────────────┐
    │    CLOCK HEADS (Unchanged)      │
    │                                 │
    │  Fast:   score candidates       │
    │  Medium: score candidates       │
    │  Slow:   score candidates       │
    │                                 │
    │  Elimination tournament         │
    └─────────────────────────────────┘
                 │
                 ↓
         ┌───────────────┐
         │  BASE MODEL   │
         │  (stateless)  │
         └───────────────┘
```

### What Changes, What Doesn't

**UNCHANGED**:
- Clock heads (scoring, state updates, decay)
- Base model forward pass (stateless)
- Chronovisor controller (coherence, P×T fields)
- Elimination tournament (candidate selection)

**NEW**:
- Session controller (manages mode, residual intent)
- Pressure computation (per-clock force calculations)
- Pause rules (mechanical decision based on pressures)
- Chunked generation loop (generate → pause → wait)
- User commands (explicit mode control)

---

## Pressure System Design

### Per-Clock Pressure Functions

Each clock computes pressure from **signals already available** (no second forward pass).

#### Fast Clock Pressure (Stability)

**Question**: "Is the system locally stable enough to proceed right now?"

**Inputs**:
- `router_entropy` ∈ [0, 1]: Routing distribution entropy
- `router_margin`: Top-1 vs top-2 expert margin
- `delta_R`: Coherence change (Kuramoto)

**Function**:
```python
P_fast = 0.5 * (-router_entropy) +
         0.3 * tanh(router_margin) +
         0.2 * tanh(delta_R)
```

**Clipped to [-1, 1]**

**Properties**:
- Mostly negative by default (damping bias)
- Spikes negative on chaos (high entropy)
- Rarely produces strong positive pressure
- Only unilateral authority: force pause on instability

**Interpretation**:
- `P_fast = -0.8`: Router chaos → immediate pause
- `P_fast = 0.2`: Stable → don't interrupt
- `P_fast > 0.5`: Rare, system very smooth

---

#### Mid Clock Pressure (Intent/Trajectory)

**Question**: "Is there meaningful unfinished business worth continuing?"

**Inputs**:
- `margin`: Routing margin (uncertainty proxy)
- `trans_prob`: Transition plausibility (from mid clock state)
- `prox_meso`: Proximity to meso basin (task/topic fit)
- `delta_R`: Coherence change

**Function**:
```python
P_mid = 0.4 * tanh(1.0 - margin) +      # Uncertainty drives curiosity
        0.3 * tanh(trans_prob) +         # Plausible next state
        0.2 * tanh(prox_meso) +          # Task coherence
        0.1 * tanh(delta_R)              # Recovery bonus
```

**Properties**:
- Main driver of multistep continuation
- High when uncertain + good structure
- Decays naturally as uncertainty resolves
- Carries "residual intent"

**Interpretation**:
- `P_mid = 0.8`: Low margin + good structure → wants to continue
- `P_mid = 0.2`: Uncertainty resolved → natural decay
- `P_mid < 0`: Topic drift or bad transition → restrain

---

#### Slow Clock Pressure (Identity/Invariants)

**Question**: "Is continuing consistent with long-term invariants?"

**Inputs**:
- `prox_macro`: Proximity to macro basin (identity fit)
- `constraint_penalty`: Summed soft penalties from constraints
- `confidence`: Slow clock confidence (basin stability)

**Function**:
```python
P_slow = confidence * (
    0.3 * tanh(prox_macro) +            # Identity alignment
    0.7 * (-tanh(constraint_penalty))   # Violations dominate
)
```

**Properties**:
- Low variance, conservative
- Usually near zero (watching, not pushing)
- Strongly negative on constraint violations
- Confidence gates expression (only speaks when certain)

**Interpretation**:
- `P_slow = -0.9`: Constraint violation + high confidence → veto
- `P_slow ≈ 0`: Usually near zero
- `P_slow = 0.3`: Rare, identity strongly aligned

---

### Pressure Weighting (Authority Allocation)

**Who is allowed to matter?**

```python
w_fast = min(0.2, 1.0 - router_entropy)    # HARD CAP at 0.2
w_mid  = 0.5 + 0.5 * residual_intent       # [0.5, 1.0]
w_slow = 0.3 * slow_confidence             # [0, 0.3]
```

**Key properties**:

1. **Fast clock hard capped at 20% authority**
   - Can interrupt, cannot dominate
   - Prevents catastrophic fast-pressure runaway

2. **Mid clock scales with residual intent**
   - Dominant in multistep mode (50-100% weight)
   - Carries momentum

3. **Slow clock grows with confidence**
   - Only speaks when certain
   - Maximum 30% authority
   - Defines "no", not "go"

### Net Pressure Computation

```python
net_pressure = w_fast * P_fast +
               w_mid * P_mid +
               w_slow * P_slow
```

**Clipped to [-1, 1]**

This is the **force balance**. Not a score, not a prediction. Pure dynamics.

---

### Pause/Continue Rules

**Mechanical, no heuristics**:

```python
if P_fast < -0.7:
    PAUSE("fast_instability")
elif net_pressure < 0:
    PAUSE("negative_pressure")
elif mode == "multistep":
    PAUSE("multistep_chunk_complete")
else:
    CONTINUE()
```

**Decision tree**:

1. **Fast clock emergency brake**: If fast pressure drops below -0.7 → immediate pause
2. **Net pressure negative**: System wants to stop → pause
3. **Multistep mode**: Always pause after chunk (yield to user)
4. **Otherwise**: Continue

**No vibes. Just force balance.**

---

### Residual Intent

**Definition**: Scalar ∈ [0, 1] representing "how much unfinished business?"

**Not a symbolic plan.** Just "completion pressure."

**Computation**:
```python
residual_intent = f(
    mid_pressure,         # Base signal
    margin,               # Uncertainty
    delta_R,              # Coherence trend
    chunk_fullness,       # How much of chunk used
)
```

**Detailed formula**:
```python
intent_base = max(0, P_mid)
chunk_factor = 1.0 - (tokens_generated / max_chunk_size)
margin_factor = 1.0 + (1.0 - margin)  # Boost if uncertain
recovery_factor = 1.0 if delta_R < 0 else max(0.5, 1.0 - delta_R)

residual_intent = intent_base * chunk_factor * margin_factor * recovery_factor
```

**Clipped to [0, 1]**

**Properties**:
- High when mid pressure is positive and margin is low
- Decays as chunk fills up
- Decays as coherence recovers
- Carries across chunks in multistep mode
- Reset to 0 on "end loop" command

---

## User Interface

### Session Modes

**Single-turn mode** (default):
```
User: "Explain quantum mechanics"
System: [generates to completion]
       [no pause, standard behavior]
```

**Multistep mode**:
```
User: "multistep on"
System: [mode = multistep]

User: "Explain quantum mechanics"
System: [chunk 1: 100 tokens]
        [PAUSE: multistep_chunk_complete]
        [residual_intent = 0.7]

User: "go on"
System: [chunk 2: 100 tokens]
        [PAUSE: multistep_chunk_complete]
        [residual_intent = 0.4]

User: "more technical"
System: [chunk 3: 100 tokens, adjusted by context]
        [PAUSE: multistep_chunk_complete]
        [residual_intent = 0.2]

User: "end loop"
System: [residual_intent → 0]
```

### User Commands

**Mode control**:
- `"multistep on"`: Enable chunked generation
- `"multistep off"`: Return to single-turn
- `"end loop"`: Collapse residual intent, signal completion

**Continuation** (in multistep mode):
- Any user input after pause → append to context, continue generation
- User input treated as trajectory perturbation, not restart

**No auto-continue**. Every chunk requires user action to proceed.

---

## Telemetry (Non-Anthropomorphic)

### What's Exposed

```json
{
  "mode": "multistep",
  "chunk_id": 3,
  "pause_reason": "multistep_chunk_complete",
  "residual_intent": 0.7,

  "pressures": {
    "fast": {
      "value": -0.2,
      "weight": 0.15,
      "signals": {
        "router_entropy": 0.3,
        "router_margin": 0.8,
        "delta_R": 0.1
      }
    },
    "mid": {
      "value": 0.6,
      "weight": 0.85,
      "signals": {
        "margin": 0.4,
        "trans_prob": 0.7,
        "prox_meso": 0.8,
        "delta_R": 0.1
      }
    },
    "slow": {
      "value": 0.1,
      "weight": 0.27,
      "signals": {
        "prox_macro": 0.5,
        "constraint_penalty": 0.0,
        "confidence": 0.9
      }
    },
    "net": 0.42
  },

  "clocks": {
    "fast": {"current_basin": 12, "confidence": 0.6},
    "mid": {"current_basin": 7, "confidence": 0.8},
    "slow": {"current_basin": 2, "confidence": 0.9}
  }
}
```

**No feelings. Just numbers.**

Users (or UIs) can display:
- "System stable" (fast pressure > 0)
- "Unfinished business" (residual intent > 0.5)
- "Identity stable" (slow confidence > 0.8)

---

## Code Changes

### New Components

#### 1. `SessionController` Class

**Location**: `src/chronomoe/session_controller.py`

**Responsibilities**:
- Mode management (single_turn / multistep)
- Residual intent tracking
- Chunked generation loop
- Pause decision execution
- User command handling

**Key methods**:
```python
class SessionController:
    def __init__(self, model, chunk_size=100)
    def set_mode(self, mode: str)
    def end_loop(self)
    def generate_with_pauses(self, input_ids, max_chunks)
    def _generate_chunk(self, input_ids, max_tokens)
```

#### 2. `PressureSystem` Module

**Location**: `src/chronomoe/pressure.py`

**Functions**:
```python
def compute_fast_pressure(router_entropy, router_margin, delta_R) -> float
def compute_mid_pressure(margin, trans_prob, prox_meso, delta_R) -> float
def compute_slow_pressure(prox_macro, constraint_penalty, confidence) -> float

def compute_pressure_weights(router_entropy, residual_intent, slow_confidence) -> Tuple[float, float, float]
def compute_net_pressure(P_fast, P_mid, P_slow, w_fast, w_mid, w_slow) -> float

def should_pause(P_fast, net_pressure, mode) -> Tuple[bool, str]
def compute_residual_intent(P_mid, margin, delta_R, chunk_tokens, max_chunk) -> float
```

#### 3. Extended `ClockGatedMixtralForCausalLM`

**Location**: `src/chronomoe/clock_gated_generation.py` (extended)

**New methods**:
```python
class ClockGatedMixtralForCausalLM:
    # Existing methods unchanged

    # New: Extract signals for pressure computation
    def get_pressure_signals(self) -> dict

    # New: Wrap with session controller
    def generate_multistep(self, input_ids, mode="single_turn", **kwargs)
```

### Modified Components

**None**. Clocks, base model, Chronovisor controller remain unchanged.

---

## Integration Points

### How Pressures Access Clock State

Pressures use **signals already computed**:

**From Chronovisor controller**:
- `router_entropy`: Entropy of routing distribution
- `router_margin`: Expert margin
- `delta_R`: Coherence change

**From clock heads**:
- `trans_prob`: Mid clock transition probability
- `prox_meso`: Mid clock meso basin proximity
- `prox_macro`: Slow clock macro basin proximity
- `constraint_penalty`: Slow clock constraint violations
- `confidence`: Slow clock confidence

**From base model**:
- `margin`: Routing margin (already tracked)

**No second forward pass. No extra embeddings.**

### Data Flow

```
Generation step:
    Base model → logits
    Clocks → candidate scores
    Tournament → selected token

After chunk complete:
    Extract signals (router_entropy, margin, etc.)
    Compute pressures (P_fast, P_mid, P_slow)
    Compute weights (w_fast, w_mid, w_slow)
    Compute net pressure
    Compute residual intent
    Check pause rules
    → PAUSE or CONTINUE
```

---

## Guarantees

### What We Can Prove

1. **Non-agentic by construction**
   - Multistep mode requires user input to continue
   - No background processing
   - State updates only during generation, not during pause

2. **Pressure is monotonic in observable signals**
   - All pressure functions are compositions of monotonic functions
   - Reviewers can verify algebra

3. **Authority separation is enforced**
   - Fast clock weight hard capped at 0.2
   - Only fast clock can force immediate pause
   - Only slow clock + consensus can veto

4. **Pause decisions are deterministic**
   - Given pressures and mode, pause decision is mechanical
   - No hidden state, no randomness

5. **No second forward pass**
   - All signals already computed
   - Pressure computation is O(1) linear algebra

### What We Measure

**Update selectivity**: Pressures concentrate in low-margin regions

**In-distribution stability**: Pressures remain bounded under normal text

**OOD sensitivity**: Pressures respond to distribution shift

**Mode adherence**: Multistep mode always pauses after chunk

**User override**: Commands work deterministically

---

## Expected Behavior Changes

### Before (Single-turn only)

```
User: "Explain quantum entanglement"
System: [generates 500 tokens to completion]
        [includes introduction, formalism, examples, implications]
        [user has no control mid-generation]
```

### After (Multistep mode)

```
User: "multistep on"
User: "Explain quantum entanglement"
System: [chunk 1: basics, 100 tokens]
        [P_mid = 0.7, residual_intent = 0.8]
        [PAUSE]

User: "go on"
System: [chunk 2: formalism, 100 tokens]
        [P_mid = 0.5, residual_intent = 0.6]
        [PAUSE]

User: "skip the math, just the implications"
System: [chunk 3: implications, adjusted, 80 tokens]
        [P_mid = 0.2, residual_intent = 0.3]
        [PAUSE]

User: "end loop"
System: [residual_intent → 0]
```

**Key differences**:
- User can steer mid-generation
- Natural breakpoints at semantic boundaries
- "Trajectory perturbation" instead of "restart from scratch"
- Less completion pressure per turn

---

## Failure Modes and Mitigations

### Potential Failure: Fast Clock Runaway

**Symptom**: Fast pressure constantly negative, system always pausing

**Cause**: Router instability, high entropy

**Mitigation**: Hard cap fast weight at 0.2, requires net pressure < 0 to pause

### Potential Failure: Infinite Loop

**Symptom**: Residual intent never decays, system always wants to continue

**Cause**: Mid pressure stays high, margin never increases

**Mitigation**:
- Max chunks limit
- Manual "end loop" command
- Residual intent decays with chunk fullness

### Potential Failure: Premature Stop

**Symptom**: System pauses too early, feels abrupt

**Cause**: Mid pressure drops too quickly

**Mitigation**: Tune mid pressure function, adjust weights

### Potential Failure: Slow Clock Over-Vetoing

**Symptom**: Slow clock blocks all novelty

**Cause**: Constraint penalties too aggressive

**Mitigation**:
- Slow weight capped at 0.3
- Requires consensus for veto (fast + medium must agree)
- Constraints are soft penalties, not hard rules

---

## Testing Strategy

### Unit Tests

1. **Pressure functions are bounded**
   - All inputs → all pressures ∈ [-1, 1]

2. **Weights sum correctly**
   - Given valid inputs → w_fast + w_mid + w_slow is sensible
   - Fast weight never exceeds 0.2

3. **Pause rules are deterministic**
   - Same inputs → same pause decision

4. **Residual intent decays**
   - As chunk fills → residual_intent decreases

### Integration Tests

1. **Multistep mode pauses after chunk**
   - Generate with mode=multistep → always pauses

2. **Single-turn mode completes**
   - Generate with mode=single_turn → runs to EOS

3. **User commands work**
   - "multistep on" → mode changes
   - "end loop" → residual_intent resets

4. **Pressures use existing signals**
   - No second forward pass occurs

### Behavioral Tests

1. **Update selectivity**: High-pressure updates concentrate in low-margin regions

2. **Stability**: Pressures bounded under normal text

3. **OOD response**: Pressures respond to noise/corruption

4. **No auto-continue**: After pause, no generation without user input

---

## Reviewer-Proof Claims

### Paper-Safe Statements

> "We introduce a pressure-based termination system where generation continues or pauses based on a weighted combination of temporal coherence signals already computed during the forward pass."

> "Fast clock pressure monitors local stability and can force pause on instability, but is hard-capped at 20% authority to prevent runaway."

> "Mid clock pressure carries residual intent - a scalar representing estimated completion pressure - which decays naturally as uncertainty resolves."

> "Slow clock pressure enforces long-term constraints, producing strongly negative pressure on identity violations, but requires consensus with faster clocks to veto."

> "In multistep mode, the system pauses after each chunk unconditionally, requiring user input to continue. This ensures non-agentic behavior by construction."

> "All pressure computations use signals already produced by the base model, clocks, or controller. No second forward pass is required."

### What Reviewers Cannot Attack

✓ Pressures are explicit, monotonic functions of observable signals
✓ Pause decisions are deterministic given pressures and mode
✓ Authority separation is mechanically enforced via weight caps
✓ Non-agentic property is guaranteed by control flow
✓ No hidden state, no anthropomorphic reasoning
✓ Telemetry exposes all decision factors

---

## Implementation Checklist

### Phase 1: Pressure System (Standalone)

- [ ] `src/chronomoe/pressure.py`
  - [ ] `compute_fast_pressure()`
  - [ ] `compute_mid_pressure()`
  - [ ] `compute_slow_pressure()`
  - [ ] `compute_pressure_weights()`
  - [ ] `compute_net_pressure()`
  - [ ] `should_pause()`
  - [ ] `compute_residual_intent()`
- [ ] Unit tests for all pressure functions
- [ ] Verify all outputs bounded to [-1, 1]

### Phase 2: Session Controller

- [ ] `src/chronomoe/session_controller.py`
  - [ ] `SessionController.__init__()`
  - [ ] `set_mode()`
  - [ ] `end_loop()`
  - [ ] `generate_with_pauses()`
  - [ ] `_generate_chunk()`
  - [ ] `_extract_signals()`
- [ ] Integration with `ClockGatedMixtralForCausalLM`
- [ ] User command parsing

### Phase 3: Integration

- [ ] Extend `ClockGatedMixtralForCausalLM.get_pressure_signals()`
- [ ] Wire pressure system into generation loop
- [ ] Add telemetry logging
- [ ] Test multistep mode end-to-end

### Phase 4: Validation

- [ ] Behavioral tests (pause after chunk, no auto-continue)
- [ ] Pressure response tests (OOD, stability)
- [ ] User command tests
- [ ] Failure mode tests

---

## Open Questions (To Resolve During Implementation)

1. **Chunk size**: 100 tokens default, or adaptive based on semantic boundaries?

2. **Mid pressure tuning**: Are the weights (0.4, 0.3, 0.2, 0.1) optimal, or should we tune empirically?

3. **Residual intent floor**: Should residual intent have a minimum threshold below which it's forced to 0?

4. **Telemetry format**: JSON? Structured logs? UI-friendly format?

5. **Fast clock emergency threshold**: Is -0.7 the right threshold for forced pause, or should it be tunable?

---

## Summary

**What this adds**: Session-level pause control via temporal pressure dynamics

**What this changes**: Nothing in clocks, base model, or Chronovisor controller

**What this enables**:
- Chunked generation with natural breakpoints
- User steering mid-generation (trajectory perturbation)
- Non-agentic multistep mode (requires user input)
- Measurable, reviewable pressure dynamics

**What this guarantees**:
- No second forward pass
- Non-agentic by construction
- Deterministic pause rules
- Authority separation enforced
- Monotonic pressure functions

**What this prevents**:
- Fast clock runaway (hard cap at 20%)
- Slow clock over-vetoing (requires consensus)
- Auto-continue (multistep mode demands user input)
- Thesis-per-turn completion pressure

---

**Status**: Design complete, ready for implementation
**Next**: Implement pressure system, then session controller
**Date**: December 2024
