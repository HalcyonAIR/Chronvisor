# Multistep Pressure System - Implementation Status

**Date**: December 2024
**Status**: ✓ Core implementation complete, ready for testing
**Next**: Phase 1 validation (TEST_PLAN.md)

---

## What Was Implemented

### 1. Pressure System (`src/chronomoe/pressure.py`)

Complete implementation of pressure-based pause/continue logic.

**Features**:
- Three pressure functions (fast/mid/slow) with bounded outputs [-1, 1]
- Authority-separated weights (fast capped at 20%, mid dominant 50-100%, slow capped at 30%)
- Deterministic pause rules (no randomness, fully reproducible)
- Residual intent tracking for momentum across chunks
- Monotonic, reviewer-proof pressure computations

**Key functions**:
```python
compute_fast_pressure(router_entropy, router_margin, delta_R)
compute_mid_pressure(margin, mid_transition_prob, mid_proximity_meso, delta_R)
compute_slow_pressure(slow_confidence, slow_proximity_macro, slow_constraint_penalty)
compute_pressure_weights(router_entropy, mid_residual_intent, slow_confidence)
compute_net_pressure(...)
should_pause(fast_pressure, net_pressure, mode)
compute_residual_intent(...)
```

**Validation**: ✓ All unit tests pass (pressure.py:main)

---

### 2. Session Controller (`src/chronomoe/session_controller.py`)

Manages multistep generation mode and telemetry.

**Features**:
- Mode switching (single_turn vs multistep)
- User command parsing ("multistep on/off", "continue", "end loop", "reset")
- Signal extraction from existing computation (NO second forward pass)
- Chunk-level telemetry with pressure values
- Session-level aggregation (avg chunk length, pause reason distribution)
- JSON export for analysis

**Non-agentic guarantee**: Generation halts after chunk in multistep mode. No auto-continuation.

**Validation**: ✓ All tests pass (session_controller.py:main)

---

### 3. Corrected Clock Heads (`src/chronomoe/clock_heads_corrected.py`)

**CRITICAL FIXES** from Halcyon's review:

#### Fix #1: No Re-Embedding Trap
**Old code** (BUGGY):
```python
def compute_score(context_ids, candidate_id, model_embeddings):
    x = embed(context + candidate)  # RE-RUNS FULL CONTEXT EVERY TIME!
```

**New code** (CORRECTED):
```python
def compute_score(h_t, logp_candidate, margin, router_margin, router_entropy, ...):
    # Uses existing signals only (NO re-embedding)
    z = project_to_clock_space(h_t, logp_candidate, margin, ...)
```

#### Fix #2: Event-Gated Updates
**Old code** (BUGGY):
```python
def update(...):
    # Always updates on every token (learns commas!)
    state.update(winner, outcome)
```

**New code** (CORRECTED):
```python
def update(...):
    should_learn, weight = should_update(margin, coherence, surprisal, is_correction)
    if not should_learn:
        return  # High margin = confident = don't learn

    # Only updates on uncertainty/surprisal/corrections
    state.update(winner, outcome, weight)
```

**Additional improvements**:
- Hierarchical attractors (micro/meso/macro) for multi-scale patterns
- State-conditioned transitions P(A→B | margin, coherence)
- Conditioned value surfaces per task mode (technical, creative, terse, explanatory)
- Decay laws preserve temporal distinction (fast=5, medium=50, slow=500)

**Validation**: ✓ All tests pass (clock_heads_corrected.py:main)

---

### 4. Integrated Model (`src/chronomoe/clock_gated_multistep.py`)

Complete integration of all components.

**Architecture stack**:
```
User Input
    ↓
SessionController (mode, commands, telemetry)
    ↓
ClockGatedMultistepModel
    ├── Base Model (stateless Mixtral)
    ├── Corrected Clock Heads (temporal arbitration)
    └── Pressure System (pause decisions)
    ↓
Generated Output + Pause Decision
```

**Key method**:
```python
model.generate_multistep(
    input_ids,
    mode=SessionMode.MULTISTEP,
    chunk_size=50,
    max_chunks=10,
)
# Returns: (generated_ids, session_telemetry)
```

**Validation**: ✓ Integration tests pass (clock_gated_multistep.py:main)

---

### 5. Test Plan (`TEST_PLAN.md`)

Comprehensive validation checklist from Halcyon's test framework.

**5 phases**:
- **Phase 0**: Wiring & invariants (no second forward pass, no auto-continue)
- **Phase 1**: Controller sanity (deterministic pauses, monotonic pressures)
- **Phase 2**: Mixtral tests (entropy profile, thesis pressure, user perturbation)
- **Phase 3**: DeepSeek tests (smoothness, manifold preservation, novelty handling)
- **Phase 4**: Failure injection (fast pressure abuse, identity violation)
- **Phase 5**: Plots & logging (paper-grade telemetry)

**Exit criteria**: GO if pauses explainable, fast never steers, multistep removes thesis-per-turn

---

## Files Created

1. `src/chronomoe/pressure.py` (322 lines)
2. `src/chronomoe/session_controller.py` (399 lines)
3. `src/chronomoe/clock_heads_corrected.py` (827 lines)
4. `src/chronomoe/clock_gated_multistep.py` (477 lines)
5. `TEST_PLAN.md` (183 lines)
6. `docs/006-multistep-pressure-system.md` (created earlier, comprehensive design doc)

**Total new code**: ~2200 lines
**All tests passing**: ✓

---

## What Changed From Original Clock Heads

| Aspect | Original (Buggy) | Corrected |
|--------|------------------|-----------|
| **Scoring input** | Re-embeds context + candidate | Uses existing signals (h_t, margin, coherence) |
| **Update trigger** | Every token | Event-gated (uncertainty, surprisal, corrections) |
| **Attractor structure** | Flat (single level) | Hierarchical (micro/meso/macro) |
| **Transitions** | Fixed P(A→B) matrix | State-conditioned P(A→B \| features) |
| **Value surfaces** | Single global | Per task mode (technical, creative, etc.) |
| **Computational cost** | O(context_len × candidates × steps) | O(1) per token (uses precomputed signals) |

**Key insight**: The original code would have looked fine until production, then become a performance disaster and learn commas.

---

## How to Use

### Basic usage (single-turn mode):

```python
from chronomoe.clock_gated_multistep import ClockGatedMultistepModel, SessionMode
from chronomoe.chronovisor_mixtral_bridge import MixtralConfig

# Create model
config = MixtralConfig(...)
model = ClockGatedMultistepModel(config)

# Generate (normal behavior)
generated, telemetry = model.generate_multistep(
    input_ids,
    mode=SessionMode.SINGLE_TURN,
)
```

### Multistep mode with pauses:

```python
# Enable multistep
generated, telemetry = model.generate_multistep(
    input_ids,
    mode=SessionMode.MULTISTEP,
    chunk_size=50,
    max_chunks=10,
    verbose=True,
)

# Pauses after first chunk (non-agentic)
print(f"Paused after {telemetry.total_chunks} chunks")
print(f"Reason: {telemetry.pause_reasons}")

# User decides to continue
# (would require interactive loop in production)
```

### Access telemetry:

```python
# Session-level stats
print(f"Total tokens: {telemetry.total_tokens}")
print(f"Avg chunk length: {telemetry.avg_chunk_length:.2f}")
print(f"Avg net pressure: {telemetry.avg_net_pressure:+.4f}")

# Per-chunk details
for chunk in telemetry.chunks:
    print(f"Chunk {chunk.chunk_index}:")
    print(f"  Net pressure: {chunk.net_pressure:+.4f}")
    print(f"  Fast: {chunk.fast_pressure:+.4f} (w={chunk.fast_weight:.2f})")
    print(f"  Mid:  {chunk.mid_pressure:+.4f} (w={chunk.mid_weight:.2f})")
    print(f"  Slow: {chunk.slow_pressure:+.4f} (w={chunk.slow_weight:.2f})")

# Export to JSON
json_str = telemetry.to_json(indent=2)
```

---

## What's NOT Implemented Yet

### 1. Full Signal Extraction

Current status: Uses placeholder values for some signals.

**Needed**:
- Compute router margin from expert usage distributions
- Extract basin proximities from clock state
- Compute constraint penalties from clock constraints
- Extract confidence metrics from basin spreads

**Priority**: Medium (placeholders work for Phase 1 testing)

### 2. Interactive User Loop

Current: Model pauses, returns control to caller.

**Needed**:
- CLI interface for user commands during generation
- "Continue", "modify trajectory", "end loop" handling
- User feedback integration

**Priority**: Medium (can test non-interactively first)

### 3. Outcome Signals

Current: Uses outcome=1.0 (assume good).

**Needed**:
- Low margin + continuation = positive reinforcement
- Regeneration/correction = strong negative signal
- Perplexity-based outcome scoring

**Priority**: Medium (affects value surface learning, not pause decisions)

### 4. Full Mixtral Integration

Current: Tested on small synthetic Mixtral.

**Needed**:
- Test on real Mixtral-8x7B
- Verify signal extraction works with real model
- Benchmark computational overhead

**Priority**: High (next step)

### 5. DeepSeek Integration

Current: None.

**Needed**:
- Adapter for DeepSeek architecture
- Compare pressure smoothness (Mixtral vs DeepSeek)
- Verify manifold preservation claims

**Priority**: High (key experiment for paper)

---

## Next Steps (In Order)

### Immediate (This Week):

1. **Run Phase 0-1 tests** (TEST_PLAN.md)
   - Verify no second forward pass
   - Verify non-agentic behavior (no auto-continue)
   - Verify pressure monotonicity
   - Verify deterministic pauses

2. **Implement full signal extraction**
   - Router margin from expert usage
   - Clock confidence from basin spreads
   - Constraint penalties

3. **Test on small real Mixtral**
   - Verify integration works end-to-end
   - Measure computational overhead
   - Check memory usage

### Short-term (Next Week):

4. **Run Phase 2 tests on Mixtral**
   - Router entropy profiles (single vs multistep)
   - Thesis pressure test (decay curves)
   - User perturbation steering

5. **Implement outcome signals**
   - Perplexity-based scoring
   - Correction detection
   - Low-margin learning signals

### Medium-term (Next 2 Weeks):

6. **DeepSeek adapter**
   - Modify for DeepSeek architecture
   - Run Phase 3 comparison tests
   - Generate pressure smoothness plots

7. **Create visualization tools**
   - Pressure trajectory plots
   - Basin transition graphs
   - Coherence vs pressure correlation

8. **Write results section**
   - Mixtral vs DeepSeek comparison
   - Multistep removes thesis pressure
   - Non-agentic guarantee proof

---

## Known Issues / Limitations

### 1. Placeholder Router Stats

**Issue**: Router margin computed as placeholder (0.5) instead of from expert usage.

**Impact**: Mid/slow pressure less accurate, but pressure system still functional.

**Fix**: Extract actual expert usage distributions from chrono_state.expert_usage.

### 2. No Interactive Loop

**Issue**: Model pauses but requires caller to handle user interaction.

**Impact**: Can't test full multistep UX yet.

**Fix**: Implement CLI wrapper with command parsing.

### 3. Value Surfaces Not Learning

**Issue**: Outcome=1.0 constant, so value surfaces don't distinguish good/bad.

**Impact**: Value component of score doesn't improve over time.

**Fix**: Implement perplexity-based outcome signals.

### 4. Task Mode Not Inferred

**Issue**: Task mode defaults to TaskMode.DEFAULT, not inferred from context.

**Impact**: Conditioned value surfaces not utilized.

**Fix**: Add task mode detection (could use keyword matching or learned classifier).

---

## Validation Status

| Component | Unit Tests | Integration Tests | Ready for Phase 1 |
|-----------|------------|-------------------|-------------------|
| Pressure system | ✓ Pass | ✓ Pass | ✓ Yes |
| Session controller | ✓ Pass | ✓ Pass | ✓ Yes |
| Corrected clocks | ✓ Pass | ✓ Pass | ✓ Yes |
| Multistep model | ✓ Pass | ✓ Pass | ✓ Yes (small Mixtral) |
| Signal extraction | ⚠ Partial | ⚠ Partial | ⚠ Needs work |

**Overall status**: Core system ready for Phase 1 testing (TEST_PLAN.md).

---

## Design Guarantees

The following properties are **guaranteed by construction** (not just tested):

### 1. Non-Agentic
```python
if mode == SessionMode.MULTISTEP:
    return True, "multistep_chunk_complete"
    # HARD STOP - no continuation without user input
```

Control flow prevents auto-continuation. Reviewer can verify by inspection.

### 2. No Second Forward Pass
```python
def compute_score(h_t, logp, margin, ...):  # All pre-computed
    z = project_to_clock_space(...)  # Just a matrix multiply
    # NO calls to model.forward()
```

Clocks never call model. Performance overhead is O(1) per token.

### 3. Monotonic Pressures
```python
def compute_fast_pressure(entropy, margin, delta_R):
    # All terms use tanh/exp - smooth, bounded, monotonic
    return clip(pressure, -1.0, 1.0)
```

Reviewer can verify algebra. No hidden nonlinearities.

### 4. Authority Separation
```python
fast_weight = min(0.2, 1.0 - entropy)  # HARD CAP at 20%
mid_weight = 0.5 + 0.5 * residual      # Range [0.5, 1.0]
slow_weight = min(0.3, confidence * 0.3)  # HARD CAP at 30%
```

Fast cannot dominate. Mid is primary driver. Slow can veto with consensus.

### 5. Event-Gated Learning
```python
if margin > 0.5:  # Confident = don't learn
    return False, 0.0
```

High-margin tokens (like commas) are explicitly skipped. Prevents degeneracy.

---

## Questions for Review

1. **Pressure formulas**: Are the weightings (0.4, 0.3, 0.2, 0.1) appropriate, or should they be tunable?

2. **Fast clock cap**: Is 20% the right limit, or should it be even lower?

3. **Event gating thresholds**: Margin < 0.5 triggers learning. Is this too permissive?

4. **Chunk size**: Default 50 tokens. Does this make sense for Mixtral/DeepSeek?

5. **Residual intent decay**: Currently 50% retention on pause. Too aggressive?

6. **Outcome signals**: Should we use perplexity, coherence recovery, or both?

---

## Acknowledgments

**Design**: Jeff (Chronovisor), Halcyon (pressure system, grenades, storage spec), Claude Cloud (clocks concept)

**Implementation**: Claude Sonnet 4.5 (this session)

**Critical catches**: Halcyon identified both major bugs (re-embedding trap, update-on-every-token) before production

---

**Status**: ✓ Implementation complete
**Next**: Phase 1 validation (TEST_PLAN.md)
**ETA for full testing**: 1-2 weeks

---

## Summary for Halcyon

We now have:

1. ✓ Pressure system (monotonic, bounded, authority-separated)
2. ✓ Event-gated clock updates (no comma learning)
3. ✓ No re-embedding trap (uses existing signals)
4. ✓ Hierarchical attractors (micro/meso/macro)
5. ✓ State-conditioned transitions
6. ✓ Conditioned value surfaces
7. ✓ Non-agentic multistep (hard stop after chunk)
8. ✓ Reviewer-proof design (no hidden nonlinearities)
9. ✓ Complete test plan (5 phases, exit criteria)

**The walls stay put. We filter what comes out instead.**

Ready for Phase 1?
