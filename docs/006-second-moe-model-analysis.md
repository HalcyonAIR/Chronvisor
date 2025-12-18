# Second MoE Model: Replication Analysis

**Goal:** Demonstrate that P×T coupling generalizes beyond the current Mixtral implementation.

**Constraint:** Boring and surgical. High value-to-effort ratio.

---

## Option 1: Scaled Mixtral (4 layers, 16 experts)

### Architecture
- Same as current implementation, just bigger
- 4 layers instead of 2
- 16 experts instead of 8
- Same hidden_dim=256 (keep model small)
- Same top-2 routing

### Implementation Effort
**Very Low** - Just change config parameters:
```python
config = MixtralConfig(
    num_layers=4,      # was 2
    num_experts=16,    # was 8
    ...
)
```

### Scientific Value
**Medium-High**
- ✓ Tests architectural scaling (depth + breadth)
- ✓ Same routing mechanism, so clean comparison
- ✓ Tests if stable basin (η=0.015, P=0.5) transfers
- ✓ Shows P×T coupling isn't specific to small models
- ✗ Doesn't test generalization to different routing strategies

### Prediction
Stable basin should transfer with minimal tuning. May need to adjust:
- η might need to be lower (more experts = more inertia)
- Pressure scale might need tuning
- But same basic regime should work

### Validation Protocol
1. Start with η=0.015, P=0.5 (transfer hypothesis)
2. If unstable, do small sweep: η ∈ {0.01, 0.015, 0.02}, P ∈ {0.3, 0.5, 0.7}
3. Test 3 seeds on long conversations
4. Look for ≥67% robustness

**Estimated Time:** 1 day (mostly compute time)

---

## Option 2: Switch Transformer (Top-1 Routing)

### Architecture
Google's Switch Transformer uses **top-1 routing** instead of top-k:
- Each token routed to exactly 1 expert (not 2)
- Simpler routing, more expert specialization pressure
- Different load balancing dynamics

### Implementation Effort
**Low-Medium**

Changes needed:
```python
class SwitchRouter(nn.Module):
    """Top-1 routing instead of top-k."""
    def __init__(self, config):
        self.num_experts = config.num_experts
        self.top_k = 1  # KEY DIFFERENCE
        self.gate = nn.Linear(config.hidden_dim, config.num_experts, bias=False)
```

Add auxiliary load balancing loss (Switch Transformer uses this):
```python
def load_balancing_loss(router_probs, expert_indices):
    """Encourage uniform expert usage."""
    # Fraction of tokens routed to each expert
    f_i = torch.bincount(expert_indices, minlength=num_experts) / batch_size
    # Average router probability for each expert
    P_i = router_probs.mean(dim=0)
    # Loss = num_experts * sum(f_i * P_i)
    return num_experts * (f_i * P_i).sum()
```

### Scientific Value
**High**
- ✓ Different routing strategy (top-1 vs top-2)
- ✓ Tests if P×T coupling works with stronger specialization pressure
- ✓ Different load balancing dynamics
- ✓ Published architecture (reproducible baseline)
- ✓ Tests generalization across routing mechanisms

### Prediction
P×T coupling should still work, but:
- Top-1 routing has higher variance (each expert sees fewer tokens)
- May need different pressure dynamics (more aggressive balancing)
- Geological temperature might evolve differently

**Key scientific question:** Does P×T coupling stabilize top-1 routing better than standard approaches?

### Validation Protocol
1. Implement SwitchRouter with P×T fields
2. Start with η=0.015, P=0.5 (transfer test)
3. If unstable, sweep: η ∈ {0.01, 0.02, 0.03}, P ∈ {0.5, 1.0, 1.5}
4. Compare load balancing with and without P×T
5. Test 3 seeds on long conversations

**Estimated Time:** 2-3 days (implementation + validation)

---

## Option 3: Expert Choice Routing

### Architecture
**Expert Choice Routing** (Zhou et al., 2022) inverts the problem:
- Instead of tokens choosing experts, **experts choose tokens**
- Each expert selects top-k tokens to process
- Better load balancing, no auxiliary loss needed

### Implementation Effort
**Medium-High**

This is a fundamentally different routing paradigm:
```python
class ExpertChoiceRouter(nn.Module):
    def forward(self, hidden_states):
        # Compute affinity: (batch*seq, num_experts)
        affinity = self.gate(hidden_states)

        # Each expert picks top-k tokens (instead of each token picking top-k experts)
        # This requires different tensor operations and expert processing loop
        ...
```

### Scientific Value
**Very High**
- ✓ Completely different routing mechanism
- ✓ Inverted control (experts choose vs tokens choose)
- ✓ Tests if P×T coupling concept generalizes beyond token→expert routing
- ✓ Novel research direction (Expert Choice is newer, less explored)
- ✗ Higher implementation complexity
- ✗ Less clear how pressure/temperature apply in inverted paradigm

### Prediction
**Unclear.** The inversion makes it non-obvious how P×T coupling applies:
- What does "pressure on experts" mean when experts are doing the choosing?
- Temperature might apply to expert selection sharpness
- Might need conceptual rethinking of the mechanism

### Validation Protocol
Would require:
1. Conceptual design of how P×T applies to expert-choice
2. Implementation
3. Extensive validation

**Estimated Time:** 1-2 weeks (too long for "boring and surgical")

---

## Option 4: Real Pretrained Mixtral-8x7B

### Architecture
Load actual HuggingFace `mistralai/Mixtral-8x7B-v0.1`:
- 32 layers, 8 experts per layer
- 4096 hidden dim
- Pretrained on real data
- Test on actual language modeling

### Implementation Effort
**Medium**

Wrap existing model:
```python
from transformers import MixtralForCausalLM

# Load pretrained
base_model = MixtralForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

# Wrap MoE layers with Chronovisor
for layer in base_model.model.layers:
    # Inject P×T fields into layer.block_sparse_moe
    layer.block_sparse_moe = wrap_with_chronovisor(layer.block_sparse_moe)
```

### Scientific Value
**Very High**
- ✓ Real pretrained model (strongest validation)
- ✓ Actual language modeling performance
- ✓ Production-scale architecture
- ✓ Direct comparison to published baselines
- ✗ Requires significant compute (8x7B is large)
- ✗ Harder to isolate P×T effects (many confounds)

### Challenges
1. **Compute:** 8x7B requires ~45GB memory, expensive to train
2. **Baseline:** Pretrained model is already very good, small improvements harder to show
3. **Isolation:** Hard to attribute changes to P×T vs other factors
4. **Evaluation:** Need real language modeling metrics (perplexity, few-shot, etc.)

### Validation Protocol
1. Freeze base weights, only train Chronovisor controller
2. Test on held-out language modeling benchmarks
3. Compare perplexity with/without P×T coupling
4. Would need multi-GPU setup

**Estimated Time:** 1-2 weeks + significant compute budget

---

## Recommendation Matrix

| Option | Effort | Scientific Value | Time | Compute | Risk |
|--------|--------|-----------------|------|---------|------|
| **1. Scaled Mixtral** | Very Low | Medium-High | 1 day | Low | Very Low |
| **2. Switch Transformer** | Low-Medium | High | 2-3 days | Low | Low |
| **3. Expert Choice** | Medium-High | Very High | 1-2 weeks | Medium | Medium |
| **4. Pretrained Mixtral** | Medium | Very High | 1-2 weeks | High | Medium |

---

## Recommended Path: Two-Stage Approach

### Stage 1: Scaled Mixtral (Immediate)
**Do this first.** It's 1 day, low risk, and provides:
- Architectural scaling evidence
- Quick confirmation that current results aren't specific to 2-layer/8-expert
- Clean comparison (same routing mechanism)

If stable basin transfers → strong evidence of robustness.

### Stage 2: Switch Transformer (After Stage 1)
**If Stage 1 succeeds**, invest 2-3 days in Switch:
- Different routing mechanism (top-1 vs top-2)
- Published architecture (reproducible)
- Tests generalization across routing strategies
- Meaningful scientific contribution

This gives you:
1. ✓ Architectural scaling (4 layers, 16 experts)
2. ✓ Routing mechanism generalization (top-1 vs top-2)
3. ✓ Both achievable in <1 week
4. ✓ Both "boring and surgical"

---

## Future Work (Post-Publication)

**Expert Choice Routing:**
- Very interesting research direction
- Requires conceptual work on how P×T applies
- Good follow-up paper

**Pretrained Mixtral:**
- Strongest validation
- Requires more resources
- Could be extension/application paper
- Partner with lab that has compute

---

## What Goes in the Paper

**If you do Scaled Mixtral + Switch Transformer:**

**Section 5: Generalization**
```
5.1 Architectural Scaling
    - 4-layer, 16-expert Mixtral
    - Stable basin transfers: η=0.015, P=0.5
    - Results: [robustness %, Δloss, Δsep]

5.2 Routing Mechanism Generalization
    - Switch Transformer (top-1 routing)
    - P×T coupling adapts to different routing strategy
    - Results: [comparison to top-2]
```

This demonstrates:
1. Robustness to architecture size
2. Generalization across routing mechanisms
3. P×T coupling is not specific to one configuration

**Strong publication story without overcommitting.**

---

## Decision Point

**Start with Option 1 (Scaled Mixtral).** It's:
- 1 day of work
- Low risk
- High value (shows scaling)
- Prerequisite for deciding on next step

**After scaling results:**
- If stable basin transfers cleanly → proceed to Switch Transformer
- If requires tuning but finds basin → document tuning, then decide on Switch
- If fails completely → reassess (unlikely given current robustness)

**Don't overthink.** Start the scaling experiment now.
