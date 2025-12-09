# Chronovisor + Mixtral MoE Integration

**Status:** Prototype ready for development
**Date:** 2025-12-10

---

## Overview

This document describes the integration of Chronovisor's geometric control layer with a from-scratch implementation of Mixtral's Mixture-of-Experts architecture.

**What we built:**
1. Full Mixtral MoE architecture (attention, routing, experts)
2. Chronovisor geometric control layer integration
3. Closed-loop system where routing behavior informs lens updates

**What remains:**
1. Embeddings and language model head (currently operates on hidden states)
2. Training infrastructure and tokenization
3. Real-world validation and tuning
4. Performance optimization

---

## Architecture Components

### 1. Mixtral Core (`mixtral_core.py`)

A complete implementation of Mixtral's architecture built from scratch:

#### Attention Mechanism
- **Grouped-Query Attention (GQA)**: 32 query heads share 8 key-value heads
- **Rotary Position Embeddings (RoPE)**: Encodes position via rotation instead of addition
- **Multi-head attention** with configurable head dimensions

```python
config = MixtralConfig(
    num_attention_heads=32,
    num_key_value_heads=8,  # 4:1 ratio reduces memory
    head_dim=128,
)
```

#### Sparse Mixture-of-Experts
- **8 experts per layer** (configurable)
- **Top-2 routing**: Each token routed to 2 experts
- **SwiGLU activation**: Gated feedforward network in each expert

```python
# Expert is a SwiGLU FFN
expert_output = (Swish(x @ W1) ⊙ (x @ V)) @ W2
```

#### Router
- Learned gating network: `hidden_dim → num_experts` logits
- Top-k selection with softmax normalization
- **Chronovisor pressure injection point**: Add bias to logits before softmax

```python
router_logits = gate(hidden_states)
router_logits += pressure_bias  # ← Chronovisor control
weights, experts = top_k(router_logits, k=2)
```

#### Decoder Layer
Standard transformer decoder structure:
1. RMSNorm
2. Grouped-Query Attention
3. Residual connection
4. RMSNorm
5. Sparse MoE
6. Residual connection

### 2. Chronovisor Integration (`chronovisor_mixtral_bridge.py`)

The bridge connects Mixtral's routing behavior to Chronovisor's geometric layer.

#### Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│  Mixtral Forward Pass                                   │
│  ────────────────────                                   │
│  Input → Attention → MoE Routing → Expert Selection     │
│                         ↓                                │
│                   Routing Stats                          │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Chronovisor Controller                                 │
│  ──────────────────────                                 │
│  1. Collect routing statistics (expert usage)           │
│  2. Compute coherence (routing entropy)                 │
│  3. Compute Δcoherence (change since last tick)         │
│  4. Update lens geometry (on macro clock)               │
│  5. Compute pressure bias (encourage/discourage experts)│
└─────────────────────────┬───────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Next Forward Pass                                      │
│  ────────────────                                       │
│  Router logits += pressure_bias * scale                 │
│  → Routing behavior adapts over time                    │
└─────────────────────────────────────────────────────────┘
```

#### Key Components

**MixtralLens** (per layer)
- `vector`: Position in expert geometry space `(n_experts,)`
- `pressure`: Bias to apply to routing logits
- `gamma`: Adaptation rate

Pressure computation:
```python
actual_dist = expert_usage / total_usage
ideal_dist = uniform(n_experts)
pressure = (ideal_dist - actual_dist) * 2.0  # Encourage underutilized
```

**ChronovisorMixtralController**
- Manages multi-scale clocks (fast, micro, macro)
- One lens per layer
- Tracks coherence over time
- Updates pressure on micro boundaries (every 5 ticks)
- Updates lens position on macro boundaries (every 20 ticks)

**Coherence Metric**
```python
# Coherence = 1 - normalized_routing_entropy
# High coherence = experts settled into stable patterns
# Low coherence = routing is chaotic/uncertain

entropy = -Σ p_i log(p_i)  # for expert distribution
coherence = 1 - (entropy / log(n_experts))
```

**Δcoherence** (delta coherence)
- Change in coherence since last measurement
- Positive Δ = system stabilizing → open gate (allow lens updates)
- Negative Δ = system destabilizing → close gate (dampen lens)

```python
coherence_gate = sigmoid(5 * delta_coherence)
lens.update(drift, gate=coherence_gate)
```

#### Integration Points

1. **Router Pressure Injection**
   ```python
   # In MixtralRouter.forward():
   router_logits = self.gate(hidden_states)
   if pressure_bias is not None:
       router_logits += pressure_bias
   ```

2. **Statistics Collection**
   ```python
   # After each forward pass:
   routing_stats = {
       'layer_idx': layer_idx,
       'selected_experts': top_k_indices,
       'routing_weights': normalized_weights,
   }
   ```

3. **Controller Update**
   ```python
   # After model forward:
   chronovisor_state = controller.tick(all_routing_stats)
   # chronovisor_state contains:
   #   - coherence, delta_coherence
   #   - lens positions and pressures
   #   - expert usage distributions
   ```

---

## What We've Validated

✓ **Mixtral architecture works**
- Attention mechanism (GQA + RoPE)
- MoE routing (top-k sparse)
- Forward passes produce correct tensor shapes

✓ **Chronovisor integration works**
- Controller ticks successfully
- Coherence computation from routing stats
- Pressure updates propagate to router
- Δcoherence tracked over time

✓ **Closed loop operates**
- Routing → statistics → coherence → lens → pressure → routing
- System evolves over multiple forward passes

---

## What's Missing for a Complete Prototype

### 1. Embeddings and Language Modeling

Currently the system operates on raw hidden states. To make it a real language model:

**Add:**
- Token embeddings: `vocab_size → hidden_dim`
- Position embeddings (or rely on RoPE alone)
- Language model head: `hidden_dim → vocab_size`
- Output projection and softmax

**File:** `src/chronomoe/mixtral_lm.py`
```python
class MixtralLanguageModel(nn.Module):
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.model = ChronovisorMixtralModel(config)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        hidden = self.embed_tokens(input_ids)
        hidden, chrono_state = self.model(hidden)
        logits = self.lm_head(hidden)
        return logits, chrono_state
```

### 2. Training Infrastructure

**Requirements:**
- Tokenizer (can use HuggingFace's Mixtral tokenizer)
- Dataset loading and batching
- Loss computation (cross-entropy)
- Optimizer (AdamW with warmup)
- Gradient accumulation (model is large)
- Checkpointing

**Chronovisor-specific:**
- Decide when to enable Chronovisor (after pretraining? from start?)
- Log coherence metrics during training
- Visualize lens trajectories
- Monitor expert specialization

**File:** `src/chronomoe/training.py`

### 3. Evaluation and Tuning

**Experiments to run:**

1. **Baseline comparison**
   - Mixtral without Chronovisor (pressure_scale=0)
   - Mixtral with Chronovisor (pressure_scale=0.1)
   - Compare: perplexity, expert usage balance, training stability

2. **Pressure scaling**
   - Test different `pressure_scale` values: 0.01, 0.05, 0.1, 0.5
   - Find sweet spot where specialization improves without degrading performance

3. **Clock periods**
   - Test different `micro_period` and `macro_period`
   - Faster updates vs. more stable lens

4. **Coherence analysis**
   - Track coherence during training
   - Correlate with validation loss
   - Look for phase transitions (settling vs. exploration)

5. **Expert specialization**
   - Measure expert-cluster alignment
   - Check if experts develop clear specializations
   - Visualize expert usage heatmaps

### 4. Efficiency and Scaling

**Current implementation is unoptimized:**

Memory optimizations:
- Gradient checkpointing for deep models
- 8-bit or 4-bit quantization
- Efficient attention (Flash Attention 2)

Compute optimizations:
- Compile with `torch.compile()`
- Expert batching (batch tokens routed to same expert)
- Kernel fusion

Distributed training:
- Model parallelism (experts on different GPUs)
- Data parallelism
- FSDP (Fully Sharded Data Parallel)

### 5. Integration with Existing Chronovisor Experiments

**Connect to existing code:**

The original Chronovisor has several experiment frameworks:
- `experiment.py`: Baseline pressure experiments
- `experiment_v7.py`: Structural alignment tracking
- `experiment_knob.py`: Meta-knob for adaptive control
- `experiment_temperature.py`: Temperature field modulation

**Adaptation needed:**
- These use simulated MoE (numpy arrays)
- Mixtral uses real PyTorch MoE
- Need to bridge the two worlds

**Option 1:** Adapt existing experiments to use Mixtral
**Option 2:** Create parallel Mixtral-specific experiments

### 6. Documentation and Visualization

**Add:**
- Training tutorial notebook
- Lens trajectory visualization
- Expert usage heatmaps over time
- Coherence plots (coherence vs. training step)
- Comparison dashboards (baseline vs. Chronovisor)

**Tools:**
- Weights & Biases for experiment tracking
- matplotlib/plotly for visualizations
- TensorBoard for real-time monitoring

---

## File Structure

```
Chronvisor/
├── src/
│   ├── chronovisor/          # Original geometric control layer
│   │   ├── controller.py
│   │   ├── lens.py
│   │   ├── expert_harness.py
│   │   └── simulation*.py
│   │
│   └── chronomoe/            # MoE integration
│       ├── moe.py            # Simulated MoE (numpy)
│       ├── router.py
│       ├── bridge.py
│       ├── alignment.py
│       ├── knob.py
│       ├── experiment*.py
│       │
│       ├── mixtral_core.py         # ← NEW: Mixtral architecture
│       ├── chronovisor_mixtral_bridge.py  # ← NEW: Integration
│       ├── mixtral_lm.py           # TODO: Language model wrapper
│       └── training.py             # TODO: Training infrastructure
│
├── tests/
│   ├── test_chronomoe.py
│   ├── test_chronovisor.py
│   └── test_mixtral.py       # TODO: Mixtral-specific tests
│
├── docs/
│   ├── 001-first-results.md
│   └── 002-mixtral-integration.md  # ← This document
│
└── experiments/              # TODO: Training runs
    ├── baseline/
    ├── chronovisor/
    └── analysis/
```

---

## Quick Start Guide

### Install Dependencies

```bash
cd Chronvisor
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch transformers accelerate
pip install -e .
```

### Run Tests

```bash
# Original Chronovisor tests
pytest tests/test_chronovisor.py -v

# ChronoMoE tests (simulated)
pytest tests/test_chronomoe.py -v

# Test Mixtral core
python src/chronomoe/mixtral_core.py

# Test integration
python src/chronomoe/chronovisor_mixtral_bridge.py
```

### Example Usage

```python
from chronomoe.mixtral_core import MixtralConfig
from chronomoe.chronovisor_mixtral_bridge import ChronovisorMixtralModel

# Create config (small for testing)
config = MixtralConfig(
    vocab_size=1000,
    hidden_dim=256,
    num_layers=4,
    num_experts=4,
    enable_chronovisor=True,
)

# Create model
model = ChronovisorMixtralModel(config)

# Forward pass
import torch
batch_size, seq_len = 2, 16
hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)

output, chrono_state = model(hidden_states)

print(f"Coherence: {chrono_state.coherence:.4f}")
print(f"Δ Coherence: {chrono_state.delta_coherence:.4f}")
print(f"Clock: fast={chrono_state.fast_clock}, micro={chrono_state.micro_clock}")
```

---

## Key Design Decisions

### 1. Internal Implementation vs. Wrapper

**Decision:** Build Mixtral from scratch rather than wrapping HuggingFace model

**Rationale:**
- Need direct access to router logits for pressure injection
- Want full control over expert selection logic
- Easier to integrate with Chronovisor's architecture
- Can optimize for research needs (observability, modularity)

**Tradeoff:**
- Can't use pretrained Mixtral weights (yet)
- Need to train from scratch or implement weight loading

### 2. Per-Layer Lenses vs. Global Lens

**Decision:** One lens per MoE layer (32 lenses for Mixtral-8x7B)

**Rationale:**
- Different layers may specialize differently
- Allows local adaptation
- Matches original Chronovisor design (layer-specific control)

**Alternative considered:**
- Single global lens affecting all layers uniformly
- Would be simpler but less expressive

### 3. Pressure Scale

**Decision:** Configurable `pressure_scale` parameter (default 0.1)

**Rationale:**
- Pressure is in [-1, 1], router logits are typically larger
- Small pressure (0.1) gently biases routing without forcing
- Can tune experimentally

**To explore:**
- Adaptive pressure scaling (based on training phase)
- Per-layer pressure scales
- Annealing schedule (high pressure early, decay over time)

### 4. Coherence Metric

**Decision:** Use entropy-based coherence

**Rationale:**
- Simple to compute from routing statistics
- Directly measures routing concentration
- Bounded in [0, 1]

**Alternatives to consider:**
- Variance-based coherence (from original simulations)
- Cross-layer alignment (do layers agree on routing patterns?)
- Temporal stability (is routing consistent across tokens?)

### 5. Clock Periods

**Decision:** micro=5, macro=20 (inherited from original Chronovisor)

**Rationale:**
- Tested in toy simulations
- Micro for frequent pressure updates
- Macro for slower lens movement

**To explore:**
- Adaptive clock periods based on coherence
- Different periods for different layers
- Relationship to batch size and sequence length

---

## Next Steps (Priority Order)

1. **Add language modeling head** (`mixtral_lm.py`)
   - Embeddings and output projection
   - Test with actual tokens

2. **Create minimal training loop**
   - Use small dataset (WikiText-2)
   - Train for 1 epoch to verify everything works
   - Compare baseline vs. Chronovisor

3. **Logging and visualization**
   - Track coherence during training
   - Plot expert usage distributions
   - Visualize lens trajectories

4. **Hyperparameter search**
   - Pressure scale: [0.01, 0.05, 0.1, 0.5]
   - Clock periods: [(5,20), (10,40), (3,10)]
   - Gamma (lens adaptation rate)

5. **Optimization**
   - Gradient checkpointing
   - Flash Attention
   - Mixed precision training

6. **Scaling experiments**
   - Increase model size gradually
   - Test on larger datasets
   - Benchmark vs. standard Mixtral

7. **Write paper** 📝
   - Document findings
   - Compare to related work (MoE gating, adapter methods)
   - Release code and checkpoints

---

## Questions for Halcyon AI

1. **Architectural validation**
   - Is the Mixtral implementation correct? (GQA, RoPE, SwiGLU)
   - Are we missing any critical components?

2. **Chronovisor integration**
   - Does the pressure injection make sense?
   - Should pressure affect routing differently? (e.g., temperature modulation)
   - Alternative coherence metrics?

3. **Training strategy**
   - Enable Chronovisor from the start or after pretraining?
   - Should lens state be saved/loaded with checkpoints?
   - How to handle Chronovisor during evaluation?

4. **Expected behavior**
   - What does "good" coherence look like during training?
   - Should experts specialize strongly or maintain diversity?
   - How quickly should the lens move?

5. **Connections to other work**
   - How does this relate to DRAI, AOSL, Memory Tender?
   - Can lens state inform these other systems?
   - Cross-pollination opportunities?

---

## Resources

**Code:**
- Mixtral core: `src/chronomoe/mixtral_core.py`
- Integration bridge: `src/chronomoe/chronovisor_mixtral_bridge.py`
- Original MoE experiments: `src/chronomoe/experiment*.py`

**Papers:**
- Mixtral of Experts (Mistral AI): https://arxiv.org/abs/2401.04088
- Switch Transformers (Google): https://arxiv.org/abs/2101.03961
- GQA (Grouped-Query Attention): https://arxiv.org/abs/2305.13245
- RoFormer (RoPE): https://arxiv.org/abs/2104.09864

**Implementations:**
- HuggingFace Mixtral: https://github.com/huggingface/transformers/tree/main/src/transformers/models/mixtral
- Official Mistral code: https://github.com/mistralai/mistral-src

---

*This document will be updated as the prototype evolves.*
