# ChronoMoE Training Infrastructure

This document describes the training infrastructure designed for ChronoMoE models, which leverages the P×T geometric control layer for curriculum learning and adaptive optimization.

## Design Principles

Traditional MoE training treats expert routing as a discrete optimization problem. ChronoMoE training instead treats it as **geological formation** - we let the landscape of expert specializations emerge naturally through the interplay of pressure, temperature, and structural memory.

### Key Innovations

1. **Coherence-Gated Training**: Gradient updates are modulated by Kuramoto coherence R
2. **Geological Curriculum**: Structural temperature T̄ landscape forms over training
3. **Meta-Knob Scheduling**: κ anneals from exploration to exploitation
4. **Valley-Aware Loss**: Penalizes pathological expert entrenchment
5. **Geological Checkpointing**: Preserves T̄ landscape across training runs

## Components

### ChronoMoELoss

Combined loss function with four components:

```python
L = L_lm + λ_balance * L_balance + λ_coherence * L_coherence + λ_valley * L_valley
```

| Component | Purpose | Default Weight |
|-----------|---------|----------------|
| `L_lm` | Cross-entropy language modeling | 1.0 |
| `L_balance` | Load balancing (even expert usage) | 0.01 |
| `L_coherence` | Reward high Kuramoto R | 0.001 |
| `L_valley` | Penalize unhealthy valleys | 0.0001 |

**Key insight**: The coherence loss encourages the expert ensemble to synchronize, while the valley loss ensures that entrenchment happens for the *right* experts.

### KnobScheduler

Schedules the meta-knob κ ∈ [-1, +1] throughout training:

```
κ > 0  →  exploration (high pressure, high temperature)
κ < 0  →  exploitation (low pressure, low temperature)
κ = 0  →  baseline behavior
```

**Schedule Options:**

| Schedule | Behavior |
|----------|----------|
| `linear` | Linear interpolation from initial to final κ |
| `cosine` | Smoother cosine annealing |
| `step` | Discrete steps at 1/3 and 2/3 progress |
| `adaptive` | Rule-based controller reacts to training dynamics |

**Curriculum Philosophy:**
- **Early training**: High κ explores expert space, discovers specializations
- **Mid training**: κ anneals, landscape starts forming valleys/ridges
- **Late training**: Low κ exploits discovered structure

### CoherenceGatedOptimizer

Wraps any optimizer to gate gradients based on Kuramoto coherence:

```python
if R < min_coherence:
    gradient_scale = 0  # Fully gated
else:
    gradient_scale = (R - min_coherence) / (1 - min_coherence)
```

**Rationale**: When experts are incoherent (low R), the routing decisions are essentially random. Gradient updates based on random routing may harm training. By gating gradients when coherence is low, we only update when the ensemble has reached consensus.

### ChronovisorCheckpoint

Preserves the full geological state:

- `structural_T_global`: Global structural temperature (slowest timescale)
- `lens_states`: Per-layer lens positions and temperatures
- `coherence_history`: Historical Kuramoto R values
- `meta_knob_history`: Historical κ decisions

**Why this matters**: The structural temperature landscape takes thousands of steps to form. Without geological checkpointing, resuming training would start with a flat landscape, losing all the learned expert structure.

## Training Configuration

```python
TrainingConfig(
    # Basic training
    learning_rate=1e-4,
    weight_decay=0.01,
    max_epochs=10,
    batch_size=8,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,

    # Loss weights
    lambda_balance=0.01,
    lambda_coherence=0.001,
    lambda_valley=0.0001,

    # Meta-knob scheduling
    initial_kappa=0.5,      # Start exploratory
    final_kappa=-0.3,       # End exploitative
    kappa_warmup_steps=1000,
    kappa_schedule="cosine",

    # Coherence gating
    use_coherence_gating=True,
    min_coherence=0.2,
    gating_strength=0.5,

    # Checkpointing
    checkpoint_dir="checkpoints",
    save_every_n_steps=1000,
    save_chronovisor_state=True,
)
```

## Usage Example

```python
from chronomoe.training import (
    ChronoMoETrainer,
    TrainingConfig,
)
from chronomoe.chronovisor_mixtral_bridge import ChronovisorMixtralForCausalLM
from chronomoe.mixtral_core import MixtralConfig

# Create model
model_config = MixtralConfig(
    vocab_size=32000,
    hidden_dim=512,
    num_layers=8,
    num_experts=8,
    enable_chronovisor=True,
)
model = ChronovisorMixtralForCausalLM(model_config)

# Training config
train_config = TrainingConfig(
    learning_rate=1e-4,
    max_steps=100000,
    initial_kappa=0.5,
    final_kappa=-0.3,
    kappa_schedule="cosine",
    use_coherence_gating=True,
)

# Create trainer
trainer = ChronoMoETrainer(
    model=model,
    config=train_config,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
)

# Train
results = trainer.train()

# Resume from checkpoint
trainer.load_checkpoint("step_50000")
trainer.train()  # Continues with preserved geology
```

## Monitoring Training

The trainer logs key metrics:

```
Step 1000 | Loss: 4.2345 | κ: 0.48 (warmup_explore) | R: 0.32 | Gate: 0.67
Step 2000 | Loss: 3.8901 | κ: 0.42 (cosine_anneal) | R: 0.45 | Gate: 0.83
...
  Landscape: var=0.0234, valleys=[2, 5], ridges=[1, 7]
```

**Key metrics to watch:**

| Metric | Healthy Range | Meaning |
|--------|---------------|---------|
| Kuramoto R | 0.3 - 0.8 | Expert synchronization |
| T̄ variance | 0.01 - 0.1 | Landscape differentiation |
| Gate value | 0.5 - 1.0 | Gradient flow allowed |
| Valleys | 1-3 experts | Stable specializations |
| Ridges | 1-3 experts | Exploration regions |

## The Training Dynamics

### Phase 1: Exploration (steps 0 - warmup)
- κ is high (0.5), pressure is strong
- Temperature is elevated, routing is diffuse
- All experts receive roughly equal traffic
- T̄ landscape is flat (no valleys/ridges)
- Kuramoto R is low (experts desynchronized)

### Phase 2: Landscape Formation (warmup - 70%)
- κ anneals downward
- Experts that perform well on certain domains see lower T_fast
- T̄ begins to differentiate via EMA
- First valleys appear (reliable specialists)
- First ridges appear (generalists/explorers)
- Kuramoto R increases as patterns stabilize

### Phase 3: Exploitation (70% - end)
- κ is low (-0.3), pressure is gentle
- Temperature landscape is well-formed
- Valleys have low T̄ (sharp routing to specialists)
- Ridges have high T̄ (diffuse routing for exploration)
- Kuramoto R is high (coherent ensemble)
- Valley health should be good (healthy valleys)

## Troubleshooting

### Loss not decreasing
- Check if coherence gating is too aggressive (`min_coherence` too high)
- Try `use_coherence_gating=False` to diagnose
- Increase `lambda_balance` if experts are very imbalanced

### No landscape formation
- Training may be too short for T̄ to evolve
- Check `eta_structural_T_global` and `eta_structural_T_local` in controller
- Consider lower `final_kappa` for more exploitation

### Unhealthy valleys detected
- The reliability → T_fast → T̄ feedback may be slow
- Check valley health diagnostics: `controller.get_valley_health_diagnostics()`
- Self-correction should eventually fill in bad valleys

### Kuramoto R stuck low
- Experts may be fighting over the same tokens
- Check load balancing loss
- Try higher `initial_kappa` for more exploration early

## Files

- `src/chronomoe/training.py` - Training infrastructure implementation
- `src/chronomoe/chronovisor_mixtral_bridge.py` - Model with P×T geometry
- `src/chronomoe/knob.py` - Meta-knob and rule-based controller
