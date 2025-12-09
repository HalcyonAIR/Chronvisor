# Claude Code Instructions: ChronoMoE Development

This document provides context for continuing ChronoMoE development in a local Claude Code session.

---

## Project Overview

**ChronoMoE** is a pressure-driven MoE (Mixture of Experts) routing framework that adds Chronovisor-style dynamics to standard MoE routing. Instead of static learned gates, routing decisions are influenced by:

1. **Pressure (b_k)** - Directional force from trust, lens coherence, cultural memory
2. **Fast Temperature (T_fast)** - Instantaneous terrain permeability based on uncertainty
3. **Structural Temperature (T̄_k)** - Slow geological memory via EMA

The routing equation:
```
p_k = softmax((logits_k + b_k) / T_effective_k)
where T_effective_k = T_fast_k × T̄_k
```

---

## Key Concepts

### The Physical Metaphor

| Component | Metaphor | Effect |
|-----------|----------|--------|
| Logits | Marble position | Where routing starts |
| Pressure b_k | Wind | Pushes toward/away from experts |
| T_fast | Current terrain softness | Exploration vs exploitation |
| T̄_k (structural) | Geology (erosion) | Long-term terrain memory |

### The Seven Guarantees

The architecture provides formal stability guarantees (see `docs/005-seven-guarantees.md`):

1. **Direction** - Temperature modulates but never reverses meaning
2. **Geological Safety** - Bad valleys cannot form
3. **Anti-Collapse** - Pressure prevents permanent over-specialization
4. **Exploration-Exploitation** - Uncertainty triggers exploration
5. **Topological Diversity** - Multiple stable niches emerge
6. **Reversibility** - Wrong terrain self-corrects
7. **Meaning-Preserving** - Topology cannot override semantics

---

## Codebase Structure

```
src/
├── chronovisor/           # Core Chronovisor dynamics
│   └── simulation_v6.py   # Controller, experts, lens, cultural evolution
│
└── chronomoe/             # MoE integration layer
    ├── bridge.py          # ChronoMoEBridge - main integration point
    ├── router.py          # Router with pressure injection + temperature warping
    ├── knob.py            # Meta-knob (κ ∈ [-1, +1]) for LLM control
    ├── alignment.py       # V7 structural alignment
    ├── moe.py             # Simulated MoE layer
    └── experiment_*.py    # Various experiments

docs/
├── 001-first-results.md
├── 002-chronomoe-temperature-geometry.md
├── 003-ontology-robustness.md
├── 004-valley-self-correction.md
└── 005-seven-guarantees.md

tests/
├── test_temperature.py    # 40 tests for temperature geometry
└── test_*.py              # Other test files (318 total tests)
```

---

## Key Classes and Methods

### ChronoMoEBridge (`src/chronomoe/bridge.py`)

The main integration point between MoE and Chronovisor:

```python
# Create bridge
bridge = ChronoMoEBridge.create(
    n_experts=8,
    eta_structural_T=0.01,  # Geological timescale
)

# Get pressure field (wind direction)
pressure = bridge.get_pressure_bias()
# pressure.combined → np.ndarray of shape (n_experts,)

# Get temperature field (terrain permeability)
temp_field = bridge.get_temperature_field()
# temp_field.effective_temperatures → np.ndarray of shape (n_experts,)
# temp_field.fast_temperatures → instantaneous
# temp_field.structural_temperatures → geological

# Feed routing stats back to Chronovisor
bridge.feed_routing_stats(stats, num_chronovisor_ticks=20)

# Diagnostics
bridge.get_valley_health_diagnostics()  # Monitor self-correction
bridge.get_structural_temperature_diagnostics()  # Landscape formation
bridge.get_diagnostics()  # General state
```

### Router (`src/chronomoe/router.py`)

```python
router = Router(input_dim=64, n_experts=8)

# Inject pressure
router.inject_pressure(pressure.combined)

# Route with temperature warping
gate_weights = router.forward_with_temperature(
    x,  # Input tensor
    temp_field.effective_temperatures,
    pressure_scale=factors.pressure_scale,
    temp_scale=factors.temp_scale,
)
```

### MetaKnob (`src/chronomoe/knob.py`)

```python
knob = MetaKnob()

# Set κ ∈ [-1, +1]
# Negative = exploit (sharper routing, lower temperature)
# Positive = explore (softer routing, higher temperature)
factors = knob.set_kappa(0.3)

# factors.pressure_scale → modulates pressure strength
# factors.temp_scale → modulates temperature field
```

---

## Running Tests

```bash
# All tests (318)
PYTHONPATH=src python -m pytest tests/ -v

# Temperature tests only (40)
PYTHONPATH=src python -m pytest tests/test_temperature.py -v

# Specific test class
PYTHONPATH=src python -m pytest tests/test_temperature.py::TestValleyHealthDiagnostics -v
```

---

## Running Experiments

```bash
# Landscape formation (structural temperature evolution)
PYTHONPATH=src python src/chronomoe/experiment_structural.py

# Temperature field comparison
PYTHONPATH=src python src/chronomoe/experiment_temperature.py

# Meta-knob strategies
PYTHONPATH=src python src/chronomoe/experiment_knob.py
```

---

## Next Step: Mixtral Integration

The next major task is integrating ChronoMoE with a real Mixtral model.

### Integration Points

1. **Hook the router logits** - Before Mixtral's softmax, add pressure:
   ```python
   logits = mixtral_router(hidden_states)
   logits = logits + bridge.get_pressure_bias().combined
   ```

2. **Apply temperature warping** - Divide by effective temperature:
   ```python
   temp_field = bridge.get_temperature_field()
   logits = logits / temp_field.effective_temperatures
   ```

3. **Feed stats back** - After each forward pass:
   ```python
   stats = RoutingStats(
       expert_usage=expert_counts,
       mean_gate_weights=gate_weights.mean(dim=0),
       batch_loss=loss.item(),
   )
   bridge.feed_routing_stats(stats)
   ```

### Mixtral Architecture Notes

Mixtral 8x7B uses:
- 8 experts per layer
- Top-2 routing (sparse)
- Router is a learned linear layer

For integration:
- Create one ChronoMoEBridge per layer, OR
- Share a single bridge across layers (simpler, may be sufficient)

### Expected Effects

Short-term (Day 1):
- More stable routing signatures
- Better expert utilization distribution

Mid-term (millions of tokens):
- Semantic rather than lexical expert clustering
- Emergent expert ecology (valleys/ridges)

Long-term (100M+ tokens):
- Crystallized expert identities
- Semantic inertia across turns

---

## Design Decisions Already Made

### Why No Asymmetric Erosion?

The reliability → T_fast → T̄ feedback loop handles bad valleys naturally:
- Bad expert → low reliability → high T_fast → T̄ rises → valley fills
- See `docs/004-valley-self-correction.md`

### Why Multiplicative Temperature?

`T_effective = T_fast × T̄` ensures:
- Fast temperature can override structural valleys during uncertainty
- Structural temperature provides memory without trapping

### Why Symmetric η_T?

The EMA rate is the same for rising and falling structural temperature.
This is sufficient because bad experts naturally run hot (high T_fast).

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eta_structural_T` | 0.01 | Geological timescale (τ ≈ 100 steps) |
| `T_min` | 0.3 | Minimum effective temperature |
| `T_max` | 3.0 | Maximum effective temperature |
| `beta_R` | 0.5 | Coherence → temperature sensitivity |
| `beta_drift` | 0.3 | Drift → temperature sensitivity |
| `beta_reliability` | 0.2 | Reliability → temperature sensitivity |
| `alpha_T` | 0.3 | Trust weight in pressure |
| `alpha_P` | 0.2 | Lens pressure weight |
| `alpha_C` | 0.1 | Cultural weight in pressure |

---

## Useful Commands

```bash
# Check current branch
git branch

# Run specific experiment
PYTHONPATH=src python src/chronomoe/experiment_structural.py

# Quick test run
PYTHONPATH=src python -m pytest tests/test_temperature.py -x -v

# See landscape formation
PYTHONPATH=src python -c "
from chronomoe import ChronoMoEBridge
bridge = ChronoMoEBridge.create(n_experts=8, eta_structural_T=0.05)
for _ in range(200):
    bridge.get_temperature_field()
print(bridge.get_structural_temperature_diagnostics())
"
```

---

## Contact

This project is part of HalcyonAIR research. The theoretical framework was developed collaboratively between Halcyon AI and Claude.

---

*Last updated after implementing: Pressure × Temperature geometry, structural temperature (geology), valley health diagnostics, and the Seven Guarantees.*
