# Plan: Switch Transformer Integration for Path Wear Testing

**Status**: Ready to implement
**Model**: google/switch-base-8
**Rationale**: Small (1-2B params), pretrained, clean router access

## Why Switch-base-8 is Ideal

✓ **Size**: ~1-2B parameters (loads on reasonable hardware)
✓ **Pretrained**: Learned routing patterns (not random)
✓ **Router access**: Explicit `router_logits` in forward pass
✓ **Clean architecture**: Research-grade, well-documented
✓ **Observable**: Perfect for our routing metrics

**Not testing language quality - testing routing topology deformation.**

## What We Keep (Unchanged)

- ✓ Same telemetry system
- ✓ Same routing metrics (ΔKL, ΔCos, ΔJaccard, entropy)
- ✓ Same experimental protocol (A×N → B, reset control)
- ✓ Same falsification tests

**We're not changing the experiment. We're running it in the proper regime.**

## Implementation Steps

### Step 1: Load Switch Model & Verify Structure

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/switch-base-8",
    device_map="cpu",  # or "cuda" if available
)
tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")

# Test with real text
text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(text, return_tensors="pt")

# Extract routing
outputs = model(**inputs, output_router_logits=True)
router_logits = outputs.router_logits  # List per layer

# Verify structure exists
routing_probs = torch.softmax(router_logits[0], dim=-1)
entropy = compute_entropy(routing_probs)

# Confirm: entropy << max (should be ~1-2, not 4+)
```

**Expected**: Entropy 1-3 (learned patterns), not 4+ (random)

### Step 2: Baseline Control (No Intervention)

Run A×N → B **without any modification**:

```python
# Measure B virgin
routing_B_virgin = extract_routing(model, text_B)

# Run A repeatedly (no state modification)
for _ in range(100):
    _ = model(tokenizer(text_A, return_tensors="pt"))

# Measure B after
routing_B_after = extract_routing(model, text_B)

# Should see: no movement (pretrained model is stateless)
assert routing_B_virgin ≈ routing_B_after
```

### Step 3: Integrate ChronoMoE Layer

**Key decision**: How to inject T̄ into pretrained Switch?

**Option A**: Wrap router with ChronoMoE adapter (non-invasive)

```python
class ChronoMoEAdapter:
    """Wraps pretrained router, injects T̄ bias."""

    def __init__(self, original_router, num_experts):
        self.router = original_router
        self.T_bar = np.ones(num_experts)
        self.eta = 0.015

    def forward(self, hidden_states):
        # Get original logits
        logits = self.router(hidden_states)

        # Inject T̄ bias (pressure)
        pressure = compute_pressure(self.T_bar)
        logits_biased = logits + pressure

        # Update T̄ based on routing
        routing = torch.softmax(logits_biased, dim=-1)
        self.update_T_bar(routing)

        return logits_biased
```

**Option B**: Train adapter layer (more complex, skip for v0)

### Step 4: Test Path Wear with ChronoMoE

```python
# Enable ChronoMoE adapter
model_chrono = wrap_with_chronovisor(model)

# Measure B virgin
routing_B_virgin = extract_routing(model_chrono, text_B)

# Wear path A
for _ in range(100):
    _ = model_chrono(text_A, update_chronovisor=True)

# Measure B after
routing_B_after = extract_routing(model_chrono, text_B)

# Compute differential
metrics = compute_routing_metrics_suite(
    routing_B_virgin, routing_B_after, routing_A
)

# Look for: delta_kl, delta_cos, delta_jaccard > 0
```

### Step 5: Compare Vanilla vs ChronoMoE

```
                    ΔKL      ΔCos     ΔJac
Vanilla (Step 2):   0.000    0.000    0.000
ChronoMoE (Step 4): ???      ???      ???

If differential > 0 → Path wear detected
If differential = 0 → Boundary extends to trained models
```

## Minimal Coherent Inputs

Don't overthink - we want semantic stability, not complexity:

```python
text_A = "The cat sat on the mat. It was a sunny day."
text_B = "The dog lay on the rug. It was a cloudy day."

# Similar structure, different content
# Should activate similar (but not identical) routing patterns
```

**Not testing generalization - testing routing drift.**

## Expected Outcomes

### Outcome 1: Structure Exists, No Wear
- Entropy: 1-2 (learned patterns) ✓
- Vanilla differential: 0.000 ✓
- ChronoMoE differential: 0.000
- **Conclusion**: Boundary extends to trained models

### Outcome 2: Structure Exists, Weak Wear
- Entropy: 1-2 ✓
- Vanilla differential: 0.000 ✓
- ChronoMoE differential: 0.01-0.05
- **Conclusion**: Signal detected, amplification experiments warranted

### Outcome 3: Structure Exists, Strong Wear
- Entropy: 1-2 ✓
- Vanilla differential: 0.000 ✓
- ChronoMoE differential: 0.1+
- **Conclusion**: Path wear confirmed, map full phase diagram

## What We Learn (Regardless of Outcome)

**If no wear**: "Path wear requires stronger mechanism than T̄ adaptation at η=0.015"

**If weak wear**: "Path wear exists in learned routing, requires amplification"

**If strong wear**: "Path wear confirmed, proceed to Mixtral/DeepSeek validation"

**All outcomes are publishable results.**

## Technical Notes

### Router Extraction for Switch

```python
def extract_switch_routing(model, inputs, layer_idx=0):
    """Extract routing from Switch Transformer."""
    outputs = model(**inputs, output_router_logits=True)

    # router_logits is list of [batch, seq, num_experts]
    logits = outputs.router_logits[layer_idx]

    # Softmax + average over sequence
    probs = torch.softmax(logits, dim=-1)
    routing = probs.mean(dim=1).squeeze().detach().cpu().numpy()

    return routing
```

### Memory Considerations

Switch-base-8 in FP32: ~3GB
Switch-base-8 in FP16: ~1.5GB
CPU inference: Slow but viable

If memory constrained: Use FP16, CPU only, small batch

## Next Steps After This

1. **If wear detected**: Validate on Mixtral/DeepSeek
2. **If no wear**: Sweep η (0.015 → 0.1), sweep n_repetitions
3. **If still no wear**: Document boundary, propose alternative mechanisms

## Timeline Estimate

- Step 1 (verify structure): 30 min
- Step 2 (baseline control): 30 min
- Step 3 (ChronoMoE adapter): 2-3 hours
- Step 4 (path wear test): 30 min
- Step 5 (analysis): 30 min

**Total: ~5 hours for complete protocol**

## Fallback: Train Tiny MoE

If Switch integration proves difficult:

```python
# 100M param, 4 experts, WikiText-2
# Train just until entropy < max
# Then run same protocol
```

This is methodologically cleaner and fully under our control.

---

**The instrumentation is ready. The protocol is validated. We just need the right substrate.**

Now executing: Load Switch-base-8, verify structure, proceed.
