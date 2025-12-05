# Chronovisor

**Experts sense. Lens shapes. Controller decides.**

Chronovisor is an experimental architecture layer that sits above expert heads in a Transformer-like model. It introduces a shared, deformable geometry — a *lens* — that reshapes how experts see semantic space, without touching tokens, embeddings, or static weights.

The goal: help experts specialise, stabilise, and coordinate over time — not by making them smarter, but by giving them a better view.

---

## The Problem

Expert heads in modern architectures tend to drift, interfere, and fight. Attention heads wander into garbage space. Mixture-of-Experts systems need brittle gating. Fine-tuning is expensive and often destabilising.

Most solutions try to fix this by changing *what the model knows* — editing weights, patching activations, adding adapters.

Chronovisor takes a different approach: it changes *how the model sees*.

---

## Architecture

Chronovisor separates concerns cleanly into three components:

### Experts (Sensors)

Expert heads become passive. They no longer decide when or how to change their behaviour. Their job is simple:

- Read through the shared lens surface
- Expose a small interface: local gain, tilt, stability score, out-of-tolerance flags
- Operate under whatever geometry the controller provides

Experts don't self-regulate. They just sense.

### Lens (Geometry)

The lens is a shared, deformable semantic surface that all experts read through. It can:

- Brighten regions where coherent, stabilising decisions occur
- Dim regions where instability grows
- Apply small affine shifts (rotations, translations) to keep experts aligned
- Produce local "bumps" — controlled warps under regions with sustained resonance

The lens never touches tokens or weights. It only reshapes the space experts perceive.

### Controller (Decision-Maker)

All decision-making lives here. The controller:

- Runs three clocks operating at different timescales (fast, micro, macro)
- Measures Δcoherence between micro-turns
- Interprets which expert signals are meaningful vs. noise
- Decides when lens adjustments are permitted
- Modulates the magnitude and locality of geometric shifts
- Prevents overreaction and runaway warping

The controller doesn't evaluate *content* — it evaluates *trajectory stability*. Did the geometry tighten or loosen? Did coherence improve or degrade?

**Δcoherence is the reinforcement signal that drives everything.**

---

## Why This Matters

This architecture introduces something new: a layer of control that operates on *geometry* rather than *weights*.

- Experts specialise with clearer boundaries (they're not fighting over territory)
- The system adapts without fine-tuning (the lens moves; the weights don't)
- Micro-drift is allowed — even required — for global stability
- Only patterns that genuinely improve coherence get reinforced

Classic failure modes this avoids:

- Expert interference and competition
- Attention heads drifting into degenerate subspaces
- Brittle attractors that shatter under perturbation
- Runaway oscillations from overreactive gating
- The need for expensive retraining to course-correct

---

## What Chronovisor Is Not

- **Not a token-patching trick** — we don't modify activations or embeddings
- **Not LoRA or weight editing** — static weights remain untouched
- **Not RLHF** — no reward model, no policy gradients
- **Not a routing mechanism** — experts don't compete for selection
- **Not MoE gating** — there's no sparse activation or top-k selection

Chronovisor never rewrites weights. It only reshapes the semantic space experts see.

---

## Implementation Plan

Starting minimal:

```
/src/controller.py    # Clock management, Δcoherence tracking, lens update logic
/src/lens.py          # Shared geometry surface, affine transforms, local warps
/src/expert_harness.py # Interface that experts expose to the controller
/src/expert_stub.py   # Toy experts for initial testing
README.md
```

**Phase 1:** Build the controller skeleton, define the lens object, implement harness stubs.

**Phase 2:** Simulate fake "experts" in a toy environment. Test the loop: expert signals → Δcoherence measurement → lens update → stability check.

**Phase 3:** Integrate with a real small model. Validate that lens adjustments improve coherence without destabilising generation.

---

## The Name

*Chronovisor* — not the Vatican legend.

This one "views" and "shifts" the geometry of meaning across micro-turns (*chrono*). It doesn't change the past. It adjusts the semantic frame the future grows in.

---

## Status

Architecture is stable. Ready for prototyping.

The core insight — that geometric adjustment can substitute for weight modification — is testable with minimal infrastructure. We'll refine as we build.

---

## Contributing

This is early-stage research. If the framing resonates, open an issue or reach out. We're particularly interested in:

- Prior work on geometric approaches to representation learning
- Connections to manifold hypothesis and semantic topology
- Practical constraints from production MoE systems

---

*Chronovisor is part of the HalcyonAIR research programme.*
