#!/usr/bin/env python3
"""
Analyze Expert Selection Patterns Across η Sweep

Key question: Is concentration going to the same experts (global collapse)
or does the chosen set depend on prompt distribution (selective carving)?

Also: Track per-expert logit drift to test H1 (bias accumulation).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path
import json

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("ERROR: transformers not installed")
    sys.exit(1)

from chronovisor_switch_adapter import wrap_switch_model_with_chronovisor


def extract_routing_with_timeline(model, inputs, adapter, n_repetitions=100, layer_idx=1):
    """
    Extract routing at multiple checkpoints during A×N repetitions.

    Returns:
        timeline: List of (step, routing_probs, router_logits) tuples
    """
    timeline = []

    # Checkpoint schedule: log spacing for efficiency
    checkpoints = [0, 1, 2, 5, 10, 20, 50, 100]

    captured_outputs = []

    def hook_fn(module, input, output):
        router_mask, router_probs, router_logits = output
        captured_outputs.append(router_logits.detach().cpu().numpy())

    target_block = model.encoder.block[layer_idx]
    router_module = target_block.layer[-1].mlp.router
    handle = router_module.register_forward_hook(hook_fn)

    try:
        for step in range(n_repetitions + 1):
            if step in checkpoints:
                # Measure routing at this checkpoint
                captured_outputs.clear()

                decoder_input_ids = inputs['input_ids'].clone()

                with torch.no_grad():
                    _ = model(
                        input_ids=inputs['input_ids'],
                        decoder_input_ids=decoder_input_ids,
                    )

                logits = captured_outputs[0].squeeze(0)  # [seq_len, num_experts]
                probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                probs = probs / probs.sum(axis=-1, keepdims=True)

                timeline.append({
                    'step': step,
                    'routing_logits': logits.copy(),
                    'routing_probs': probs.copy(),
                })

            # Run one repetition (unless this is the last checkpoint)
            if step < n_repetitions:
                captured_outputs.clear()
                decoder_input_ids = inputs['input_ids'].clone()

                with torch.no_grad():
                    _ = model(
                        input_ids=inputs['input_ids'],
                        decoder_input_ids=decoder_input_ids,
                    )

        return timeline

    finally:
        handle.remove()


def analyze_expert_drift(timeline):
    """
    Analyze how per-expert logits drift over time.

    Tests H1: Logit bias accumulation.
    """
    num_experts = timeline[0]['routing_logits'].shape[-1]

    # Average logits over sequence for each checkpoint
    mean_logits_by_step = []

    for checkpoint in timeline:
        mean_logits = checkpoint['routing_logits'].mean(axis=0)  # [num_experts]
        mean_logits_by_step.append({
            'step': checkpoint['step'],
            'mean_logits': mean_logits,
        })

    # Compute per-expert drift (final - initial)
    initial_logits = mean_logits_by_step[0]['mean_logits']
    final_logits = mean_logits_by_step[-1]['mean_logits']

    expert_drift = final_logits - initial_logits

    # Check if drift is monotonic and additive
    # For each expert, fit linear trend
    slopes = []
    for expert_idx in range(num_experts):
        steps = [d['step'] for d in mean_logits_by_step]
        logits = [d['mean_logits'][expert_idx] for d in mean_logits_by_step]

        # Linear fit
        if len(steps) > 1:
            slope = (logits[-1] - logits[0]) / (steps[-1] - steps[0]) if steps[-1] != steps[0] else 0
            slopes.append(slope)
        else:
            slopes.append(0)

    return {
        'expert_drift': expert_drift,
        'expert_slopes': np.array(slopes),
        'mean_logits_timeline': mean_logits_by_step,
    }


def analyze_expert_selection(timeline):
    """
    Analyze which experts are selected and how selection evolves.

    Tests: Global collapse vs selective carving.
    """
    num_experts = timeline[0]['routing_probs'].shape[-1]

    # For each checkpoint, compute sequence-averaged routing
    expert_usage_by_step = []

    for checkpoint in timeline:
        probs = checkpoint['routing_probs']  # [seq_len, num_experts]
        avg_probs = probs.mean(axis=0)  # [num_experts]

        # Top-2 experts
        top2_indices = np.argsort(avg_probs)[::-1][:2]
        top2_probs = avg_probs[top2_indices]

        expert_usage_by_step.append({
            'step': checkpoint['step'],
            'avg_probs': avg_probs,
            'top2_experts': top2_indices.tolist(),
            'top2_probs': top2_probs.tolist(),
        })

    # Check consistency: do the same experts win?
    initial_top2 = set(expert_usage_by_step[0]['top2_experts'])
    final_top2 = set(expert_usage_by_step[-1]['top2_experts'])

    consistency = len(initial_top2 & final_top2) / 2.0  # Fraction of overlap

    return {
        'expert_usage_timeline': expert_usage_by_step,
        'top2_consistency': consistency,
        'initial_top2': list(initial_top2),
        'final_top2': list(final_top2),
    }


def main():
    print("="*70)
    print("EXPERT SELECTION ANALYSIS")
    print("="*70)
    print()
    print("Testing:")
    print("  H1: Logit bias accumulation (additive drift)")
    print("  Global collapse vs selective carving")
    print()

    if not HAS_TRANSFORMERS:
        print("ERROR: transformers not available")
        return

    # Load model
    print("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/switch-base-8",
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
    print("✓ Model loaded\n")

    # Wrap with ChronoMoE
    adapters = wrap_switch_model_with_chronovisor(
        model,
        layer_indices=[1],
        eta_structural_T=0.05,  # Use η where we crossed detection
    )
    adapter = adapters[1]
    print()

    # Test with multiple prompts to check selectivity
    prompts = {
        'A': "The cat sat on the mat. It was a sunny day.",
        'B': "Machine learning models process data efficiently.",
        'C': "Quantum computers leverage superposition for parallel computation.",
    }

    results = {}

    for label, text in prompts.items():
        print(f"{'='*70}")
        print(f"PROMPT {label}: \"{text[:50]}...\"")
        print(f"{'='*70}")
        print()

        inputs = tokenizer(text, return_tensors="pt")

        # Reset adapter
        adapter.reset()
        adapter.enable_adaptation()

        # Extract routing timeline
        print("Running A×100 with timeline capture...")
        timeline = extract_routing_with_timeline(
            model, inputs, adapter,
            n_repetitions=100,
            layer_idx=1
        )

        # Analyze drift
        drift_analysis = analyze_expert_drift(timeline)

        # Analyze selection
        selection_analysis = analyze_expert_selection(timeline)

        print(f"\nExpert drift (final - initial logits):")
        for i, drift in enumerate(drift_analysis['expert_drift']):
            print(f"  Expert {i}: {drift:+.4f}")

        print(f"\nTop-2 experts:")
        print(f"  Initial: {selection_analysis['initial_top2']} "
              f"({selection_analysis['expert_usage_timeline'][0]['top2_probs']})")
        print(f"  Final:   {selection_analysis['final_top2']} "
              f"({selection_analysis['expert_usage_timeline'][-1]['top2_probs']})")
        print(f"  Consistency: {selection_analysis['top2_consistency']:.1%}")

        # Check monotonicity
        slopes = drift_analysis['expert_slopes']
        monotonic_count = sum(1 for s in slopes if abs(s) > 0.001)

        print(f"\nMonotonic drift: {monotonic_count}/{len(slopes)} experts")
        print()

        results[label] = {
            'drift_analysis': drift_analysis,
            'selection_analysis': selection_analysis,
        }

    # Cross-prompt analysis
    print("="*70)
    print("CROSS-PROMPT ANALYSIS")
    print("="*70)
    print()

    # Check if same experts win across prompts
    final_experts_by_prompt = {
        label: set(results[label]['selection_analysis']['final_top2'])
        for label in prompts.keys()
    }

    # Pairwise overlap
    from itertools import combinations

    print("Top-2 expert overlap (final state):")
    for label1, label2 in combinations(prompts.keys(), 2):
        overlap = len(final_experts_by_prompt[label1] & final_experts_by_prompt[label2]) / 2.0
        print(f"  {label1} ∩ {label2}: {overlap:.1%}")

    print()

    # If overlap is low → selective carving
    # If overlap is high → global collapse

    all_overlaps = [
        len(final_experts_by_prompt[l1] & final_experts_by_prompt[l2]) / 2.0
        for l1, l2 in combinations(prompts.keys(), 2)
    ]
    mean_overlap = np.mean(all_overlaps)

    print(f"Mean cross-prompt overlap: {mean_overlap:.1%}")
    print()

    if mean_overlap < 0.3:
        print("✓ SELECTIVE CARVING")
        print("  Different prompts concentrate on different expert sets.")
        print("  Wear is encoding prompt-specific patterns.")
    elif mean_overlap > 0.7:
        print("⚠ GLOBAL COLLAPSE")
        print("  Same experts win regardless of prompt.")
        print("  Wear may be exploiting pre-existing bias, not prompt structure.")
    else:
        print("~ MIXED PATTERN")
        print("  Some selectivity, but also shared preferences.")

    print()

    # Save results
    output_dir = Path(__file__).parent.parent / "takens_data"
    output_dir.mkdir(exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {}
    for label, data in results.items():
        results_serializable[label] = {
            'final_top2': data['selection_analysis']['final_top2'],
            'top2_consistency': data['selection_analysis']['top2_consistency'],
            'expert_drift': data['drift_analysis']['expert_drift'].tolist(),
            'expert_slopes': data['drift_analysis']['expert_slopes'].tolist(),
        }

    with open(output_dir / "expert_selection_analysis.json", 'w') as f:
        json.dump({
            'prompts': prompts,
            'results': results_serializable,
            'mean_cross_prompt_overlap': float(mean_overlap),
        }, f, indent=2)

    print(f"Results saved to: {output_dir / 'expert_selection_analysis.json'}")
    print()


if __name__ == '__main__':
    main()
