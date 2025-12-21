#!/usr/bin/env python3
"""
Self-Gated Hysteresis: Testing Phase 2 Hypothesis

Phase 1 Result: Unconditional pressure cannot deform the decision manifold.
    - Zero hysteresis in A→B vs B→A transitions
    - Boundaries rigid under stress (η up to 0.25)
    - Constraint re-projection dominates

Phase 2 Hypothesis: Self-gated state can create path-dependent routing.
    - Margin-conditioned influence: strong when uncertain, weak when confident
    - State with momentum can coast through gaps unreachable from standing start
    - "Ball rolling" can cross boundaries that "static push" cannot

Test Design:
    1. Run noise ramp A→B (low→high) with self-gated adapter
    2. Run noise ramp B→A (high→low) with self-gated adapter
    3. Compare transition points and expert selection
    4. Compare to Phase 1 baseline (unconditional pressure)

Success Criteria:
    ✓ Transition points shift with approach direction (hysteresis detected)
    ✓ In-distribution routing preserved (no corruption of clean inputs)
    ✗ Pathological collapse to Expert 2 (trap created)

Three Possible Outcomes:
    1. Shift without lock-in → Self-gating creates adaptive routing ✓
    2. Pathological lock-in → Built a trap ✗
    3. No shift at all → Walls absolute even with state ○
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path
import json
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("ERROR: transformers not installed")
    sys.exit(1)

from chronovisor_switch_self_gated import wrap_switch_model_with_self_gated_chronovisor


def create_noise_interpolated_input(tokenizer, text, noise_fraction, seed=42):
    """Create input with specified fraction of tokens replaced by random."""
    real_inputs = tokenizer(text, return_tensors="pt")
    token_ids = real_inputs['input_ids'][0].tolist()

    random.seed(seed)
    num_tokens = len(token_ids)
    num_to_replace = int(num_tokens * noise_fraction)
    positions_to_replace = set(random.sample(range(num_tokens), num_to_replace))

    vocab_size = len(tokenizer)
    mixed_ids = [
        random.randint(0, vocab_size - 1) if i in positions_to_replace else token_id
        for i, token_id in enumerate(token_ids)
    ]

    return {
        'input_ids': torch.tensor([mixed_ids]),
        'attention_mask': torch.ones_like(torch.tensor([mixed_ids]))
    }


def extract_routing_stats(model, inputs, layer_idx=1):
    """Extract routing statistics including margin."""
    captured_outputs = []

    def hook_fn(module, input, output):
        router_mask, router_probs, router_logits = output
        captured_outputs.append(router_logits.detach().cpu().numpy())

    target_block = model.encoder.block[layer_idx]
    router_module = target_block.layer[-1].mlp.router
    handle = router_module.register_forward_hook(hook_fn)

    try:
        decoder_input_ids = inputs['input_ids'].clone()

        with torch.no_grad():
            _ = model(
                input_ids=inputs['input_ids'],
                decoder_input_ids=decoder_input_ids,
            )

        logits = captured_outputs[0].squeeze(0)
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)

        # Compute margin
        sorted_logits = np.sort(logits, axis=-1)[:, ::-1]
        margins = sorted_logits[:, 0] - sorted_logits[:, 1]
        avg_margin = float(np.mean(margins))

        # Compute entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
        avg_entropy = float(np.mean(entropy))

        # Top expert
        top_expert = int(np.argmax(probs.mean(axis=0)))

        return {
            'probs': probs,
            'logits': logits,
            'margin': avg_margin,
            'entropy': avg_entropy,
            'top_expert': top_expert,
        }

    finally:
        handle.remove()


def run_noise_ramp(
    model,
    adapter,
    tokenizer,
    text,
    noise_levels,
    direction,
    carry_forward_T=False,
    n_repetitions=100,
):
    """
    Run noise ramp with self-gated adapter.

    Args:
        carry_forward_T: If True, don't reset T̄ between noise levels
    """
    results = []

    for i, noise_frac in enumerate(noise_levels):
        # Reset T̄ unless carrying forward
        if not carry_forward_T or i == 0:
            adapter.reset()

        adapter.enable_adaptation()

        inputs = create_noise_interpolated_input(tokenizer, text, noise_frac, seed=42)

        # Measure initial (stiffness check)
        stats_initial = extract_routing_stats(model, inputs, layer_idx=1)

        # Run wear with self-gated adaptation
        for rep in range(n_repetitions):
            with torch.no_grad():
                decoder_input_ids = inputs['input_ids'].clone()
                _ = model(
                    input_ids=inputs['input_ids'],
                    decoder_input_ids=decoder_input_ids,
                )

        # Measure final
        stats_final = extract_routing_stats(model, inputs, layer_idx=1)

        # Get adapter state
        adapter_state = adapter.get_state()

        results.append({
            'noise_fraction': noise_frac,
            'margin_initial': stats_initial['margin'],
            'margin_final': stats_final['margin'],
            'entropy_initial': stats_initial['entropy'],
            'entropy_final': stats_final['entropy'],
            'top_expert_initial': stats_initial['top_expert'],
            'top_expert_final': stats_final['top_expert'],
            'T_bar': adapter_state['T_bar'].tolist(),
            'pressure': adapter_state['pressure'].tolist(),
            'mean_gate': adapter_state.get('mean_gate', None),
            'mean_margin': adapter_state.get('mean_margin', None),
        })

    return results


def analyze_transitions(results):
    """Find transition boundaries in results."""
    experts = [r['top_expert_final'] for r in results]
    noise_levels = [r['noise_fraction'] for r in results]

    # Find 7→4 boundary
    boundary_7_to_4 = None
    for i in range(len(experts) - 1):
        if experts[i] == 7 and experts[i+1] != 7:
            boundary_7_to_4 = noise_levels[i+1]  # First non-7
            break

    # Find 4→2 boundary
    boundary_4_to_2 = None
    for i in range(len(experts) - 1):
        if experts[i] == 4 and experts[i+1] == 2:
            boundary_4_to_2 = noise_levels[i+1]  # First 2 after 4
            break

    # Find any 7→2 direct transition (bad sign)
    direct_7_to_2 = None
    for i in range(len(experts) - 1):
        if experts[i] == 7 and experts[i+1] == 2:
            direct_7_to_2 = noise_levels[i+1]
            break

    # Expert 4 width
    expert_4_levels = [noise_levels[i] for i, e in enumerate(experts) if e == 4]
    if expert_4_levels:
        expert_4_width = max(expert_4_levels) - min(expert_4_levels) + 0.05
    else:
        expert_4_width = 0.0

    return {
        'boundary_7_to_4': boundary_7_to_4,
        'boundary_4_to_2': boundary_4_to_2,
        'direct_7_to_2': direct_7_to_2,
        'expert_4_width': expert_4_width,
        'expert_sequence': experts,
    }


def check_pathological_collapse(results):
    """Check if clean inputs still route to Expert 7."""
    low_noise_experts = [
        r['top_expert_final'] for r in results
        if r['noise_fraction'] <= 0.20
    ]

    if not low_noise_experts:
        return None

    expert_7_count = sum(1 for e in low_noise_experts if e == 7)
    is_healthy = expert_7_count >= len(low_noise_experts) * 0.8

    return {
        'low_noise_expert_7_fraction': expert_7_count / len(low_noise_experts),
        'is_healthy': is_healthy,
    }


def main():
    print("="*70)
    print("SELF-GATED HYSTERESIS: Phase 2 Hypothesis Test")
    print("="*70)
    print()
    print("Testing: Can margin-conditioned state create path dependence")
    print("         where unconditional pressure could not?")
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

    # Wrap with self-gated ChronoMoE
    print("Wrapping with self-gated adapter...")
    adapters = wrap_switch_model_with_self_gated_chronovisor(
        model,
        layer_indices=[1],
        eta_structural_T=0.05,  # Use η from Phase 1 (detection threshold)
        gate_scale=2.0,  # Moderate sensitivity
        gate_offset=0.5,  # 50% gate at margin=0.5
    )
    adapter = adapters[1]
    print()

    # Test parameters
    text = "Machine learning models process data efficiently."
    noise_levels = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    print(f"Text: {text}")
    print(f"Noise levels: 0% → 60% (5% steps)")
    print(f"n_repetitions: 100")
    print(f"η: 0.05 (Phase 1 detection threshold)")
    print()

    # ================================================================
    # EXPERIMENT 1: A→B Ramp (Low→High Noise, Carry-forward T̄)
    # ================================================================
    print("="*70)
    print("EXPERIMENT 1: A→B Ramp (Low→High, Carry T̄)")
    print("="*70)
    print()

    results_up_carry = run_noise_ramp(
        model, adapter, tokenizer, text,
        noise_levels=noise_levels,
        direction='up',
        carry_forward_T=True,
        n_repetitions=100,
    )

    print("Complete.\n")

    # ================================================================
    # EXPERIMENT 2: B→A Ramp (High→Low Noise, Carry-forward T̄)
    # ================================================================
    print("="*70)
    print("EXPERIMENT 2: B→A Ramp (High→Low, Carry T̄)")
    print("="*70)
    print()

    results_down_carry = run_noise_ramp(
        model, adapter, tokenizer, text,
        noise_levels=list(reversed(noise_levels)),
        direction='down',
        carry_forward_T=True,
        n_repetitions=100,
    )

    # Reverse back for comparison
    results_down_carry = list(reversed(results_down_carry))

    print("Complete.\n")

    # ================================================================
    # EXPERIMENT 3: Fresh T̄ Ramps (Control)
    # ================================================================
    print("="*70)
    print("EXPERIMENT 3: Fresh T̄ Ramps (Control)")
    print("="*70)
    print()

    print("Running A→B (fresh)...")
    results_up_fresh = run_noise_ramp(
        model, adapter, tokenizer, text,
        noise_levels=noise_levels,
        direction='up',
        carry_forward_T=False,
        n_repetitions=100,
    )

    print("Running B→A (fresh)...")
    results_down_fresh = run_noise_ramp(
        model, adapter, tokenizer, text,
        noise_levels=list(reversed(noise_levels)),
        direction='down',
        carry_forward_T=False,
        n_repetitions=100,
    )
    results_down_fresh = list(reversed(results_down_fresh))

    print("Complete.\n")

    # ================================================================
    # ANALYSIS: Phase Diagram
    # ================================================================
    print("="*70)
    print("ANALYSIS: Self-Gated Phase Diagram")
    print("="*70)
    print()

    # Analyze transitions
    trans_up_carry = analyze_transitions(results_up_carry)
    trans_down_carry = analyze_transitions(results_down_carry)
    trans_up_fresh = analyze_transitions(results_up_fresh)
    trans_down_fresh = analyze_transitions(results_down_fresh)

    # Display comparison
    print("CARRY-FORWARD T̄ (State with Momentum):")
    print()
    print(f"{'Noise %':<10} {'Up→Expert':<12} {'Down→Expert':<12} {'Match':<8}")
    print("-"*50)

    for i, noise in enumerate(noise_levels):
        expert_up = results_up_carry[i]['top_expert_final']
        expert_down = results_down_carry[i]['top_expert_final']
        match = "✓" if expert_up == expert_down else "✗ HYSTERESIS"

        print(f"{int(noise*100):<10} {expert_up:<12} {expert_down:<12} {match:<8}")

    print()

    # Boundary comparison
    print("Transition Boundaries (Carry-forward T̄):")
    print(f"  Up (A→B):   7→? at {int(trans_up_carry['boundary_7_to_4']*100) if trans_up_carry['boundary_7_to_4'] else 'N/A'}%")
    print(f"  Down (B→A): 7→? at {int(trans_down_carry['boundary_7_to_4']*100) if trans_down_carry['boundary_7_to_4'] else 'N/A'}%")
    print()

    print("Expert 4 Width:")
    print(f"  Up (A→B):   {trans_up_carry['expert_4_width']*100:.0f}%")
    print(f"  Down (B→A): {trans_down_carry['expert_4_width']*100:.0f}%")
    print()

    # Check for hysteresis
    up_experts = [r['top_expert_final'] for r in results_up_carry]
    down_experts = [r['top_expert_final'] for r in results_down_carry]
    matches = sum(1 for u, d in zip(up_experts, down_experts) if u == d)
    total = len(up_experts)

    print(f"Expert selection matches: {matches}/{total}")
    print()

    if matches < total:
        print("✓ HYSTERESIS DETECTED")
        print(f"  {total - matches} noise levels show different routing")
        print("  Self-gated state creates path dependence")
        hysteresis_detected = True
    else:
        print("○ NO HYSTERESIS")
        print("  Up/down transitions identical")
        hysteresis_detected = False

    print()

    # ================================================================
    # SAFETY CHECK: Pathological Collapse
    # ================================================================
    print("="*70)
    print("SAFETY CHECK: Pathological Collapse")
    print("="*70)
    print()

    collapse_up = check_pathological_collapse(results_up_carry)
    collapse_down = check_pathological_collapse(results_down_carry)

    if collapse_up and collapse_down:
        print(f"In-distribution routing (0-20% noise):")
        print(f"  Up ramp:   {collapse_up['low_noise_expert_7_fraction']*100:.0f}% → Expert 7")
        print(f"  Down ramp: {collapse_down['low_noise_expert_7_fraction']*100:.0f}% → Expert 7")
        print()

        if collapse_up['is_healthy'] and collapse_down['is_healthy']:
            print("✓ No pathological collapse")
            print("  Clean inputs still route to Expert 7")
            pathological = False
        else:
            print("⚠ PATHOLOGICAL COLLAPSE")
            print("  Clean inputs corrupted by self-gating")
            pathological = True
    else:
        print("⚠ Cannot determine collapse status")
        pathological = None

    print()

    # ================================================================
    # COMPARISON TO PHASE 1 BASELINE
    # ================================================================
    print("="*70)
    print("COMPARISON TO PHASE 1 BASELINE")
    print("="*70)
    print()

    print("Phase 1 (Unconditional Pressure):")
    print("  Hysteresis: NONE (0/9 mismatches)")
    print("  Boundaries: 7→4 at 35%, 4→2 at 45% (rigid)")
    print("  Mechanism: Constraint re-projection")
    print()

    print("Phase 2 (Self-Gated State):")
    print(f"  Hysteresis: {'DETECTED' if hysteresis_detected else 'NONE'} ({total - matches}/{total} mismatches)")

    if trans_up_carry['boundary_7_to_4'] and trans_down_carry['boundary_7_to_4']:
        boundary_shift = abs(trans_up_carry['boundary_7_to_4'] - trans_down_carry['boundary_7_to_4'])
        print(f"  Boundary shift: {boundary_shift*100:.0f}% noise")
    else:
        print(f"  Boundary shift: Unable to determine")

    print(f"  Pathological: {'YES (FAILURE)' if pathological else 'NO'}")
    print()

    # ================================================================
    # OUTCOME CLASSIFICATION
    # ================================================================
    print("="*70)
    print("OUTCOME CLASSIFICATION")
    print("="*70)
    print()

    if hysteresis_detected and not pathological:
        print("✓ OUTCOME 1: Shift without lock-in")
        print("  Self-gating creates adaptive routing with controllable hysteresis")
        print("  State with momentum can cross boundaries pressure alone cannot")
        print("  Hypothesis VALIDATED")
        outcome = "success"
    elif hysteresis_detected and pathological:
        print("✗ OUTCOME 2: Pathological lock-in")
        print("  Built a trap - clean inputs corrupted")
        print("  Self-gating too aggressive")
        print("  Hypothesis FAILED (bad side effect)")
        outcome = "pathological"
    else:
        print("○ OUTCOME 3: No shift")
        print("  Walls absolute even with self-gated state")
        print("  Constraint manifold dominates all adaptive mechanisms")
        print("  Hypothesis FALSIFIED")
        outcome = "null"

    print()

    # ================================================================
    # VISUALIZATION
    # ================================================================
    print("="*70)
    print("VISUALIZATION")
    print("="*70)
    print()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Expert selection (carry-forward)
    ax1 = axes[0, 0]
    noise_pct = [n * 100 for n in noise_levels]
    ax1.plot(noise_pct, up_experts, marker='o', linewidth=2, label='Up (A→B)', color='blue')
    ax1.plot(noise_pct, down_experts, marker='s', linewidth=2, label='Down (B→A)', color='red', linestyle='--')
    ax1.set_xlabel('Noise %')
    ax1.set_ylabel('Top Expert')
    ax1.set_title('Self-Gated Expert Selection (Carry T̄)')
    ax1.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Margin evolution
    ax2 = axes[0, 1]
    margins_up = [r['mean_margin'] for r in results_up_carry if r['mean_margin'] is not None]
    margins_down = [r['mean_margin'] for r in results_down_carry if r['mean_margin'] is not None]
    ax2.plot(noise_pct[:len(margins_up)], margins_up, marker='o', label='Up', color='blue')
    ax2.plot(noise_pct[:len(margins_down)], margins_down, marker='s', label='Down', color='red', linestyle='--')
    ax2.set_xlabel('Noise %')
    ax2.set_ylabel('Mean Margin During Wear')
    ax2.set_title('Routing Confidence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Gate activity
    ax3 = axes[1, 0]
    gates_up = [r['mean_gate'] for r in results_up_carry if r['mean_gate'] is not None]
    gates_down = [r['mean_gate'] for r in results_down_carry if r['mean_gate'] is not None]
    ax3.plot(noise_pct[:len(gates_up)], gates_up, marker='o', label='Up', color='blue')
    ax3.plot(noise_pct[:len(gates_down)], gates_down, marker='s', label='Down', color='red', linestyle='--')
    ax3.set_xlabel('Noise %')
    ax3.set_ylabel('Mean Gate (T̄ Influence)')
    ax3.set_title('Self-Gating Activity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Fresh vs carry comparison (up direction only)
    ax4 = axes[1, 1]
    experts_up_fresh = [r['top_expert_final'] for r in results_up_fresh]
    ax4.plot(noise_pct, up_experts, marker='o', linewidth=2, label='Carry T̄', color='blue')
    ax4.plot(noise_pct, experts_up_fresh, marker='^', linewidth=2, label='Fresh T̄', color='green', linestyle=':')
    ax4.set_xlabel('Noise %')
    ax4.set_ylabel('Top Expert')
    ax4.set_title('Effect of T̄ Carry-forward (Up Ramp)')
    ax4.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = Path(__file__).parent.parent / "takens_data"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "self_gated_hysteresis.png", dpi=150)

    print(f"Plot saved to: {output_dir / 'self_gated_hysteresis.png'}")
    print()

    # Helper to convert numpy types to Python types
    def convert_numpy(obj):
        """Recursively convert numpy types to Python types."""
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj

    # Save results
    with open(output_dir / "self_gated_hysteresis_results.json", 'w') as f:
        json.dump(convert_numpy({
            'text': text,
            'noise_levels': noise_levels,
            'eta': 0.05,
            'gate_scale': 2.0,
            'gate_offset': 0.5,
            'n_repetitions': 100,
            'results_up_carry': results_up_carry,
            'results_down_carry': results_down_carry,
            'results_up_fresh': results_up_fresh,
            'results_down_fresh': results_down_fresh,
            'transitions': {
                'up_carry': trans_up_carry,
                'down_carry': trans_down_carry,
                'up_fresh': trans_up_fresh,
                'down_fresh': trans_down_fresh,
            },
            'outcome': outcome,
            'hysteresis_detected': hysteresis_detected,
            'pathological': pathological,
        }), f, indent=2)

    print(f"Results saved to: {output_dir / 'self_gated_hysteresis_results.json'}")
    print()

    print("="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
