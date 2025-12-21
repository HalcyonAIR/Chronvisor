#!/usr/bin/env python3
"""
Telemetry Calibration Test: Pause-Point Detection Protocol

Implements the "weekend project" calibration experiment:
1. Run inference with pause detection
2. Emit telemetry JSON at candidate stillness points
3. Resume without new input to test if structure continues

This creates the observable substrate for phenomenology reports.

Protocol:
  - System runs until candidate_stillness detected
  - Emits telemetry snapshot
  - Pauses for human/model review
  - Optionally resumes to test true vs false stillness
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

from chronomoe.deepseek_core import DeepSeekConfig
from chronomoe.chronovisor_deepseek_bridge import ChronovisorDeepSeekForCausalLM
from chronomoe.telemetry import (
    ChronoMoETelemetryCollector,
    StillnessThresholds,
    TelemetrySnapshot,
)


def run_inference_with_pauses(
    model,
    input_ids: torch.Tensor,
    collector: ChronoMoETelemetryCollector,
    max_pauses: int = 3,
    max_steps_per_burst: int = 20,
    resume_on_pause: bool = True,
    resume_steps: int = 5,
) -> List[TelemetrySnapshot]:
    """
    Run inference with pause-point detection.

    Args:
        model: ChronoMoE model
        input_ids: Input token IDs
        collector: Telemetry collector
        max_pauses: Maximum number of pauses to allow
        max_steps_per_burst: Maximum steps before forced pause
        resume_on_pause: If True, resume after each pause to test structure
        resume_steps: How many steps to run after pause

    Returns:
        List of telemetry snapshots at each pause
    """
    snapshots = []
    pause_count = 0
    step_count = 0

    print(f"\n{'='*70}")
    print("INFERENCE WITH PAUSE DETECTION")
    print(f"{'='*70}\n")

    print(f"Configuration:")
    print(f"  Max pauses: {max_pauses}")
    print(f"  Max steps per burst: {max_steps_per_burst}")
    print(f"  Resume on pause: {resume_on_pause}")
    print(f"  Resume steps: {resume_steps}")
    print(f"  Stillness thresholds: fast={collector.thresholds.fast}, "
          f"medium={collector.thresholds.medium}, slow={collector.thresholds.slow}")
    print()

    while pause_count < max_pauses:
        print(f"\n--- Burst {pause_count + 1} ---")

        # Run inference for up to max_steps_per_burst
        for step in range(max_steps_per_burst):
            with torch.no_grad():
                _, chrono_state, _ = model(input_ids, update_chronovisor=True)

            step_count += 1

            # Check for candidate stillness
            should_pause, reason = collector.should_pause(chrono_state)

            # Also check if we hit max steps
            if step >= max_steps_per_burst - 1:
                should_pause = True
                reason = "max_steps"

            if should_pause:
                print(f"  Pause detected at step {step_count} (reason: {reason})")

                # Capture snapshot
                snapshot = collector.capture_snapshot(
                    chrono_state=chrono_state,
                    pause_reason=reason,
                )
                snapshots.append(snapshot)

                # Print telemetry
                print(f"\n{'='*70}")
                print(f"TELEMETRY SNAPSHOT #{pause_count + 1}")
                print(f"{'='*70}")
                print(snapshot.to_json())
                print()

                # Interpret stillness
                if snapshot.stillness_flags['fast'] and snapshot.stillness_flags['medium']:
                    print("  ✓ Strong stillness signal (fast + medium)")
                elif snapshot.stillness_flags['fast']:
                    print("  ~ Weak stillness signal (fast only)")
                else:
                    print("  ○ No stillness (forced pause)")
                print()

                pause_count += 1

                # Resume test (if enabled)
                if resume_on_pause and pause_count < max_pauses:
                    print(f"  Resuming for {resume_steps} steps to test structure...")

                    initial_entropy = snapshot.routing_entropy
                    initial_pc2 = snapshot.pc2

                    # Run a few more steps without new input
                    for resume_step in range(resume_steps):
                        with torch.no_grad():
                            _, chrono_state_resume, _ = model(input_ids, update_chronovisor=True)

                    # Capture post-resume snapshot
                    snapshot_after = collector.capture_snapshot(
                        chrono_state=chrono_state_resume,
                        pause_reason="post_resume",
                    )

                    # Compare
                    delta_entropy = snapshot_after.routing_entropy - initial_entropy
                    delta_pc2 = snapshot_after.pc2 - initial_pc2

                    print(f"    Δ entropy: {delta_entropy:+.4f}")
                    print(f"    Δ PC2: {delta_pc2:+.4f}")

                    if abs(delta_pc2) > 0.05 or abs(delta_entropy) > 0.1:
                        print("    → Structure continued (false stillness?)")
                    else:
                        print("    → No new structure (true stillness)")
                    print()

                break  # Exit inner loop, start new burst

        else:
            # Didn't hit pause naturally
            continue

        # Check if we should stop
        if pause_count >= max_pauses:
            break

    print(f"\n{'='*70}")
    print(f"INFERENCE COMPLETE")
    print(f"{'='*70}")
    print(f"Total pauses: {pause_count}")
    print(f"Total steps: {step_count}")
    print()

    return snapshots


def main():
    print("="*70)
    print("TELEMETRY CALIBRATION TEST")
    print("Pause-Point Detection Protocol")
    print("="*70)
    print()
    print("This test demonstrates:")
    print("  1. Inference with pause detection (candidate stillness)")
    print("  2. Telemetry emission at pause points")
    print("  3. Resume test (does structure continue?)")
    print()
    print("Output: JSON snapshots suitable for phenomenology review")
    print()

    # Create model
    config = DeepSeekConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=1024,
        num_layers=2,
        num_shared_experts=2,
        num_routed_experts=64,
        num_experts_per_token=6,
        enable_chronovisor=True,
    )

    torch.manual_seed(42)
    model = ChronovisorDeepSeekForCausalLM(config)
    model.eval()

    print(f"Model created:")
    print(f"  Architecture: DeepSeek-MoE + ChronoMoE")
    print(f"  Routed experts: {config.num_routed_experts}")
    print(f"  Chronovisor: enabled")
    print()

    # Create telemetry collector
    thresholds = StillnessThresholds(
        fast=0.01,
        medium=0.02,
        slow=0.005,
    )

    collector = ChronoMoETelemetryCollector(
        session_id="calibration_v0",
        stillness_thresholds=thresholds,
        window_size=10,
    )

    print(f"Telemetry collector created:")
    print(f"  Session ID: {collector.session_id}")
    print(f"  Thresholds: fast={thresholds.fast}, medium={thresholds.medium}, slow={thresholds.slow}")
    print()

    # Create input
    input_ids = torch.randint(0, config.vocab_size, (1, 50))
    print(f"Input created: {input_ids.shape}")
    print()

    # Run inference with pauses
    snapshots = run_inference_with_pauses(
        model=model,
        input_ids=input_ids,
        collector=collector,
        max_pauses=3,
        max_steps_per_burst=20,
        resume_on_pause=True,
        resume_steps=5,
    )

    # Summary
    print(f"{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    print(f"Total snapshots captured: {len(snapshots)}")
    print()

    print("Snapshot summary:")
    for i, snapshot in enumerate(snapshots):
        print(f"\nSnapshot {i+1}:")
        print(f"  Pause reason: {snapshot.pause_reason}")
        print(f"  PC1 (momentum): {snapshot.pc1:.4f}")
        print(f"  PC2 (exploration): {snapshot.pc2:.4f}")
        print(f"  dPC1: {snapshot.d_pc1:+.4f}")
        print(f"  dPC2: {snapshot.d_pc2:+.4f}")
        print(f"  Routing entropy: {snapshot.routing_entropy:.4f}")
        print(f"  Expert switch rate: {snapshot.expert_switch_rate:.4f}")
        print(f"  Stillness: fast={snapshot.stillness_flags['fast']}, "
              f"medium={snapshot.stillness_flags['medium']}, "
              f"slow={snapshot.stillness_flags['slow']}")

    # Save snapshots
    output_dir = Path(__file__).parent.parent / "takens_data"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "telemetry_snapshots.json"
    with open(output_file, 'w') as f:
        snapshots_data = [snapshot.__dict__ for snapshot in snapshots]
        import json
        json.dump(snapshots_data, f, indent=2)

    print(f"\n\nSnapshots saved to: {output_file}")
    print()

    print("="*70)
    print("NEXT STEPS")
    print("="*70)
    print()
    print("For phenomenology calibration:")
    print()
    print("1. Share the telemetry snapshots with an observer (human or model)")
    print()
    print("2. Observer provides felt-state labels:")
    print("   {")
    print('     "snapshot_id": 1,')
    print('     "felt_state": "complete | premature | stuck",')
    print('     "confidence": 0.8,')
    print('     "notes": "This felt like natural completion, not truncation"')
    print("   }")
    print()
    print("3. Compare:")
    print("   - System stillness flags (what the system thinks)")
    print("   - Observer phenomenology (what it felt like)")
    print("   - Resume test results (what actually happened)")
    print()
    print("4. Build confusion matrix:")
    print("   - True stillness: system=done, felt=complete, resume=filler")
    print("   - False positive: system=done, felt=premature, resume=structure")
    print("   - False negative: system=not done, felt=complete")
    print()
    print("5. Tune thresholds between sessions (not mid-run)")
    print()
    print("This is calibration research, not consciousness research.")
    print()


if __name__ == '__main__':
    main()
