#!/usr/bin/env python3
"""
Telemetry Calibration Test (Modular Version)

Demonstrates the refactored telemetry module with clean separation:
  - observer.py: Read-only state access
  - metrics.py: Pure measurement functions
  - stillness.py: Multi-timescale stillness detection
  - pause.py: Candidate pause proposals
  - schema.py: JSON format + glossary
  - recorder.py: Main orchestrator

This module does not decide when the system is done.
It only proposes pause points and exposes the evidence.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path
from typing import List

from chronomoe.deepseek_core import DeepSeekConfig
from chronomoe.chronovisor_deepseek_bridge import ChronovisorDeepSeekForCausalLM

from chronomoe_telemetry import (
    ChronoMoEObserver,
    TelemetryRecorder,
    StillnessThresholds,
    TelemetrySnapshot,
)


def run_inference_with_telemetry(
    model,
    input_ids: torch.Tensor,
    recorder: TelemetryRecorder,
    max_pauses: int = 3,
    max_steps_per_burst: int = 20,
    resume_on_pause: bool = True,
    resume_steps: int = 5,
) -> List[TelemetrySnapshot]:
    """
    Run inference with telemetry-based pause detection.

    Three-layer validation:
      1. System belief: stillness flags (what the system thinks)
      2. Execution phenomenology: (observer reports - not implemented here)
      3. Reality check: resume test (does structure continue?)
    """
    snapshots = []
    pause_count = 0
    step_count = 0

    print(f"\n{'='*70}")
    print("INFERENCE WITH TELEMETRY")
    print(f"{'='*70}\n")

    print(f"Configuration:")
    print(f"  Max pauses: {max_pauses}")
    print(f"  Max steps per burst: {max_steps_per_burst}")
    print(f"  Resume test: {resume_on_pause}")
    print()

    while pause_count < max_pauses:
        print(f"\n--- Burst {pause_count + 1} ---")

        # Run inference burst
        for step in range(max_steps_per_burst):
            with torch.no_grad():
                _, chrono_state, _ = model(input_ids, update_chronovisor=True)

            step_count += 1

            # Capture telemetry snapshot
            snapshot = recorder.capture_snapshot(
                chrono_state=chrono_state,
                pause_reason="step_update",
            )

            # Check if pause should be proposed
            should_pause, reason = recorder.should_pause()

            # Also check max steps
            if step >= max_steps_per_burst - 1:
                should_pause = True
                reason = "max_steps"

            if should_pause:
                print(f"  Pause proposed at step {step_count} (reason: {reason})")

                # Update pause reason in snapshot
                snapshot.pause_reason = reason

                # Print telemetry
                print(f"\n{'='*70}")
                print(f"TELEMETRY SNAPSHOT #{pause_count + 1}")
                print(f"{'='*70}")
                print(snapshot.to_json())
                print()

                # Interpret stillness
                flags = snapshot.stillness_flags
                if flags['fast'] and flags['medium'] and flags['slow']:
                    print("  ✓✓✓ All timescales show stillness (very strong signal)")
                elif flags['fast'] and flags['medium']:
                    print("  ✓✓ Strong stillness signal (fast + medium)")
                elif flags['fast']:
                    print("  ✓ Weak stillness signal (fast only)")
                else:
                    print("  ○ No stillness (forced pause)")
                print()

                snapshots.append(snapshot)
                pause_count += 1

                # Resume test (Layer 3: Reality check)
                if resume_on_pause and pause_count < max_pauses:
                    print(f"  Resume test: Running {resume_steps} steps without new input...")

                    initial_entropy = snapshot.routing_entropy
                    initial_pc2 = snapshot.pc2

                    # Continue inference
                    for _ in range(resume_steps):
                        with torch.no_grad():
                            _, chrono_state_resume, _ = model(input_ids, update_chronovisor=True)

                    # Capture post-resume snapshot
                    snapshot_after = recorder.capture_snapshot(
                        chrono_state=chrono_state_resume,
                        pause_reason="post_resume",
                    )

                    # Compare
                    delta_entropy = snapshot_after.routing_entropy - initial_entropy
                    delta_pc2 = snapshot_after.pc2 - initial_pc2

                    print(f"    Δ entropy: {delta_entropy:+.4f}")
                    print(f"    Δ PC2: {delta_pc2:+.4f}")

                    if abs(delta_pc2) > 0.05 or abs(delta_entropy) > 0.1:
                        print("    → Structure continued (possible false stillness)")
                    else:
                        print("    → No new structure (true stillness confirmed)")
                    print()

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
    print("TELEMETRY CALIBRATION TEST (MODULAR VERSION)")
    print("="*70)
    print()
    print("This module does not decide when the system is done.")
    print("It only proposes pause points and exposes the evidence.")
    print()
    print("Three-layer validation:")
    print("  1. System belief: stillness flags")
    print("  2. Execution phenomenology: observer reports")
    print("  3. Reality check: resume test")
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

    print(f"Model: DeepSeek-MoE + ChronoMoE")
    print(f"  Experts: {config.num_routed_experts}")
    print()

    # Create telemetry recorder
    observer = ChronoMoEObserver()
    thresholds = StillnessThresholds(fast=0.01, medium=0.02, slow=0.005)

    recorder = TelemetryRecorder(
        session_id="calibration_modular_v1",
        observer=observer,
        stillness_thresholds=thresholds,
    )

    print(f"Telemetry recorder created")
    print(f"  Session: {recorder.session_id}")
    print(f"  Thresholds: fast={thresholds.fast}, medium={thresholds.medium}, slow={thresholds.slow}")
    print()

    # Create input
    input_ids = torch.randint(0, config.vocab_size, (1, 50))
    print(f"Input: {input_ids.shape}")
    print()

    # Run inference
    snapshots = run_inference_with_telemetry(
        model=model,
        input_ids=input_ids,
        recorder=recorder,
        max_pauses=3,
        max_steps_per_burst=20,
        resume_on_pause=True,
        resume_steps=5,
    )

    # Summary
    print(f"{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    print(f"Snapshots captured: {len(snapshots)}")
    print()

    for i, snapshot in enumerate(snapshots):
        print(f"Snapshot {i+1}:")
        print(f"  Reason: {snapshot.pause_reason}")
        print(f"  PC1={snapshot.pc1:.4f}, PC2={snapshot.pc2:.4f}")
        print(f"  dPC1={snapshot.d_pc1:+.4f}, dPC2={snapshot.d_pc2:+.4f}")
        print(f"  Entropy={snapshot.routing_entropy:.4f}")
        print(f"  Stillness: F={snapshot.stillness_flags['fast']}, "
              f"M={snapshot.stillness_flags['medium']}, "
              f"S={snapshot.stillness_flags['slow']}")
        print()

    # Save
    output_dir = Path(__file__).parent.parent / "takens_data"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "telemetry_modular_snapshots.json"
    with open(output_file, 'w') as f:
        import json
        snapshots_data = [s.to_dict() for s in snapshots]
        json.dump(snapshots_data, f, indent=2)

    print(f"Saved to: {output_file}")
    print()

    print("="*70)
    print("NEXT: PHENOMENOLOGY CALIBRATION")
    print("="*70)
    print()
    print("For each snapshot, observer provides:")
    print("  {")
    print('    "snapshot_id": 1,')
    print('    "felt_state": "complete | premature | stuck",')
    print('    "confidence": 0.8,')
    print('    "notes": "Explanation..."')
    print("  }")
    print()
    print("Then compare:")
    print("  System belief (stillness flags)")
    print("  vs Phenomenology (felt state)")
    print("  vs Reality (resume test)")
    print()
    print("Build confusion matrix → tune thresholds between sessions")
    print()


if __name__ == '__main__':
    main()
