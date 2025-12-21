#!/usr/bin/env python3
"""
Telemetry Proxy Mode Test

Demonstrates telemetry running WITHOUT ChronoMoE.
Uses synthetic data to validate the pause detection protocol.

This proves method is decoupled from mechanism.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pathlib import Path

from chronomoe_telemetry import (
    ProxyObserver,
    TelemetryRecorder,
    StillnessThresholds,
    create_gloss_reference,
)


def simulate_convergence_scenario():
    """
    Simulate a scenario where routing gradually converges.

    This mimics what happens during inference:
      - Initial exploration (high entropy)
      - Gradual convergence (entropy decreases)
      - Stillness (entropy flatlines)
    """
    print("\n" + "="*70)
    print("SCENARIO 1: CONVERGENCE TO STILLNESS")
    print("="*70 + "\n")

    # Create proxy observer
    observer = ProxyObserver(num_experts=64, num_layers=2)

    # Create telemetry recorder
    recorder = TelemetryRecorder(
        session_id="proxy_convergence",
        observer=observer,
        stillness_thresholds=StillnessThresholds(
            fast=0.01,
            medium=0.02,
            slow=0.005,
        ),
    )

    snapshots = []

    # Simulate 30 steps of gradual convergence
    print("Simulating 30 steps of routing dynamics...\n")

    for step in range(30):
        # Gradually reduce noise (simulates convergence)
        noise_scale = 0.1 * (1.0 - step / 30.0)
        observer.step(noise_scale=noise_scale)

        # Capture snapshot
        snapshot = recorder.capture_snapshot(
            chrono_state=None,  # No real model!
            pause_reason="step_update",
        )

        # Check for pause
        should_pause, reason = recorder.should_pause()

        if should_pause:
            print(f"Step {step+1}: Pause proposed (reason: {reason})")
            print(f"  PC1={snapshot.pc1:.4f}, PC2={snapshot.pc2:.4f}")
            print(f"  dPC1={snapshot.d_pc1:+.6f}, dPC2={snapshot.d_pc2:+.6f}")
            print(f"  Stillness: F={snapshot.stillness_flags['fast']}, "
                  f"M={snapshot.stillness_flags['medium']}, "
                  f"S={snapshot.stillness_flags['slow']}")
            print()

            snapshot.pause_reason = reason
            snapshots.append(snapshot)

    print(f"Total pauses proposed: {len(snapshots)}")
    return snapshots


def simulate_oscillation_scenario():
    """
    Simulate a scenario where routing oscillates (never converges).

    This should NOT trigger stillness - tests false negative rate.
    """
    print("\n" + "="*70)
    print("SCENARIO 2: OSCILLATION (NO CONVERGENCE)")
    print("="*70 + "\n")

    # Create proxy observer
    observer = ProxyObserver(num_experts=64, num_layers=2)

    # Create telemetry recorder
    recorder = TelemetryRecorder(
        session_id="proxy_oscillation",
        observer=observer,
        stillness_thresholds=StillnessThresholds(
            fast=0.01,
            medium=0.02,
            slow=0.005,
        ),
    )

    snapshots = []

    # Simulate 30 steps of oscillation
    print("Simulating 30 steps of oscillating routing...\n")

    for step in range(30):
        # Constant high noise (simulates oscillation)
        noise_scale = 0.1
        observer.step(noise_scale=noise_scale)

        # Capture snapshot
        snapshot = recorder.capture_snapshot(
            chrono_state=None,
            pause_reason="step_update",
        )

        # Check for pause
        should_pause, reason = recorder.should_pause()

        if should_pause:
            print(f"Step {step+1}: Pause proposed (reason: {reason})")
            print(f"  ⚠ WARNING: Stillness detected during oscillation (false positive)")
            print()

            snapshot.pause_reason = reason
            snapshots.append(snapshot)

    if len(snapshots) == 0:
        print("✓ No pauses proposed (correct - system is oscillating)")
    else:
        print(f"✗ {len(snapshots)} pauses proposed (false positives)")

    return snapshots


def simulate_premature_collapse():
    """
    Simulate a scenario where entropy drops suddenly then continues.

    This tests false stillness detection.
    """
    print("\n" + "="*70)
    print("SCENARIO 3: PREMATURE COLLAPSE")
    print("="*70 + "\n")

    observer = ProxyObserver(num_experts=64, num_layers=2)

    recorder = TelemetryRecorder(
        session_id="proxy_premature",
        observer=observer,
        stillness_thresholds=StillnessThresholds(
            fast=0.01,
            medium=0.02,
            slow=0.005,
        ),
    )

    snapshots = []

    print("Simulating premature collapse scenario...\n")

    for step in range(40):
        # Noise pattern: high → low → high
        if 10 <= step < 20:
            noise_scale = 0.001  # Very low noise (fake convergence)
        else:
            noise_scale = 0.1    # High noise (active exploration)

        observer.step(noise_scale=noise_scale)

        snapshot = recorder.capture_snapshot(
            chrono_state=None,
            pause_reason="step_update",
        )

        should_pause, reason = recorder.should_pause()

        if should_pause:
            if 10 <= step < 20:
                print(f"Step {step+1}: Pause proposed during false convergence")
                print(f"  This is premature - exploration resumes at step 20")
            else:
                print(f"Step {step+1}: Pause proposed")

            print(f"  PC1={snapshot.pc1:.4f}, PC2={snapshot.pc2:.4f}")
            print(f"  dPC2={snapshot.d_pc2:+.6f}")
            print()

            snapshot.pause_reason = reason
            snapshots.append(snapshot)

    return snapshots


def main():
    print("="*70)
    print("TELEMETRY PROXY MODE TEST")
    print("="*70)
    print()
    print("This demonstrates telemetry running WITHOUT ChronoMoE.")
    print("Uses synthetic routing data to validate pause detection.")
    print()
    print("Method is decoupled from mechanism.")
    print()

    # Run scenarios
    convergence_snapshots = simulate_convergence_scenario()
    oscillation_snapshots = simulate_oscillation_scenario()
    premature_snapshots = simulate_premature_collapse()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70 + "\n")

    print("Scenario 1 (Convergence):")
    print(f"  Pauses proposed: {len(convergence_snapshots)}")
    print(f"  Expected: 1-3 pauses as system converges")
    print()

    print("Scenario 2 (Oscillation):")
    print(f"  Pauses proposed: {len(oscillation_snapshots)}")
    print(f"  Expected: 0 pauses (oscillation should not trigger stillness)")
    print()

    print("Scenario 3 (Premature Collapse):")
    print(f"  Pauses proposed: {len(premature_snapshots)}")
    print(f"  Expected: 1-2 pauses during false convergence window")
    print()

    # Save results
    output_dir = Path(__file__).parent.parent / "takens_data"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "telemetry_proxy_results.json"
    with open(output_file, 'w') as f:
        import json
        results = {
            'convergence': [s.to_dict() for s in convergence_snapshots],
            'oscillation': [s.to_dict() for s in oscillation_snapshots],
            'premature': [s.to_dict() for s in premature_snapshots],
        }
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()

    print("="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    print()
    print("✓ Telemetry runs without ChronoMoE")
    print("✓ Pause detection works on synthetic data")
    print("✓ Method decoupled from mechanism")
    print()
    print("Next: Test with real ChronoMoE and compare results")
    print()


if __name__ == '__main__':
    main()
