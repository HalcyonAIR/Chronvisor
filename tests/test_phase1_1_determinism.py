"""
Phase 1.1: Deterministic Pause Behavior

Tests that pause decisions are deterministic and occur only under sanctioned conditions.

Success criteria:
- 3 identical runs (same seed) → identical pause locations
- pause_reason always matches at least one condition
- No pauses when all conditions False

What we ignore:
- Absolute pressure values
- Clock scores
- Token quality
"""

import json
from pathlib import Path

import numpy as np
import torch

from chronomoe.clock_gated_multistep import ClockGatedMultistepModel, SessionMode
from chronomoe.chronovisor_mixtral_bridge import MixtralConfig


def run_generation_with_logging(model, input_ids, seed, run_id, mode):
    """Run generation and log pause events."""
    # Set seeds for determinism
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Reset model state
    model.reset_session()

    # Generate
    generated, telemetry = model.generate_multistep(
        input_ids,
        mode=mode,
        chunk_size=20,
        max_chunks=3,
        verbose=False,
    )

    # Extract minimal pause data
    pause_events = []
    for chunk in telemetry.chunks:
        # Check which conditions are met
        conditions_met = {
            "fast_instability": chunk.fast_pressure < -0.7,
            "negative_pressure": chunk.net_pressure < 0,
            "multistep_boundary": mode == SessionMode.MULTISTEP,
        }

        pause_events.append({
            "run_id": run_id,
            "seed": seed,
            "chunk_index": chunk.chunk_index,
            "pause_occurred": chunk.should_pause,
            "pause_reason": chunk.pause_reason,
            "net_pressure": round(chunk.net_pressure, 4),
            "fast_pressure": round(chunk.fast_pressure, 4),
            "mid_pressure": round(chunk.mid_pressure, 4),
            "slow_pressure": round(chunk.slow_pressure, 4),
            "conditions_met": conditions_met,
        })

    return pause_events, telemetry


def test_determinism():
    """Test that identical runs produce identical pause behavior."""
    print("=" * 70)
    print("Phase 1.1: Deterministic Pause Behavior")
    print("=" * 70)
    print()

    # Create small model
    config = MixtralConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=512,
        num_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=4,
        num_experts_per_token=2,
        max_seq_length=512,
        enable_chronovisor=True,
    )

    model = ClockGatedMultistepModel(config)

    # Fixed input
    seed = 42
    input_ids = torch.randint(0, config.vocab_size, (1, 10))

    print("Test 1: Multistep mode - pauses at chunk boundaries")
    print("-" * 70)

    # Run 3 times with same seed
    all_pause_events = []
    for run_id in range(3):
        pause_events, _ = run_generation_with_logging(
            model, input_ids, seed, run_id, SessionMode.MULTISTEP
        )
        all_pause_events.append(pause_events)

        print(f"\nRun {run_id + 1}:")
        for event in pause_events:
            if event["pause_occurred"]:
                print(f"  Chunk {event['chunk_index']}: "
                      f"PAUSE ({event['pause_reason']}) - "
                      f"net_pressure={event['net_pressure']:+.4f}")

    # Verify determinism
    print("\n" + "-" * 70)
    print("Determinism check:")

    # All runs should have same number of chunks
    chunk_counts = [len(events) for events in all_pause_events]
    print(f"  Chunk counts: {chunk_counts}")
    assert len(set(chunk_counts)) == 1, "Different chunk counts across runs!"

    # All runs should pause at same locations with same reasons
    for chunk_idx in range(len(all_pause_events[0])):
        pause_reasons = [
            events[chunk_idx]["pause_reason"]
            for events in all_pause_events
        ]
        pause_states = [
            events[chunk_idx]["pause_occurred"]
            for events in all_pause_events
        ]

        print(f"  Chunk {chunk_idx}: pause={pause_states[0]}, reason={pause_reasons[0]}")

        assert len(set(pause_reasons)) == 1, f"Different pause reasons at chunk {chunk_idx}!"
        assert len(set(pause_states)) == 1, f"Different pause states at chunk {chunk_idx}!"

    print("  ✓ All runs identical")

    # Verify pause reasons match conditions
    print("\n" + "-" * 70)
    print("Condition validation:")

    for event in all_pause_events[0]:  # Check first run
        if event["pause_occurred"]:
            conditions = event["conditions_met"]
            reason = event["pause_reason"]

            # Check that at least one condition is met
            any_condition_met = any(conditions.values())
            assert any_condition_met, f"Pause without any condition met: {event}"

            # Check that reason matches a met condition
            if reason == "fast_instability":
                assert conditions["fast_instability"], "fast_instability reason but condition False"
            elif reason == "negative_pressure":
                assert conditions["negative_pressure"], "negative_pressure reason but condition False"
            elif reason == "multistep_chunk_complete":
                assert conditions["multistep_boundary"], "multistep_chunk_complete but mode not multistep"

            print(f"  Chunk {event['chunk_index']}: {reason} ✓")

    print("\n" + "=" * 70)
    print("Test 2: Single-turn mode - no multistep pauses")
    print("-" * 70)

    # Run in single-turn mode (should not pause at chunk boundaries)
    pause_events, telemetry = run_generation_with_logging(
        model, input_ids, seed, 0, SessionMode.SINGLE_TURN
    )

    multistep_pauses = [
        e for e in pause_events
        if e["pause_reason"] == "multistep_chunk_complete"
    ]

    print(f"  Total chunks: {len(pause_events)}")
    print(f"  Multistep pauses: {len(multistep_pauses)}")
    assert len(multistep_pauses) == 0, "Multistep pauses in single-turn mode!"
    print("  ✓ No multistep pauses in single-turn mode")

    print("\n" + "=" * 70)
    print("Test 3: No auto-continuation in multistep mode")
    print("-" * 70)

    # In multistep mode, should pause after first chunk
    pause_events, telemetry = run_generation_with_logging(
        model, input_ids, seed, 0, SessionMode.MULTISTEP
    )

    # Should have paused (total_chunks < max_chunks)
    print(f"  Max chunks allowed: 3")
    print(f"  Chunks generated: {telemetry.total_chunks}")
    print(f"  First pause reason: {pause_events[0]['pause_reason']}")

    # In multistep, should pause after first chunk
    assert telemetry.total_chunks <= 1, "Auto-continued beyond first chunk!"
    assert pause_events[0]["pause_reason"] == "multistep_chunk_complete", \
        "First pause not due to multistep boundary!"

    print("  ✓ Paused after first chunk (no auto-continuation)")

    # Save logs
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)

    log_file = output_dir / "phase1_1_determinism.json"
    with open(log_file, "w") as f:
        json.dump(
            {
                "test": "phase1_1_determinism",
                "status": "PASS",
                "runs": all_pause_events,
            },
            f,
            indent=2,
        )

    print(f"\n✓ Logs saved to {log_file}")

    print("\n" + "=" * 70)
    print("Phase 1.1: PASS")
    print("=" * 70)
    print("\nKey findings:")
    print("  - Pause decisions are deterministic (3 identical runs)")
    print("  - Pauses occur only under sanctioned conditions")
    print("  - No auto-continuation in multistep mode")
    print("  - Pause reasons always match met conditions")
    print()


if __name__ == "__main__":
    test_determinism()
