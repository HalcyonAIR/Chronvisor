"""
Longer Sequence Test

Test entropy and pressure evolution over extended generation.

Goals:
- See if entropy collapses over time (single-turn)
- Check if pressure patterns emerge
- Validate signal extraction at scale
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from chronomoe.clock_gated_multistep import ClockGatedMultistepModel, SessionMode
from chronomoe.chronovisor_mixtral_bridge import MixtralConfig


def run_long_sequence_test():
    """Run generation over longer sequences."""
    print("=" * 70)
    print("Longer Sequence Test: Entropy & Pressure Evolution")
    print("=" * 70)
    print()

    # Create model
    config = MixtralConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=512,
        num_layers=4,
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

    print("Test: Long sequence generation (50 chunks)")
    print("-" * 70)
    print()

    # Run long generation
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.reset_session()

    generated, telemetry = model.generate_multistep(
        input_ids,
        mode=SessionMode.SINGLE_TURN,
        chunk_size=10,  # Smaller chunks for more data points
        max_chunks=50,  # Many chunks to see evolution
        verbose=False,
    )

    print(f"Generated {telemetry.total_chunks} chunks")
    print(f"Total tokens: {telemetry.total_tokens}")
    print()

    # Extract time series
    entropies = [chunk.router_entropy for chunk in telemetry.chunks]
    pressures_net = [chunk.net_pressure for chunk in telemetry.chunks]
    pressures_fast = [chunk.fast_pressure for chunk in telemetry.chunks]
    pressures_mid = [chunk.mid_pressure for chunk in telemetry.chunks]
    pressures_slow = [chunk.slow_pressure for chunk in telemetry.chunks]
    residual_intent = [chunk.residual_intent for chunk in telemetry.chunks]

    # Analysis
    print("Time Series Analysis:")
    print("-" * 70)

    # Entropy
    print(f"Entropy:")
    print(f"  Initial (first 10 chunks): {np.mean(entropies[:10]):.4f}")
    print(f"  Final (last 10 chunks):    {np.mean(entropies[-10:]):.4f}")
    print(f"  Overall trend:             {np.polyfit(range(len(entropies)), entropies, 1)[0]:+.6f}")
    print()

    # Pressure
    print(f"Net Pressure:")
    print(f"  Initial: {np.mean(pressures_net[:10]):+.4f}")
    print(f"  Final:   {np.mean(pressures_net[-10:]):+.4f}")
    print(f"  Min:     {np.min(pressures_net):+.4f} (chunk {np.argmin(pressures_net)})")
    print(f"  Max:     {np.max(pressures_net):+.4f} (chunk {np.argmax(pressures_net)})")
    print()

    # Residual intent
    print(f"Residual Intent:")
    print(f"  Initial: {residual_intent[0]:.4f}")
    print(f"  Final:   {residual_intent[-1]:.4f}")
    print(f"  Peak:    {np.max(residual_intent):.4f} (chunk {np.argmax(residual_intent)})")
    print()

    # Check for late-stage collapse
    early_entropy = np.mean(entropies[:int(len(entropies) * 0.3)])
    late_entropy = np.mean(entropies[int(len(entropies) * 0.7):])
    entropy_collapse = early_entropy - late_entropy

    print(f"Entropy Collapse Check:")
    print(f"  Early stage (first 30%): {early_entropy:.4f}")
    print(f"  Late stage (last 30%):   {late_entropy:.4f}")
    print(f"  Collapse magnitude:      {entropy_collapse:+.4f}")

    if entropy_collapse > 0.05:
        print(f"  → Significant collapse detected!")
    elif entropy_collapse < -0.05:
        print(f"  → Entropy increasing (unusual)")
    else:
        print(f"  → Entropy relatively stable")
    print()

    # Visualize
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Entropy over time
    ax = axes[0, 0]
    ax.plot(entropies, marker=".", alpha=0.6, markersize=3)
    ax.axhline(y=early_entropy, color="green", linestyle="--", alpha=0.5, label="Early mean")
    ax.axhline(y=late_entropy, color="red", linestyle="--", alpha=0.5, label="Late mean")
    ax.set_xlabel("Chunk Index")
    ax.set_ylabel("Router Entropy")
    ax.set_title("Entropy Evolution Over Time")
    ax.legend()
    ax.grid(alpha=0.3)

    # Net pressure over time
    ax = axes[0, 1]
    ax.plot(pressures_net, marker=".", alpha=0.6, markersize=3, color="purple")
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax.set_xlabel("Chunk Index")
    ax.set_ylabel("Net Pressure")
    ax.set_title("Net Pressure Evolution")
    ax.grid(alpha=0.3)

    # Pressure components
    ax = axes[1, 0]
    ax.plot(pressures_fast, marker="^", alpha=0.5, markersize=2, label="Fast")
    ax.plot(pressures_mid, marker="o", alpha=0.5, markersize=2, label="Mid")
    ax.plot(pressures_slow, marker="s", alpha=0.5, markersize=2, label="Slow")
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax.set_xlabel("Chunk Index")
    ax.set_ylabel("Pressure")
    ax.set_title("Pressure Components")
    ax.legend()
    ax.grid(alpha=0.3)

    # Residual intent
    ax = axes[1, 1]
    ax.plot(residual_intent, marker=".", alpha=0.6, markersize=3, color="orange")
    ax.set_xlabel("Chunk Index")
    ax.set_ylabel("Residual Intent")
    ax.set_title("Residual Intent Evolution")
    ax.grid(alpha=0.3)

    # Entropy vs Net Pressure
    ax = axes[2, 0]
    ax.scatter(entropies, pressures_net, alpha=0.4, s=10)
    ax.set_xlabel("Router Entropy")
    ax.set_ylabel("Net Pressure")
    ax.set_title("Entropy vs Pressure Correlation")
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax.grid(alpha=0.3)

    # Histogram of entropies
    ax = axes[2, 1]
    ax.hist(entropies, bins=30, alpha=0.7, edgecolor="black")
    ax.axvline(x=early_entropy, color="green", linestyle="--", label="Early mean")
    ax.axvline(x=late_entropy, color="red", linestyle="--", label="Late mean")
    ax.set_xlabel("Router Entropy")
    ax.set_ylabel("Frequency")
    ax.set_title("Entropy Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    plot_file = output_dir / "long_sequence_analysis.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"✓ Plots saved to {plot_file}")
    print()

    # Save data
    results = {
        "test": "long_sequence_evolution",
        "config": {
            "chunk_size": 10,
            "total_chunks": telemetry.total_chunks,
            "total_tokens": telemetry.total_tokens,
        },
        "analysis": {
            "entropy": {
                "early": float(early_entropy),
                "late": float(late_entropy),
                "collapse": float(entropy_collapse),
                "trend": float(np.polyfit(range(len(entropies)), entropies, 1)[0]),
            },
            "pressure": {
                "initial": float(np.mean(pressures_net[:10])),
                "final": float(np.mean(pressures_net[-10:])),
                "min": float(np.min(pressures_net)),
                "max": float(np.max(pressures_net)),
            },
            "residual_intent": {
                "initial": float(residual_intent[0]),
                "final": float(residual_intent[-1]),
                "peak": float(np.max(residual_intent)),
            },
        },
        "time_series": {
            "entropies": entropies,
            "pressures_net": pressures_net,
            "pressures_fast": pressures_fast,
            "pressures_mid": pressures_mid,
            "pressures_slow": pressures_slow,
            "residual_intent": residual_intent,
        },
    }

    log_file = output_dir / "long_sequence_results.json"
    with open(log_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to {log_file}")
    print()

    # Interpretation
    print("=" * 70)
    print("Interpretation:")
    print("-" * 70)

    if entropy_collapse > 0.05:
        print("⚠ Entropy collapse detected in late stage")
        print("  This suggests router convergence (potential monopoly)")
        print("  Multistep mode should reduce this effect")
    else:
        print("✓ Entropy relatively stable across generation")
        print("  Router maintains diversity throughout")

    print()
    print("Next steps:")
    print("  1. Run same test in multistep mode (with continuation)")
    print("  2. Compare entropy collapse between modes")
    print("  3. Check if multistep reduces late-stage convergence")
    print()

    print("=" * 70)


if __name__ == "__main__":
    run_long_sequence_test()
