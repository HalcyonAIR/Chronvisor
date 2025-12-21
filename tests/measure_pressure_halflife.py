"""
Pressure Half-Life Measurement

Measures natural pressure decay dynamics in multistep mode.

Per Halcyon's guidance:
- "You're not testing 'longer sequences' yet. You're testing pressure half-life."
- "Take a single prompt. Run multistep. Do not intervene. Just let it pause.
   Measure how mid-pressure decays across chunks if you keep allowing 'continue'
   with zero semantic perturbation. That gives you a natural decay curve."

This is NOT about answer quality.
This is about: "How much force does it take to keep it thinking?"

Goals:
1. Measure natural mid-pressure decay (no perturbation)
2. Compute half-life (chunks until pressure drops 50%)
3. Identify decay shape (exponential, linear, cliff?)
4. Log trajectory for Mixtral baseline

Future:
- Add neutral perturbation ("go on" equivalent)
- Measure delta (external energy required)
- Compare Mixtral vs DeepSeek half-lives
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from chronomoe.clock_gated_multistep import ClockGatedMultistepModel, SessionMode
from chronomoe.chronovisor_mixtral_bridge import MixtralConfig


class PressureHalfLifeMeasurement:
    """Measures natural pressure decay dynamics."""

    def __init__(self, config):
        self.model = ClockGatedMultistepModel(config)
        self.config = config

    def measure_natural_decay(
        self, input_ids: torch.Tensor, max_chunks: int = 50, chunk_size: int = 10, seed: int = 42
    ):
        """
        Measure pressure decay with zero semantic perturbation.

        Each chunk is pure continuation - no intervention, no modification.
        We measure how pressure decays naturally.

        Args:
            input_ids: Initial prompt
            max_chunks: How many chunks to run
            chunk_size: Tokens per chunk
            seed: Random seed for reproducibility

        Returns:
            dict with pressure trajectory and half-life analysis
        """
        print("=" * 70)
        print("Pressure Half-Life Measurement: Natural Decay")
        print("=" * 70)
        print()
        print("Protocol:")
        print("  - Zero semantic perturbation")
        print("  - Pure continuation at each pause")
        print("  - Measure mid-pressure decay")
        print()
        print(f"Configuration:")
        print(f"  Input length: {input_ids.shape[1]} tokens")
        print(f"  Chunk size: {chunk_size} tokens")
        print(f"  Max chunks: {max_chunks}")
        print(f"  Seed: {seed}")
        print()

        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Reset session
        self.model.reset_session()

        # Track trajectory
        mid_pressures = []
        net_pressures = []
        fast_pressures = []
        slow_pressures = []
        entropies = []
        residual_intents = []
        chunk_indices = []

        current_ids = input_ids.clone()

        # Generate chunks with pure continuation
        for chunk_idx in range(max_chunks):
            # Generate one chunk
            generated, telemetry = self.model.generate_multistep(
                current_ids,
                mode=SessionMode.MULTISTEP,
                chunk_size=chunk_size,
                max_chunks=1,  # One chunk at a time
                verbose=False,
            )

            # Update for next iteration (zero perturbation - just continue)
            current_ids = generated

            # Extract signals from this chunk
            chunk = telemetry.chunks[0]
            mid_pressures.append(chunk.mid_pressure)
            net_pressures.append(chunk.net_pressure)
            fast_pressures.append(chunk.fast_pressure)
            slow_pressures.append(chunk.slow_pressure)
            entropies.append(chunk.router_entropy)
            residual_intents.append(chunk.residual_intent)
            chunk_indices.append(chunk_idx)

            # Progress
            if (chunk_idx + 1) % 10 == 0:
                print(
                    f"  Chunk {chunk_idx + 1}/{max_chunks}: "
                    f"mid_pressure = {chunk.mid_pressure:+.4f}, "
                    f"entropy = {chunk.router_entropy:.4f}"
                )

        print()
        print(f"✓ Completed {max_chunks} chunks")
        print()

        # Analyze decay
        analysis = self._analyze_decay(mid_pressures)

        # Return trajectory
        return {
            "protocol": "natural_decay",
            "perturbation": "zero",
            "config": {
                "chunk_size": chunk_size,
                "max_chunks": max_chunks,
                "total_tokens": int(current_ids.shape[1]),
                "seed": seed,
            },
            "trajectory": {
                "chunk_indices": chunk_indices,
                "mid_pressures": mid_pressures,
                "net_pressures": net_pressures,
                "fast_pressures": fast_pressures,
                "slow_pressures": slow_pressures,
                "entropies": entropies,
                "residual_intents": residual_intents,
            },
            "analysis": analysis,
        }

    def _analyze_decay(self, mid_pressures):
        """
        Analyze pressure decay shape.

        Returns half-life, decay shape, and trajectory stats.
        """
        mid_pressures = np.array(mid_pressures)

        # Initial and final
        p_initial = mid_pressures[0]
        p_final = mid_pressures[-1]
        p_delta = p_final - p_initial

        # Half-life: when does pressure drop to 50% of initial?
        p_half = p_initial * 0.5
        half_life_idx = None
        for i, p in enumerate(mid_pressures):
            if p <= p_half:
                half_life_idx = i
                break

        # Decay shape: fit exponential vs linear
        chunks = np.arange(len(mid_pressures))

        # Linear fit
        linear_slope, linear_intercept = np.polyfit(chunks, mid_pressures, 1)

        # Try exponential fit: p(t) = p0 * exp(-t / tau)
        # log(p) = log(p0) - t/tau
        # If pressures go negative, can't fit exponential directly
        # Use absolute value for fitting shape
        abs_pressures = np.abs(mid_pressures - mid_pressures.min()) + 0.01
        try:
            log_pressures = np.log(abs_pressures)
            exp_slope, exp_intercept = np.polyfit(chunks, log_pressures, 1)
            tau = -1.0 / exp_slope if exp_slope < 0 else None
        except:
            tau = None

        # Detect cliffs: large single-chunk drops
        deltas = np.diff(mid_pressures)
        max_drop_idx = int(np.argmin(deltas))
        max_drop = float(deltas[max_drop_idx])

        return {
            "initial_pressure": float(p_initial),
            "final_pressure": float(p_final),
            "total_decay": float(p_delta),
            "half_life_chunks": half_life_idx,
            "decay_shape": {
                "linear_slope": float(linear_slope),
                "exponential_tau": float(tau) if tau else None,
                "max_single_drop": float(max_drop),
                "max_drop_at_chunk": int(max_drop_idx),
            },
            "statistics": {
                "mean": float(np.mean(mid_pressures)),
                "std": float(np.std(mid_pressures)),
                "min": float(np.min(mid_pressures)),
                "max": float(np.max(mid_pressures)),
            },
        }

    def plot_decay(self, results, output_file=None):
        """Plot pressure decay trajectory."""
        trajectory = results["trajectory"]
        analysis = results["analysis"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        chunks = trajectory["chunk_indices"]
        mid_pressures = trajectory["mid_pressures"]
        net_pressures = trajectory["net_pressures"]
        fast_pressures = trajectory["fast_pressures"]
        slow_pressures = trajectory["slow_pressures"]
        entropies = trajectory["entropies"]

        # Mid-pressure decay
        ax = axes[0, 0]
        ax.plot(chunks, mid_pressures, marker="o", markersize=4, label="Mid pressure")
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # Mark half-life
        if analysis["half_life_chunks"] is not None:
            half_idx = analysis["half_life_chunks"]
            ax.axvline(x=half_idx, color="red", linestyle="--", alpha=0.5, label=f"Half-life ({half_idx} chunks)")
            ax.scatter([half_idx], [mid_pressures[half_idx]], color="red", s=100, zorder=5)

        # Mark initial
        ax.scatter([0], [mid_pressures[0]], color="green", s=100, zorder=5, label="Initial")

        ax.set_xlabel("Chunk Index")
        ax.set_ylabel("Mid Pressure")
        ax.set_title("Mid-Pressure Decay (Natural)")
        ax.legend()
        ax.grid(alpha=0.3)

        # Net pressure
        ax = axes[0, 1]
        ax.plot(chunks, net_pressures, marker=".", markersize=3, color="purple", alpha=0.7)
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax.set_xlabel("Chunk Index")
        ax.set_ylabel("Net Pressure")
        ax.set_title("Net Pressure Evolution")
        ax.grid(alpha=0.3)

        # Pressure components
        ax = axes[1, 0]
        ax.plot(chunks, fast_pressures, marker="^", markersize=2, alpha=0.6, label="Fast")
        ax.plot(chunks, mid_pressures, marker="o", markersize=2, alpha=0.6, label="Mid")
        ax.plot(chunks, slow_pressures, marker="s", markersize=2, alpha=0.6, label="Slow")
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax.set_xlabel("Chunk Index")
        ax.set_ylabel("Pressure")
        ax.set_title("Pressure Components")
        ax.legend()
        ax.grid(alpha=0.3)

        # Entropy
        ax = axes[1, 1]
        ax.plot(chunks, entropies, marker=".", markersize=3, color="orange", alpha=0.7)
        ax.set_xlabel("Chunk Index")
        ax.set_ylabel("Router Entropy")
        ax.set_title("Entropy Evolution")
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            print(f"✓ Plot saved to {output_file}")

        return fig

    def report_findings(self, results):
        """Report half-life findings."""
        analysis = results["analysis"]
        config = results["config"]

        print("=" * 70)
        print("Half-Life Analysis")
        print("-" * 70)
        print()

        print("Pressure Trajectory:")
        print(f"  Initial: {analysis['initial_pressure']:+.4f}")
        print(f"  Final:   {analysis['final_pressure']:+.4f}")
        print(f"  Decay:   {analysis['total_decay']:+.4f}")
        print()

        print("Half-Life:")
        if analysis["half_life_chunks"] is not None:
            hl = analysis["half_life_chunks"]
            print(f"  {hl} chunks (pressure dropped to 50% of initial)")
        else:
            print(f"  Not reached (pressure didn't drop to 50%)")
        print()

        print("Decay Shape:")
        shape = analysis["decay_shape"]
        print(f"  Linear slope:        {shape['linear_slope']:+.6f} per chunk")
        if shape["exponential_tau"]:
            print(f"  Exponential tau:     {shape['exponential_tau']:.2f} chunks")
        print(f"  Max single drop:     {shape['max_single_drop']:+.4f} (chunk {shape['max_drop_at_chunk']})")
        print()

        # Interpretation
        print("Interpretation:")
        if shape["linear_slope"] > -0.001:
            print("  → Pressure is stable (minimal decay)")
            print("  → System maintains intrinsic continuation pressure")
        elif abs(shape["max_single_drop"]) > 0.1 and abs(shape["max_single_drop"]) > abs(shape["linear_slope"]) * 5:
            print("  → Cliff-like decay (sharp drop)")
            print("  → System loses pressure suddenly, not gradually")
        elif shape["exponential_tau"]:
            print(f"  → Exponential decay (tau ≈ {shape['exponential_tau']:.1f} chunks)")
            print("  → Classic half-life behavior")
        else:
            print("  → Linear decay")
            print("  → Pressure drains steadily")
        print()

        print("Next Steps:")
        print("  1. Run with neutral perturbation ('go on' equivalent)")
        print("  2. Measure delta (external energy required)")
        print("  3. Run on full Mixtral (current: toy model)")
        print("  4. Compare Mixtral vs DeepSeek half-lives")
        print()
        print("=" * 70)


def main():
    """Run pressure half-life measurement."""
    # Create toy model (same as other tests)
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
        max_seq_length=1024,
        enable_chronovisor=True,
    )

    measurement = PressureHalfLifeMeasurement(config)

    # Initial prompt (random tokens for now)
    input_ids = torch.randint(0, config.vocab_size, (1, 10))

    print("Measuring natural pressure decay...")
    print("(Pure continuation, zero perturbation)")
    print()

    # Measure decay
    results = measurement.measure_natural_decay(
        input_ids,
        max_chunks=50,
        chunk_size=10,
        seed=42,
    )

    # Save results
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / "pressure_halflife_natural.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {results_file}")
    print()

    # Plot
    plot_file = output_dir / "pressure_halflife_natural.png"
    measurement.plot_decay(results, output_file=plot_file)
    print()

    # Report
    measurement.report_findings(results)


if __name__ == "__main__":
    main()
