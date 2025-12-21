"""
Pressure Half-Life Measurement: Full Mixtral-8x7B

Measures natural pressure decay on real Mixtral model with semantic prompts.

This is the critical test per Halcyon's guidance:
> "Next concrete move: measure pressure half-life first, before any tuning,
>  on Mixtral alone. DeepSeek comes after, not as a crutch but as a contrast."

Differences from toy model test:
1. Uses pre-trained Mixtral-8x7B (real weights)
2. Uses real explanatory prompts (not random tokens)
3. Real routing behavior (semantic convergence)
4. May show natural pressure decay (vs infinite half-life on toy)

Protocol:
- Zero semantic perturbation (pure continuation)
- Measure mid-pressure decay over N chunks
- Compute half-life and decay shape
- Compare to toy baseline
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chronomoe.external_mixtral_adapter import ExternalMixtralAdapter, ExternalMixtralConfig
from chronomoe.session_controller import SessionMode


class MixtralHalfLifeMeasurement:
    """Measures pressure half-life on full Mixtral."""

    def __init__(self, model_name: str = "mistralai/Mixtral-8x7B-v0.1", load_in_8bit: bool = True):
        print("=" * 70)
        print("Mixtral Pressure Half-Life Measurement")
        print("=" * 70)
        print()

        # Create config
        config = ExternalMixtralConfig(
            model_name=model_name,
            enable_chronovisor=True,
            enable_clock_heads=True,
            load_in_8bit=load_in_8bit,
        )

        # Load model
        print("Loading Mixtral model...")
        print(f"  Model: {model_name}")
        print(f"  8-bit quantization: {load_in_8bit}")
        print()

        self.adapter = ExternalMixtralAdapter(config)
        print()
        print("✓ Model loaded successfully")
        print()

    def measure_natural_decay(
        self,
        prompt: str,
        max_chunks: int = 50,
        chunk_size: int = 20,
        seed: int = 42,
    ):
        """
        Measure pressure decay with zero semantic perturbation.

        Args:
            prompt: Initial prompt (explanatory task)
            max_chunks: How many chunks to generate
            chunk_size: Tokens per chunk
            seed: Random seed

        Returns:
            dict with trajectory and analysis
        """
        print("Protocol: Natural Decay (Zero Perturbation)")
        print("-" * 70)
        print()
        print(f"Prompt: {prompt}")
        print()
        print(f"Configuration:")
        print(f"  Chunk size: {chunk_size} tokens")
        print(f"  Max chunks: {max_chunks}")
        print(f"  Seed: {seed}")
        print()

        # Tokenize prompt
        input_ids = self.adapter.tokenize(prompt)
        print(f"Input length: {input_ids.shape[1]} tokens")
        print()

        # Set seed
        import torch

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Reset session
        self.adapter.reset_session()

        print("Generating chunks...")
        print()

        # Generate with multistep mode
        generated, telemetry = self.adapter.generate_multistep(
            input_ids,
            mode=SessionMode.MULTISTEP,
            chunk_size=chunk_size,
            max_chunks=max_chunks,
            verbose=True,
        )

        print()
        print(f"✓ Completed {telemetry.total_chunks} chunks")
        print()

        # Extract trajectory
        mid_pressures = [chunk.mid_pressure for chunk in telemetry.chunks]
        net_pressures = [chunk.net_pressure for chunk in telemetry.chunks]
        fast_pressures = [chunk.fast_pressure for chunk in telemetry.chunks]
        slow_pressures = [chunk.slow_pressure for chunk in telemetry.chunks]
        entropies = [chunk.router_entropy for chunk in telemetry.chunks]
        residual_intents = [chunk.residual_intent for chunk in telemetry.chunks]

        # Analyze decay
        analysis = self._analyze_decay(mid_pressures)

        # Decode generated text
        generated_text = self.adapter.decode(generated)

        return {
            "protocol": "natural_decay",
            "model": self.adapter.config.model_name,
            "prompt": prompt,
            "config": {
                "chunk_size": chunk_size,
                "max_chunks": max_chunks,
                "total_chunks": telemetry.total_chunks,
                "total_tokens": telemetry.total_tokens,
                "seed": seed,
            },
            "trajectory": {
                "chunk_indices": list(range(len(mid_pressures))),
                "mid_pressures": mid_pressures,
                "net_pressures": net_pressures,
                "fast_pressures": fast_pressures,
                "slow_pressures": slow_pressures,
                "entropies": entropies,
                "residual_intents": residual_intents,
            },
            "analysis": analysis,
            "generated_text": generated_text,
        }

    def _analyze_decay(self, mid_pressures):
        """Analyze pressure decay shape and half-life."""
        mid_pressures = np.array(mid_pressures)

        p_initial = mid_pressures[0]
        p_final = mid_pressures[-1]
        p_delta = p_final - p_initial

        # Half-life
        p_half = p_initial * 0.5
        half_life_idx = None
        for i, p in enumerate(mid_pressures):
            if p <= p_half:
                half_life_idx = i
                break

        # Linear fit
        chunks = np.arange(len(mid_pressures))
        linear_slope, linear_intercept = np.polyfit(chunks, mid_pressures, 1)

        # Exponential fit (if possible)
        try:
            abs_pressures = np.abs(mid_pressures - mid_pressures.min()) + 0.01
            log_pressures = np.log(abs_pressures)
            exp_slope, exp_intercept = np.polyfit(chunks, log_pressures, 1)
            tau = -1.0 / exp_slope if exp_slope < 0 else None
        except:
            tau = None

        # Detect cliffs
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

    def plot_comparison(self, mixtral_results, toy_results, output_file=None):
        """Plot Mixtral vs toy model comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Extract trajectories
        m_chunks = mixtral_results["trajectory"]["chunk_indices"]
        m_mid = mixtral_results["trajectory"]["mid_pressures"]
        m_entropy = mixtral_results["trajectory"]["entropies"]

        t_chunks = toy_results["trajectory"]["chunk_indices"]
        t_mid = toy_results["trajectory"]["mid_pressures"]
        t_entropy = toy_results["trajectory"]["entropies"]

        # Mid-pressure comparison
        ax = axes[0, 0]
        ax.plot(m_chunks, m_mid, marker="o", markersize=4, label="Mixtral-8x7B", alpha=0.7)
        ax.plot(t_chunks, t_mid, marker="s", markersize=3, label="Toy model", alpha=0.7)
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax.set_xlabel("Chunk Index")
        ax.set_ylabel("Mid Pressure")
        ax.set_title("Mid-Pressure: Mixtral vs Toy")
        ax.legend()
        ax.grid(alpha=0.3)

        # Half-life markers
        m_hl = mixtral_results["analysis"]["half_life_chunks"]
        t_hl = toy_results["analysis"]["half_life_chunks"]
        if m_hl is not None:
            ax.axvline(x=m_hl, color="blue", linestyle="--", alpha=0.5)
            ax.text(m_hl, ax.get_ylim()[1] * 0.9, f"Mixtral HL={m_hl}", rotation=90)
        if t_hl is not None:
            ax.axvline(x=t_hl, color="orange", linestyle="--", alpha=0.5)
            ax.text(t_hl, ax.get_ylim()[1] * 0.8, f"Toy HL={t_hl}", rotation=90)

        # Entropy comparison
        ax = axes[0, 1]
        ax.plot(m_chunks, m_entropy, marker="o", markersize=4, label="Mixtral-8x7B", alpha=0.7)
        ax.plot(t_chunks, t_entropy, marker="s", markersize=3, label="Toy model", alpha=0.7)
        ax.set_xlabel("Chunk Index")
        ax.set_ylabel("Router Entropy")
        ax.set_title("Entropy: Mixtral vs Toy")
        ax.legend()
        ax.grid(alpha=0.3)

        # Pressure decay rate
        ax = axes[1, 0]
        m_slope = mixtral_results["analysis"]["decay_shape"]["linear_slope"]
        t_slope = toy_results["analysis"]["decay_shape"]["linear_slope"]
        ax.bar(
            ["Mixtral-8x7B", "Toy model"],
            [m_slope, t_slope],
            color=["blue", "orange"],
            alpha=0.7,
        )
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)
        ax.set_ylabel("Decay Rate (pressure/chunk)")
        ax.set_title("Pressure Decay Rate Comparison")
        ax.grid(alpha=0.3, axis="y")

        # Half-life comparison
        ax = axes[1, 1]
        half_lives = []
        labels = []
        if m_hl is not None:
            half_lives.append(m_hl)
            labels.append("Mixtral-8x7B")
        else:
            half_lives.append(50)  # Max chunks (not reached)
            labels.append("Mixtral-8x7B\n(not reached)")

        if t_hl is not None:
            half_lives.append(t_hl)
            labels.append("Toy model")
        else:
            half_lives.append(50)
            labels.append("Toy model\n(not reached)")

        ax.bar(labels, half_lives, color=["blue", "orange"], alpha=0.7)
        ax.set_ylabel("Half-Life (chunks)")
        ax.set_title("Pressure Half-Life Comparison")
        ax.grid(alpha=0.3, axis="y")

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            print(f"✓ Comparison plot saved to {output_file}")

        return fig

    def report_findings(self, results, toy_baseline=None):
        """Report findings and compare to toy baseline."""
        analysis = results["analysis"]

        print("=" * 70)
        print("Mixtral Pressure Half-Life Analysis")
        print("-" * 70)
        print()

        print("Model:", results["model"])
        print("Prompt:", results["prompt"])
        print()

        print("Pressure Trajectory:")
        print(f"  Initial: {analysis['initial_pressure']:+.4f}")
        print(f"  Final:   {analysis['final_pressure']:+.4f}")
        print(f"  Decay:   {analysis['total_decay']:+.4f}")
        print()

        print("Half-Life:")
        if analysis["half_life_chunks"] is not None:
            hl = analysis["half_life_chunks"]
            print(f"  {hl} chunks")
        else:
            print(f"  Not reached in {results['config']['max_chunks']} chunks")
        print()

        print("Decay Shape:")
        shape = analysis["decay_shape"]
        print(f"  Linear slope:    {shape['linear_slope']:+.6f} per chunk")
        if shape["exponential_tau"]:
            print(f"  Exponential tau: {shape['exponential_tau']:.2f} chunks")
        print(f"  Max single drop: {shape['max_single_drop']:+.4f}")
        print()

        # Compare to toy baseline
        if toy_baseline:
            print("Comparison to Toy Baseline:")
            print("-" * 70)

            toy_analysis = toy_baseline["analysis"]

            m_hl = analysis["half_life_chunks"]
            t_hl = toy_analysis["half_life_chunks"]

            print(f"  Half-life:")
            print(f"    Mixtral: {m_hl if m_hl else 'Not reached'}")
            print(f"    Toy:     {t_hl if t_hl else 'Not reached'}")

            m_slope = shape["linear_slope"]
            t_slope = toy_analysis["decay_shape"]["linear_slope"]

            print(f"  Decay rate:")
            print(f"    Mixtral: {m_slope:+.6f} per chunk")
            print(f"    Toy:     {t_slope:+.6f} per chunk")
            print(f"    Ratio:   {abs(m_slope/t_slope) if t_slope != 0 else 'inf'}x")
            print()

            if m_hl and not t_hl:
                print("  → Mixtral shows natural decay, toy model does not")
            elif not m_hl and not t_hl:
                print("  → Both models show infinite half-life")
            elif m_hl and t_hl:
                if m_hl < t_hl:
                    print(f"  → Mixtral decays faster ({m_hl} vs {t_hl} chunks)")
                else:
                    print(f"  → Toy decays faster ({t_hl} vs {m_hl} chunks)")

        print()
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Measure Mixtral pressure half-life")
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Mixtral model name",
    )
    parser.add_argument("--chunks", type=int, default=50, help="Max chunks to generate")
    parser.add_argument("--chunk-size", type=int, default=20, help="Tokens per chunk")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--8bit", action="store_true", help="Load in 8-bit mode")
    parser.add_argument(
        "--toy-baseline",
        type=str,
        default="test_results/pressure_halflife_natural.json",
        help="Path to toy baseline results",
    )

    args = parser.parse_args()

    # Explanatory prompt (per Halcyon: use real semantic content)
    prompt = """Explain how photosynthesis works in plants, covering the light-dependent and light-independent reactions:"""

    # Create measurement
    measurement = MixtralHalfLifeMeasurement(model_name=args.model, load_in_8bit=args.8bit)

    # Measure decay
    results = measurement.measure_natural_decay(
        prompt=prompt, max_chunks=args.chunks, chunk_size=args.chunk_size, seed=args.seed
    )

    # Save results
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / "mixtral_pressure_halflife.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {results_file}")
    print()

    # Load toy baseline
    toy_baseline = None
    if Path(args.toy_baseline).exists():
        with open(args.toy_baseline) as f:
            toy_baseline = json.load(f)
        print(f"✓ Loaded toy baseline from {args.toy_baseline}")
        print()

    # Plot comparison
    if toy_baseline:
        plot_file = output_dir / "mixtral_vs_toy_comparison.png"
        measurement.plot_comparison(results, toy_baseline, output_file=plot_file)
        print()

    # Report
    measurement.report_findings(results, toy_baseline)

    # Show generated text
    print()
    print("Generated Text:")
    print("-" * 70)
    print(results["generated_text"])
    print("-" * 70)


if __name__ == "__main__":
    main()
