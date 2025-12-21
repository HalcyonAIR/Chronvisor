"""
Adaptive Phase 2 Test Runner

Instead of flat scripts, this provides a conversational interface for
exploring Phase 2 behavior.

Can be used:
1. Interactively (respond to what's found)
2. With agents (Task tool for exploration)
3. Adaptively (adjust based on results)

Not a rigid test - a flexible investigation framework.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from chronomoe.clock_gated_multistep import ClockGatedMultistepModel, SessionMode
from chronomoe.chronovisor_mixtral_bridge import MixtralConfig


class AdaptivePhase2Runner:
    """
    Adaptive test runner for Phase 2.

    Can explore different scenarios based on what it finds.
    """

    def __init__(self, config: Optional[MixtralConfig] = None):
        if config is None:
            # Default small config for testing
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

        self.config = config
        self.model = ClockGatedMultistepModel(config)
        self.results = []
        self.output_dir = Path("test_results")
        self.output_dir.mkdir(exist_ok=True)

    def investigate_entropy_behavior(
        self,
        input_ids: torch.Tensor,
        mode: SessionMode,
        chunk_size: int = 20,
        max_chunks: int = 10,
        seed: int = 42,
    ) -> Dict:
        """
        Investigate entropy behavior for a given mode.

        Returns dict with findings and suggestions for next steps.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model.reset_session()

        # Run generation
        generated, telemetry = self.model.generate_multistep(
            input_ids,
            mode=mode,
            chunk_size=chunk_size,
            max_chunks=max_chunks,
            verbose=False,
        )

        # Extract entropy data
        entropies = [chunk.router_entropy for chunk in telemetry.chunks]
        pressures_net = [chunk.net_pressure for chunk in telemetry.chunks]
        pressures_fast = [chunk.fast_pressure for chunk in telemetry.chunks]
        pressures_mid = [chunk.mid_pressure for chunk in telemetry.chunks]

        # Analyze
        analysis = {
            "mode": mode.value,
            "total_chunks": telemetry.total_chunks,
            "total_tokens": telemetry.total_tokens,
            "entropies": entropies,
            "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
            "entropy_trend": self._compute_trend(entropies),
            "pressures": {
                "net": pressures_net,
                "fast": pressures_fast,
                "mid": pressures_mid,
            },
            "pause_reasons": telemetry.pause_reasons,
        }

        # Determine next steps based on findings
        suggestions = []

        if mode == SessionMode.MULTISTEP and telemetry.total_chunks == 1:
            suggestions.append(
                "Multistep paused after 1 chunk (expected non-agentic behavior)"
            )
            suggestions.append(
                "To compare with single-turn, need to either:"
            )
            suggestions.append("  1. Run single-turn for comparison")
            suggestions.append("  2. Implement interactive continuation")

        if len(entropies) > 3:
            trend = analysis["entropy_trend"]
            if abs(trend) < 0.01:
                suggestions.append(
                    f"Entropy appears constant ({np.mean(entropies):.4f}) - may be using placeholder values"
                )
                suggestions.append(
                    "Check signal extraction in clock_gated_multistep.py"
                )
            elif trend < -0.05:
                suggestions.append(
                    f"Entropy decreasing (trend={trend:.4f}) - potential collapse"
                )
            elif trend > 0.05:
                suggestions.append(
                    f"Entropy increasing (trend={trend:.4f}) - growing uncertainty"
                )

        analysis["suggestions"] = suggestions
        self.results.append(analysis)

        return analysis

    def _compute_trend(self, values: List[float]) -> float:
        """Compute linear trend (slope) of values."""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        y = np.array(values)

        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    def compare_modes(
        self,
        input_ids: torch.Tensor,
        chunk_size: int = 20,
        seed: int = 42,
    ) -> Dict:
        """
        Compare single-turn vs multistep mode.

        Returns comparison analysis with actionable insights.
        """
        print("Comparing single-turn vs multistep...")
        print()

        # Run single-turn
        print("Running single-turn mode...")
        single_analysis = self.investigate_entropy_behavior(
            input_ids,
            mode=SessionMode.SINGLE_TURN,
            chunk_size=chunk_size,
            max_chunks=10,
            seed=seed,
        )

        # Run multistep
        print("Running multistep mode...")
        multi_analysis = self.investigate_entropy_behavior(
            input_ids,
            mode=SessionMode.MULTISTEP,
            chunk_size=chunk_size,
            max_chunks=10,
            seed=seed,
        )

        # Compare
        comparison = {
            "single_turn": single_analysis,
            "multistep": multi_analysis,
            "differences": {
                "chunks_generated": {
                    "single_turn": single_analysis["total_chunks"],
                    "multistep": multi_analysis["total_chunks"],
                    "difference": single_analysis["total_chunks"]
                    - multi_analysis["total_chunks"],
                },
                "mean_entropy": {
                    "single_turn": single_analysis["mean_entropy"],
                    "multistep": multi_analysis["mean_entropy"],
                    "difference": single_analysis["mean_entropy"]
                    - multi_analysis["mean_entropy"],
                },
                "entropy_trend": {
                    "single_turn": single_analysis["entropy_trend"],
                    "multistep": multi_analysis["entropy_trend"],
                    "difference": single_analysis["entropy_trend"]
                    - multi_analysis["entropy_trend"],
                },
            },
        }

        # Generate insights
        insights = []

        if multi_analysis["total_chunks"] < single_analysis["total_chunks"]:
            insights.append(
                f"✓ Non-agentic verified: multistep paused after {multi_analysis['total_chunks']} chunk(s)"
            )

        if abs(comparison["differences"]["mean_entropy"]["difference"]) < 0.01:
            insights.append(
                "⚠ Entropy values similar - may be using placeholder signals"
            )
            insights.append(
                "  Action: Verify signal extraction from chrono_state.routing_entropy"
            )

        comparison["insights"] = insights
        comparison["next_steps"] = [
            "If entropy is placeholder: Fix signal extraction",
            "If multistep paused early: Implement interactive continuation",
            "If ready for real testing: Run on full Mixtral with real prompts",
        ]

        return comparison

    def visualize_comparison(self, comparison: Dict, save_path: Optional[Path] = None):
        """Create adaptive visualization based on what was found."""
        single = comparison["single_turn"]
        multi = comparison["multistep"]

        # Determine what to plot based on data
        has_entropy_variation = (
            np.std(single["entropies"]) > 0.01
            if single["entropies"]
            else False
        )

        if not has_entropy_variation:
            print("Note: Entropy appears constant - plotting structure only")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Entropy comparison
        ax1 = axes[0, 0]
        if single["entropies"]:
            ax1.plot(
                single["entropies"], marker="o", label="Single-turn", alpha=0.7
            )
        if multi["entropies"]:
            ax1.plot(
                multi["entropies"],
                marker="s",
                label="Multistep",
                alpha=0.7,
                color="orange",
            )
        ax1.set_xlabel("Chunk Index")
        ax1.set_ylabel("Router Entropy")
        ax1.set_title("Entropy Profile Comparison")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Net pressure comparison
        ax2 = axes[0, 1]
        if single["pressures"]["net"]:
            ax2.plot(
                single["pressures"]["net"],
                marker="o",
                label="Single-turn",
                alpha=0.7,
            )
        if multi["pressures"]["net"]:
            ax2.plot(
                multi["pressures"]["net"],
                marker="s",
                label="Multistep",
                alpha=0.7,
                color="orange",
            )
        ax2.set_xlabel("Chunk Index")
        ax2.set_ylabel("Net Pressure")
        ax2.set_title("Pressure Profile Comparison")
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.axhline(y=0, color="black", linestyle="--", alpha=0.3)

        # Pressure components (single-turn)
        ax3 = axes[1, 0]
        if single["pressures"]["fast"]:
            ax3.plot(
                single["pressures"]["fast"],
                marker="^",
                label="Fast",
                alpha=0.7,
            )
            ax3.plot(
                single["pressures"]["mid"],
                marker="o",
                label="Mid",
                alpha=0.7,
            )
        ax3.set_xlabel("Chunk Index")
        ax3.set_ylabel("Pressure")
        ax3.set_title("Single-Turn: Pressure Components")
        ax3.legend()
        ax3.grid(alpha=0.3)
        ax3.axhline(y=0, color="black", linestyle="--", alpha=0.3)

        # Summary stats
        ax4 = axes[1, 1]
        ax4.axis("off")

        summary_text = f"""
        Comparison Summary:

        Chunks Generated:
          Single-turn: {single['total_chunks']}
          Multistep:   {multi['total_chunks']}

        Mean Entropy:
          Single-turn: {single['mean_entropy']:.4f}
          Multistep:   {multi['mean_entropy']:.4f}

        Entropy Trend:
          Single-turn: {single['entropy_trend']:+.4f}
          Multistep:   {multi['entropy_trend']:+.4f}

        Insights:
        """

        for insight in comparison["insights"]:
            summary_text += f"\n  {insight}"

        ax4.text(
            0.1,
            0.9,
            summary_text,
            transform=ax4.transAxes,
            fontsize=9,
            verticalalignment="top",
            family="monospace",
        )

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "adaptive_phase2_comparison.png"

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n✓ Visualization saved to {save_path}")

        return save_path

    def save_results(self, comparison: Dict, filename: str = "adaptive_phase2_results.json"):
        """Save results with adaptive recommendations."""
        output_file = self.output_dir / filename

        with open(output_file, "w") as f:
            json.dump(comparison, f, indent=2)

        print(f"✓ Results saved to {output_file}")

        return output_file

    def report_findings(self, comparison: Dict):
        """Generate a conversational report of findings."""
        print("\n" + "=" * 70)
        print("ADAPTIVE PHASE 2 FINDINGS")
        print("=" * 70)
        print()

        print("What I found:")
        print("-" * 70)

        single = comparison["single_turn"]
        multi = comparison["multistep"]

        print(f"Single-turn mode:")
        print(f"  Generated {single['total_chunks']} chunks")
        print(f"  Mean entropy: {single['mean_entropy']:.4f}")
        print(f"  Entropy trend: {single['entropy_trend']:+.4f}")
        print()

        print(f"Multistep mode:")
        print(f"  Generated {multi['total_chunks']} chunks")
        print(f"  Mean entropy: {multi['mean_entropy']:.4f}")
        print(f"  Entropy trend: {multi['entropy_trend']:+.4f}")
        print()

        print("Insights:")
        print("-" * 70)
        for insight in comparison["insights"]:
            print(f"  {insight}")
        print()

        print("Suggestions for next steps:")
        print("-" * 70)
        for i, step in enumerate(comparison["next_steps"], 1):
            print(f"  {i}. {step}")
        print()

        print("=" * 70)


def main():
    """Run adaptive Phase 2 investigation."""
    print("Adaptive Phase 2 Test Runner")
    print("=" * 70)
    print()

    runner = AdaptivePhase2Runner()

    # Generate test input
    input_ids = torch.randint(0, runner.config.vocab_size, (1, 10))

    # Run adaptive comparison
    comparison = runner.compare_modes(input_ids, chunk_size=15, seed=42)

    # Visualize
    runner.visualize_comparison(comparison)

    # Save
    runner.save_results(comparison)

    # Report
    runner.report_findings(comparison)


if __name__ == "__main__":
    main()
