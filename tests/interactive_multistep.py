"""
Interactive Multistep Generation

Allows manual continuation to test multistep mode over many chunks.

Usage:
    python interactive_multistep.py

Commands:
    continue (c)  - Generate next chunk
    status (s)    - Show current status
    plot (p)      - Show live plots
    end (e)       - Stop generation
    auto N        - Auto-continue for N chunks
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from chronomoe.clock_gated_multistep import ClockGatedMultistepModel, SessionMode
from chronomoe.chronovisor_mixtral_bridge import MixtralConfig


class InteractiveSession:
    """Interactive multistep generation session."""

    def __init__(self, config):
        self.model = ClockGatedMultistepModel(config)
        self.config = config

        # Session state
        self.current_ids = None
        self.all_telemetry = []
        self.total_chunks = 0

        # Time series
        self.entropies = []
        self.pressures_net = []
        self.pressures_fast = []
        self.pressures_mid = []
        self.pressures_slow = []
        self.residual_intents = []

    def start(self, input_ids: torch.Tensor, seed: int = 42):
        """Initialize session."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.current_ids = input_ids.clone()
        self.model.reset_session()

        print("=" * 70)
        print("Interactive Multistep Session")
        print("=" * 70)
        print(f"Input length: {input_ids.shape[1]} tokens")
        print(f"Seed: {seed}")
        print()

    def generate_chunk(self, chunk_size: int = 10) -> bool:
        """
        Generate one chunk.

        Returns:
            True if should continue, False if session ended
        """
        # Generate
        generated, telemetry = self.model.generate_multistep(
            self.current_ids,
            mode=SessionMode.MULTISTEP,
            chunk_size=chunk_size,
            max_chunks=1,  # One chunk at a time
            verbose=False,
        )

        # Update state
        self.current_ids = generated
        self.all_telemetry.append(telemetry)
        self.total_chunks += telemetry.total_chunks

        # Extract stats
        for chunk in telemetry.chunks:
            self.entropies.append(chunk.router_entropy)
            self.pressures_net.append(chunk.net_pressure)
            self.pressures_fast.append(chunk.fast_pressure)
            self.pressures_mid.append(chunk.mid_pressure)
            self.pressures_slow.append(chunk.slow_pressure)
            self.residual_intents.append(chunk.residual_intent)

        # Show chunk info
        chunk = telemetry.chunks[0]
        print(f"\nChunk {self.total_chunks}:")
        print(f"  Tokens: {chunk.tokens_generated}")
        print(f"  Entropy: {chunk.router_entropy:.4f}")
        print(f"  Net pressure: {chunk.net_pressure:+.4f}")
        print(f"  Residual intent: {chunk.residual_intent:.4f}")
        print(f"  Pause reason: {chunk.pause_reason}")

        return True

    def show_status(self):
        """Display current session status."""
        print("\n" + "=" * 70)
        print("Session Status")
        print("-" * 70)
        print(f"Total chunks: {self.total_chunks}")
        print(f"Total tokens: {self.current_ids.shape[1]}")

        if len(self.entropies) > 0:
            print(f"\nEntropy:")
            print(f"  Current: {self.entropies[-1]:.4f}")
            print(f"  Mean: {np.mean(self.entropies):.4f}")

            if len(self.entropies) >= 10:
                early = np.mean(self.entropies[:5])
                recent = np.mean(self.entropies[-5:])
                print(f"  Early (first 5): {early:.4f}")
                print(f"  Recent (last 5): {recent:.4f}")
                print(f"  Collapse: {early - recent:+.4f}")

        print(f"\nPressure:")
        if len(self.pressures_net) > 0:
            print(f"  Net: {self.pressures_net[-1]:+.4f}")
            print(f"  Fast: {self.pressures_fast[-1]:+.4f}")
            print(f"  Mid: {self.pressures_mid[-1]:+.4f}")
            print(f"  Slow: {self.pressures_slow[-1]:+.4f}")

        print(f"\nResidual Intent: {self.residual_intents[-1]:.4f}" if self.residual_intents else "")
        print("=" * 70)

    def plot_live(self):
        """Show live plots of evolution."""
        if len(self.entropies) < 2:
            print("Not enough data to plot (need at least 2 chunks)")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Entropy
        ax = axes[0, 0]
        ax.plot(self.entropies, marker=".", markersize=4)
        ax.set_xlabel("Chunk")
        ax.set_ylabel("Entropy")
        ax.set_title("Entropy Evolution")
        ax.grid(alpha=0.3)

        # Net pressure
        ax = axes[0, 1]
        ax.plot(self.pressures_net, marker=".", markersize=4, color="purple")
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax.set_xlabel("Chunk")
        ax.set_ylabel("Net Pressure")
        ax.set_title("Net Pressure Evolution")
        ax.grid(alpha=0.3)

        # Pressure components
        ax = axes[1, 0]
        ax.plot(self.pressures_fast, marker="^", markersize=3, alpha=0.7, label="Fast")
        ax.plot(self.pressures_mid, marker="o", markersize=3, alpha=0.7, label="Mid")
        ax.plot(self.pressures_slow, marker="s", markersize=3, alpha=0.7, label="Slow")
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax.set_xlabel("Chunk")
        ax.set_ylabel("Pressure")
        ax.set_title("Pressure Components")
        ax.legend()
        ax.grid(alpha=0.3)

        # Residual intent
        ax = axes[1, 1]
        ax.plot(self.residual_intents, marker=".", markersize=4, color="orange")
        ax.set_xlabel("Chunk")
        ax.set_ylabel("Residual Intent")
        ax.set_title("Residual Intent")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

        print("\n✓ Live plot displayed (close window to continue)")

    def save_results(self):
        """Save session results."""
        import json

        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)

        results = {
            "mode": "interactive_multistep",
            "total_chunks": self.total_chunks,
            "total_tokens": int(self.current_ids.shape[1]),
            "time_series": {
                "entropies": self.entropies,
                "pressures_net": self.pressures_net,
                "pressures_fast": self.pressures_fast,
                "pressures_mid": self.pressures_mid,
                "pressures_slow": self.pressures_slow,
                "residual_intents": self.residual_intents,
            },
        }

        # Analysis
        if len(self.entropies) >= 10:
            early_cutoff = max(1, int(len(self.entropies) * 0.3))
            late_start = max(1, int(len(self.entropies) * 0.7))

            early_entropy = float(np.mean(self.entropies[:early_cutoff]))
            late_entropy = float(np.mean(self.entropies[late_start:]))

            results["analysis"] = {
                "entropy": {
                    "early": early_entropy,
                    "late": late_entropy,
                    "collapse": early_entropy - late_entropy,
                    "trend": float(
                        np.polyfit(range(len(self.entropies)), self.entropies, 1)[0]
                    ),
                },
                "pressure": {
                    "initial": float(self.pressures_net[0]),
                    "final": float(self.pressures_net[-1]),
                    "mean": float(np.mean(self.pressures_net)),
                },
            }

        output_file = output_dir / "interactive_multistep_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {output_file}")


def main():
    """Run interactive session."""
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

    session = InteractiveSession(config)

    # Start
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    session.start(input_ids, seed=42)

    print("Commands:")
    print("  continue (c) - Generate next chunk")
    print("  status (s)   - Show status")
    print("  plot (p)     - Show live plots")
    print("  end (e)      - Stop and save")
    print("  auto N       - Auto-continue for N chunks")
    print()

    # Interactive loop
    while True:
        try:
            command = input(f"\n[Chunk {session.total_chunks}] > ").strip().lower()

            if command in ["", "c", "continue"]:
                session.generate_chunk(chunk_size=10)

            elif command in ["s", "status"]:
                session.show_status()

            elif command in ["p", "plot"]:
                session.plot_live()

            elif command in ["e", "end"]:
                print("\nEnding session...")
                break

            elif command.startswith("auto "):
                try:
                    n_chunks = int(command.split()[1])
                    print(f"\nAuto-continuing for {n_chunks} chunks...")
                    for i in range(n_chunks):
                        session.generate_chunk(chunk_size=10)
                        if (i + 1) % 10 == 0:
                            print(f"  ... {i + 1}/{n_chunks} chunks generated")
                    print(f"✓ Auto-continue complete ({n_chunks} chunks)")
                except (IndexError, ValueError):
                    print("Usage: auto N (where N is number of chunks)")

            else:
                print("Unknown command. Try: continue, status, plot, end, auto N")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Saving...")
            break

        except Exception as e:
            print(f"\nError: {e}")
            import traceback

            traceback.print_exc()

    # Save on exit
    session.save_results()
    session.show_status()

    print("\n" + "=" * 70)
    print("Session complete. Results saved.")
    print("=" * 70)


if __name__ == "__main__":
    main()
