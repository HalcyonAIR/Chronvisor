"""
Geological Seismograph: Watch ChronoMoE Dynamics on Frozen Weights

This experiment demonstrates that Chronovisor's geometric control layer
produces emergent behavior even when model weights are frozen.

The "mountain" (Mixtral weights) stays static.
The "seismograph" (Chronovisor diagnostics) records geological activity.

Metrics monitored:
- Kuramoto R (expert coherence)
- Structural T̄ variance (landscape formation)
- Valley health (self-correction)
- Fast T spikes (behavioral responses)
- Routing entropy (specialization)
- Expert usage (traffic patterns)
- Pressure magnitudes (force fields)

The hypothesis: geometric dynamics emerge from sequence stressors alone.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field

from chronomoe.chronovisor_mixtral_bridge import (
    ChronovisorMixtralForCausalLM,
    ChronovisorMixtralController,
)
from chronomoe.mixtral_core import MixtralConfig


@dataclass
class SeismographReading:
    """Single timepoint of geological measurements."""
    step: int
    prompt_idx: int
    prompt_type: str

    # Coherence
    kuramoto_R: float
    delta_R: float
    mean_phase: float

    # Landscape
    structural_T_variance: float
    structural_T_mean: float
    num_valleys: int
    num_ridges: int
    landscape_formed: bool

    # Valley health
    num_healthy_valleys: int
    num_unhealthy_valleys: int
    self_correction_working: bool

    # Temperature dynamics
    fast_T_mean: float
    fast_T_std: float
    effective_T_mean: float

    # Routing
    routing_entropy: float
    expert_usage_cv: float  # Coefficient of variation

    # Pressure
    avg_pressure_magnitude: float

    # Meta-knob
    kappa: float


@dataclass
class StressorSequence:
    """A sequence designed to stress the routing system."""
    name: str
    prompts: List[str]
    expected_behavior: str
    stressor_type: str  # "domain_shift", "complexity_spike", "chaos", "alternating"


# Define stressor sequences
STRESSOR_SEQUENCES = [
    StressorSequence(
        name="domain_shift",
        prompts=[
            "Explain the legal doctrine of precedent in common law.",
            "Explain the legal concept of mens rea in criminal law.",
            "Explain the legal principle of consideration in contract law.",
            "Explain quantum entanglement and its implications.",
            "Describe the Higgs boson and its role in physics.",
            "Explain string theory and extra dimensions.",
        ],
        expected_behavior="R should drop during shift (step 3→4), then recover. Structural T̄ variance should increase as specialists emerge.",
        stressor_type="domain_shift"
    ),

    StressorSequence(
        name="complexity_spike",
        prompts=[
            "Count to five: 1, 2, 3, 4, 5.",
            "List primary colors: red, blue, yellow.",
            "Name three fruits: apple, banana, orange.",
            "Derive the Euler-Lagrange equation from Hamilton's principle of stationary action for a field theory with Lorentz symmetry.",
            "Name three animals: cat, dog, bird.",
            "List days of week: Monday, Tuesday, Wednesday.",
        ],
        expected_behavior="Fast T should spike at step 4 (complexity). Valley variance should increase.",
        stressor_type="complexity_spike"
    ),

    StressorSequence(
        name="chaos",
        prompts=[
            "Explain photosynthesis.",
            "Write a haiku about silence.",
            "Solve 2+2.",
            "Describe Gothic architecture.",
            "Recipe for pancakes.",
            "Quantum field theory basics.",
            "List US presidents.",
            "Sonnet about moonlight.",
        ],
        expected_behavior="Structural T̄ should remain flat (no pattern). R stays low. System behaves as generalist.",
        stressor_type="chaos"
    ),

    StressorSequence(
        name="alternating",
        prompts=[
            "Explain Newton's first law of motion.",
            "Write a poem about dawn breaking.",
            "Explain conservation of momentum.",
            "Write a poem about stars fading.",
            "Explain kinetic energy formula.",
            "Write a poem about ocean waves.",
            "Explain gravitational acceleration.",
            "Write a poem about mountain peaks.",
        ],
        expected_behavior="Two valley systems should form (physics and poetry). Bimodal landscape. R should stabilize.",
        stressor_type="alternating"
    ),
]


class GeologicalSeismograph:
    """
    Seismograph for monitoring Chronovisor geometric dynamics.

    Runs sequences through a frozen model and records geological evolution.
    """

    def __init__(
        self,
        model: ChronovisorMixtralForCausalLM,
        output_dir: str = "seismograph_output",
        fast_geology: bool = False,
    ):
        """
        Initialize seismograph.

        Args:
            model: ChronoMoE model with frozen weights
            output_dir: Where to save plots and data
            fast_geology: If True, speed up structural T evolution 10x for visualization
        """
        self.model = model
        self.controller = model.model.controller
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.readings: List[SeismographReading] = []
        self.step = 0
        self.fast_geology = fast_geology

        # Apply fast geology if requested
        if fast_geology:
            self.controller.eta_structural_T_global = 0.05  # 10x faster (was 0.005)
            # Update local η for all lenses
            for lens in self.controller.lenses.values():
                lens.eta_structural_T = 0.1  # 10x faster (was 0.01)

        # Freeze all weights
        for param in self.model.parameters():
            param.requires_grad = False

        print("🌋 Geological Seismograph initialized")
        print(f"   Model: {model.config.num_layers} layers, {model.config.num_experts} experts/layer")
        print(f"   Weights frozen: {sum(1 for p in model.parameters() if not p.requires_grad)} params")
        if fast_geology:
            print("   " + "=" * 60)
            print("   ⚠️  FAST GEOLOGY MODE (VISUALIZATION ONLY)")
            print("   η_global=0.05, η_local=0.1 (10x production speed)")
            print("   For debugging valley formation - NOT realistic timescales")
            print("   " + "=" * 60)
        else:
            print(f"   🐌 Realistic geology: η_global=0.005, η_local=0.01")
        print(f"   Output: {self.output_dir}")

    def run_sequence(
        self,
        sequence: StressorSequence,
        num_ticks_per_prompt: int = 20,
        set_kappa: float = 0.0,
    ) -> List[SeismographReading]:
        """
        Run a stressor sequence and record geological activity.

        Args:
            sequence: StressorSequence to run
            num_ticks_per_prompt: How many Chronovisor ticks per prompt
            set_kappa: Meta-knob setting (0 = baseline)

        Returns:
            List of SeismographReading measurements
        """
        print(f"\n{'='*60}")
        print(f"Running sequence: {sequence.name}")
        print(f"Stressor type: {sequence.stressor_type}")
        print(f"Expected: {sequence.expected_behavior}")
        print(f"{'='*60}\n")

        # Set meta-knob
        self.controller.set_knob(set_kappa, f"experiment_{sequence.name}")

        sequence_readings = []

        for prompt_idx, prompt in enumerate(sequence.prompts):
            print(f"[{prompt_idx+1}/{len(sequence.prompts)}] {prompt[:60]}...")

            # Tokenize (simulate - we don't have a real tokenizer)
            # For demo purposes, use random token IDs
            batch_size = 2
            seq_len = 16
            input_ids = torch.randint(0, self.model.config.vocab_size, (batch_size, seq_len))

            # Forward pass (no gradient)
            with torch.no_grad():
                logits, chrono_state = self.model(
                    input_ids,
                    update_chronovisor=True,
                )

            # Tick Chronovisor multiple times to let geology evolve
            for tick in range(num_ticks_per_prompt - 1):
                # Simulate continued routing by re-running with same input
                with torch.no_grad():
                    _, chrono_state = self.model(
                        input_ids,
                        update_chronovisor=True,
                    )

            # Take reading
            reading = self._take_reading(
                prompt_idx=prompt_idx,
                prompt_type=sequence.stressor_type,
                chrono_state=chrono_state,
            )

            sequence_readings.append(reading)
            self.readings.append(reading)

            # Print snapshot
            print(f"   R={reading.kuramoto_R:.3f} | "
                  f"T̄_var={reading.structural_T_variance:.4f} | "
                  f"valleys={reading.num_valleys} | "
                  f"ridges={reading.num_ridges}")

            self.step += 1

        print(f"\n✓ Sequence complete: {len(sequence_readings)} readings")
        return sequence_readings

    def _take_reading(
        self,
        prompt_idx: int,
        prompt_type: str,
        chrono_state: Any,
    ) -> SeismographReading:
        """Take a seismograph reading from current controller state."""

        # Get diagnostics
        diag = self.controller.get_diagnostics()
        st_diag = diag["structural_T_diagnostics"]
        vh_diag = diag["valley_health"]

        # Compute routing entropy
        routing_entropy = 0.0
        for layer_idx, usage in self.controller.expert_usage.items():
            total = usage.sum()
            if total > 0:
                p = usage / total
                p = p[p > 0]
                layer_entropy = -np.sum(p * np.log(p + 1e-10))
                routing_entropy += layer_entropy
        routing_entropy /= len(self.controller.expert_usage) if self.controller.expert_usage else 1

        # Compute expert usage CV
        all_usage = []
        for usage in self.controller.expert_usage.values():
            all_usage.append(usage)
        if all_usage:
            total_usage = np.stack(all_usage).sum(axis=0)
            mean_usage = total_usage.mean()
            cv = total_usage.std() / mean_usage if mean_usage > 0 else 0.0
        else:
            cv = 0.0

        # Aggregate fast T across layers
        fast_T_all = []
        for lens in self.controller.lenses.values():
            fast_T_all.append(lens.temperature_fast)
        fast_T_mean = np.mean(fast_T_all) if fast_T_all else 1.0
        fast_T_std = np.std(fast_T_all) if fast_T_all else 0.0

        # Aggregate effective T
        effective_T_all = []
        for lens in self.controller.lenses.values():
            effective_T_all.append(lens.temperature_effective)
        effective_T_mean = np.mean(effective_T_all) if effective_T_all else 1.0

        # Get meta-knob
        knob_factors = self.controller.get_knob_factors()

        return SeismographReading(
            step=self.step,
            prompt_idx=prompt_idx,
            prompt_type=prompt_type,
            kuramoto_R=diag["kuramoto_R"],
            delta_R=diag["delta_R"],
            mean_phase=diag["mean_phase_psi"],
            structural_T_variance=st_diag["variance"],
            structural_T_mean=st_diag["mean"],
            num_valleys=len(st_diag["valleys"]),
            num_ridges=len(st_diag["ridges"]),
            landscape_formed=st_diag["landscape_formed"],
            num_healthy_valleys=len(vh_diag["healthy_valleys"]),
            num_unhealthy_valleys=len(vh_diag["unhealthy_valleys"]),
            self_correction_working=vh_diag["self_correction_working"],
            fast_T_mean=float(fast_T_mean),
            fast_T_std=float(fast_T_std),
            effective_T_mean=float(effective_T_mean),
            routing_entropy=float(routing_entropy),
            expert_usage_cv=float(cv),
            avg_pressure_magnitude=float(diag.get("avg_pressure_magnitude", 0.0)) if "avg_pressure_magnitude" in str(diag) else 0.0,
            kappa=knob_factors.kappa,
        )

    def plot_evolution(self, sequence_name: str = "all"):
        """
        Plot geological evolution over time.

        Args:
            sequence_name: Which sequence to plot, or "all"
        """
        if not self.readings:
            print("No readings to plot")
            return

        # Filter readings if needed
        if sequence_name != "all":
            readings = [r for r in self.readings if sequence_name in r.prompt_type]
        else:
            readings = self.readings

        if not readings:
            print(f"No readings for sequence: {sequence_name}")
            return

        steps = [r.step for r in readings]

        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        fig.suptitle(f"Geological Seismograph: {sequence_name}", fontsize=16)

        # Plot 1: Kuramoto R
        ax = axes[0, 0]
        ax.plot(steps, [r.kuramoto_R for r in readings], 'b-', linewidth=2)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel("Kuramoto R")
        ax.set_title("Expert Coherence")
        ax.grid(True, alpha=0.3)

        # Plot 2: Structural T̄ variance
        ax = axes[0, 1]
        ax.plot(steps, [r.structural_T_variance for r in readings], 'r-', linewidth=2)
        ax.axhline(0.01, color='gray', linestyle='--', alpha=0.5, label='Formation threshold')
        ax.set_ylabel("Variance")
        ax.set_title("Landscape Formation (Structural T̄)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Valleys and Ridges
        ax = axes[0, 2]
        ax.plot(steps, [r.num_valleys for r in readings], 'g-', linewidth=2, label='Valleys')
        ax.plot(steps, [r.num_ridges for r in readings], 'orange', linewidth=2, label='Ridges')
        ax.set_ylabel("Count")
        ax.set_title("Valleys vs Ridges")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Fast Temperature
        ax = axes[1, 0]
        fast_T_means = [r.fast_T_mean for r in readings]
        fast_T_stds = [r.fast_T_std for r in readings]
        ax.plot(steps, fast_T_means, 'purple', linewidth=2, label='Mean')
        ax.fill_between(steps,
                         [m - s for m, s in zip(fast_T_means, fast_T_stds)],
                         [m + s for m, s in zip(fast_T_means, fast_T_stds)],
                         alpha=0.3, color='purple')
        ax.set_ylabel("Temperature")
        ax.set_title("Fast Temperature")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 5: Valley Health
        ax = axes[1, 1]
        ax.plot(steps, [r.num_healthy_valleys for r in readings], 'g-', linewidth=2, label='Healthy')
        ax.plot(steps, [r.num_unhealthy_valleys for r in readings], 'r-', linewidth=2, label='Unhealthy')
        ax.set_ylabel("Count")
        ax.set_title("Valley Health")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 6: Routing Entropy
        ax = axes[1, 2]
        ax.plot(steps, [r.routing_entropy for r in readings], 'brown', linewidth=2)
        ax.set_ylabel("Entropy")
        ax.set_title("Routing Entropy (Specialization)")
        ax.grid(True, alpha=0.3)

        # Plot 7: Expert Usage CV
        ax = axes[2, 0]
        ax.plot(steps, [r.expert_usage_cv for r in readings], 'teal', linewidth=2)
        ax.set_ylabel("CV (std/mean)")
        ax.set_title("Expert Usage Balance")
        ax.grid(True, alpha=0.3)

        # Plot 8: ΔR (coherence change)
        ax = axes[2, 1]
        ax.plot(steps, [r.delta_R for r in readings], 'navy', linewidth=2)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel("ΔR")
        ax.set_title("Coherence Rate of Change")
        ax.grid(True, alpha=0.3)

        # Plot 9: Landscape Formation Flag
        ax = axes[2, 2]
        landscape_formed = [1 if r.landscape_formed else 0 for r in readings]
        ax.fill_between(steps, landscape_formed, alpha=0.5, color='green')
        ax.set_ylabel("Formed (bool)")
        ax.set_title("Landscape Formed")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)

        # Common x-label
        for ax in axes[2, :]:
            ax.set_xlabel("Step")

        plt.tight_layout()

        # Save
        filename = self.output_dir / f"seismograph_{sequence_name}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\n📊 Plot saved: {filename}")

    def print_summary(self):
        """Print summary of all readings."""
        if not self.readings:
            print("No readings recorded")
            return

        print(f"\n{'='*60}")
        print("GEOLOGICAL SEISMOGRAPH SUMMARY")
        print(f"{'='*60}")
        print(f"Total readings: {len(self.readings)}")
        print(f"Total steps: {self.step}")

        # Overall trends
        initial = self.readings[0]
        final = self.readings[-1]

        print(f"\nInitial → Final:")
        print(f"  Kuramoto R:       {initial.kuramoto_R:.3f} → {final.kuramoto_R:.3f}")
        print(f"  T̄ variance:       {initial.structural_T_variance:.4f} → {final.structural_T_variance:.4f}")
        print(f"  Valleys:          {initial.num_valleys} → {final.num_valleys}")
        print(f"  Ridges:           {initial.num_ridges} → {final.num_ridges}")
        print(f"  Landscape formed: {initial.landscape_formed} → {final.landscape_formed}")

        # Valley health
        unhealthy_count = sum(1 for r in self.readings if r.num_unhealthy_valleys > 0)
        print(f"\nValley Health:")
        print(f"  Steps with unhealthy valleys: {unhealthy_count}/{len(self.readings)}")
        print(f"  Self-correction working: {final.self_correction_working}")

        # Coherence stats
        R_values = [r.kuramoto_R for r in self.readings]
        print(f"\nKuramoto R:")
        print(f"  Mean: {np.mean(R_values):.3f}")
        print(f"  Std:  {np.std(R_values):.3f}")
        print(f"  Min:  {np.min(R_values):.3f}")
        print(f"  Max:  {np.max(R_values):.3f}")

        print(f"\n{'='*60}\n")


def run_experiment():
    """Run the full geological seismograph experiment."""

    print("🌋 Geological Seismograph Experiment")
    print("=" * 60)
    print("Demonstrating ChronoMoE dynamics on frozen weights")
    print("=" * 60)

    # Create small model for fast experimentation
    config = MixtralConfig(
        vocab_size=32000,
        hidden_dim=256,
        intermediate_dim=1024,
        num_layers=4,
        num_experts=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,  # hidden_dim / num_attention_heads = 256 / 8
        enable_chronovisor=True,
    )

    print("\nInitializing model...")
    model = ChronovisorMixtralForCausalLM(config)

    # Create seismograph with fast geology for demonstration
    seismograph = GeologicalSeismograph(
        model=model,
        output_dir="seismograph_output",
        fast_geology=True,  # 10x faster for visualization
    )

    # Run each stressor sequence with more ticks to see valley formation
    for sequence in STRESSOR_SEQUENCES:
        seismograph.run_sequence(
            sequence=sequence,
            num_ticks_per_prompt=50,  # Increased from 20 to see landscape form
            set_kappa=0.0,  # Baseline behavior
        )

    # Print summary
    seismograph.print_summary()

    # Plot evolution
    print("\nGenerating plots...")
    seismograph.plot_evolution("all")

    # Plot individual sequences
    for sequence in STRESSOR_SEQUENCES:
        seismograph.plot_evolution(sequence.stressor_type)

    print("\n✓ Experiment complete!")
    print(f"  Results saved to: {seismograph.output_dir}")

    return seismograph


if __name__ == "__main__":
    seismograph = run_experiment()
