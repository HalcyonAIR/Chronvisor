"""
Proto-Role Detector: Turn-Level Expert Usage Analysis

Builds a 7 × N_experts matrix showing which experts prefer which conversation phases.

Expected outcomes:
1. Rows look the same → Experts still interchangeable, need stronger geometry
2. Rows clearly differ → Proto-roles exist, T̄_var=0 is a wiring issue

Turn phases:
0. inquiry
1. premise
2. complication
3. contradiction
4. exception
5. concession
6. synthesis
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

from chronomoe.chronovisor_mixtral_bridge import ChronovisorMixtralForCausalLM
from chronomoe.mixtral_core import MixtralConfig

# Try relative import first, fall back to absolute
try:
    from conversational_dataset import ConversationalDataset, ALL_CONVERSATIONAL_DOMAINS as ALL_DOMAINS
except ImportError:
    try:
        from experiments.conversational_dataset import ConversationalDataset, ALL_CONVERSATIONAL_DOMAINS as ALL_DOMAINS
    except ImportError:
        # If neither works, these are only used in __main__, not in TurnUsageAnalyzer
        ConversationalDataset = None
        ALL_DOMAINS = None


class ThreeDomainDatasetPyTorch(Dataset):
    """PyTorch Dataset wrapper for 3-domain synthetic data."""

    def __init__(self, sequences: List[Dict], vocab_size: int):
        self.sequences = sequences
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = self.sequences[idx]
        token_ids = torch.tensor(item["token_ids"], dtype=torch.long)
        labels = torch.cat([token_ids[1:], torch.tensor([-100])])
        return {
            "input_ids": token_ids,
            "labels": labels,
            "domain": item["domain"],
            "turn_boundaries": item.get("turn_boundaries", [])  # List of token indices where turns start
        }


class TurnUsageAnalyzer:
    """
    Analyzes expert usage per conversation turn.

    Hooks into the router to capture which experts are selected for each token,
    then aggregates by turn position to detect phase-specific specialization.
    """

    def __init__(self, model: ChronovisorMixtralForCausalLM, num_turns: int = 7):
        self.model = model
        self.num_turns = num_turns
        self.num_experts = model.config.num_experts
        self.num_layers = model.config.num_layers

        # Storage for routing decisions
        # routing_log[layer_idx] = list of (token_idx, expert_indices)
        self.routing_log = defaultdict(list)

        # Per-turn usage matrix: usage[layer][turn][expert] = count
        self.turn_usage = np.zeros((self.num_layers, num_turns, self.num_experts))

        # Register hooks on routers
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on all MoE routers to capture routing decisions."""

        def make_hook(layer_idx):
            def hook(module, input, output):
                # output = (routing_weights, selected_experts)
                # routing_weights: [batch, seq_len, top_k]
                # selected_experts: [batch, seq_len, top_k]
                routing_weights, selected_experts = output

                # Store the selected experts for this layer
                # We'll match them to turn positions later
                batch_size, seq_len, top_k = selected_experts.shape

                for b in range(batch_size):
                    for t in range(seq_len):
                        experts = selected_experts[b, t].cpu().numpy()  # top_k expert indices
                        self.routing_log[layer_idx].append({
                            'batch_idx': b,
                            'token_idx': t,
                            'experts': experts
                        })

            return hook

        # Hook into each layer's router
        for layer_idx, layer in enumerate(self.model.model.layers):
            layer.moe.router.register_forward_hook(make_hook(layer_idx))

    def analyze_batch(self, batch: Dict):
        """
        Run inference on batch and update turn usage statistics.

        Args:
            batch: Dataset batch with input_ids and turn_boundaries
        """
        input_ids = batch["input_ids"]
        turn_boundaries_batch = batch.get("turn_boundaries", [])
        batch_size, seq_len = input_ids.shape

        # Clear routing log for this batch
        self.routing_log.clear()

        # Forward pass (triggers hooks)
        with torch.no_grad():
            logits, chrono_state = self.model(input_ids, update_chronovisor=False)

        # Process routing decisions
        for layer_idx in range(self.num_layers):
            decisions = self.routing_log[layer_idx]

            for decision in decisions:
                batch_idx = decision['batch_idx']
                token_idx = decision['token_idx']
                experts = decision['experts']

                # Get turn boundaries for this sample
                if batch_idx < len(turn_boundaries_batch) and len(turn_boundaries_batch[batch_idx]) > 0:
                    boundaries = turn_boundaries_batch[batch_idx]

                    # Map token_idx to turn_idx using boundaries
                    # Boundaries[i] = start of turn i
                    # Find the largest boundary index where token_idx >= boundary
                    turn_idx = 0
                    for i in range(len(boundaries)):
                        if token_idx >= boundaries[i]:
                            turn_idx = i
                        else:
                            break

                    # Clamp to valid range
                    turn_idx = min(turn_idx, self.num_turns - 1)
                else:
                    # Fallback to approximate mapping if boundaries not available
                    tokens_per_turn = seq_len // self.num_turns
                    turn_idx = min(token_idx // tokens_per_turn, self.num_turns - 1)

                # Update usage counts
                for expert_id in experts:
                    self.turn_usage[layer_idx, turn_idx, expert_id] += 1

    def get_usage_matrix(self, layer_idx: int = 0, normalize: bool = True) -> np.ndarray:
        """
        Get turn × expert usage matrix for a specific layer.

        Args:
            layer_idx: Which layer to analyze
            normalize: If True, normalize rows to sum to 1

        Returns:
            usage_matrix: [num_turns, num_experts] array
        """
        usage = self.turn_usage[layer_idx].copy()

        if normalize:
            row_sums = usage.sum(axis=1, keepdims=True) + 1e-8
            usage = usage / row_sums

        return usage

    def print_usage_summary(self, layer_idx: int = 0):
        """Print text summary of turn-level expert preferences."""
        usage = self.get_usage_matrix(layer_idx, normalize=True)

        turn_names = ["Inquiry", "Premise", "Complication", "Contradiction",
                      "Exception", "Concession", "Synthesis"]

        print(f"\n{'=' * 70}")
        print(f"TURN-LEVEL EXPERT USAGE: Layer {layer_idx}")
        print(f"{'=' * 70}")

        for turn_idx, turn_name in enumerate(turn_names):
            probs = usage[turn_idx]
            top_experts = np.argsort(probs)[-3:][::-1]  # Top 3 experts

            print(f"\n{turn_name} (Turn {turn_idx}):")
            print(f"  Top experts: {top_experts[0]} ({probs[top_experts[0]]:.3f}), "
                  f"{top_experts[1]} ({probs[top_experts[1]]:.3f}), "
                  f"{top_experts[2]} ({probs[top_experts[2]]:.3f})")
            print(f"  Full distribution: {' '.join([f'{p:.2f}' for p in probs])}")

    def plot_usage_heatmap(self, output_path: str = "turn_usage_heatmap.png"):
        """Plot heatmap of turn × expert usage for all layers."""

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Expert Usage by Conversation Turn (Proto-Role Detection)", fontsize=16)

        turn_names = ["Inquiry", "Premise", "Complication", "Contradiction",
                      "Exception", "Concession", "Synthesis"]

        for layer_idx in range(min(4, self.num_layers)):
            ax = axes[layer_idx // 2, layer_idx % 2]

            usage = self.get_usage_matrix(layer_idx, normalize=True)

            # Plot heatmap
            sns.heatmap(usage, annot=True, fmt='.2f', cmap='YlOrRd',
                       ax=ax, cbar_kws={'label': 'Usage Probability'},
                       xticklabels=[f'E{i}' for i in range(self.num_experts)],
                       yticklabels=turn_names)

            ax.set_title(f"Layer {layer_idx}")
            ax.set_xlabel("Expert")
            ax.set_ylabel("Conversation Phase")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\n📊 Turn usage heatmap saved: {output_path}")

    def diagnose_specialization(self, layer_idx: int = 0, threshold: float = 0.05):
        """
        Diagnose whether proto-roles have formed.

        Checks if turn rows are significantly different from each other.

        Args:
            layer_idx: Which layer to analyze
            threshold: Minimum std deviation to consider "specialized"
        """
        usage = self.get_usage_matrix(layer_idx, normalize=True)

        print(f"\n{'=' * 70}")
        print(f"PROTO-ROLE DIAGNOSIS: Layer {layer_idx}")
        print(f"{'=' * 70}")

        # Compute per-expert variance across turns
        expert_variance = usage.var(axis=0)  # Variance across turns for each expert

        print("\nPer-expert variance across turns:")
        for expert_id in range(self.num_experts):
            var = expert_variance[expert_id]
            status = "✓ SPECIALIZED" if var > threshold else "✗ Uniform"
            print(f"  Expert {expert_id}: var={var:.6f} {status}")

        # Compute turn distinctiveness
        turn_std = usage.std(axis=1)  # Std within each turn's distribution

        print("\nTurn distinctiveness (std of expert preferences):")
        turn_names = ["Inquiry", "Premise", "Complication", "Contradiction",
                      "Exception", "Concession", "Synthesis"]
        for turn_idx, turn_name in enumerate(turn_names):
            std = turn_std[turn_idx]
            status = "✓ DISTINCT" if std > 0.1 else "✗ Uniform"
            print(f"  {turn_name}: std={std:.4f} {status}")

        # Overall diagnosis
        max_variance = expert_variance.max()
        mean_variance = expert_variance.mean()

        print(f"\n{'=' * 70}")
        print("VERDICT:")
        print(f"{'=' * 70}")

        if max_variance > threshold:
            print(f"✅ PROTO-ROLES DETECTED")
            print(f"   Max expert variance: {max_variance:.6f} (threshold: {threshold})")
            print(f"   Mean expert variance: {mean_variance:.6f}")
            print("\n   → Experts have phase-specific preferences")
            print("   → T̄_var=0 is likely a wiring/logging issue")
            print("   → Next: Investigate structural T update logic")
        else:
            print(f"❌ NO SPECIALIZATION YET")
            print(f"   Max expert variance: {max_variance:.6f} (threshold: {threshold})")
            print(f"   Mean expert variance: {mean_variance:.6f}")
            print("\n   → Experts still interchangeable across turns")
            print("   → Router hasn't learned phase-specific roles")
            print("   → Next: Increase geometry influence or add data asymmetry")


def run_turn_analysis(model_checkpoint: str = None, num_samples: int = 500):
    """
    Run turn-level usage analysis on trained or fresh model.

    Args:
        model_checkpoint: Path to trained model (if None, use fresh model)
        num_samples: Number of samples to analyze
    """

    print("=" * 70)
    print("TURN-LEVEL EXPERT USAGE ANALYSIS")
    print("Proto-Role Detector for Conversational MoE")
    print("=" * 70)

    # Generate dataset
    print("\n📊 Generating conversational dataset...")
    dataset_gen = ConversationalDataset(seq_length=128, vocab_size=1000)
    full_dataset = dataset_gen.generate_dataset(num_sequences=num_samples, balanced=True)

    test_sequences = full_dataset["sequences"]
    test_dataset = ThreeDomainDatasetPyTorch(test_sequences, vocab_size=1000)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Create or load model
    print("\n🏗️  Setting up model...")
    config = MixtralConfig(
        vocab_size=1000,
        hidden_dim=512,
        intermediate_dim=2048,
        num_layers=4,
        num_experts=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        enable_chronovisor=True,
    )

    model = ChronovisorMixtralForCausalLM(config)

    if model_checkpoint:
        print(f"   Loading checkpoint: {model_checkpoint}")
        model.load_state_dict(torch.load(model_checkpoint))
    else:
        print("   Using fresh (untrained) model")

    model.eval()

    # Create analyzer
    analyzer = TurnUsageAnalyzer(model, num_turns=7)

    # Analyze batches
    print(f"\n🔍 Analyzing {num_samples} conversational samples...")

    for batch_idx, batch in enumerate(test_loader):
        analyzer.analyze_batch(batch)

        if (batch_idx + 1) % 25 == 0:
            print(f"   Processed {(batch_idx + 1) * 4} samples...")

    # Generate reports
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)

    # Print summary for layer 0
    analyzer.print_usage_summary(layer_idx=0)

    # Diagnose specialization
    analyzer.diagnose_specialization(layer_idx=0, threshold=0.01)

    # Plot heatmap
    output_dir = Path("conversational_results")
    output_dir.mkdir(exist_ok=True)
    analyzer.plot_usage_heatmap(str(output_dir / "turn_usage_heatmap.png"))

    print("\n✅ Turn-level analysis complete!")
    print("   Check conversational_results/turn_usage_heatmap.png")


if __name__ == "__main__":
    # Analyze fresh model to see baseline (random) usage
    print("\n" + "=" * 70)
    print("BASELINE: Analyzing UNTRAINED model")
    print("(Should show uniform usage across all turns)")
    print("=" * 70)
    run_turn_analysis(model_checkpoint=None, num_samples=500)

    # TODO: After training completes, run with trained checkpoint
    # run_turn_analysis(model_checkpoint="conversational_results/model_final.pt", num_samples=500)
