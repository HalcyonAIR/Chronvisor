"""
ChronoMoE Training on 3-Domain Synthetic Dataset

This is Option A: demonstrate the full closed loop.
- weights update → routing shifts → valleys deepen → bad valleys heal → R stabilizes

Monitors three critical curves:
1. LM Loss (should decrease)
2. Kuramoto R (should increase and stabilize)
3. Structural T̄ variance (should escape zero, then stabilize)

Expected behavior:
Step 0:    Loss=high, R=low,  T̄_var=0.0
Step 500:  Loss↓,     R↑,     T̄_var↑     ← Valleys forming
Step 2000: Loss↓↓,    R=high, T̄_var=stable ← Specialized

If valleys form correctly, we should see 3 clusters:
- Legal experts (blue)
- Physics experts (green)
- Poetry experts (red)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import json

from chronomoe.chronovisor_mixtral_bridge import ChronovisorMixtralForCausalLM
from chronomoe.mixtral_core import MixtralConfig
from chronomoe.training import ChronoMoETrainer, TrainingConfig, ChronoMoELoss
from experiments.conversational_dataset import ConversationalDataset, ALL_CONVERSATIONAL_DOMAINS as ALL_DOMAINS


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

        # Labels are shifted input_ids for language modeling
        # For next-token prediction: input[0,1,2,...,n-1] -> label[1,2,3,...,n]
        # We shift labels left by 1 and pad the last position with -100 (ignore)
        labels = torch.cat([token_ids[1:], torch.tensor([-100])])

        return {
            "input_ids": token_ids,
            "labels": labels,
            "domain": item["domain"]
        }


class ThreeDomainTrainer:
    """
    Custom trainer for 3-domain experiment with enhanced monitoring.

    Tracks and plots:
    - LM loss over time
    - Kuramoto R over time
    - Structural T̄ variance over time
    - Per-domain expert usage (valley clusters)
    """

    def __init__(
        self,
        model: ChronovisorMixtralForCausalLM,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        output_dir: str = "three_domain_results",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.controller = model.model.controller

        # Metrics tracking
        self.history = {
            "step": [],
            "loss": [],
            "kuramoto_R": [],
            "structural_T_variance": [],
            "structural_T_mean": [],
            "num_valleys": [],
            "num_ridges": [],
            "learning_rate": [],
        }

        # Per-domain expert usage tracking
        self.domain_expert_usage = {
            domain.name: {i: [] for i in range(model.config.num_experts)}
            for domain in ALL_DOMAINS
        }

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Loss function
        self.loss_fn = ChronoMoELoss(
            lambda_balance=config.lambda_balance,
            lambda_coherence=config.lambda_coherence,
            lambda_valley=config.lambda_valley,
        )

        self.global_step = 0

    def train(self, num_steps: int):
        """Run training for specified number of steps."""
        print("=" * 70)
        print("CHRONOMOE 3-DOMAIN TRAINING")
        print("=" * 70)
        print(f"Model: {self.model.config.num_layers} layers, {self.model.config.num_experts} experts")
        print(f"Training steps: {num_steps}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Fast geology: {self.config.fast_geology}")
        print("=" * 70)

        self.model.train()
        train_iter = iter(self.train_loader)

        for step in range(num_steps):
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Forward pass
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            domains = batch["domain"]

            logits, chrono_state = self.model(input_ids, update_chronovisor=True)

            # Compute loss
            loss, loss_components = self.loss_fn.compute(
                logits=logits,
                labels=labels,
                chrono_state=chrono_state,
                controller=self.controller,
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            # Record metrics
            self._record_metrics(step, loss, chrono_state, domains)

            # Logging
            if step % self.config.log_every_n_steps == 0:
                self._log_step(step, loss_components, chrono_state)

            # Plot
            if step % 500 == 0 and step > 0:
                self._plot_progress()

            self.global_step += 1

        # Final plots
        self._plot_progress()
        self._plot_expert_specialization()
        self._save_results()

        # Save model checkpoint
        print("\n💾 Saving model checkpoint...")
        checkpoint_path = self.output_dir / "model_final.pt"
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"   Model saved: {checkpoint_path}")

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)

    def _record_metrics(self, step, loss, chrono_state, domains):
        """Record metrics for plotting."""
        # Get structural T diagnostics
        st_diag = self.controller.get_structural_temperature_diagnostics()

        self.history["step"].append(step)
        self.history["loss"].append(loss.item())
        self.history["kuramoto_R"].append(chrono_state.coherence)
        self.history["structural_T_variance"].append(st_diag["variance"])
        self.history["structural_T_mean"].append(st_diag["mean"])
        self.history["num_valleys"].append(len(st_diag["valleys"]))
        self.history["num_ridges"].append(len(st_diag["ridges"]))
        self.history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])

        # Track per-domain expert usage
        if chrono_state.expert_usage:
            for domain_name in self.domain_expert_usage.keys():
                # Get expert usage for this domain from the batch
                domain_mask = [d == domain_name for d in domains]
                if any(domain_mask):
                    # Aggregate usage across layers for this domain
                    for expert_id in range(self.model.config.num_experts):
                        usage = 0.0
                        for layer_usage in chrono_state.expert_usage.values():
                            usage += layer_usage[expert_id].item()
                        self.domain_expert_usage[domain_name][expert_id].append(usage)

    def _log_step(self, step, loss_components, chrono_state):
        """Log training progress."""
        log_msg = f"Step {step:5d}"
        log_msg += f" | Loss: {loss_components['total_loss']:.4f}"
        log_msg += f" | R: {chrono_state.coherence:.3f}"
        log_msg += f" | T̄_var: {self.history['structural_T_variance'][-1]:.6f}"
        log_msg += f" | Valleys: {self.history['num_valleys'][-1]}"
        print(log_msg)

    def _plot_progress(self):
        """Plot the three critical curves."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("ChronoMoE Training Progress: 3-Domain Dataset", fontsize=16)

        steps = self.history["step"]

        # Plot 1: LM Loss
        ax = axes[0, 0]
        ax.plot(steps, self.history["loss"], 'b-', linewidth=2)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.set_title("Language Modeling Loss (should ↓)")
        ax.grid(True, alpha=0.3)

        # Plot 2: Kuramoto R
        ax = axes[0, 1]
        ax.plot(steps, self.history["kuramoto_R"], 'g-', linewidth=2)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='R=0.5')
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Kuramoto R")
        ax.set_title("Expert Coherence (should ↑ then stabilize)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Structural T̄ Variance
        ax = axes[1, 0]
        ax.plot(steps, self.history["structural_T_variance"], 'r-', linewidth=2)
        ax.axhline(0.01, color='gray', linestyle='--', alpha=0.5, label='Formation threshold')
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Variance")
        ax.set_title("Landscape Formation (should escape 0, then stabilize)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Valleys and Ridges
        ax = axes[1, 1]
        ax.plot(steps, self.history["num_valleys"], 'g-', linewidth=2, label='Valleys')
        ax.plot(steps, self.history["num_ridges"], 'orange', linewidth=2, label='Ridges')
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Count")
        ax.set_title("Valley/Ridge Formation")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = self.output_dir / "training_progress.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\n📊 Progress plot saved: {filename}")

    def _plot_expert_specialization(self):
        """Plot expert specialization by domain (the key result!)."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Expert Specialization by Domain", fontsize=16)

        domain_colors = {"legal": "blue", "physics": "green", "poetry": "red"}

        for idx, domain in enumerate(ALL_DOMAINS):
            ax = axes[idx]

            # Plot expert usage over time for this domain
            for expert_id in range(self.model.config.num_experts):
                usage_history = self.domain_expert_usage[domain.name][expert_id]
                if usage_history:
                    ax.plot(usage_history, alpha=0.7, label=f"Expert {expert_id}")

            ax.set_xlabel("Batch")
            ax.set_ylabel("Expert Usage")
            ax.set_title(f"{domain.name.upper()} Domain", color=domain_colors[domain.name])
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = self.output_dir / "expert_specialization.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"📊 Specialization plot saved: {filename}")

    def _save_results(self):
        """Save training results to JSON."""
        results = {
            "history": self.history,
            "config": {
                "num_layers": self.model.config.num_layers,
                "num_experts": self.model.config.num_experts,
                "learning_rate": self.config.learning_rate,
                "fast_geology": self.config.fast_geology,
            },
            "final_metrics": {
                "loss": self.history["loss"][-1],
                "kuramoto_R": self.history["kuramoto_R"][-1],
                "structural_T_variance": self.history["structural_T_variance"][-1],
                "num_valleys": self.history["num_valleys"][-1],
                "num_ridges": self.history["num_ridges"][-1],
            }
        }

        filename = self.output_dir / "results.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved: {filename}")


def run_experiment(fast_geology: bool = True, num_steps: int = 2000):
    """Run the 3-domain training experiment."""

    print("\n🌋 ChronoMoE 3-Domain Experiment")
    print("=" * 70)

    # Generate dataset
    print("\n1. Generating conversational dataset with phase shifts...")
    dataset_gen = ConversationalDataset(seq_length=128, vocab_size=1000)
    dataset_gen.print_examples(num_examples=1)

    full_dataset = dataset_gen.generate_dataset(
        num_sequences=10000,
        balanced=True
    )

    # Split train/val
    num_val = 1000  # 10% validation
    train_sequences = full_dataset["sequences"][num_val:]
    val_sequences = full_dataset["sequences"][:num_val]

    # Create PyTorch datasets
    train_dataset = ThreeDomainDatasetPyTorch(train_sequences, vocab_size=1000)
    val_dataset = ThreeDomainDatasetPyTorch(val_sequences, vocab_size=1000)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Reduced for longer sequences
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    print(f"\n2. Creating toy Mixtral model...")
    config = MixtralConfig(
        vocab_size=1000,
        hidden_dim=512,
        intermediate_dim=2048,
        num_layers=4,
        num_experts=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,  # 512 / 8
        enable_chronovisor=True,
    )

    model = ChronovisorMixtralForCausalLM(config)
    print(f"   Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # Training config
    train_config = TrainingConfig(
        learning_rate=1e-4,
        max_steps=num_steps,
        batch_size=4,  # Reduced for longer sequences (128 tokens)
        lambda_balance=0.01,
        lambda_coherence=0.001,
        lambda_valley=0.0001,
        fast_geology=fast_geology,
        log_every_n_steps=100,
        save_every_n_steps=1000,
        use_coherence_gating=False,  # Disable for first pass to see raw dynamics
    )

    print(f"\n3. Training for {num_steps} steps...")
    trainer = ThreeDomainTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        output_dir="conversational_results",
    )

    trainer.train(num_steps=num_steps)

    print("\n✅ Experiment complete!")
    print("   Check conversational_results/ for plots and results")


if __name__ == "__main__":
    # Run with fast geology first
    run_experiment(fast_geology=True, num_steps=2000)
