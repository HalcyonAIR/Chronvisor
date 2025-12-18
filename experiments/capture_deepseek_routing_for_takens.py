#!/usr/bin/env python3
"""
Capture DeepSeek Routing Trajectories for Takens Analysis

Tests the critical question: With 64 routed experts (vs Mixtral's 8),
does the attractor dimension stay at d=2?

If YES → Attractor dimension is about routing STRATEGY, not expert COUNT
If NO → More experts = more degrees of freedom in routing geometry

This experiment will reveal whether the d=2 finding is fundamental or
an artifact of small expert pools.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from pathlib import Path

from chronomoe.deepseek_core import DeepSeekConfig
from chronomoe.chronovisor_deepseek_bridge import ChronovisorDeepSeekForCausalLM
from generate_long_geeky_conversations import LongGeekyConversationGenerator


class DeepSeekRoutingCapture:
    """
    Capture routing entropy from DeepSeek's ROUTED experts only.

    Shared experts are always activated, so we only track the competitive
    routing decisions in the routed expert pool.
    """

    def __init__(self, sample_every: int = 5):
        self.entropy_history = []
        self.sample_every = sample_every
        self.call_counter = 0

    def record(self, router_probs: torch.Tensor, step_idx: int):
        """
        Record routing entropy from router probabilities.

        Args:
            router_probs: [batch, seq_len, num_routed_experts]
            step_idx: Current training step
        """
        self.call_counter += 1

        # Sample at specified cadence
        if self.call_counter % self.sample_every != 0:
            return

        # Compute entropy over routed experts
        eps = 1e-10
        entropy = -torch.sum(router_probs * torch.log(router_probs + eps), dim=-1)

        # Average over batch and sequence
        mean_entropy = entropy.mean().item()

        self.entropy_history.append(mean_entropy)

    def save(self, filepath: Path):
        """Save captured trajectory."""
        np.save(filepath, np.array(self.entropy_history))
        print(f"Saved {len(self.entropy_history)} steps to {filepath}")


def capture_deepseek_routing(
    config,
    condition_name: str,
    num_conversations: int = 10,
    num_epochs: int = 20,
    sample_every: int = 5
):
    """
    Capture routing trajectory for DeepSeek model.

    Args:
        config: DeepSeekConfig with Chronovisor enabled
        condition_name: Name for this condition (e.g., "DeepSeek+Chronovisor")
        num_conversations: Number of training conversations
        num_epochs: Number of training epochs
        sample_every: Sample every N forward passes
    """
    print(f"\nCapturing routing for: {condition_name}")
    print(f"  Routed experts: {config.num_routed_experts}")
    print(f"  Top-k: {config.num_experts_per_token}")
    print(f"  Sample cadence: every {sample_every} forward passes")

    # Create model
    torch.manual_seed(42)
    model = ChronovisorDeepSeekForCausalLM(config)

    # Generate conversations
    gen = LongGeekyConversationGenerator(vocab_size=config.vocab_size)
    dataset = gen.generate_dataset(num_conversations=num_conversations)
    conversations = [seq['input_ids'].tolist() for seq in dataset['sequences']]

    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create routing capture
    capture = DeepSeekRoutingCapture(sample_every=sample_every)

    # Training loop with routing capture
    print(f"  Training for {num_epochs} epochs...")
    step_idx = 0

    for epoch in range(num_epochs):
        for conv_tokens in conversations:
            input_ids = torch.tensor(conv_tokens[:-1]).unsqueeze(0)
            target_ids = torch.tensor(conv_tokens[1:]).unsqueeze(0)

            # Forward pass
            logits, chrono_state, aux_loss = model(input_ids, update_chronovisor=True)

            # Compute loss
            lm_loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, config.vocab_size),
                target_ids[:, 1:].reshape(-1),
                ignore_index=-100
            )
            total_loss = lm_loss + aux_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # CAPTURE ROUTING from chrono_state
            # Routing entropy already computed in forward pass
            if chrono_state is not None and chrono_state.routing_entropy:
                # Get entropy from layer 0 (representative)
                entropy = chrono_state.routing_entropy.get(0, 0.0)

                # Sample at specified cadence
                capture.call_counter += 1
                if capture.call_counter % capture.sample_every == 0:
                    capture.entropy_history.append(entropy)

            step_idx += 1

    print(f"  Captured {len(capture.entropy_history)} samples")

    # Save trajectory
    output_dir = Path(__file__).parent.parent / "takens_data"
    output_dir.mkdir(exist_ok=True)

    filename = f"deepseek_{condition_name.lower().replace(' ', '_').replace('+', '_')}_routing.npy"
    capture.save(output_dir / filename)

    return capture.entropy_history


def main():
    print("=" * 70)
    print("DEEPSEEK ROUTING CAPTURE FOR TAKENS ANALYSIS")
    print("=" * 70)
    print()
    print("Testing the critical question:")
    print("  With 64 routed experts, does d stay around 2?")
    print()
    print("If YES → Attractor dimension is about STRATEGY, not expert COUNT")
    print("If NO  → More experts = more degrees of freedom")
    print()

    # DeepSeek config with full 64 routed experts
    deepseek_config = DeepSeekConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=1024,
        num_layers=2,
        num_shared_experts=2,
        num_routed_experts=64,  # FULL SCALE - this is the test
        num_experts_per_token=6,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        max_seq_length=2048,
        enable_chronovisor=True,
        router_aux_loss_coef=0.001,
        router_z_loss_coef=0.001,
    )

    # Capture with Chronovisor
    print("CONDITION 1: DeepSeek + Chronovisor (64 routed experts)")
    capture_deepseek_routing(
        config=deepseek_config,
        condition_name="DeepSeek+Chronovisor",
        num_conversations=10,
        num_epochs=20,
        sample_every=5
    )

    # Capture baseline (Chronovisor enabled but pressure=0, no coupling)
    print("\nCONDITION 2: DeepSeek Baseline (64 routed experts, no P×T coupling)")
    baseline_config = DeepSeekConfig(
        vocab_size=deepseek_config.vocab_size,
        hidden_dim=deepseek_config.hidden_dim,
        intermediate_dim=deepseek_config.intermediate_dim,
        num_layers=deepseek_config.num_layers,
        num_shared_experts=deepseek_config.num_shared_experts,
        num_routed_experts=deepseek_config.num_routed_experts,
        num_experts_per_token=deepseek_config.num_experts_per_token,
        num_attention_heads=deepseek_config.num_attention_heads,
        num_key_value_heads=deepseek_config.num_key_value_heads,
        head_dim=deepseek_config.head_dim,
        max_seq_length=deepseek_config.max_seq_length,
        enable_chronovisor=True,  # Keep enabled to collect stats
        chronovisor_pressure_scale=0.0,  # But disable coupling
        router_aux_loss_coef=deepseek_config.router_aux_loss_coef,
        router_z_loss_coef=deepseek_config.router_z_loss_coef,
    )

    # Note: We'll set pressure_scale=0 in capture function to disable P×T
    # This keeps routing statistics collection while disabling coupling

    # Hack: create model, then disable pressure
    import torch
    torch.manual_seed(42)
    baseline_model = ChronovisorDeepSeekForCausalLM(baseline_config)
    baseline_model.model.controller.pressure_scale = 0.0

    # Now capture with disabled coupling
    print("  Note: Chronovisor enabled for statistics, but pressure_scale=0 (no coupling)")

    # Generate conversations
    from generate_long_geeky_conversations import LongGeekyConversationGenerator
    gen = LongGeekyConversationGenerator(vocab_size=baseline_config.vocab_size)
    dataset = gen.generate_dataset(num_conversations=10)
    conversations = [seq['input_ids'].tolist() for seq in dataset['sequences']]

    # Setup
    optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=1e-4)
    baseline_capture = DeepSeekRoutingCapture(sample_every=5)

    print(f"  Training for 20 epochs...")
    step_idx = 0

    for epoch in range(20):
        for conv_tokens in conversations:
            input_ids = torch.tensor(conv_tokens[:-1]).unsqueeze(0)
            target_ids = torch.tensor(conv_tokens[1:]).unsqueeze(0)

            # Forward
            logits, chrono_state, aux_loss = baseline_model(input_ids, update_chronovisor=True)

            # Loss
            lm_loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, baseline_config.vocab_size),
                target_ids[:, 1:].reshape(-1),
                ignore_index=-100
            )
            total_loss = lm_loss + aux_loss

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # CAPTURE
            if chrono_state is not None and chrono_state.routing_entropy:
                entropy = chrono_state.routing_entropy.get(0, 0.0)
                baseline_capture.call_counter += 1
                if baseline_capture.call_counter % baseline_capture.sample_every == 0:
                    baseline_capture.entropy_history.append(entropy)

            step_idx += 1

    print(f"  Captured {len(baseline_capture.entropy_history)} samples")

    # Save
    from pathlib import Path
    output_dir = Path(__file__).parent.parent / "takens_data"
    output_dir.mkdir(exist_ok=True)
    filename = "deepseek_baseline_routing.npy"
    baseline_capture.save(output_dir / filename)

    print()
    print("=" * 70)
    print("CAPTURE COMPLETE")
    print("=" * 70)
    print()
    print("Next step: Run analyze_routing_comprehensive.py on these captures")
    print("Key question: Does FNN converge at d=2 even with 64 experts?")
    print()
    print("If d ≈ 2:")
    print("  → Routing manifold is low-dimensional regardless of expert count")
    print("  → P×T coupling operates at fundamental geometric level")
    print("  → Attractor dimension reflects STRATEGY, not choice space size")
    print()
    print("If d >> 2:")
    print("  → More experts → higher-dimensional attractor")
    print("  → Previous d=2 finding may be artifact of small expert pool")
    print("  → Would need to reconsider geometric interpretation")
    print()


if __name__ == '__main__':
    main()
