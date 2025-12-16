#!/usr/bin/env python3
"""
Capture Routing Trajectories for Takens Analysis

Purpose: Save complete routing probability histories for FNN diagnostics.
NOT a performance test - just data capture.

Runs three conditions:
1. Mixtral 2L/8E + Chronovisor (η=0.015, P=0.5)
2. Mixtral 2L/8E baseline (no Chronovisor)
3. Switch 2L/8E + Chronovisor (top-1 routing)

Output per condition:
- routing_probs: [num_steps, batch, seq_len, num_experts]
- metadata: step indices, timestamps, config

Single seed (42) for reproducibility.
Minimal epochs (20) - just need trajectory, not convergence.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch
import numpy as np
from pathlib import Path
import time
from typing import Dict, List

from chronomoe.mixtral_core import MixtralConfig
from chronomoe.switch_core import SwitchConfig
from chronomoe.chronovisor_mixtral_bridge import ChronovisorMixtralForCausalLM
from chronomoe.chronovisor_switch_bridge import ChronovisorSwitchForCausalLM
from generate_long_geeky_conversations import LongGeekyConversationGenerator


class RoutingCapture:
    """
    Captures routing probabilities during training.

    Hooks into router to save probabilities at sampled intervals.
    """

    def __init__(self, sample_every: int = 1):
        """
        Args:
            sample_every: Record every N forward passes (1 = every pass)
        """
        self.entropy_history = []  # Scalar entropy per step
        self.step_indices = []
        self.timestamps = []
        self.start_time = time.time()
        self.call_counter = 0  # Total number of forward passes
        self.sample_every = sample_every

    def record(self, routing_probs: torch.Tensor, step_idx: int):
        """
        Record routing entropy if sampling condition is met.

        Args:
            routing_probs: [batch, seq_len, num_experts] tensor (RAW, no normalization)
            step_idx: Actual training step index
        """
        self.call_counter += 1

        # Only record every sample_every calls
        if self.call_counter % self.sample_every != 0:
            return

        # Compute mean routing entropy for this step
        eps = 1e-10
        entropy = -torch.sum(routing_probs * torch.log(routing_probs + eps), dim=-1)
        mean_entropy = entropy.mean().item()  # Average over batch and sequence

        # Save scalar entropy and metadata
        self.entropy_history.append(mean_entropy)
        self.step_indices.append(step_idx)
        self.timestamps.append(time.time() - self.start_time)

    def save(self, output_path: str, metadata: Dict = None):
        """
        Save captured data.

        Args:
            output_path: Where to save (without extension)
            metadata: Additional info to save
        """
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert to arrays
        entropy_array = np.array(self.entropy_history)  # [num_steps]
        steps_array = np.array(self.step_indices)
        timestamps_array = np.array(self.timestamps)

        # Save data
        np.save(f"{output_path}_entropy.npy", entropy_array)
        np.save(f"{output_path}_steps.npy", steps_array)
        np.save(f"{output_path}_timestamps.npy", timestamps_array)

        # Save metadata
        if metadata is None:
            metadata = {}
        metadata['num_steps'] = len(self.entropy_history)
        metadata['entropy_mean'] = np.mean(entropy_array)
        metadata['entropy_std'] = np.std(entropy_array)

        np.save(f"{output_path}_metadata.npy", metadata)

        print(f"\n✓ Saved routing trajectory:")
        print(f"  Path: {output_path}")
        print(f"  Steps: {len(self.step_indices)}")
        print(f"  Mean entropy: {metadata['entropy_mean']:.4f} ± {metadata['entropy_std']:.4f}")


def capture_mixtral_routing(
    enable_chronovisor: bool,
    eta: float = 0.015,
    pressure_scale: float = 0.5,
    seed: int = 42,
    num_conversations: int = 10,
    num_epochs: int = 20,
    sample_every: int = 5
) -> RoutingCapture:
    """
    Capture routing from Mixtral model.

    Args:
        enable_chronovisor: Whether to enable P×T coupling
        eta: Geological learning rate
        pressure_scale: Pressure scaling
        seed: Random seed
        num_conversations: Number of training sequences
        num_epochs: Number of epochs
        sample_every: Record every N forward passes (avoids oversampling)

    Returns:
        capture: RoutingCapture object with history
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Configuration
    config = MixtralConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=1024,
        num_layers=2,
        num_experts=8,
        num_experts_per_token=2,  # top-2
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        max_seq_length=2048,
        enable_chronovisor=enable_chronovisor,
    )

    # Create model
    model = ChronovisorMixtralForCausalLM(config)

    # Configure Chronovisor if enabled
    if enable_chronovisor:
        controller = model.model.controller
        controller.eta_structural_T_local = eta
        controller.eta_structural_T_global = eta / 2.0
        controller.pressure_scale = pressure_scale

        for lens in controller.lenses.values():
            lens.eta_structural_T = eta

    # Generate data
    gen = LongGeekyConversationGenerator(vocab_size=config.vocab_size)
    dataset = gen.generate_dataset(num_conversations=num_conversations)
    conversations = dataset['sequences']

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Capture object
    capture = RoutingCapture(sample_every=sample_every)

    # Hook to capture routing (will be called every forward pass)
    # Note: global_step tracking happens in training loop
    current_step = {'value': 0}  # Mutable container for closure

    def make_routing_hook(layer_idx):
        def hook(module, input, output):
            # output = (routing_weights, selected_experts)
            # We want the full softmax probabilities before top-k
            # These are in the router's forward pass
            # For now, capture the routing weights (top-k selected)
            routing_weights, selected_experts = output

            # Record with actual step index (capture handles sampling)
            # Save RAW routing_weights - no normalization
            capture.record(routing_weights, step_idx=current_step['value'])

        return hook

    # Register hook on first layer's router
    layer0_router = model.model.layers[0].moe.router
    layer0_router.register_forward_hook(make_routing_hook(0))

    # Training loop
    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        for conv_idx, conv_data in enumerate(conversations):
            input_ids = torch.from_numpy(conv_data['input_ids']).long().unsqueeze(0)
            labels = torch.from_numpy(conv_data['labels']).long().unsqueeze(0)

            # Update step counter for hook
            current_step['value'] = global_step

            # Forward pass
            optimizer.zero_grad()
            logits, chrono_state = model(input_ids, update_chronovisor=enable_chronovisor)

            # Loss
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            loss = torch.nn.functional.cross_entropy(logits_flat, labels_flat, ignore_index=-100)

            # Backward
            loss.backward()
            optimizer.step()

            global_step += 1

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Steps captured: {len(capture.entropy_history)}")

    return capture


def capture_switch_routing(
    eta: float = 0.015,
    pressure_scale: float = 0.5,
    seed: int = 42,
    num_conversations: int = 10,
    num_epochs: int = 20,
    sample_every: int = 5
) -> RoutingCapture:
    """
    Capture routing from Switch Transformer (top-1).

    Args:
        eta: Geological learning rate
        pressure_scale: Pressure scaling
        seed: Random seed
        num_conversations: Number of training sequences
        num_epochs: Number of epochs
        sample_every: Record every N forward passes (avoids oversampling)

    Returns:
        capture: RoutingCapture object with history
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Configuration
    config = SwitchConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=1024,
        num_layers=2,
        num_experts=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        max_seq_length=2048,
        enable_chronovisor=True,  # Always enable for Switch (to test failure mode)
        router_aux_loss_coef=0.01,
    )

    # Create model
    model = ChronovisorSwitchForCausalLM(config)

    # Configure Chronovisor
    controller = model.model.controller
    controller.eta_structural_T_local = eta
    controller.eta_structural_T_global = eta / 2.0
    controller.pressure_scale = pressure_scale

    for lens in controller.lenses.values():
        lens.eta_structural_T = eta

    # Generate data
    gen = LongGeekyConversationGenerator(vocab_size=config.vocab_size)
    dataset = gen.generate_dataset(num_conversations=num_conversations)
    conversations = dataset['sequences']

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Capture object
    capture = RoutingCapture(sample_every=sample_every)

    # Hook to capture routing (top-1 probabilities)
    # Note: global_step tracking happens in training loop
    current_step = {'value': 0}  # Mutable container for closure

    def make_routing_hook(layer_idx):
        def hook(module, input, output):
            routing_weights, selected_experts, router_probs, aux_loss = output
            # For Switch, capture full router_probs (before top-1 selection)
            # Save RAW router_probs - no normalization
            capture.record(router_probs, step_idx=current_step['value'])

        return hook

    # Register hook on first layer's router
    layer0_router = model.model.layers[0].moe.router
    layer0_router.register_forward_hook(make_routing_hook(0))

    # Training loop
    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        for conv_idx, conv_data in enumerate(conversations):
            input_ids = torch.from_numpy(conv_data['input_ids']).long().unsqueeze(0)
            labels = torch.from_numpy(conv_data['labels']).long().unsqueeze(0)

            # Update step counter for hook
            current_step['value'] = global_step

            # Forward pass (Switch returns total_loss, aux_loss)
            optimizer.zero_grad()
            total_loss, aux_loss = model.compute_loss(input_ids, labels)

            # Backward
            total_loss.backward()
            optimizer.step()

            global_step += 1

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item():.4f}, Steps captured: {len(capture.entropy_history)}")

    return capture


def main():
    """
    Capture routing trajectories for all three conditions.
    """
    print("=" * 70)
    print("ROUTING TRAJECTORY CAPTURE FOR TAKENS DIAGNOSTICS")
    print("=" * 70)
    print()
    print("Capturing routing probabilities for FNN analysis.")
    print("Three conditions: Mixtral+Chrono, Mixtral baseline, Switch top-1")
    print()

    seed = 42
    num_conversations = 10
    num_epochs = 20  # Minimal - just need trajectory
    sample_every = 5  # Avoid oversampling, per Halcyon's feedback

    output_dir = Path("takens_data")
    output_dir.mkdir(exist_ok=True)

    # Condition 1: Mixtral with Chronovisor
    print("-" * 70)
    print("CONDITION 1: Mixtral 2L/8E + Chronovisor (top-2)")
    print("-" * 70)
    print(f"Config: η=0.015, P=0.5, seed={seed}, sample_every={sample_every}")
    print()

    capture1 = capture_mixtral_routing(
        enable_chronovisor=True,
        eta=0.015,
        pressure_scale=0.5,
        seed=seed,
        num_conversations=num_conversations,
        num_epochs=num_epochs,
        sample_every=sample_every
    )

    capture1.save(
        str(output_dir / "mixtral_chronovisor"),
        metadata={
            'condition': 'mixtral_2l8e_chronovisor',
            'eta': 0.015,
            'pressure_scale': 0.5,
            'seed': seed,
            'routing_type': 'top-2',
        }
    )

    # Condition 2: Mixtral baseline
    print("\n" + "-" * 70)
    print("CONDITION 2: Mixtral 2L/8E Baseline (no Chronovisor)")
    print("-" * 70)
    print(f"Config: seed={seed}, sample_every={sample_every}")
    print()

    capture2 = capture_mixtral_routing(
        enable_chronovisor=False,
        seed=seed,
        num_conversations=num_conversations,
        num_epochs=num_epochs,
        sample_every=sample_every
    )

    capture2.save(
        str(output_dir / "mixtral_baseline"),
        metadata={
            'condition': 'mixtral_2l8e_baseline',
            'seed': seed,
            'routing_type': 'top-2',
        }
    )

    # Condition 3: Switch top-1
    print("\n" + "-" * 70)
    print("CONDITION 3: Switch Transformer 2L/8E (top-1)")
    print("-" * 70)
    print(f"Config: η=0.015, P=0.5, seed={seed}, sample_every={sample_every}")
    print()

    capture3 = capture_switch_routing(
        eta=0.015,
        pressure_scale=0.5,
        seed=seed,
        num_conversations=num_conversations,
        num_epochs=num_epochs,
        sample_every=sample_every
    )

    capture3.save(
        str(output_dir / "switch_top1"),
        metadata={
            'condition': 'switch_2l8e_top1',
            'eta': 0.015,
            'pressure_scale': 0.5,
            'seed': seed,
            'routing_type': 'top-1',
        }
    )

    # Summary
    print("\n" + "=" * 70)
    print("CAPTURE COMPLETE")
    print("=" * 70)
    print()
    print("Saved routing trajectories:")
    print(f"  {output_dir / 'mixtral_chronovisor_routing.npy'}")
    print(f"  {output_dir / 'mixtral_baseline_routing.npy'}")
    print(f"  {output_dir / 'switch_top1_routing.npy'}")
    print()
    print("Next: Run takens_diagnostics.py with real_data=True")
    print()


if __name__ == '__main__':
    main()
