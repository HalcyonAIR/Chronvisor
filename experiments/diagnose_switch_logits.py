#!/usr/bin/env python3
"""
Diagnose Switch Router Logits

Question from Halcyon: Do Switch router logits show near-ties before argmax,
or is it already a clean winner most of the time?

This tells us:
- Clean winner (high gap) → genuine cliff, softmax doesn't matter
- Near ties (low gap) → forced cliff, argmax destroying real competition

Output:
- Logit gap statistics (max - second_max)
- Softmax probability distribution (before top-1 selection)
- Comparison to Mixtral's top-2 logits
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from typing import List, Dict

from chronomoe.switch_core import SwitchConfig
from chronomoe.mixtral_core import MixtralConfig
from chronomoe.chronovisor_switch_bridge import ChronovisorSwitchForCausalLM
from chronomoe.chronovisor_mixtral_bridge import ChronovisorMixtralForCausalLM
from generate_long_geeky_conversations import LongGeekyConversationGenerator


class RouterLogitAnalyzer:
    """Captures and analyzes router logits before selection."""

    def __init__(self):
        self.logit_gaps = []  # max - second_max
        self.softmax_entropies = []  # Shannon entropy of full softmax
        self.top1_probs = []  # Probability of top expert
        self.top2_prob_sum = []  # Sum of top-2 probabilities

    def analyze_logits(self, router_logits: torch.Tensor):
        """
        Analyze router logits before top-k selection.

        Args:
            router_logits: [batch, seq_len, num_experts] raw logits
        """
        # Flatten to [N, num_experts]
        logits_flat = router_logits.reshape(-1, router_logits.size(-1))

        # Sort logits to get gaps
        sorted_logits, _ = torch.sort(logits_flat, dim=-1, descending=True)
        gap = sorted_logits[:, 0] - sorted_logits[:, 1]  # max - second_max

        # Compute softmax
        probs = torch.softmax(logits_flat, dim=-1)

        # Entropy
        eps = 1e-10
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)

        # Top-1 probability
        top1_prob = probs.max(dim=-1)[0]

        # Top-2 probability sum
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top2_sum = sorted_probs[:, 0] + sorted_probs[:, 1]

        # Record statistics
        self.logit_gaps.extend(gap.cpu().numpy())
        self.softmax_entropies.extend(entropy.cpu().numpy())
        self.top1_probs.extend(top1_prob.cpu().numpy())
        self.top2_prob_sum.extend(top2_sum.cpu().numpy())

    def summary(self) -> Dict[str, float]:
        """Compute summary statistics."""
        return {
            'gap_mean': np.mean(self.logit_gaps),
            'gap_std': np.std(self.logit_gaps),
            'gap_median': np.median(self.logit_gaps),
            'gap_min': np.min(self.logit_gaps),
            'gap_max': np.max(self.logit_gaps),
            'entropy_mean': np.mean(self.softmax_entropies),
            'entropy_std': np.std(self.softmax_entropies),
            'top1_prob_mean': np.mean(self.top1_probs),
            'top1_prob_std': np.std(self.top1_probs),
            'top2_sum_mean': np.mean(self.top2_prob_sum),
            'top2_sum_std': np.std(self.top2_prob_sum),
            'num_samples': len(self.logit_gaps),
        }


def diagnose_switch_logits(seed: int = 42, num_steps: int = 50) -> RouterLogitAnalyzer:
    """
    Run Switch model and capture router logits.

    Args:
        seed: Random seed
        num_steps: Number of forward passes to sample

    Returns:
        analyzer: RouterLogitAnalyzer with captured statistics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Configuration (same as test_switch_transformer.py)
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
        enable_chronovisor=True,
        router_aux_loss_coef=0.01,
    )

    model = ChronovisorSwitchForCausalLM(config)

    # Configure Chronovisor (same as tests)
    controller = model.model.controller
    controller.eta_structural_T_local = 0.015
    controller.eta_structural_T_global = 0.0075
    controller.pressure_scale = 0.5

    for lens in controller.lenses.values():
        lens.eta_structural_T = 0.015

    # Generate data
    gen = LongGeekyConversationGenerator(vocab_size=config.vocab_size)
    dataset = gen.generate_dataset(num_conversations=5)
    conversations = dataset['sequences']

    # Analyzer
    analyzer = RouterLogitAnalyzer()

    # Hook to capture logits BEFORE top-1 selection
    def make_logit_hook():
        def hook(module, input, output):
            # In Switch router, we need to intercept the logits before softmax/argmax
            # The router computes: logits = W_router @ hidden_states
            # Then: probs = softmax(logits)
            # Then: selected = argmax(probs)

            # output contains routing_weights, selected_experts, router_probs, aux_loss
            # We want to analyze router_probs (full softmax)
            routing_weights, selected_experts, router_probs, aux_loss = output

            # Reconstruct logits from probs (approximate via log)
            # Better: capture directly in router forward if possible
            # For now, analyze the router_probs to see if it's peaked
            analyzer.analyze_logits(torch.log(router_probs + 1e-10))

        return hook

    # Register hook on first layer
    layer0_router = model.model.layers[0].moe.router
    layer0_router.register_forward_hook(make_logit_hook())

    # Run forward passes
    model.eval()
    with torch.no_grad():
        for step in range(min(num_steps, len(conversations))):
            conv_data = conversations[step % len(conversations)]
            input_ids = torch.from_numpy(conv_data['input_ids']).long().unsqueeze(0)

            # Forward pass (Switch returns logits, chrono_state, aux_loss)
            logits, chrono_state, aux_loss = model(input_ids, update_chronovisor=False)

    return analyzer


def diagnose_mixtral_logits(seed: int = 42, num_steps: int = 50) -> RouterLogitAnalyzer:
    """
    Run Mixtral model and capture router logits for comparison.

    Args:
        seed: Random seed
        num_steps: Number of forward passes to sample

    Returns:
        analyzer: RouterLogitAnalyzer with captured statistics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Configuration (same as baseline tests)
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
        enable_chronovisor=True,
    )

    model = ChronovisorMixtralForCausalLM(config)

    # Configure Chronovisor
    controller = model.model.controller
    controller.eta_structural_T_local = 0.015
    controller.eta_structural_T_global = 0.0075
    controller.pressure_scale = 0.5

    for lens in controller.lenses.values():
        lens.eta_structural_T = 0.015

    # Generate data
    gen = LongGeekyConversationGenerator(vocab_size=config.vocab_size)
    dataset = gen.generate_dataset(num_conversations=5)
    conversations = dataset['sequences']

    # Analyzer
    analyzer = RouterLogitAnalyzer()

    # Hook to capture logits BEFORE top-k selection
    def make_logit_hook():
        def hook(module, input, output):
            # For Mixtral, output is (routing_weights, selected_experts)
            # We need to intercept the router to get full logits
            # For now, approximate from routing_weights
            routing_weights, selected_experts = output

            # Routing weights are already softmax probabilities
            # Approximate logits via log (not perfect but indicative)
            analyzer.analyze_logits(torch.log(routing_weights + 1e-10))

        return hook

    # Register hook on first layer
    layer0_router = model.model.layers[0].moe.router
    layer0_router.register_forward_hook(make_logit_hook())

    # Run forward passes
    model.eval()
    with torch.no_grad():
        for step in range(min(num_steps, len(conversations))):
            conv_data = conversations[step % len(conversations)]
            input_ids = torch.from_numpy(conv_data['input_ids']).long().unsqueeze(0)

            # Forward pass
            logits, chrono_state = model(input_ids, update_chronovisor=False)

    return analyzer


def main():
    """
    Diagnose router logits for Switch vs Mixtral.
    """
    print("=" * 70)
    print("SWITCH ROUTER LOGIT DIAGNOSTICS")
    print("=" * 70)
    print()
    print("Question: Are there near-ties in router logits, or clean winners?")
    print()
    print("Interpretation:")
    print("  • Large gap (>2) → genuine cliff, already peaked")
    print("  • Small gap (<0.5) → forced cliff, argmax destroys competition")
    print("  • Top-1 prob >0.9 → very confident, no real alternatives")
    print("  • Top-2 sum <0.7 → distributed, argmax forcing decision")
    print()

    seed = 42
    num_steps = 50

    # Switch analysis
    print("-" * 70)
    print("SWITCH TRANSFORMER (top-1)")
    print("-" * 70)
    print(f"Analyzing {num_steps} forward passes...")
    print()

    switch_analyzer = diagnose_switch_logits(seed=seed, num_steps=num_steps)
    switch_stats = switch_analyzer.summary()

    print(f"Router Logit Statistics:")
    print(f"  Logit gap (max - 2nd):  {switch_stats['gap_mean']:.4f} ± {switch_stats['gap_std']:.4f}")
    print(f"    Median: {switch_stats['gap_median']:.4f}")
    print(f"    Range:  [{switch_stats['gap_min']:.4f}, {switch_stats['gap_max']:.4f}]")
    print()
    print(f"  Softmax entropy:        {switch_stats['entropy_mean']:.4f} ± {switch_stats['entropy_std']:.4f}")
    print(f"  Top-1 probability:      {switch_stats['top1_prob_mean']:.4f} ± {switch_stats['top1_prob_std']:.4f}")
    print(f"  Top-2 prob sum:         {switch_stats['top2_sum_mean']:.4f} ± {switch_stats['top2_sum_std']:.4f}")
    print(f"  Samples analyzed:       {switch_stats['num_samples']}")
    print()

    # Mixtral comparison
    print("-" * 70)
    print("MIXTRAL (top-2) - For Comparison")
    print("-" * 70)
    print(f"Analyzing {num_steps} forward passes...")
    print()

    mixtral_analyzer = diagnose_mixtral_logits(seed=seed, num_steps=num_steps)
    mixtral_stats = mixtral_analyzer.summary()

    print(f"Router Logit Statistics:")
    print(f"  Logit gap (max - 2nd):  {mixtral_stats['gap_mean']:.4f} ± {mixtral_stats['gap_std']:.4f}")
    print(f"    Median: {mixtral_stats['gap_median']:.4f}")
    print(f"    Range:  [{mixtral_stats['gap_min']:.4f}, {mixtral_stats['gap_max']:.4f}]")
    print()
    print(f"  Softmax entropy:        {mixtral_stats['entropy_mean']:.4f} ± {mixtral_stats['entropy_std']:.4f}")
    print(f"  Top-1 probability:      {mixtral_stats['top1_prob_mean']:.4f} ± {mixtral_stats['top1_prob_std']:.4f}")
    print(f"  Top-2 prob sum:         {mixtral_stats['top2_sum_mean']:.4f} ± {mixtral_stats['top2_sum_std']:.4f}")
    print(f"  Samples analyzed:       {mixtral_stats['num_samples']}")
    print()

    # Interpretation
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()

    gap_ratio = switch_stats['gap_mean'] / (mixtral_stats['gap_mean'] + 1e-10)
    entropy_ratio = switch_stats['entropy_mean'] / (mixtral_stats['entropy_mean'] + 1e-10)

    print(f"Switch vs Mixtral:")
    print(f"  Gap ratio:     {gap_ratio:.2f}x")
    print(f"  Entropy ratio: {entropy_ratio:.2f}x")
    print()

    if switch_stats['gap_mean'] > 2.0:
        print("✓ GENUINE CLIFF")
        print("  Switch logits are already peaked (large gap).")
        print("  Softmax doesn't matter - it's already a winner-take-all.")
        print("  Temperature modulation has nothing to work with.")
    elif switch_stats['gap_mean'] < 0.5:
        print("✓ FORCED CLIFF")
        print("  Switch logits show competition (small gap).")
        print("  Argmax destroys real alternatives that softmax preserves.")
        print("  Temperature modulation WOULD work, but argmax prevents it.")
    else:
        print("~ MIXED")
        print("  Switch shows moderate gaps - some competition exists.")
        print("  Argmax still forces discrete decisions where softmax allows grading.")
        print(f"  Compare to Mixtral gap: {mixtral_stats['gap_mean']:.4f}")

    print()

    if switch_stats['top2_sum_mean'] < 0.7:
        print("Additionally: Routing is distributed (top-2 sum < 0.7)")
        print("  This means argmax is forcing decisions among genuinely competitive experts.")
        print("  The cliff is forced, not genuine.")
    elif switch_stats['top1_prob_mean'] > 0.9:
        print("Additionally: Routing is very confident (top-1 prob > 0.9)")
        print("  This means logits are already peaked - genuine winner emerges.")
        print("  The cliff exists in the logits, argmax just makes it explicit.")

    print()


if __name__ == '__main__':
    main()
