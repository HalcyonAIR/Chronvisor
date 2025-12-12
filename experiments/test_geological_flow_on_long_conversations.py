"""
Geological Flow Test on Long Conversations

Tests if geological coupling stabilizes when tracking full conversational arcs
instead of short fragments.

Hypothesis: The lens system is designed to track conversational *flow*.
Short fragments don't give geology time to converge. Full conversations (500-1000 tokens)
should show:
1. More stable geological patterns
2. Better seed robustness
3. Clearer turn-based expert specialization

Compares frozen vs live on long geeky conversations.
"""

import sys
sys.path.insert(0, 'experiments')

import torch
import numpy as np
from torch.optim import AdamW
from chronomoe.chronovisor_mixtral_bridge import ChronovisorMixtralForCausalLM
from chronomoe.mixtral_core import MixtralConfig
from generate_long_geeky_conversations import LongGeekyConversationGenerator
from analyze_turn_usage import TurnUsageAnalyzer


class LongConversationDataset(torch.utils.data.Dataset):
    """PyTorch dataset for long conversations."""

    def __init__(self, conversations, vocab_size):
        self.conversations = conversations
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conv = self.conversations[idx]
        return {
            "input_ids": torch.from_numpy(conv["input_ids"]).long(),
            "labels": torch.from_numpy(conv["labels"]).long(),
            "turn_boundaries": conv["turn_boundaries"],
        }


def run_config(enable_chronovisor, eta_local, pressure_scale, conversations, seed):
    """Run single configuration with long conversations."""

    torch.manual_seed(seed + 1000)
    np.random.seed(seed + 1000)

    config = MixtralConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=1024,
        num_layers=2,
        num_experts=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        enable_chronovisor=enable_chronovisor,
    )

    model = ChronovisorMixtralForCausalLM(config)

    if enable_chronovisor:
        controller = model.model.controller
        controller.eta_structural_T_local = eta_local
        controller.eta_structural_T_global = eta_local / 2
        controller.pressure_scale = pressure_scale

        for lens in controller.lenses.values():
            lens.eta_structural_T = eta_local

    optimizer = AdamW(model.parameters(), lr=1e-4)
    dataset = LongConversationDataset(conversations, vocab_size=1000)

    # Training - each epoch processes all conversations
    model.train()
    losses = []
    tbar_vars = []
    num_epochs = 50  # 50 epochs * 10 conversations = 500 forward passes

    for epoch in range(num_epochs):
        for conv_idx in range(len(dataset)):
            batch = dataset[conv_idx]

            input_ids = batch["input_ids"].unsqueeze(0)
            labels = batch["labels"].unsqueeze(0)

            optimizer.zero_grad()
            logits, chrono_state = model(input_ids, update_chronovisor=enable_chronovisor)

            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            loss = torch.nn.functional.cross_entropy(logits_flat, labels_flat, ignore_index=-100)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Track T̄
            if chrono_state and hasattr(chrono_state, 'T_bar') and chrono_state.T_bar is not None:
                T_bar = chrono_state.T_bar
                if isinstance(T_bar, np.ndarray) and len(T_bar) > 0:
                    tbar_vars.append(T_bar.var())
                else:
                    tbar_vars.append(0.0)
            else:
                tbar_vars.append(0.0)

    # Post-training analysis
    model.eval()
    analyzer = TurnUsageAnalyzer(model, num_turns=7)

    for conv_idx in range(len(dataset)):
        batch = dataset[conv_idx]
        batch_expanded = {
            "input_ids": batch["input_ids"].unsqueeze(0),
            "labels": batch["labels"].unsqueeze(0),
            "turn_boundaries": [batch["turn_boundaries"]]
        }
        analyzer.analyze_batch(batch_expanded)

    usage_matrix = analyzer.get_usage_matrix(layer_idx=0)
    usage_normalized = usage_matrix / (usage_matrix.sum(axis=1, keepdims=True) + 1e-9)
    expert_variance = usage_normalized.var(axis=0)
    turn_separation = expert_variance.sum()

    # Final metrics
    final_loss = np.mean(losses[-50:])
    final_tbar_var = tbar_vars[-1] if tbar_vars else 0.0
    max_tbar_var = max(tbar_vars) if tbar_vars else 0.0

    return {
        "final_loss": final_loss,
        "turn_separation": turn_separation,
        "tbar_variance": final_tbar_var,
        "tbar_variance_max": max_tbar_var,
        "loss_history": losses,
    }


def main():
    print("="*70)
    print("GEOLOGICAL FLOW TEST ON LONG CONVERSATIONS")
    print("="*70)
    print("\nTesting if full conversational arcs stabilize geological coupling\n")

    # Configuration
    eta_local = 0.01
    pressure_scale = 0.5
    seeds = [42, 12345, 67890]  # Test 3 seeds

    print(f"Configuration: η={eta_local:.3f}, P={pressure_scale:.1f}")
    print(f"Seeds: {seeds}")
    print(f"Conversations: 10 full arcs (500-1000 tokens each)\n")

    results = []

    for seed in seeds:
        print(f"{'='*70}")
        print(f"Seed: {seed}")
        print(f"{'='*70}\n")

        # Generate long conversations with this seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        gen = LongGeekyConversationGenerator(vocab_size=1000, min_length=500, max_length=1000)
        dataset = gen.generate_dataset(num_conversations=10)
        conversations = dataset["sequences"]

        print(f"Generated {len(conversations)} conversations")
        print(f"  Avg length: {dataset['avg_length']:.1f} tokens")
        print(f"  Total tokens: {sum(c['length'] for c in conversations)}\n")

        # Run frozen baseline
        print("Running frozen baseline...")
        frozen = run_config(
            enable_chronovisor=False,
            eta_local=eta_local,
            pressure_scale=pressure_scale,
            conversations=conversations,
            seed=seed,
        )
        print(f"  Loss: {frozen['final_loss']:.6f}, Sep: {frozen['turn_separation']:.6f}")

        # Run live geology
        print("Running live geology...")
        live = run_config(
            enable_chronovisor=True,
            eta_local=eta_local,
            pressure_scale=pressure_scale,
            conversations=conversations,
            seed=seed,
        )
        print(f"  Loss: {live['final_loss']:.6f}, Sep: {live['turn_separation']:.6f}, T̄: {live['tbar_variance']:.6f}")

        # Compute deltas
        delta_loss = live['final_loss'] - frozen['final_loss']
        delta_sep = live['turn_separation'] - frozen['turn_separation']
        delta_loss_pct = (delta_loss / frozen['final_loss']) * 100
        delta_sep_pct = (delta_sep / frozen['turn_separation']) * 100

        is_pareto = delta_loss < 0 and delta_sep > 0

        result_data = {
            'seed': seed,
            'frozen_loss': frozen['final_loss'],
            'frozen_sep': frozen['turn_separation'],
            'live_loss': live['final_loss'],
            'live_sep': live['turn_separation'],
            'tbar_var': live['tbar_variance'],
            'delta_loss_pct': delta_loss_pct,
            'delta_sep_pct': delta_sep_pct,
            'is_pareto': is_pareto,
        }
        results.append(result_data)

        print(f"\n  Δ Loss: {delta_loss_pct:+.1f}%")
        print(f"  Δ Sep:  {delta_sep_pct:+.1f}%")
        print(f"  Pareto: {'✅ YES' if is_pareto else '❌ NO'}\n")

    # Summary
    print("="*70)
    print("LONG CONVERSATION FLOW RESULTS")
    print("="*70)

    print(f"\n{'Seed':>8s} {'Δ Loss':>10s} {'Δ Sep':>10s} {'T̄ var':>10s} {'Pareto':>8s}")
    print("-"*70)

    for r in results:
        pareto_marker = "✅" if r['is_pareto'] else "❌"
        print(f"{r['seed']:8d} {r['delta_loss_pct']:+9.1f}% {r['delta_sep_pct']:+9.1f}% "
              f"{r['tbar_var']:10.6f} {pareto_marker:>8s}")

    # Statistics
    pareto_count = sum(1 for r in results if r['is_pareto'])
    avg_delta_loss = np.mean([r['delta_loss_pct'] for r in results])
    avg_delta_sep = np.mean([r['delta_sep_pct'] for r in results])
    avg_tbar_var = np.mean([r['tbar_var'] for r in results])

    std_delta_loss = np.std([r['delta_loss_pct'] for r in results])
    std_delta_sep = np.std([r['delta_sep_pct'] for r in results])

    print(f"\n{'='*70}")
    print("STATISTICS")
    print("="*70)

    print(f"\nPareto-better seeds: {pareto_count}/{len(seeds)} ({pareto_count/len(seeds)*100:.0f}%)")
    print(f"\nΔ Loss (mean ± std): {avg_delta_loss:+.1f}% ± {std_delta_loss:.1f}%")
    print(f"Δ Sep  (mean ± std): {avg_delta_sep:+.1f}% ± {std_delta_sep:.1f}%")
    print(f"T̄ var  (mean):       {avg_tbar_var:.6f}")

    # Verdict
    print(f"\n{'='*70}")
    print("VERDICT: FRAGMENTS vs FULL CONVERSATIONS")
    print("="*70)

    print(f"\nLong conversations (500-1000 tokens):")
    print(f"  Pareto-better: {pareto_count}/{len(seeds)} ({pareto_count/len(seeds)*100:.0f}%)")
    print(f"  Δ Loss: {avg_delta_loss:+.1f}% ± {std_delta_loss:.1f}%")
    print(f"  Δ Sep:  {avg_delta_sep:+.1f}% ± {std_delta_sep:.1f}%")

    print(f"\nShort fragments (128 tokens, from earlier test):")
    print(f"  Pareto-better: 1/5 (20%)")
    print(f"  Δ Loss: +5.3% ± 10.3%")
    print(f"  Δ Sep:  +9.2% ± 27.0%")

    if pareto_count >= len(seeds) * 0.67:
        print(f"\n✅ HYPOTHESIS CONFIRMED")
        print(f"   Long conversational arcs improve geological stability")
        print(f"   The lens system tracks *flow*, not fragments")
    elif pareto_count > 1:
        print(f"\n⚠️  PARTIAL IMPROVEMENT")
        print(f"   Long conversations help but don't fully stabilize")
    else:
        print(f"\n❌ NO IMPROVEMENT")
        print(f"   Long conversations don't resolve seed sensitivity")

    # Save results
    with open("long_conversation_flow_test.txt", "w") as f:
        f.write("Long Conversation Geological Flow Test\n")
        f.write("="*70 + "\n\n")
        f.write(f"Configuration: η={eta_local:.3f}, P={pressure_scale:.1f}\n")
        f.write(f"Conversations: 10 full arcs (500-1000 tokens each)\n\n")
        f.write(f"Pareto-better: {pareto_count}/{len(seeds)} ({pareto_count/len(seeds)*100:.0f}%)\n")
        f.write(f"Δ Loss: {avg_delta_loss:+.1f}% ± {std_delta_loss:.1f}%\n")
        f.write(f"Δ Sep:  {avg_delta_sep:+.1f}% ± {std_delta_sep:.1f}%\n")
        f.write(f"T̄ var:  {avg_tbar_var:.6f}\n")

    print(f"\n✅ Results saved to long_conversation_flow_test.txt")


if __name__ == "__main__":
    main()
