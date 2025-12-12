"""
Sweet Spot Validation on Long Conversations

Tests the top 3 configs from refined sweet spot sweep on full conversational arcs
to determine which configurations are both STRONG and ROBUST.

Configs tested:
- η=0.010, P=1.0 (best combined: -14.4% loss, +16.4% sep on fragments)
- η=0.015, P=0.5 (best loss: -24.7% loss, +4.9% sep on fragments)
- η=0.010, P=0.7 (best separation: -4.0% loss, +17.4% sep on fragments)

Seeds: [42, 12345, 67890]
Conversations: 10 full arcs (500-1000 tokens each)
Training: 50 epochs = 500 forward passes

This is the critical test: where fragment peaks meet conversational stability.
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
    print("SWEET SPOT VALIDATION ON LONG CONVERSATIONS")
    print("="*70)
    print("\nTesting top 3 configs from refined sweep on conversational flow\n")

    # Configurations to test (from refined sweet spot sweep)
    configs = [
        {"eta": 0.010, "pressure": 1.0, "name": "Best Combined"},
        {"eta": 0.015, "pressure": 0.5, "name": "Best Loss"},
        {"eta": 0.010, "pressure": 0.7, "name": "Best Separation"},
    ]

    seeds = [42, 12345, 67890]

    print(f"Configs: {len(configs)}")
    for cfg in configs:
        print(f"  - {cfg['name']}: η={cfg['eta']:.3f}, P={cfg['pressure']:.1f}")
    print(f"\nSeeds: {seeds}")
    print(f"Conversations: 10 full arcs (500-1000 tokens each)\n")

    all_results = []

    for cfg in configs:
        eta_local = cfg["eta"]
        pressure_scale = cfg["pressure"]
        cfg_name = cfg["name"]

        print(f"{'='*70}")
        print(f"CONFIG: {cfg_name} (η={eta_local:.3f}, P={pressure_scale:.1f})")
        print(f"{'='*70}\n")

        config_results = []

        for seed in seeds:
            print(f"  Seed {seed}:")

            # Generate long conversations with this seed
            torch.manual_seed(seed)
            np.random.seed(seed)

            gen = LongGeekyConversationGenerator(vocab_size=1000, min_length=500, max_length=1000)
            dataset = gen.generate_dataset(num_conversations=10)
            conversations = dataset["sequences"]

            print(f"    Generated {len(conversations)} conversations (avg {dataset['avg_length']:.0f} tokens)")

            # Run frozen baseline
            print(f"    Running frozen baseline...", end=" ")
            frozen = run_config(
                enable_chronovisor=False,
                eta_local=eta_local,
                pressure_scale=pressure_scale,
                conversations=conversations,
                seed=seed,
            )
            print(f"Loss: {frozen['final_loss']:.6f}, Sep: {frozen['turn_separation']:.6f}")

            # Run live geology
            print(f"    Running live geology...", end=" ")
            live = run_config(
                enable_chronovisor=True,
                eta_local=eta_local,
                pressure_scale=pressure_scale,
                conversations=conversations,
                seed=seed,
            )
            print(f"Loss: {live['final_loss']:.6f}, Sep: {live['turn_separation']:.6f}, T̄: {live['tbar_variance']:.6f}")

            # Compute deltas
            delta_loss = live['final_loss'] - frozen['final_loss']
            delta_sep = live['turn_separation'] - frozen['turn_separation']
            delta_loss_pct = (delta_loss / frozen['final_loss']) * 100
            delta_sep_pct = (delta_sep / frozen['turn_separation']) * 100

            is_pareto = delta_loss < 0 and delta_sep > 0

            result_data = {
                'config': cfg_name,
                'eta': eta_local,
                'pressure': pressure_scale,
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
            config_results.append(result_data)

            print(f"    → Δ Loss: {delta_loss_pct:+.1f}%, Δ Sep: {delta_sep_pct:+.1f}%, Pareto: {'✅' if is_pareto else '❌'}\n")

        # Config summary
        pareto_count = sum(1 for r in config_results if r['is_pareto'])
        avg_delta_loss = np.mean([r['delta_loss_pct'] for r in config_results])
        avg_delta_sep = np.mean([r['delta_sep_pct'] for r in config_results])
        std_delta_loss = np.std([r['delta_loss_pct'] for r in config_results])
        std_delta_sep = np.std([r['delta_sep_pct'] for r in config_results])
        avg_tbar_var = np.mean([r['tbar_var'] for r in config_results])

        print(f"  Summary for {cfg_name}:")
        print(f"    Pareto-better: {pareto_count}/{len(seeds)} ({pareto_count/len(seeds)*100:.0f}%)")
        print(f"    Δ Loss: {avg_delta_loss:+.1f}% ± {std_delta_loss:.1f}%")
        print(f"    Δ Sep:  {avg_delta_sep:+.1f}% ± {std_delta_sep:.1f}%")
        print(f"    T̄ var:  {avg_tbar_var:.6f}\n")

        all_results.extend(config_results)

    # Final comparison
    print(f"{'='*70}")
    print("FINAL COMPARISON: SWEET SPOT CONFIGS ON LONG CONVERSATIONS")
    print(f"{'='*70}\n")

    # Group by config
    for cfg in configs:
        cfg_name = cfg['name']
        cfg_results = [r for r in all_results if r['config'] == cfg_name]

        pareto_count = sum(1 for r in cfg_results if r['is_pareto'])
        avg_delta_loss = np.mean([r['delta_loss_pct'] for r in cfg_results])
        avg_delta_sep = np.mean([r['delta_sep_pct'] for r in cfg_results])
        std_delta_loss = np.std([r['delta_loss_pct'] for r in cfg_results])
        std_delta_sep = np.std([r['delta_sep_pct'] for r in cfg_results])

        robustness = pareto_count / len(seeds)

        print(f"{cfg_name:20s} (η={cfg['eta']:.3f}, P={cfg['pressure']:.1f}):")
        print(f"  Robustness: {pareto_count}/{len(seeds)} ({robustness*100:.0f}%)")
        print(f"  Δ Loss:     {avg_delta_loss:+.1f}% ± {std_delta_loss:.1f}%")
        print(f"  Δ Sep:      {avg_delta_sep:+.1f}% ± {std_delta_sep:.1f}%")

        # Quality marker
        if robustness >= 0.67 and abs(avg_delta_loss) > 1.0:
            print(f"  → 🎯 STABLE BASIN (high robustness + strong effect)")
        elif robustness >= 0.67:
            print(f"  → ✅ ROBUST (high robustness, modest effect)")
        elif abs(avg_delta_loss) > 5.0:
            print(f"  → ⚠️  ERRATIC PEAK (strong effect, low robustness)")
        else:
            print(f"  → ❌ WEAK (low robustness, modest effect)")
        print()

    # Save results
    with open("sweet_spot_long_conversations.txt", "w") as f:
        f.write("Sweet Spot Validation on Long Conversations\n")
        f.write("="*70 + "\n\n")
        f.write(f"Testing top 3 configs from refined sweep on conversational flow\n\n")

        for cfg in configs:
            cfg_name = cfg['name']
            cfg_results = [r for r in all_results if r['config'] == cfg_name]

            pareto_count = sum(1 for r in cfg_results if r['is_pareto'])
            avg_delta_loss = np.mean([r['delta_loss_pct'] for r in cfg_results])
            avg_delta_sep = np.mean([r['delta_sep_pct'] for r in cfg_results])
            std_delta_loss = np.std([r['delta_loss_pct'] for r in cfg_results])
            std_delta_sep = np.std([r['delta_sep_pct'] for r in cfg_results])

            f.write(f"{cfg_name} (η={cfg['eta']:.3f}, P={cfg['pressure']:.1f}):\n")
            f.write(f"  Pareto-better: {pareto_count}/{len(seeds)} ({pareto_count/len(seeds)*100:.0f}%)\n")
            f.write(f"  Δ Loss: {avg_delta_loss:+.1f}% ± {std_delta_loss:.1f}%\n")
            f.write(f"  Δ Sep:  {avg_delta_sep:+.1f}% ± {std_delta_sep:.1f}%\n\n")

    print(f"✅ Results saved to sweet_spot_long_conversations.txt")


if __name__ == "__main__":
    main()
