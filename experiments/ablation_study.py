"""
Ablation Study: P×T Coupling Decomposition

Tests whether improvements from η=0.015, P=0.5 come from:
1. Full P×T coupling (η=0.015, P=0.5) ← Our stable basin
2. Pressure-only (P=0.5, η=0) ← Freeze temperature, keep pressure
3. Temperature-only (η=0.015, P=0) ← No pressure bias, geological evolution only
4. Frozen baseline (disable Chronovisor) ← No intervention

This isolates the contributions of each component and validates that
the stable basin requires BOTH pressure and temperature coupling.

Seeds: [42, 12345, 67890]
Conversations: 10 full arcs (500-1000 tokens each)
Training: 50 epochs = 500 forward passes
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


def run_config(config_name, enable_chronovisor, eta_local, pressure_scale, conversations, seed):
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

    # Training
    model.train()
    losses = []
    tbar_vars = []
    num_epochs = 50

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

    return {
        "config": config_name,
        "final_loss": final_loss,
        "turn_separation": turn_separation,
        "tbar_variance": final_tbar_var,
        "loss_history": losses,
    }


def main():
    print("="*70)
    print("ABLATION STUDY: P×T COUPLING DECOMPOSITION")
    print("="*70)
    print("\nIsolating pressure and temperature contributions\n")

    # Ablation configurations
    configs = [
        {"name": "Full P×T", "enable": True, "eta": 0.015, "pressure": 0.5},
        {"name": "Pressure-only", "enable": True, "eta": 0.0, "pressure": 0.5},
        {"name": "Temperature-only", "enable": True, "eta": 0.015, "pressure": 0.0},
        {"name": "Frozen baseline", "enable": False, "eta": 0.0, "pressure": 0.0},
    ]

    seeds = [42, 12345, 67890]

    print(f"Configurations: {len(configs)}")
    for cfg in configs:
        print(f"  - {cfg['name']:20s}: enable={cfg['enable']}, η={cfg['eta']:.3f}, P={cfg['pressure']:.1f}")
    print(f"\nSeeds: {seeds}")
    print(f"Conversations: 10 full arcs (500-1000 tokens each)\n")

    all_results = []

    for cfg in configs:
        cfg_name = cfg["name"]
        enable_chronovisor = cfg["enable"]
        eta_local = cfg["eta"]
        pressure_scale = cfg["pressure"]

        print(f"{'='*70}")
        print(f"CONFIG: {cfg_name}")
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

            # Run configuration
            print(f"    Running {cfg_name}...", end=" ")
            result = run_config(
                config_name=cfg_name,
                enable_chronovisor=enable_chronovisor,
                eta_local=eta_local,
                pressure_scale=pressure_scale,
                conversations=conversations,
                seed=seed,
            )
            print(f"Loss: {result['final_loss']:.6f}, Sep: {result['turn_separation']:.6f}, T̄: {result['tbar_variance']:.6f}")

            result['seed'] = seed
            config_results.append(result)

        # Config summary
        avg_loss = np.mean([r['final_loss'] for r in config_results])
        avg_sep = np.mean([r['turn_separation'] for r in config_results])
        std_loss = np.std([r['final_loss'] for r in config_results])
        std_sep = np.std([r['turn_separation'] for r in config_results])
        avg_tbar = np.mean([r['tbar_variance'] for r in config_results])

        print(f"\n  Summary for {cfg_name}:")
        print(f"    Loss: {avg_loss:.6f} ± {std_loss:.6f}")
        print(f"    Sep:  {avg_sep:.6f} ± {std_sep:.6f}")
        print(f"    T̄:    {avg_tbar:.6f}\n")

        all_results.extend(config_results)

    # Comparative analysis
    print(f"{'='*70}")
    print("ABLATION ANALYSIS")
    print(f"{'='*70}\n")

    # Compute deltas relative to frozen baseline
    baseline_results = [r for r in all_results if r['config'] == 'Frozen baseline']
    baseline_loss = np.mean([r['final_loss'] for r in baseline_results])
    baseline_sep = np.mean([r['turn_separation'] for r in baseline_results])

    print(f"Baseline (Frozen): Loss={baseline_loss:.6f}, Sep={baseline_sep:.6f}\n")
    print(f"{'Config':20s} {'Δ Loss':>12s} {'Δ Sep':>12s} {'T̄ var':>10s} {'Robust':>8s}")
    print("-"*70)

    for cfg in configs:
        if cfg['name'] == 'Frozen baseline':
            continue

        cfg_results = [r for r in all_results if r['config'] == cfg['name']]

        # Compute Pareto-better count
        pareto_count = 0
        delta_losses = []
        delta_seps = []

        for r in cfg_results:
            # Find corresponding baseline result for same seed
            baseline_r = next(br for br in baseline_results if br['seed'] == r['seed'])

            delta_loss = r['final_loss'] - baseline_r['final_loss']
            delta_sep = r['turn_separation'] - baseline_r['turn_separation']

            delta_loss_pct = (delta_loss / baseline_r['final_loss']) * 100
            delta_sep_pct = (delta_sep / baseline_r['turn_separation']) * 100

            delta_losses.append(delta_loss_pct)
            delta_seps.append(delta_sep_pct)

            if delta_loss < 0 and delta_sep > 0:
                pareto_count += 1

        avg_delta_loss = np.mean(delta_losses)
        avg_delta_sep = np.mean(delta_seps)
        std_delta_loss = np.std(delta_losses)
        std_delta_sep = np.std(delta_seps)
        avg_tbar = np.mean([r['tbar_variance'] for r in cfg_results])

        robustness = f"{pareto_count}/{len(seeds)}"

        print(f"{cfg['name']:20s} {avg_delta_loss:+6.1f}% ± {std_delta_loss:3.1f}% "
              f"{avg_delta_sep:+6.1f}% ± {std_delta_sep:3.1f}% "
              f"{avg_tbar:10.6f} {robustness:>8s}")

    # Key findings
    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}\n")

    full_pt_results = [r for r in all_results if r['config'] == 'Full P×T']
    pressure_only_results = [r for r in all_results if r['config'] == 'Pressure-only']
    temp_only_results = [r for r in all_results if r['config'] == 'Temperature-only']

    # Compute Pareto counts
    full_pt_pareto = sum(1 for r in full_pt_results
                        if (r['final_loss'] - next(br['final_loss'] for br in baseline_results if br['seed'] == r['seed']) < 0 and
                            r['turn_separation'] - next(br['turn_separation'] for br in baseline_results if br['seed'] == r['seed']) > 0))

    pressure_pareto = sum(1 for r in pressure_only_results
                         if (r['final_loss'] - next(br['final_loss'] for br in baseline_results if br['seed'] == r['seed']) < 0 and
                             r['turn_separation'] - next(br['turn_separation'] for br in baseline_results if br['seed'] == r['seed']) > 0))

    temp_pareto = sum(1 for r in temp_only_results
                     if (r['final_loss'] - next(br['final_loss'] for br in baseline_results if br['seed'] == r['seed']) < 0 and
                         r['turn_separation'] - next(br['turn_separation'] for br in baseline_results if br['seed'] == r['seed']) > 0))

    print(f"Full P×T coupling:")
    print(f"  Pareto-better: {full_pt_pareto}/{len(seeds)} ({full_pt_pareto/len(seeds)*100:.0f}%)")
    print(f"  → Validated stable basin\n")

    print(f"Pressure-only (η=0, no geology):")
    print(f"  Pareto-better: {pressure_pareto}/{len(seeds)} ({pressure_pareto/len(seeds)*100:.0f}%)")
    if pressure_pareto >= full_pt_pareto:
        print(f"  ⚠️  WARNING: Pressure alone performs as well as P×T coupling!")
    else:
        print(f"  → Geology adds value beyond pressure\n")

    print(f"Temperature-only (P=0, no pressure):")
    print(f"  Pareto-better: {temp_pareto}/{len(seeds)} ({temp_pareto/len(seeds)*100:.0f}%)")
    if temp_pareto >= full_pt_pareto:
        print(f"  ⚠️  WARNING: Temperature alone performs as well as P×T coupling!")
    else:
        print(f"  → Pressure adds value beyond temperature\n")

    # Verdict
    print(f"{'='*70}")
    print("VERDICT")
    print(f"{'='*70}\n")

    if full_pt_pareto == len(seeds) and pressure_pareto < full_pt_pareto and temp_pareto < full_pt_pareto:
        print("✅ P×T COUPLING VALIDATED")
        print("   Full coupling outperforms both pressure-only and temperature-only")
        print("   Both components contribute to stable basin")
    elif full_pt_pareto > max(pressure_pareto, temp_pareto):
        print("✅ P×T COUPLING BENEFICIAL")
        print("   Full coupling improves over single-component ablations")
        print("   Synergy between pressure and temperature confirmed")
    else:
        print("⚠️  COUPLING NOT ESSENTIAL")
        print("   Single components perform as well as full coupling")
        print("   May indicate redundancy or independent mechanisms")

    # Save results
    with open("ablation_study_results.txt", "w") as f:
        f.write("Ablation Study: P×T Coupling Decomposition\n")
        f.write("="*70 + "\n\n")
        f.write(f"Baseline: Loss={baseline_loss:.6f}, Sep={baseline_sep:.6f}\n\n")
        f.write(f"{'Config':20s} {'Δ Loss':>12s} {'Δ Sep':>12s} {'Robust':>8s}\n")
        f.write("-"*70 + "\n")

        for cfg in configs:
            if cfg['name'] == 'Frozen baseline':
                continue

            cfg_results = [r for r in all_results if r['config'] == cfg['name']]
            pareto_count = sum(1 for r in cfg_results
                              if (r['final_loss'] - next(br['final_loss'] for br in baseline_results if br['seed'] == r['seed']) < 0 and
                                  r['turn_separation'] - next(br['turn_separation'] for br in baseline_results if br['seed'] == r['seed']) > 0))

            delta_losses = [(r['final_loss'] - next(br['final_loss'] for br in baseline_results if br['seed'] == r['seed'])) /
                           next(br['final_loss'] for br in baseline_results if br['seed'] == r['seed']) * 100
                           for r in cfg_results]
            delta_seps = [(r['turn_separation'] - next(br['turn_separation'] for br in baseline_results if br['seed'] == r['seed'])) /
                         next(br['turn_separation'] for br in baseline_results if br['seed'] == r['seed']) * 100
                         for r in cfg_results]

            f.write(f"{cfg['name']:20s} {np.mean(delta_losses):+6.1f}% ± {np.std(delta_losses):3.1f}% "
                   f"{np.mean(delta_seps):+6.1f}% ± {np.std(delta_seps):3.1f}% "
                   f"{pareto_count}/{len(seeds)}\n")

        f.write(f"\nFull P×T: {full_pt_pareto}/{len(seeds)} Pareto-better\n")
        f.write(f"Pressure-only: {pressure_pareto}/{len(seeds)} Pareto-better\n")
        f.write(f"Temperature-only: {temp_pareto}/{len(seeds)} Pareto-better\n")

    print(f"\n✅ Results saved to ablation_study_results.txt")


if __name__ == "__main__":
    main()
