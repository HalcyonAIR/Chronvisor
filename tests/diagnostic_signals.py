"""
Diagnostic: Check what signals are actually available in chrono_state.

This will help us understand why entropy is still placeholder (0.5).
"""

import torch

from chronomoe.clock_gated_multistep import ClockGatedMultistepModel
from chronomoe.chronovisor_mixtral_bridge import MixtralConfig


def inspect_chrono_state():
    """Inspect what's actually in chrono_state."""
    print("=" * 70)
    print("Diagnostic: Chrono State Signal Inspection")
    print("=" * 70)
    print()

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

    model = ClockGatedMultistepModel(config)

    # Generate some input
    input_ids = torch.randint(0, config.vocab_size, (1, 10))

    # Run one forward pass
    print("Running forward pass...")
    with torch.no_grad():
        embeddings = model.embed_tokens(input_ids)
        hidden_states, chrono_state = model.model.forward(
            embeddings, update_chronovisor=True
        )

    print("\nChrono State Contents:")
    print("-" * 70)

    # Check routing_entropy
    print(f"\nrouting_entropy keys: {list(chrono_state.routing_entropy.keys())}")
    print("routing_entropy values:")
    for layer_idx, entropy in chrono_state.routing_entropy.items():
        print(f"  Layer {layer_idx}: {entropy:.4f}")

    # Check expert_usage
    print(f"\nexpert_usage keys: {list(chrono_state.expert_usage.keys())}")
    print("expert_usage shapes:")
    for layer_idx, usage in chrono_state.expert_usage.items():
        if isinstance(usage, torch.Tensor):
            print(f"  Layer {layer_idx}: {usage.shape} (tensor)")
        else:
            print(f"  Layer {layer_idx}: {len(usage)} (array)")

    # Check coherence
    print(f"\ncoherence: {chrono_state.coherence:.4f}")
    print(f"delta_coherence: {chrono_state.delta_coherence:.4f}")

    # Try extracting signals like we do in generation
    print("\n" + "=" * 70)
    print("Signal Extraction Test")
    print("-" * 70)

    last_layer_idx = config.num_layers - 1

    routing_entropy = chrono_state.routing_entropy.get(last_layer_idx, None)
    print(f"\nLast layer ({last_layer_idx}) routing entropy: {routing_entropy}")

    usage = chrono_state.expert_usage.get(last_layer_idx, None)
    if usage is not None:
        if isinstance(usage, torch.Tensor):
            usage_np = usage.cpu().numpy()
        else:
            usage_np = usage

        print(f"Last layer expert usage: {usage_np}")

        # Compute margin
        sorted_usage = sorted(usage_np, reverse=True)
        margin = sorted_usage[0] - sorted_usage[1] if len(sorted_usage) >= 2 else 0.0
        print(f"Computed margin: {margin:.4f}")
    else:
        print("Last layer expert usage: None")

    print("\n" + "=" * 70)
    print("Diagnosis Complete")
    print("=" * 70)


if __name__ == "__main__":
    inspect_chrono_state()
