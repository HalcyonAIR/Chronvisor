#!/usr/bin/env python3
"""
Step 1: Prove Routing Structure Exists

Before testing path wear, we must establish:
1. Pretrained model has learned routing patterns
2. Coherent inputs produce structured (non-uniform) routing
3. Entropy << max, patterns repeat for similar inputs

This establishes: There is dirt, not water.

Options:
- Load pretrained Mixtral/DeepSeek from HuggingFace
- OR train a small model on toy task
- OR use synthetic "structured" inputs (repeated patterns)

We'll try all three approaches and document which works.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path

# Try to import transformers for pretrained models
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠ transformers not available - will try alternative approaches")


def test_approach_1_pretrained():
    """
    Approach 1: Load pretrained MoE from HuggingFace

    Candidates:
    - mistralai/Mixtral-8x7B-v0.1 (large)
    - mistralai/Mixtral-8x7B-Instruct-v0.1
    - deepseek-ai/deepseek-moe-16b-base
    """

    if not HAS_TRANSFORMERS:
        print("❌ Approach 1: transformers not available")
        return None

    print("="*70)
    print("APPROACH 1: Pretrained Model from HuggingFace")
    print("="*70)
    print()

    # Try to load smallest available MoE
    model_name = "mistralai/Mixtral-8x7B-v0.1"  # Will likely be too large

    print(f"Attempting to load: {model_name}")
    print("(This will likely fail due to model size)")
    print()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_8bit=True,  # Try quantization
        )

        # Test with real text
        text = "The quick brown fox jumps over the lazy dog."
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs, output_router_logits=True)

        # Try to extract routing
        if hasattr(outputs, 'router_logits'):
            router_logits = outputs.router_logits[0]  # First layer
            routing_probs = torch.softmax(router_logits, dim=-1)
            routing_np = routing_probs.mean(dim=0).cpu().numpy()  # Average over sequence

            entropy = -np.sum(routing_np * np.log(routing_np + 1e-10))
            max_entropy = np.log(len(routing_np))

            print(f"✓ Loaded pretrained model")
            print(f"  Routing entropy: {entropy:.4f} / {max_entropy:.4f}")
            print(f"  Structure: {'YES' if entropy < 0.8 * max_entropy else 'NO (still flat)'}")
            print()

            return {
                'approach': 'pretrained',
                'model_name': model_name,
                'entropy': entropy,
                'max_entropy': max_entropy,
                'routing': routing_np,
                'success': True,
            }
        else:
            print("❌ Could not extract routing from model")
            return None

    except Exception as e:
        print(f"❌ Failed to load pretrained model: {e}")
        print()
        return None


def test_approach_2_synthetic_structure():
    """
    Approach 2: Create synthetic "structured" routing

    Use our existing ChronoMoE but with:
    - Repeated patterns (not random)
    - Injected prior bias (break symmetry)
    - Verify routing develops structure
    """

    print("="*70)
    print("APPROACH 2: Synthetic Structured Inputs")
    print("="*70)
    print()

    from chronomoe.deepseek_core import DeepSeekConfig
    from chronomoe.chronovisor_deepseek_bridge import ChronovisorDeepSeekForCausalLM

    # Create model with prior bias to break symmetry
    config = DeepSeekConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=1024,
        num_layers=2,
        num_shared_experts=2,
        num_routed_experts=64,
        num_experts_per_token=6,
        enable_chronovisor=False,  # Not testing ChronoMoE yet
    )

    torch.manual_seed(42)
    model = ChronovisorDeepSeekForCausalLM(config)
    model.eval()

    # Create repeated pattern (not random)
    # Pattern: [0, 1, 2, 3, 4] repeated
    pattern = torch.tensor([0, 1, 2, 3, 4] * 10).unsqueeze(0)  # [1, 50]

    print(f"Input: Repeated pattern [0,1,2,3,4]×10")
    print()

    with torch.no_grad():
        _, chrono_state, _ = model(pattern, update_chronovisor=False)

    # Try to extract routing
    # For vanilla mode, we need to inspect the model internals
    # This is harder - let's see what we can get

    print("⚠ Synthetic approach: Cannot easily extract routing from vanilla mode")
    print("  Would need to hook into router internals")
    print()

    return None


def test_approach_3_train_small_model():
    """
    Approach 3: Train a small model on toy task

    Train on simple next-token prediction task to develop routing structure.
    """

    print("="*70)
    print("APPROACH 3: Train Small Model on Toy Task")
    print("="*70)
    print()

    print("This would require:")
    print("  1. Small dataset (e.g., repeated sequences)")
    print("  2. Training loop (optimize loss)")
    print("  3. Verify routing develops structure (entropy drops)")
    print()
    print("Not implemented in this session - would take hours to train")
    print()

    return None


def main():
    print("="*70)
    print("STEP 1: PROVE ROUTING STRUCTURE EXISTS")
    print("="*70)
    print()
    print("Goal: Establish that routing can be non-uniform with proper inputs/training")
    print()
    print("We'll try multiple approaches:")
    print("  1. Pretrained model from HuggingFace")
    print("  2. Synthetic structured inputs")
    print("  3. Train small model (not in this session)")
    print()

    results = []

    # Try approach 1
    result_1 = test_approach_1_pretrained()
    if result_1:
        results.append(result_1)

    # Try approach 2
    result_2 = test_approach_2_synthetic_structure()
    if result_2:
        results.append(result_2)

    # Try approach 3
    result_3 = test_approach_3_train_small_model()
    if result_3:
        results.append(result_3)

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()

    if len(results) == 0:
        print("❌ NO APPROACH SUCCEEDED")
        print()
        print("We cannot proceed to Step 2 without establishing structure.")
        print()
        print("Options:")
        print("  1. Install transformers and download small MoE model")
        print("  2. Modify our models to expose routing in vanilla mode")
        print("  3. Train a small model (takes time)")
        print("  4. Use ChronoMoE-enabled model and just verify baseline structure")
        print()
        print("Recommendation: Option 4 is fastest for proof-of-concept")
        print("  - Use ChronoMoE model (routing accessible via chrono_state)")
        print("  - Create structured inputs (repeated patterns)")
        print("  - Verify entropy drops below random baseline")
        print("  - Then proceed to Step 2 (control) and Step 3 (differential)")
    else:
        print(f"✓ SUCCESS: {len(results)} approach(es) worked")
        print()
        for r in results:
            print(f"Approach: {r['approach']}")
            if 'entropy' in r:
                print(f"  Entropy: {r['entropy']:.4f} / {r['max_entropy']:.4f}")
                print(f"  Structured: {'YES' if r['entropy'] < 0.8 * r['max_entropy'] else 'NO'}")
            print()

    print("Next: Implement pragmatic approach with ChronoMoE-enabled model")
    print("      and structured inputs to establish baseline structure.")
    print()


if __name__ == '__main__':
    main()
