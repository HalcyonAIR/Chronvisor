#!/usr/bin/env python3
"""Debug Switch router to understand exact return signature."""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8", device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")

# Get the router from layer 1
router = model.encoder.block[1].layer[-1].mlp.router

print(f"Router type: {type(router)}")
print(f"Router: {router}")
print()

# Create dummy input
hidden_states = torch.randn(1, 10, 768)  # [batch, seq, hidden_dim]

print("Calling router.forward(hidden_states)...")
with torch.no_grad():
    outputs = router.forward(hidden_states)

print(f"Number of return values: {len(outputs) if isinstance(outputs, tuple) else 1}")
print()

if isinstance(outputs, tuple):
    for i, output in enumerate(outputs):
        if isinstance(output, torch.Tensor):
            print(f"Output {i}: Tensor with shape {output.shape}, dtype {output.dtype}")
        else:
            print(f"Output {i}: {type(output)}")
else:
    print(f"Single output: {type(outputs)}, shape {outputs.shape if hasattr(outputs, 'shape') else 'N/A'}")
