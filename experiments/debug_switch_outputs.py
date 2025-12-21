#!/usr/bin/env python3
"""Debug Switch Transformer outputs to understand router logits structure."""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8", device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")

text = "The cat sat on the mat."
inputs = tokenizer(text, return_tensors="pt")
decoder_input_ids = inputs['input_ids'].clone()

print("\nRunning forward pass...")
with torch.no_grad():
    outputs = model(
        input_ids=inputs['input_ids'],
        decoder_input_ids=decoder_input_ids,
        output_router_logits=True,
        output_hidden_states=True,
    )

print("\nOutputs keys:")
for key in outputs.keys():
    val = outputs[key]
    if val is not None:
        if isinstance(val, torch.Tensor):
            print(f"  {key}: Tensor{val.shape}")
        elif isinstance(val, (tuple, list)):
            print(f"  {key}: {type(val).__name__} of length {len(val)}")
            if len(val) > 0:
                elem = val[0]
                if isinstance(elem, torch.Tensor):
                    print(f"    [0]: Tensor{elem.shape}")
                elif isinstance(elem, tuple):
                    print(f"    [0]: tuple of length {len(elem)}")
                    for i, sub in enumerate(elem):
                        if isinstance(sub, torch.Tensor):
                            print(f"      [{i}]: Tensor{sub.shape}, dtype={sub.dtype}")
        else:
            print(f"  {key}: {type(val)}")

# Check encoder specifically
if hasattr(model, 'encoder'):
    print("\nModel structure:")
    print(f"  Encoder blocks: {len(model.encoder.block)}")
    print(f"  First block layers: {len(model.encoder.block[0].layer)}")

    # Check if there's a router in the MLP
    for i, block in enumerate(model.encoder.block[:2]):
        if len(block.layer) > 1:
            mlp_layer = block.layer[-1]
            print(f"\n  Block {i} MLP layer type: {type(mlp_layer)}")
            if hasattr(mlp_layer, 'mlp'):
                print(f"    MLP type: {type(mlp_layer.mlp)}")
                for attr in ['router', 'switch_router', 'experts']:
                    if hasattr(mlp_layer.mlp, attr):
                        print(f"      Has {attr}: {getattr(mlp_layer.mlp, attr)}")
