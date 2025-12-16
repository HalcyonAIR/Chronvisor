#!/usr/bin/env python3
"""Quick script to capture just Switch routing (other two already done)"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from pathlib import Path
from capture_routing_for_takens import capture_switch_routing

# Run Switch capture
print("=" * 70)
print("SWITCH ROUTING CAPTURE")
print("=" * 70)
print()

output_dir = Path("takens_data")
output_dir.mkdir(exist_ok=True)

print("Capturing Switch Transformer routing...")
capture3 = capture_switch_routing(
    eta=0.015,
    pressure_scale=0.5,
    seed=42,
    num_conversations=10,
    num_epochs=20,
    sample_every=5
)

capture3.save(
    str(output_dir / "switch_top1"),
    metadata={
        'condition': 'switch_2l8e_top1',
        'eta': 0.015,
        'pressure_scale': 0.5,
        'seed': 42,
        'routing_type': 'top-1',
    }
)

print()
print("✓ Switch capture complete!")
print()
