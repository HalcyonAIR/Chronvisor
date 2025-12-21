"""
Diagnostic: Residual Intent Behavior

Investigate why residual intent is 0.0 in multistep mode.
"""

from chronomoe.pressure import compute_residual_intent

# Test scenarios

print("=" * 70)
print("Residual Intent Behavior")
print("=" * 70)
print()

# Scenario 1: Multistep pausing (positive pressure)
print("Scenario 1: Multistep mode (always pauses with positive pressure)")
print("-" * 70)

residual = 0.0
for chunk_idx in range(5):
    mid_pressure = 0.64  # Positive (wants to continue)
    net_pressure = 0.30  # Positive
    did_pause = True     # Multistep always pauses

    new_residual = compute_residual_intent(residual, mid_pressure, net_pressure, did_pause)

    print(f"  Chunk {chunk_idx}: residual {residual:.4f} → {new_residual:.4f}")
    residual = new_residual

print()
print("Result: Residual stays at 0.0 because:")
print("  - did_pause=True (multistep mode)")
print("  - net_pressure > 0 (positive)")
print("  - Logic: return current_residual * 0.5")
print("  - 0.0 * 0.5 = 0.0 forever")
print()

# Scenario 2: Single-turn continuing (no pauses)
print("Scenario 2: Single-turn mode (continuing, no pauses)")
print("-" * 70)

residual = 0.0
for chunk_idx in range(5):
    mid_pressure = 0.64  # Positive (wants to continue)
    net_pressure = 0.30  # Positive
    did_pause = False    # Single-turn continues

    new_residual = compute_residual_intent(residual, mid_pressure, net_pressure, did_pause)

    print(f"  Chunk {chunk_idx}: residual {residual:.4f} → {new_residual:.4f}")
    residual = new_residual

print()
print("Result: Residual accumulates because:")
print("  - did_pause=False (continuing)")
print("  - Logic: updated = 0.7 * old + 0.3 * mid_pressure")
print("  - Exponentially approaches mid_pressure value")
print()

# Scenario 3: Multistep with initial residual
print("Scenario 3: What if multistep started with residual > 0?")
print("-" * 70)

residual = 0.5  # Start with some residual
for chunk_idx in range(5):
    mid_pressure = 0.64  # Positive
    net_pressure = 0.30  # Positive
    did_pause = True     # Multistep pauses

    new_residual = compute_residual_intent(residual, mid_pressure, net_pressure, did_pause)

    print(f"  Chunk {chunk_idx}: residual {residual:.4f} → {new_residual:.4f}")
    residual = new_residual

print()
print("Result: Residual decays to 0.0 because:")
print("  - Each pause: residual *= 0.5")
print("  - 0.5 → 0.25 → 0.125 → 0.0625 → 0.03125")
print()

# Scenario 4: Negative pressure pause
print("Scenario 4: Pause with negative pressure (resolved intent)")
print("-" * 70)

residual = 0.8
mid_pressure = -0.3   # Negative (wants to stop)
net_pressure = -0.2   # Negative
did_pause = True

new_residual = compute_residual_intent(residual, mid_pressure, net_pressure, did_pause)

print(f"  Before: residual = {residual:.4f}")
print(f"  After:  residual = {new_residual:.4f}")
print()
print("Result: Residual resets to 0.0 because:")
print("  - net_pressure < 0 (resolved)")
print("  - Intent to continue is extinguished")
print()

print("=" * 70)
print("Conclusion")
print("-" * 70)
print()
print("Multistep mode residual intent = 0.0 is BY DESIGN:")
print()
print("  1. Each chunk pauses (did_pause=True)")
print("  2. Positive pressure → decay (residual *= 0.5)")
print("  3. Starting from 0 → stays at 0")
print()
print("This is CORRECT behavior!")
print()
print("  - Multistep = fresh evaluation each chunk")
print("  - No accumulated momentum")
print("  - Each pause is opportunity to re-evaluate")
print()
print("Single-turn accumulates residual because:")
print()
print("  - Long generation without pauses")
print("  - did_pause=False during generation")
print("  - Residual builds up → 'finish what you started' momentum")
print()
print("=" * 70)
