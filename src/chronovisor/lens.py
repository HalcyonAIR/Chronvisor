"""
Lens: Shared geometry surface, affine transforms, local warps.

The lens is a deformable semantic surface that all experts read through. It can:
- Brighten regions where coherent, stabilising decisions occur
- Dim regions where instability grows
- Apply affine shifts (rotations, translations) to keep experts aligned
- Produce local "bumps" under regions with sustained resonance
"""
