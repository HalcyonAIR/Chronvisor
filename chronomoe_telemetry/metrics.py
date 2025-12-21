"""
Metrics: PC1/PC2, velocities, entropy, switch rate.

Pure functions for computing measurements from routing state.
No side effects, no state modification.
"""

import numpy as np
from typing import Tuple, Optional, List


def compute_pc_projection(routing_weights: np.ndarray) -> Tuple[float, float]:
    """
    Compute PC1/PC2 projection from routing weights.

    For v0, use simple proxies:
    - PC1 ≈ momentum (L2 norm of routing weights)
    - PC2 ≈ exploration (entropy of routing distribution)

    Args:
        routing_weights: Probability distribution over experts [num_experts]

    Returns:
        (pc1, pc2) tuple
    """
    if routing_weights is None or len(routing_weights) == 0:
        return (0.0, 0.0)

    # PC1 proxy: L2 norm (commitment strength)
    pc1 = float(np.linalg.norm(routing_weights))

    # PC2 proxy: entropy (exploration margin)
    # Normalize first to ensure valid probability distribution
    normalized = routing_weights / (routing_weights.sum() + 1e-10)
    pc2 = -np.sum(normalized * np.log(normalized + 1e-10))

    return (pc1, float(pc2))


def compute_routing_entropy(routing_weights: np.ndarray) -> float:
    """
    Compute entropy of routing distribution.

    Args:
        routing_weights: Probability distribution over experts

    Returns:
        Shannon entropy
    """
    if routing_weights is None or len(routing_weights) == 0:
        return 0.0

    normalized = routing_weights / (routing_weights.sum() + 1e-10)
    entropy = -np.sum(normalized * np.log(normalized + 1e-10))
    return float(entropy)


def compute_expert_switch_rate(
    current_routing: np.ndarray,
    previous_routing: Optional[np.ndarray] = None,
    threshold: float = 0.01,
) -> float:
    """
    Compute rate of expert coalition changes.

    Measures what fraction of experts switched between active and inactive
    states between two routing snapshots.

    Args:
        current_routing: Current routing weights [num_experts]
        previous_routing: Previous routing weights [num_experts], or None
        threshold: Minimum weight to consider an expert "active"

    Returns:
        Fraction of experts that switched state (0.0 to 1.0)
    """
    if previous_routing is None:
        return 0.0

    if len(current_routing) != len(previous_routing):
        return 0.0

    # Determine which experts are "active" (above threshold)
    current_active = set(np.where(current_routing > threshold)[0])
    previous_active = set(np.where(previous_routing > threshold)[0])

    # Symmetric difference = experts that changed state
    switched = current_active.symmetric_difference(previous_active)

    # Total unique experts involved
    total = current_active.union(previous_active)

    if len(total) == 0:
        return 0.0

    switch_rate = len(switched) / len(total)
    return float(switch_rate)


def compute_pc_velocity(
    current_pc: Tuple[float, float],
    previous_pc: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Compute velocity in PC space.

    Args:
        current_pc: (pc1, pc2) at current step
        previous_pc: (pc1, pc2) at previous step

    Returns:
        (d_pc1, d_pc2) velocity
    """
    d_pc1 = current_pc[0] - previous_pc[0]
    d_pc2 = current_pc[1] - previous_pc[1]
    return (d_pc1, d_pc2)


def compute_pc_curvature(
    current_velocity: Tuple[float, float],
    previous_velocity: Tuple[float, float],
) -> float:
    """
    Compute curvature (second derivative) in PC space.

    Measures how much the trajectory is bending.

    Args:
        current_velocity: (d_pc1, d_pc2) at current step
        previous_velocity: (d_pc1, d_pc2) at previous step

    Returns:
        Magnitude of acceleration vector
    """
    dd_pc1 = current_velocity[0] - previous_velocity[0]
    dd_pc2 = current_velocity[1] - previous_velocity[1]

    curvature = np.sqrt(dd_pc1**2 + dd_pc2**2)
    return float(curvature)


def compute_velocity_magnitude(velocity: Tuple[float, float]) -> float:
    """
    Compute magnitude of velocity vector.

    Args:
        velocity: (d_pc1, d_pc2)

    Returns:
        ||velocity||
    """
    return float(np.sqrt(velocity[0]**2 + velocity[1]**2))
