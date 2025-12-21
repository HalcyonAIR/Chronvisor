"""
Routing Metrics Suite

Proper measurements for path wear detection.
Not outputs, not T̄ - direct routing landscape measurements.

Metrics:
1. ΔKL-to-A: Movement toward A's routing pattern
2. Cosine shift: Correlation change (stable under uniform)
3. Top-k Jaccard: Expert coalition overlap
4. Entropy: Concentration tracking
"""

import numpy as np
from typing import Tuple, Optional, Dict


def extract_routing_distribution(chrono_state, layer_idx: int = 0) -> Optional[np.ndarray]:
    """
    Extract routing probability distribution from chronovisor state.

    Args:
        chrono_state: ChronovisorState with expert_usage
        layer_idx: Which layer to extract (default: 0)

    Returns:
        Normalized routing distribution [num_experts], or None
    """
    if chrono_state is None or not hasattr(chrono_state, 'expert_usage'):
        return None

    if not chrono_state.expert_usage or layer_idx not in chrono_state.expert_usage:
        return None

    usage = chrono_state.expert_usage[layer_idx]

    # Normalize to probability distribution
    normalized = usage / (usage.sum() + 1e-10)
    return normalized


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute KL divergence KL(P || Q).

    Measures how much P differs from Q.
    """
    p = p + 1e-10
    q = q + 1e-10
    p = p / p.sum()
    q = q / q.sum()

    return float(np.sum(p * np.log(p / q)))


def cosine_similarity(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute cosine similarity between routing distributions.

    More stable than KL when distributions are near-uniform.
    Range: [-1, 1], where 1 = identical direction.
    """
    p_norm = p / (np.linalg.norm(p) + 1e-10)
    q_norm = q / (np.linalg.norm(q) + 1e-10)

    return float(np.dot(p_norm, q_norm))


def jaccard_similarity(p: np.ndarray, q: np.ndarray, k: int = 6, threshold: float = 0.01) -> float:
    """
    Compute Jaccard similarity of top-k expert sets.

    Measures expert coalition overlap.

    Args:
        p, q: Routing distributions
        k: Number of top experts to consider
        threshold: Minimum weight to consider expert "active"

    Returns:
        Jaccard index [0, 1]
    """
    # Get active experts (top-k or above threshold)
    active_p = set(np.where(p > threshold)[0])
    active_q = set(np.where(q > threshold)[0])

    # Also ensure we include top-k
    top_k_p = set(np.argsort(p)[-k:])
    top_k_q = set(np.argsort(q)[-k:])

    active_p = active_p.union(top_k_p)
    active_q = active_q.union(top_k_q)

    # Jaccard = intersection / union
    intersection = len(active_p & active_q)
    union = len(active_p | active_q)

    if union == 0:
        return 0.0

    return float(intersection / union)


def entropy(p: np.ndarray) -> float:
    """
    Compute Shannon entropy of routing distribution.

    High entropy = flat/uniform routing.
    Low entropy = concentrated routing.
    """
    p_normalized = p / (p.sum() + 1e-10)
    return float(-np.sum(p_normalized * np.log(p_normalized + 1e-10)))


def compute_routing_metrics_suite(
    routing_B_virgin: np.ndarray,
    routing_B_after: np.ndarray,
    routing_A: np.ndarray,
    top_k: int = 6,
) -> Dict[str, float]:
    """
    Compute full suite of routing metrics for path wear detection.

    Args:
        routing_B_virgin: B's routing before wear
        routing_B_after: B's routing after wearing path A
        routing_A: A's reference routing
        top_k: Number of top experts for Jaccard

    Returns:
        Dictionary of metrics with interpretation
    """

    # 1. ΔKL-to-A: Movement toward A's pattern
    kl_virgin_to_A = kl_divergence(routing_B_virgin, routing_A)
    kl_after_to_A = kl_divergence(routing_B_after, routing_A)
    delta_kl = kl_virgin_to_A - kl_after_to_A  # Positive = moved closer to A

    # 2. Cosine similarity shift
    cos_virgin = cosine_similarity(routing_B_virgin, routing_A)
    cos_after = cosine_similarity(routing_B_after, routing_A)
    delta_cos = cos_after - cos_virgin  # Positive = more similar to A

    # 3. Top-k Jaccard overlap
    jaccard_virgin = jaccard_similarity(routing_B_virgin, routing_A, k=top_k)
    jaccard_after = jaccard_similarity(routing_B_after, routing_A, k=top_k)
    delta_jaccard = jaccard_after - jaccard_virgin  # Positive = more overlap with A

    # 4. Entropy tracking (is "wear" just lower entropy?)
    entropy_virgin = entropy(routing_B_virgin)
    entropy_after = entropy(routing_B_after)
    entropy_A = entropy(routing_A)
    delta_entropy = entropy_after - entropy_virgin  # Negative = more concentrated

    # 5. Direct B change magnitude
    kl_virgin_to_after = kl_divergence(routing_B_virgin, routing_B_after)
    cos_virgin_to_after = cosine_similarity(routing_B_virgin, routing_B_after)

    return {
        # Primary wear signals
        'delta_kl_to_A': delta_kl,
        'delta_cos_to_A': delta_cos,
        'delta_jaccard_to_A': delta_jaccard,

        # Entropy tracking
        'entropy_B_virgin': entropy_virgin,
        'entropy_B_after': entropy_after,
        'entropy_A': entropy_A,
        'delta_entropy_B': delta_entropy,

        # Direct B change
        'kl_B_virgin_to_after': kl_virgin_to_after,
        'cos_B_virgin_to_after': cos_virgin_to_after,

        # Raw distances to A
        'kl_virgin_to_A': kl_virgin_to_A,
        'kl_after_to_A': kl_after_to_A,
        'cos_virgin_to_A': cos_virgin,
        'cos_after_to_A': cos_after,
        'jaccard_virgin_to_A': jaccard_virgin,
        'jaccard_after_to_A': jaccard_after,
    }


def interpret_routing_metrics(metrics: Dict[str, float]) -> str:
    """
    Interpret routing metrics and return human-readable summary.

    Returns multi-line interpretation string.
    """
    lines = []

    lines.append("ROUTING METRICS INTERPRETATION")
    lines.append("=" * 70)
    lines.append("")

    # Primary wear signals
    lines.append("Primary Wear Signals:")
    lines.append(f"  ΔKL to A:      {metrics['delta_kl_to_A']:+.6f}  (positive = moved toward A)")
    lines.append(f"  ΔCosine to A:  {metrics['delta_cos_to_A']:+.6f}  (positive = more similar to A)")
    lines.append(f"  ΔJaccard to A: {metrics['delta_jaccard_to_A']:+.6f}  (positive = more expert overlap)")
    lines.append("")

    # Entropy context
    lines.append("Entropy Context:")
    lines.append(f"  H(B_virgin): {metrics['entropy_B_virgin']:.4f}")
    lines.append(f"  H(B_after):  {metrics['entropy_B_after']:.4f}")
    lines.append(f"  H(A):        {metrics['entropy_A']:.4f}")
    lines.append(f"  ΔH(B):       {metrics['delta_entropy_B']:+.6f}  (negative = more concentrated)")
    lines.append("")

    # Direct B change
    lines.append("B Self-Change:")
    lines.append(f"  KL(B_virgin || B_after): {metrics['kl_B_virgin_to_after']:.6f}")
    lines.append(f"  Cosine(B_virgin, B_after): {metrics['cos_B_virgin_to_after']:.6f}")
    lines.append("")

    # Interpretation
    lines.append("Interpretation:")

    # Check for wear signals
    wear_detected = (
        metrics['delta_kl_to_A'] > 0.01 or
        metrics['delta_cos_to_A'] > 0.001 or
        metrics['delta_jaccard_to_A'] > 0.05
    )

    entropy_changed = abs(metrics['delta_entropy_B']) > 0.01

    if wear_detected and not entropy_changed:
        lines.append("  ✓ WEAR DETECTED: B moved toward A with stable entropy")
        lines.append("    This is genuine path wear via landscape deformation.")
    elif wear_detected and entropy_changed:
        lines.append("  ~ WEAR DETECTED: B moved toward A but entropy also changed")
        lines.append("    Effect may be entropy reduction, not pure path wear.")
    elif entropy_changed and not wear_detected:
        lines.append("  ○ ENTROPY SHIFT: B changed but not toward A")
        lines.append("    System dynamics changed, but not in A's direction.")
    else:
        lines.append("  ○ NO WEAR DETECTED: All signals below threshold")
        lines.append("    Routing landscape unchanged by repeated A passes.")

    lines.append("")

    # Near-uniform warning
    max_entropy_64 = np.log(64)  # ~4.16
    if metrics['entropy_B_virgin'] > 0.9 * max_entropy_64:
        lines.append("  ⚠ WARNING: Near-uniform routing (entropy ~4.0+)")
        lines.append("    Small biases won't register in flat landscape.")
        lines.append("    Consider entropy-controlled regime (top-k, temp, prior).")
        lines.append("")

    return "\n".join(lines)


def print_routing_comparison(
    routing_B_virgin: np.ndarray,
    routing_B_after: np.ndarray,
    routing_A: np.ndarray,
    top_k: int = 6,
):
    """
    Print detailed routing comparison for debugging.

    Shows top-k experts for each distribution.
    """
    print("\nROUTING DISTRIBUTION COMPARISON")
    print("=" * 70)
    print()

    # Top-k for each
    top_k_virgin = np.argsort(routing_B_virgin)[-top_k:][::-1]
    top_k_after = np.argsort(routing_B_after)[-top_k:][::-1]
    top_k_A = np.argsort(routing_A)[-top_k:][::-1]

    print(f"Top-{top_k} Experts:")
    print()
    print("  B (virgin):")
    for i, expert_idx in enumerate(top_k_virgin):
        print(f"    {i+1}. Expert {expert_idx:2d}: {routing_B_virgin[expert_idx]:.4f}")

    print()
    print("  B (after wear):")
    for i, expert_idx in enumerate(top_k_after):
        marker = "  ← in A" if expert_idx in top_k_A else ""
        print(f"    {i+1}. Expert {expert_idx:2d}: {routing_B_after[expert_idx]:.4f}{marker}")

    print()
    print("  A (reference):")
    for i, expert_idx in enumerate(top_k_A):
        print(f"    {i+1}. Expert {expert_idx:2d}: {routing_A[expert_idx]:.4f}")

    print()
