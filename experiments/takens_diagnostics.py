#!/usr/bin/env python3
"""
Takens Embedding Diagnostics for P×T Coupling

Tests whether routing dynamics exhibit delay-embedding structure consistent
with Takens' theorem. If true, this explains curvature and sustained dynamics
requirements.

Key tests:
1. False Nearest Neighbors (FNN) - finds optimal embedding dimension
2. Delay dimension sweep - shows when structure stabilizes
3. Top-k vs Top-1 comparison - tests smoothness requirement

References:
- Takens (1981): Detecting strange attractors in turbulence
- Kennel et al. (1992): Determining embedding dimension using FNN method
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List


class TakensAnalyzer:
    """
    Analyzes time series for delay-embedding structure.

    Uses False Nearest Neighbors to determine if a low-dimensional
    attractor exists in routing dynamics.
    """

    def __init__(self, time_series: np.ndarray, delay: int = 1):
        """
        Args:
            time_series: 1D array of scalar observations over time
            delay: Time delay for embedding (τ in Takens notation)
        """
        self.ts = time_series
        self.delay = delay
        self.N = len(time_series)

    def embed(self, dimension: int) -> np.ndarray:
        """
        Create delay-coordinate embedding.

        Args:
            dimension: Embedding dimension (d)

        Returns:
            embedded: [N - (d-1)*delay, d] array
                     Each row is [x(t), x(t-τ), x(t-2τ), ..., x(t-(d-1)τ)]
        """
        n_points = self.N - (dimension - 1) * self.delay
        embedded = np.zeros((n_points, dimension))

        for i in range(dimension):
            start_idx = i * self.delay
            end_idx = start_idx + n_points
            embedded[:, i] = self.ts[start_idx:end_idx]

        return embedded

    def false_nearest_neighbors(
        self,
        dimension: int,
        rtol: float = 15.0,  # Kennel et al. use 15
        atol: float = 2.0     # Additional absolute threshold
    ) -> float:
        """
        Compute percentage of false nearest neighbors.

        A neighbor in d-dimensional space is "false" if it's far away
        when you add one more dimension (d+1). High FNN means you need
        more dimensions to unfold the attractor.

        Args:
            dimension: Current embedding dimension
            rtol: Relative distance threshold (default 15 from Kennel)
            atol: Absolute distance threshold (in std units)

        Returns:
            fnn_percent: Percentage of false neighbors (0-100)
        """
        # Embed in d and d+1 dimensions
        Y_d = self.embed(dimension)
        Y_d1 = self.embed(dimension + 1)

        # Use minimum size (d+1 embedding has fewer points)
        n_points = min(Y_d.shape[0], Y_d1.shape[0])
        false_count = 0

        # Trim Y_d to match Y_d1 size for safe indexing
        Y_d = Y_d[:n_points]

        # For each point, find nearest neighbor in d dimensions
        for i in range(n_points):
            # Compute distances to all other points in d dimensions
            dists_d = np.linalg.norm(Y_d - Y_d[i], axis=1)
            dists_d[i] = np.inf  # Exclude self

            # Find nearest neighbor
            nn_idx = np.argmin(dists_d)
            dist_d = dists_d[nn_idx]

            # Skip if nearest neighbor is too far (isolated point)
            if dist_d > atol * np.std(self.ts):
                continue

            # Skip if nn_idx is out of bounds for Y_d1 (safety check)
            if nn_idx >= n_points:
                continue

            # Compute distance in d+1 dimensions
            dist_d1 = np.linalg.norm(Y_d1[i] - Y_d1[nn_idx])

            # Check if neighbor is "false" (far in d+1 but close in d)
            # Kennel criterion: |x_d+1(i) - x_d+1(nn)| / dist_d > rtol
            if (dist_d1 - dist_d) / dist_d > rtol:
                false_count += 1

        fnn_percent = 100.0 * false_count / n_points
        return fnn_percent

    def dimension_sweep(
        self,
        max_dim: int = 20,
        rtol: float = 15.0,
        atol: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sweep embedding dimensions and compute FNN for each.

        Returns:
            dimensions: Array of tested dimensions
            fnn_values: FNN percentage for each dimension
        """
        dimensions = np.arange(1, max_dim + 1)
        fnn_values = np.zeros(max_dim)

        for d in dimensions:
            fnn_values[d - 1] = self.false_nearest_neighbors(d, rtol, atol)
            print(f"  Dimension {d:2d}: FNN = {fnn_values[d - 1]:5.2f}%")

        return dimensions, fnn_values


def extract_routing_time_series(
    routing_history: List[np.ndarray],
    metric: str = 'entropy'
) -> np.ndarray:
    """
    Extract scalar time series from routing history.

    Args:
        routing_history: List of routing probability arrays over time
        metric: Which scalar to extract
                'entropy' - routing entropy (measures dispersion)
                'max_prob' - maximum routing probability (measures confidence)
                'gini' - Gini coefficient (measures inequality)

    Returns:
        time_series: 1D array of scalar observations
    """
    time_series = []

    for probs in routing_history:
        if metric == 'entropy':
            # Shannon entropy: -sum(p * log(p))
            eps = 1e-10
            entropy = -np.sum(probs * np.log(probs + eps))
            time_series.append(entropy)
        elif metric == 'max_prob':
            # Maximum probability (confidence)
            time_series.append(np.max(probs))
        elif metric == 'gini':
            # Gini coefficient (inequality measure)
            sorted_probs = np.sort(probs)
            n = len(probs)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_probs)) / (n * np.sum(sorted_probs)) - (n + 1) / n
            time_series.append(gini)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return np.array(time_series)


def load_routing_data_from_experiment(
    experiment_name: str,
    enable_chronovisor: bool = True
) -> np.ndarray:
    """
    Load routing statistics from a completed experiment.

    For now, we'll generate synthetic data that matches our observed patterns.
    In production, this would load from saved experiment logs.

    Args:
        experiment_name: 'mixtral_2l8e', 'mixtral_4l16e', 'switch_2l8e'
        enable_chronovisor: Whether to load chronovisor-enabled run

    Returns:
        routing_series: Time series of routing entropy over training
    """
    # TODO: Replace with actual data loading when logs are structured
    # For now, generate synthetic data matching our observations

    if experiment_name == 'mixtral_2l8e':
        if enable_chronovisor:
            # Top-2 routing with P×T: should show structure
            # Entropy decreases as routing becomes more confident
            # But oscillates due to pressure-temperature feedback
            t = np.linspace(0, 100, 500)
            trend = 2.0 - 0.5 * (1 - np.exp(-t/20))  # Decay toward lower entropy
            oscillation = 0.1 * np.sin(2 * np.pi * t / 15)  # Feedback oscillations
            noise = 0.02 * np.random.randn(500)
            routing_series = trend + oscillation + noise
        else:
            # No chronovisor: more random walk
            routing_series = 2.0 - 0.3 * np.cumsum(np.random.randn(500)) / 50
            routing_series += 0.05 * np.random.randn(500)

    elif experiment_name == 'switch_2l8e':
        # Top-1 routing: discrete jumps, no smooth structure
        # Entropy is either high (uncertain) or low (locked), no intermediate
        t = np.linspace(0, 100, 500)
        # Piecewise constant (argmax jumps)
        routing_series = np.zeros(500)
        segment_length = 50
        for i in range(0, 500, segment_length):
            routing_series[i:i+segment_length] = np.random.choice([0.5, 1.5, 2.0])
        routing_series += 0.02 * np.random.randn(500)  # Small noise

    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    return routing_series


def plot_fnn_comparison(
    results: dict,
    output_path: str = "takens_fnn_comparison.png"
):
    """
    Plot FNN results for multiple conditions.

    Args:
        results: Dict of {name: (dimensions, fnn_values)}
        output_path: Where to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: All FNN curves
    ax1 = axes[0]
    for name, (dims, fnn) in results.items():
        ax1.plot(dims, fnn, marker='o', label=name, linewidth=2)

    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5,
                label='1% threshold (typical convergence)')
    ax1.set_xlabel('Embedding Dimension', fontsize=12)
    ax1.set_ylabel('False Nearest Neighbors (%)', fontsize=12)
    ax1.set_title('FNN vs Embedding Dimension', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(max(fnn) for _, fnn in results.values()) * 1.1)

    # Right plot: Example embedded trajectory (2D projection of first condition)
    ax2 = axes[1]
    first_name = list(results.keys())[0]
    first_ts = load_routing_data_from_experiment(
        first_name.split()[0].lower().replace('-', '_'),
        'Chronovisor' in first_name
    )

    analyzer = TakensAnalyzer(first_ts, delay=1)
    embedded = analyzer.embed(dimension=2)

    # Color by time
    colors = np.arange(len(embedded))
    scatter = ax2.scatter(embedded[:, 0], embedded[:, 1],
                         c=colors, cmap='viridis', s=10, alpha=0.6)
    ax2.set_xlabel('x(t)', fontsize=12)
    ax2.set_ylabel('x(t-τ)', fontsize=12)
    ax2.set_title(f'2D Embedding: {first_name}', fontsize=14)
    plt.colorbar(scatter, ax=ax2, label='Time')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 FNN comparison saved: {output_path}")


def main():
    """
    Run Takens diagnostics on routing data.

    Tests:
    1. Mixtral top-2 with Chronovisor (should show structure)
    2. Mixtral top-2 without Chronovisor (less structure)
    3. Switch top-1 (should fail - no smoothness)
    """
    print("=" * 70)
    print("TAKENS EMBEDDING DIAGNOSTICS")
    print("=" * 70)
    print()
    print("Testing whether routing dynamics exhibit delay-embedding structure.")
    print("If FNN decreases with dimension → consistent with attractor unfolding.")
    print("If FNN stays high → no low-dimensional structure.")
    print()

    # Parameters
    max_dim = 15
    delay = 1  # Time steps (could tune this)
    rtol = 15.0  # Kennel et al. standard
    atol = 2.0   # Std units

    print(f"Parameters:")
    print(f"  Max embedding dimension: {max_dim}")
    print(f"  Time delay (τ): {delay}")
    print(f"  Relative threshold: {rtol}")
    print(f"  Absolute threshold: {atol} std")
    print()

    results = {}

    # Test 1: Mixtral top-2 WITH Chronovisor
    print("-" * 70)
    print("TEST 1: Mixtral 2L/8E (top-2) WITH Chronovisor")
    print("-" * 70)
    print("Expected: FNN should decrease (attractor unfolds)")
    print()

    ts1 = load_routing_data_from_experiment('mixtral_2l8e', enable_chronovisor=True)
    analyzer1 = TakensAnalyzer(ts1, delay=delay)
    dims1, fnn1 = analyzer1.dimension_sweep(max_dim, rtol, atol)
    results['Mixtral top-2 + Chronovisor'] = (dims1, fnn1)

    print(f"\nResult: FNN drops from {fnn1[0]:.1f}% to {fnn1[-1]:.1f}%")
    print()

    # Test 2: Mixtral top-2 WITHOUT Chronovisor
    print("-" * 70)
    print("TEST 2: Mixtral 2L/8E (top-2) WITHOUT Chronovisor")
    print("-" * 70)
    print("Expected: Less structure (higher FNN)")
    print()

    ts2 = load_routing_data_from_experiment('mixtral_2l8e', enable_chronovisor=False)
    analyzer2 = TakensAnalyzer(ts2, delay=delay)
    dims2, fnn2 = analyzer2.dimension_sweep(max_dim, rtol, atol)
    results['Mixtral top-2 (baseline)'] = (dims2, fnn2)

    print(f"\nResult: FNN drops from {fnn2[0]:.1f}% to {fnn2[-1]:.1f}%")
    print()

    # Test 3: Switch top-1
    print("-" * 70)
    print("TEST 3: Switch Transformer (top-1)")
    print("-" * 70)
    print("Expected: FNN stays high (no smooth structure)")
    print()

    ts3 = load_routing_data_from_experiment('switch_2l8e', enable_chronovisor=True)
    analyzer3 = TakensAnalyzer(ts3, delay=delay)
    dims3, fnn3 = analyzer3.dimension_sweep(max_dim, rtol, atol)
    results['Switch top-1'] = (dims3, fnn3)

    print(f"\nResult: FNN drops from {fnn3[0]:.1f}% to {fnn3[-1]:.1f}%")
    print()

    # Summary
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()

    # Find optimal embedding dimension (where FNN < 1%)
    for name, (dims, fnn) in results.items():
        optimal_idx = np.where(fnn < 1.0)[0]
        if len(optimal_idx) > 0:
            optimal_dim = dims[optimal_idx[0]]
            print(f"{name}:")
            print(f"  Optimal embedding dimension: {optimal_dim}")
            print(f"  FNN at optimal: {fnn[optimal_idx[0]]:.2f}%")
        else:
            print(f"{name}:")
            print(f"  No convergence (FNN stays above 1%)")
            print(f"  Final FNN: {fnn[-1]:.2f}%")
        print()

    # Takens interpretation
    print("Takens Interpretation:")
    print("-" * 70)

    if fnn1[-1] < 1.0 and fnn3[-1] > 10.0:
        print("✓ CONSISTENT WITH TAKENS EMBEDDING")
        print()
        print("Evidence:")
        print("  • Top-2 routing: FNN converges (smooth dynamics)")
        print("  • Top-1 routing: FNN stays high (discontinuous)")
        print("  • This matches smoothness requirement exactly")
        print()
        print("Implication:")
        print("  Temperature may be performing delay-coordinate reconstruction")
        print("  of hidden routing state. Top-1 fails because argmax destroys")
        print("  the smoothness that Takens requires.")
    elif fnn1[-1] < 1.0 and fnn3[-1] < 1.0:
        print("⚠ UNEXPECTED: Both converge")
        print()
        print("Top-1 shouldn't show embedding structure (discontinuous).")
        print("May indicate synthetic data is too simple.")
        print("Need to test on real experimental data.")
    else:
        print("✗ NOT CONSISTENT WITH TAKENS")
        print()
        print("FNN doesn't converge for either condition.")
        print("May indicate:")
        print("  • Time series too short")
        print("  • Wrong observable (try different metric)")
        print("  • Dynamics aren't low-dimensional")

    print()

    # Plot
    plot_fnn_comparison(results)

    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Replace synthetic data with real routing logs")
    print("2. Try different observables (max_prob, gini, etc.)")
    print("3. Test delay parameter (τ ∈ {1, 2, 5, 10})")
    print("4. If FNN confirms, add to Paper 1 as modest appendix")
    print()


if __name__ == '__main__':
    main()
