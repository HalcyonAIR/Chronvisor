#!/usr/bin/env python3
"""
Analyze Captured Routing Trajectories with Takens Embedding

Takes the routing probability histories from capture_routing_for_takens.py
and runs FNN diagnostics to test for attractor structure.

Expected outcomes:
- Mixtral top-2 + Chronovisor: FNN converges (smooth attractor)
- Mixtral top-2 baseline: FNN converges but less structured
- Switch top-1: FNN stays high or converges slowly (discontinuous)

This is the REAL test - unlike synthetic data, real routing with argmax
should show actual discontinuities that prevent embedding.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, List


class TakensAnalyzer:
    """
    Analyzes time series for delay-embedding structure.

    (Same implementation as takens_diagnostics.py - copied for standalone use)
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
        rtol: float = 15.0,
        atol: float = 2.0
    ) -> float:
        """
        Compute percentage of false nearest neighbors.

        Args:
            dimension: Current embedding dimension
            rtol: Relative distance threshold
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
            if (dist_d1 - dist_d) / dist_d > rtol:
                false_count += 1

        fnn_percent = 100.0 * false_count / n_points
        return fnn_percent

    def dimension_sweep(
        self,
        max_dim: int = 15,
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


def load_entropy_data(condition: str, data_dir: Path) -> Tuple[np.ndarray, dict]:
    """
    Load captured entropy data for a condition.

    Args:
        condition: 'mixtral_chronovisor', 'mixtral_baseline', or 'switch_top1'
        data_dir: Directory containing captured data

    Returns:
        entropy_series: [num_steps] array of routing entropy over time
        metadata: Dictionary with capture metadata
    """
    entropy_path = data_dir / f"{condition}_entropy.npy"
    metadata_path = data_dir / f"{condition}_metadata.npy"

    if not entropy_path.exists():
        raise FileNotFoundError(f"Entropy data not found: {entropy_path}")

    entropy_series = np.load(entropy_path)
    metadata = np.load(metadata_path, allow_pickle=True).item()

    return entropy_series, metadata


def plot_fnn_results(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    entropy_series: Dict[str, np.ndarray],
    output_dir: Path
):
    """
    Create comprehensive FNN diagnostic plots.

    Args:
        results: {condition_name: (dimensions, fnn_values)}
        entropy_series: {condition_name: entropy_time_series}
        output_dir: Where to save plots
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Top row: Entropy time series for each condition
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    entropy_axes = [ax1, ax2, ax3]
    condition_names = list(results.keys())

    for i, (name, ax) in enumerate(zip(condition_names, entropy_axes)):
        entropy = entropy_series[name]
        ax.plot(entropy, linewidth=1, alpha=0.7)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel('Step', fontsize=9)
        ax.set_ylabel('Routing Entropy', fontsize=9)
        ax.grid(True, alpha=0.3)

    # Middle row: FNN curves
    ax4 = fig.add_subplot(gs[1, :])

    for name, (dims, fnn) in results.items():
        ax4.plot(dims, fnn, marker='o', label=name, linewidth=2, markersize=4)

    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.5,
                label='1% threshold (typical convergence)')
    ax4.set_xlabel('Embedding Dimension', fontsize=12)
    ax4.set_ylabel('False Nearest Neighbors (%)', fontsize=12)
    ax4.set_title('FNN vs Embedding Dimension', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, max(max(fnn) for _, fnn in results.values()) * 1.1)

    # Bottom row: 2D embeddings
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])
    ax7 = fig.add_subplot(gs[2, 2])

    embedding_axes = [ax5, ax6, ax7]

    for i, (name, ax) in enumerate(zip(condition_names, embedding_axes)):
        entropy = entropy_series[name]
        analyzer = TakensAnalyzer(entropy, delay=1)
        embedded = analyzer.embed(dimension=2)

        # Color by time
        colors = np.arange(len(embedded))
        scatter = ax.scatter(embedded[:, 0], embedded[:, 1],
                           c=colors, cmap='viridis', s=10, alpha=0.6)
        ax.set_xlabel('x(t)', fontsize=9)
        ax.set_ylabel('x(t-τ)', fontsize=9)
        ax.set_title(f'2D Embedding: {name}', fontsize=10)
        plt.colorbar(scatter, ax=ax, label='Time')

    # Save
    output_path = output_dir / "fnn_diagnostics_real_data.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 FNN diagnostics plot saved: {output_path}")
    plt.close()


def main():
    """
    Analyze captured routing data with FNN diagnostics.
    """
    print("=" * 70)
    print("TAKENS EMBEDDING ANALYSIS - REAL ROUTING DATA")
    print("=" * 70)
    print()
    print("Analyzing captured routing trajectories with FNN diagnostics.")
    print()

    # Load data
    data_dir = Path("takens_data")
    if not data_dir.exists():
        print(f"❌ Error: Data directory not found: {data_dir}")
        print("   Run capture_routing_for_takens.py first.")
        return

    conditions = [
        ("mixtral_chronovisor", "Mixtral top-2 + Chronovisor"),
        ("mixtral_baseline", "Mixtral top-2 baseline"),
        ("switch_top1", "Switch top-1"),
    ]

    # Parameters
    max_dim = 15
    delay = 1
    rtol = 15.0
    atol = 2.0

    print(f"Parameters:")
    print(f"  Max embedding dimension: {max_dim}")
    print(f"  Time delay (τ): {delay}")
    print(f"  Relative threshold: {rtol}")
    print(f"  Absolute threshold: {atol} std")
    print()

    results = {}
    entropy_series = {}

    # Analyze each condition
    for condition_id, condition_name in conditions:
        print("-" * 70)
        print(f"ANALYZING: {condition_name}")
        print("-" * 70)

        # Load entropy data
        try:
            entropy, metadata = load_entropy_data(condition_id, data_dir)
            print(f"Loaded entropy data:")
            print(f"  Steps: {len(entropy)}")
            print(f"  Mean entropy: {np.mean(entropy):.4f}")
            print(f"  Std entropy: {np.std(entropy):.4f}")
            print(f"  Metadata: {metadata}")
            print()
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print()
            continue

        entropy_series[condition_name] = entropy

        # Run FNN analysis
        print("Running FNN dimension sweep...")
        analyzer = TakensAnalyzer(entropy, delay=delay)
        dims, fnn = analyzer.dimension_sweep(max_dim, rtol, atol)
        results[condition_name] = (dims, fnn)

        # Find optimal dimension
        optimal_idx = np.where(fnn < 1.0)[0]
        if len(optimal_idx) > 0:
            optimal_dim = dims[optimal_idx[0]]
            print(f"\n✓ Optimal embedding dimension: {optimal_dim}")
            print(f"  FNN at optimal: {fnn[optimal_idx[0]]:.2f}%")
        else:
            print(f"\n⚠ No convergence (FNN stays above 1%)")
            print(f"  Final FNN: {fnn[-1]:.2f}%")
        print()

    # Summary
    print("=" * 70)
    print("TAKENS INTERPRETATION")
    print("=" * 70)
    print()

    # Extract FNN values
    mixtral_chrono_fnn = results.get("Mixtral top-2 + Chronovisor", (None, np.array([100])))[1][-1]
    switch_fnn = results.get("Switch top-1", (None, np.array([100])))[1][-1]

    print("Final FNN values:")
    for name, (dims, fnn) in results.items():
        print(f"  {name}: {fnn[-1]:.2f}%")
    print()

    # Interpretation
    if mixtral_chrono_fnn < 1.0 and switch_fnn > 10.0:
        print("✓ CONSISTENT WITH TAKENS EMBEDDING HYPOTHESIS")
        print()
        print("Evidence:")
        print("  • Top-2 routing: FNN converges (smooth dynamics)")
        print("  • Top-1 routing: FNN stays high (discontinuous)")
        print("  • Matches smoothness requirement exactly")
        print()
        print("Interpretation:")
        print("  Temperature may be performing delay-coordinate reconstruction")
        print("  of hidden routing state. Top-1 fails because argmax destroys")
        print("  the smoothness that Takens' theorem requires.")
        print()
        print("Implication:")
        print("  The curvature requirement isn't arbitrary - it's needed for")
        print("  delay-embedding to unfold the attractor geometry.")
    elif mixtral_chrono_fnn < 1.0 and switch_fnn < 1.0:
        print("⚠ UNEXPECTED: Both converge")
        print()
        print("Both top-2 and top-1 show embedding structure.")
        print("Possible explanations:")
        print("  • Switch implementation may not be truly discontinuous")
        print("  • Routing changes slowly enough that discontinuities don't matter")
        print("  • Need to examine Switch router implementation")
    else:
        print("✗ NOT CONSISTENT WITH TAKENS")
        print()
        print("Neither condition shows clear embedding structure.")
        print("Possible issues:")
        print("  • Time series too short")
        print("  • Entropy not the right observable")
        print("  • Need to try different delay (τ)")

    print()

    # Generate plots
    print("-" * 70)
    print("Generating diagnostic plots...")
    plot_fnn_results(results, entropy_series, data_dir)

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Review FNN plots in takens_data/")
    print("2. If Switch converges unexpectedly, check router implementation")
    print("3. If results support Takens, add to Paper 1 appendix")
    print("4. Consider delay sweep (τ ∈ {1, 2, 5, 10}) for robustness")
    print()


if __name__ == '__main__':
    main()
