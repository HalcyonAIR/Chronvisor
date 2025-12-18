#!/usr/bin/env python3
"""
Analyze DeepSeek Routing with Takens Embedding

Tests the critical hypothesis: Does d=2 hold with 64 experts?

Compares:
1. Mixtral + Chronovisor (8 experts, d=2 baseline)
2. DeepSeek + Chronovisor (64 routed experts)
3. DeepSeek Baseline (64 routed experts, no P×T)

If d ≈ 2-4 for all three:
  → Strategy space is fundamental
  → Expert count doesn't matter
  → "Grammar stays simple even when vocabulary explodes"

If d >> 4 for DeepSeek:
  → More experts = higher-dimensional manifold
  → d=2 was artifact of small expert pool
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class TakensAnalyzer:
    """Takens embedding and FNN analysis."""

    def __init__(self, time_series, delay=1):
        self.time_series = np.array(time_series)
        self.N = len(self.time_series)
        self.delay = delay

    def embed(self, dimension):
        """Create delay-coordinate embedding."""
        embedded = []
        for i in range(self.N - (dimension - 1) * self.delay):
            point = [self.time_series[i + k * self.delay] for k in range(dimension)]
            embedded.append(point)
        return np.array(embedded)

    def false_nearest_neighbors(self, dimension, rtol=15.0, atol=2.0):
        """
        Compute False Nearest Neighbors percentage.

        Args:
            dimension: Current embedding dimension
            rtol: Relative tolerance threshold
            atol: Absolute tolerance threshold (in std units)

        Returns:
            Percentage of false nearest neighbors
        """
        # Embed in d and d+1 dimensions
        Y_d = self.embed(dimension)
        Y_d1 = self.embed(dimension + 1)

        # Trim to same size
        n_points = min(Y_d.shape[0], Y_d1.shape[0])
        Y_d = Y_d[:n_points]
        Y_d1 = Y_d1[:n_points]

        if n_points < 10:
            return 100.0  # Insufficient data

        # Compute pairwise distances in d dimensions
        from scipy.spatial.distance import pdist, squareform
        dist_d = squareform(pdist(Y_d))

        # Find nearest neighbor for each point
        nearest_neighbors = []
        false_count = 0

        for i in range(n_points):
            # Get sorted distances (excluding self)
            sorted_idx = np.argsort(dist_d[i])
            sorted_idx = sorted_idx[sorted_idx != i]  # Remove self

            if len(sorted_idx) == 0:
                continue

            # Nearest neighbor
            nn_idx = sorted_idx[0]
            nearest_neighbors.append(nn_idx)

            # Distance in d dimensions
            dist_i_nn_d = dist_d[i, nn_idx]

            # Distance in d+1 dimensions
            dist_i_nn_d1 = np.linalg.norm(Y_d1[i] - Y_d1[nn_idx])

            # Compute increase ratio
            if dist_i_nn_d > 1e-10:
                ratio = (dist_i_nn_d1 - dist_i_nn_d) / dist_i_nn_d
            else:
                ratio = 0.0

            # Absolute threshold (in std units)
            data_std = np.std(self.time_series)
            abs_threshold = atol * data_std

            # Check if false neighbor
            if ratio > rtol or dist_i_nn_d1 > abs_threshold:
                false_count += 1

        # Return percentage
        if len(nearest_neighbors) == 0:
            return 100.0

        return (false_count / len(nearest_neighbors)) * 100.0

    def find_optimal_dimension(self, max_dim=15, rtol=15.0, atol=2.0):
        """
        Find optimal embedding dimension via FNN.

        Returns:
            optimal_d: Dimension where FNN first drops below threshold
            fnn_curve: FNN percentages for each dimension
        """
        # Adjust max_dim based on available data
        max_feasible_dim = (self.N - 1) // self.delay
        if max_feasible_dim < max_dim:
            max_dim = max_feasible_dim

        fnn_curve = []
        for d in range(1, max_dim + 1):
            fnn = self.false_nearest_neighbors(d, rtol, atol)
            fnn_curve.append(fnn)

            # Check for convergence (FNN < 5%)
            if fnn < 5.0:
                return d, fnn_curve

        # If no convergence, return last dimension
        return max_dim, fnn_curve


def load_captures():
    """Load all routing captures."""
    data_dir = Path(__file__).parent.parent / "takens_data"

    captures = {}

    # Mixtral + Chronovisor (reference)
    mixtral_chrono_file = data_dir / "mixtral_chronovisor_entropy.npy"
    if mixtral_chrono_file.exists():
        captures['Mixtral+Chronovisor (8 experts)'] = np.load(mixtral_chrono_file)

    # DeepSeek + Chronovisor
    deepseek_chrono_file = data_dir / "deepseek_deepseek_chronovisor_routing.npy"
    if deepseek_chrono_file.exists():
        captures['DeepSeek+Chronovisor (64 routed)'] = np.load(deepseek_chrono_file)

    # DeepSeek Baseline
    deepseek_baseline_file = data_dir / "deepseek_baseline_routing.npy"
    if deepseek_baseline_file.exists():
        data = np.load(deepseek_baseline_file)
        if len(data) > 0:  # Check not empty
            captures['DeepSeek Baseline (64 routed)'] = data

    return captures


def analyze_all_conditions(captures, tau_values=[1, 2, 4, 8]):
    """Run FNN analysis on all conditions."""

    results = {}

    for name, data in captures.items():
        print(f"\n{'=' * 70}")
        print(f"{name}")
        print(f"{'=' * 70}")
        print(f"  Length: {len(data)} samples")
        print(f"  Mean: {np.mean(data):.4f}")
        print(f"  Std: {np.std(data):.4f}")
        print()

        condition_results = {}

        for tau in tau_values:
            analyzer = TakensAnalyzer(data, delay=tau)

            # Find optimal dimension
            optimal_d, fnn_curve = analyzer.find_optimal_dimension(max_dim=15)

            condition_results[tau] = {
                'optimal_d': optimal_d,
                'fnn_curve': fnn_curve,
                'fnn_at_optimal': fnn_curve[optimal_d - 1] if optimal_d <= len(fnn_curve) else fnn_curve[-1]
            }

            print(f"  τ={tau}: d_optimal={optimal_d}, FNN={condition_results[tau]['fnn_at_optimal']:.2f}%")

        results[name] = condition_results

    return results


def create_comparison_plot(results, captures):
    """Create comparison plot showing d convergence for all conditions."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    tau_values = [1, 2, 4, 8]
    conditions = list(results.keys())
    colors = ['blue', 'red', 'green', 'orange']

    for i, tau in enumerate(tau_values):
        ax = axes[i]

        for condition, color in zip(conditions, colors):
            if tau in results[condition]:
                fnn_curve = results[condition][tau]['fnn_curve']
                dimensions = list(range(1, len(fnn_curve) + 1))

                ax.plot(dimensions, fnn_curve, marker='o', label=condition, color=color, linewidth=2)

        ax.axhline(y=5.0, color='gray', linestyle='--', alpha=0.5, label='5% threshold')
        ax.set_xlabel('Embedding Dimension', fontsize=12)
        ax.set_ylabel('False Nearest Neighbors (%)', fontsize=12)
        ax.set_title(f'FNN vs Dimension (τ={tau})', fontsize=14)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-5, 105])

    plt.tight_layout()

    # Save
    output_dir = Path(__file__).parent.parent / "takens_data"
    output_file = output_dir / "deepseek_d2_hypothesis_test.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved: {output_file}")

    return fig


def interpret_results(results):
    """Interpret results and test hypothesis."""

    print("\n" + "=" * 70)
    print("HYPOTHESIS TEST: Does d=2 hold with 64 experts?")
    print("=" * 70)

    # Extract optimal d for each condition at tau=1
    tau = 1
    optimal_dims = {}

    for condition, cond_results in results.items():
        if tau in cond_results:
            optimal_dims[condition] = cond_results[tau]['optimal_d']

    print(f"\nOptimal embedding dimensions (τ={tau}):")
    print("-" * 70)
    for condition, d_opt in optimal_dims.items():
        print(f"{condition:40s} d = {d_opt}")

    # Test hypothesis
    print()
    print("=" * 70)

    if 'DeepSeek+Chronovisor (64 routed)' in optimal_dims and 'Mixtral+Chronovisor (8 experts)' in optimal_dims:
        d_deepseek = optimal_dims['DeepSeek+Chronovisor (64 routed)']
        d_mixtral = optimal_dims['Mixtral+Chronovisor (8 experts)']

        print(f"Comparison:")
        print(f"  Mixtral (8 experts):    d = {d_mixtral}")
        print(f"  DeepSeek (64 experts):  d = {d_deepseek}")
        print()

        if d_deepseek <= 4:
            print("✓ HYPOTHESIS CONFIRMED")
            print()
            print("Result: d ≈ 2-4 even with 64 experts")
            print()
            print("Interpretation:")
            print("  → Strategy space is FUNDAMENTAL")
            print("  → Expert count does NOT determine attractor dimension")
            print("  → 'Experts are vocabulary. Strategy is grammar.'")
            print("  → Grammar stays simple even when vocabulary explodes")
            print()
            print("Implication:")
            print("  P×T coupling operates at fundamental geometric level,")
            print("  independent of architecture scale.")
            print()
            print("The routing system asks the same small number of questions:")
            print("  • Explore vs Exploit?")
            print("  • Specialize vs Generalize?")
            print("  • Commit vs Hedge?")
            print()
            print("These are the dimensions of the attractor, not the number")
            print("of experts available.")

        else:
            print("✗ HYPOTHESIS REJECTED")
            print()
            print(f"Result: d = {d_deepseek} >> 4 with 64 experts")
            print()
            print("Interpretation:")
            print("  → More experts = higher-dimensional routing manifold")
            print("  → d=2 finding was artifact of small expert pool (8 experts)")
            print("  → Attractor dimension scales with choice space size")
            print()
            print("Implication:")
            print("  P×T coupling mechanism may be scale-dependent or")
            print("  architecture-specific.")

    else:
        print("⚠ Insufficient data to test hypothesis")
        print("  Missing DeepSeek or Mixtral captures")

    print("=" * 70)


def main():
    print("=" * 70)
    print("DEEPSEEK d=2 HYPOTHESIS TEST")
    print("=" * 70)
    print()
    print("Testing whether attractor dimension scales with expert count.")
    print()
    print("Hypothesis: d ≈ 2-4 regardless of expert count (8 vs 64)")
    print("Prediction: 'Grammar stays simple even when vocabulary explodes'")
    print()

    # Load captures
    captures = load_captures()

    if len(captures) == 0:
        print("✗ No captures found. Run capture_deepseek_routing_for_takens.py first.")
        return

    print(f"Loaded {len(captures)} conditions:")
    for name in captures.keys():
        print(f"  • {name}")
    print()

    # Analyze
    results = analyze_all_conditions(captures)

    # Plot
    create_comparison_plot(results, captures)

    # Interpret
    interpret_results(results)

    print()


if __name__ == '__main__':
    main()
