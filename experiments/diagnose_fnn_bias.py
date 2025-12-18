#!/usr/bin/env python3
"""
Diagnostic: Check for FNN Analysis Biases

Tests whether the d=2 finding is genuine or artifact of:
1. Early stopping threshold (5%)
2. Parameter choices (rtol, atol)
3. Insufficient data
4. Implementation bugs
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist, squareform


class FNNDiagnostic:
    """Diagnostic version of FNN with full reporting."""

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

    def false_nearest_neighbors_diagnostic(self, dimension, rtol=15.0, atol=2.0):
        """
        FNN with full diagnostic output.

        Returns:
            fnn_percent: Percentage of false neighbors
            diagnostics: Dict with detailed info
        """
        # Embed
        Y_d = self.embed(dimension)
        Y_d1 = self.embed(dimension + 1)

        # Trim
        n_points = min(Y_d.shape[0], Y_d1.shape[0])
        Y_d = Y_d[:n_points]
        Y_d1 = Y_d1[:n_points]

        diagnostics = {
            'n_points': n_points,
            'dimension': dimension,
            'rtol': rtol,
            'atol': atol,
        }

        if n_points < 10:
            diagnostics['insufficient_data'] = True
            return 100.0, diagnostics

        # Compute distances
        dist_d = squareform(pdist(Y_d))

        # Statistics
        false_count = 0
        total_count = 0
        ratio_violations = 0
        abs_violations = 0

        ratios = []
        distances_d = []
        distances_d1 = []

        data_std = np.std(self.time_series)
        abs_threshold = atol * data_std

        for i in range(n_points):
            sorted_idx = np.argsort(dist_d[i])
            sorted_idx = sorted_idx[sorted_idx != i]

            if len(sorted_idx) == 0:
                continue

            nn_idx = sorted_idx[0]
            total_count += 1

            dist_i_nn_d = dist_d[i, nn_idx]
            dist_i_nn_d1 = np.linalg.norm(Y_d1[i] - Y_d1[nn_idx])

            distances_d.append(dist_i_nn_d)
            distances_d1.append(dist_i_nn_d1)

            # Compute ratio
            if dist_i_nn_d > 1e-10:
                ratio = (dist_i_nn_d1 - dist_i_nn_d) / dist_i_nn_d
            else:
                ratio = 0.0

            ratios.append(ratio)

            # Check violations
            ratio_violation = ratio > rtol
            abs_violation = dist_i_nn_d1 > abs_threshold

            if ratio_violation:
                ratio_violations += 1
            if abs_violation:
                abs_violations += 1

            if ratio_violation or abs_violation:
                false_count += 1

        fnn_percent = (false_count / total_count * 100.0) if total_count > 0 else 100.0

        diagnostics.update({
            'total_points': total_count,
            'false_count': false_count,
            'ratio_violations': ratio_violations,
            'abs_violations': abs_violations,
            'fnn_percent': fnn_percent,
            'mean_ratio': np.mean(ratios),
            'median_ratio': np.median(ratios),
            'max_ratio': np.max(ratios),
            'mean_dist_d': np.mean(distances_d),
            'mean_dist_d1': np.mean(distances_d1),
            'data_std': data_std,
            'abs_threshold': abs_threshold,
        })

        return fnn_percent, diagnostics

    def sweep_parameters(self, dimension, rtol_values, atol_values):
        """Test FNN across parameter ranges."""
        results = {}

        for rtol in rtol_values:
            for atol in atol_values:
                fnn, diag = self.false_nearest_neighbors_diagnostic(dimension, rtol, atol)
                results[(rtol, atol)] = fnn

        return results

    def full_fnn_curve(self, max_dim=15, rtol=15.0, atol=2.0):
        """Compute FNN for all dimensions (no early stopping)."""
        max_feasible_dim = (self.N - 1) // self.delay
        max_dim = min(max_dim, max_feasible_dim)

        fnn_values = []
        diagnostics_list = []

        for d in range(1, max_dim + 1):
            fnn, diag = self.false_nearest_neighbors_diagnostic(d, rtol, atol)
            fnn_values.append(fnn)
            diagnostics_list.append(diag)

        return fnn_values, diagnostics_list


def test_data_quality(time_series, name):
    """Check if data has issues that could bias FNN."""
    print(f"\n{'=' * 70}")
    print(f"DATA QUALITY: {name}")
    print(f"{'=' * 70}")

    data = np.array(time_series)

    # Basic statistics
    print(f"  Length: {len(data)}")
    print(f"  Mean: {np.mean(data):.6f}")
    print(f"  Std: {np.std(data):.6f}")
    print(f"  Range: [{np.min(data):.6f}, {np.max(data):.6f}]")

    # Check for plateaus
    deltas = np.abs(np.diff(data))
    plateau_threshold = 1e-6
    plateau_count = np.sum(deltas < plateau_threshold)
    plateau_pct = plateau_count / len(deltas) * 100

    print(f"  Plateaus (|Δ| < {plateau_threshold}): {plateau_pct:.1f}%")

    # Check for variance
    if np.std(data) < 1e-6:
        print(f"  ⚠️ WARNING: Very low variance, FNN may be unreliable")

    # Check for periodicity
    from scipy.fft import fft
    fft_data = np.abs(fft(data - np.mean(data)))
    dominant_freq_idx = np.argmax(fft_data[1:len(fft_data)//2]) + 1
    dominant_freq = dominant_freq_idx / len(data)
    print(f"  Dominant frequency: {dominant_freq:.4f} (period ≈ {1/dominant_freq:.1f} samples)")

    return {
        'plateau_pct': plateau_pct,
        'std': np.std(data),
        'dominant_period': 1/dominant_freq if dominant_freq > 0 else np.inf,
    }


def test_parameter_sensitivity(time_series, name):
    """Test sensitivity to rtol and atol parameters."""
    print(f"\n{'=' * 70}")
    print(f"PARAMETER SENSITIVITY: {name}")
    print(f"{'=' * 70}")

    analyzer = FNNDiagnostic(time_series, delay=1)

    # Test at d=2 (claimed optimal)
    rtol_values = [5.0, 10.0, 15.0, 20.0, 30.0]
    atol_values = [1.0, 2.0, 3.0, 5.0]

    print(f"\nFNN at d=2 with different thresholds:")
    print(f"{'rtol':>8s} {'atol':>8s} {'FNN (%)':>10s}")
    print("-" * 30)

    for rtol in rtol_values:
        for atol in atol_values:
            fnn, _ = analyzer.false_nearest_neighbors_diagnostic(2, rtol, atol)
            print(f"{rtol:8.1f} {atol:8.1f} {fnn:10.2f}%")


def test_full_curves_no_early_stop(captures):
    """Compute full FNN curves without early stopping."""
    print(f"\n{'=' * 70}")
    print("FULL FNN CURVES (No Early Stopping)")
    print(f"{'=' * 70}")

    fig, axes = plt.subplots(1, len(captures), figsize=(6 * len(captures), 5))
    if len(captures) == 1:
        axes = [axes]

    for idx, (name, data) in enumerate(captures.items()):
        analyzer = FNNDiagnostic(data, delay=1)
        fnn_curve, diagnostics = analyzer.full_fnn_curve(max_dim=15)

        print(f"\n{name}:")
        print(f"  Dimension | FNN (%)  | n_points | Ratio viol. | Abs viol.")
        print("-" * 65)
        for d, fnn_val, diag in zip(range(1, len(fnn_curve) + 1), fnn_curve, diagnostics):
            if 'insufficient_data' in diag:
                print(f"  {d:9d} | {'N/A':8s} | (insufficient data)")
            else:
                print(f"  {d:9d} | {fnn_val:7.2f}% | {diag['total_points']:8d} | "
                      f"{diag['ratio_violations']:11d} | {diag['abs_violations']:10d}")

        # Plot
        ax = axes[idx]
        dimensions = list(range(1, len(fnn_curve) + 1))
        ax.plot(dimensions, fnn_curve, marker='o', linewidth=2, markersize=8)
        ax.axhline(y=5.0, color='red', linestyle='--', label='5% threshold')
        ax.set_xlabel('Embedding Dimension', fontsize=12)
        ax.set_ylabel('False Nearest Neighbors (%)', fontsize=12)
        ax.set_title(name, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-5, 105])

    plt.tight_layout()

    # Save
    output_dir = Path(__file__).parent.parent / "takens_data"
    output_file = output_dir / "fnn_bias_diagnostic.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Full curves saved: {output_file}")


def main():
    print("=" * 70)
    print("FNN BIAS DIAGNOSTIC")
    print("=" * 70)
    print()
    print("Checking for potential biases in FNN analysis:")
    print("  1. Data quality issues")
    print("  2. Parameter sensitivity")
    print("  3. Early stopping bias")
    print()

    # Load captures
    data_dir = Path(__file__).parent.parent / "takens_data"

    captures = {}

    # Mixtral
    mixtral_file = data_dir / "mixtral_chronovisor_entropy.npy"
    if mixtral_file.exists():
        captures['Mixtral (8 experts)'] = np.load(mixtral_file)

    # DeepSeek
    deepseek_file = data_dir / "deepseek_deepseek_chronovisor_routing.npy"
    if deepseek_file.exists():
        captures['DeepSeek (64 routed)'] = np.load(deepseek_file)

    if len(captures) == 0:
        print("✗ No captures found")
        return

    # Test 1: Data quality
    for name, data in captures.items():
        test_data_quality(data, name)

    # Test 2: Parameter sensitivity
    for name, data in captures.items():
        test_parameter_sensitivity(data, name)

    # Test 3: Full curves (no early stopping)
    test_full_curves_no_early_stop(captures)

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print()
    print("If d=2 is genuine:")
    print("  ✓ FNN should stay low (< 5%) at d≥2 across parameter ranges")
    print("  ✓ Full curve should show clear convergence, not just hit threshold")
    print("  ✓ Data should have sufficient variance and no major artifacts")
    print()
    print("If d=2 is artifact:")
    print("  ✗ FNN highly sensitive to parameters")
    print("  ✗ No clear convergence pattern in full curve")
    print("  ✗ Data has quality issues")
    print()


if __name__ == '__main__':
    main()
