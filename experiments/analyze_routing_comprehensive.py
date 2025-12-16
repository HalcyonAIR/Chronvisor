#!/usr/bin/env python3
"""
Comprehensive Takens Analysis with Full Diagnostics

Per Halcyon's guidance:
1. Time series plots with first differences
2. Delta histograms (plateaus vs continuous)
3. FNN curves for τ ∈ {1, 2, 4, 8}
4. Full diagnostic suite

Current limitation: We're capturing pre-argmax signal for Switch (router_probs),
which shows the smooth logits BEFORE argmax destroys them. This will likely
converge, proving the smoothness exists but is destroyed by selection.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, List


class TakensAnalyzer:
    """Analyzes time series for delay-embedding structure."""

    def __init__(self, time_series: np.ndarray, delay: int = 1):
        self.ts = time_series
        self.delay = delay
        self.N = len(time_series)

    def embed(self, dimension: int) -> np.ndarray:
        """Create delay-coordinate embedding."""
        n_points = self.N - (dimension - 1) * self.delay

        # Check if we have enough points
        if n_points <= 0:
            raise ValueError(f"Not enough points for d={dimension}, τ={self.delay}. Need at least {(dimension-1)*self.delay + 1} points, have {self.N}.")

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
        """Compute percentage of false nearest neighbors."""
        Y_d = self.embed(dimension)
        Y_d1 = self.embed(dimension + 1)

        n_points = min(Y_d.shape[0], Y_d1.shape[0])
        false_count = 0

        Y_d = Y_d[:n_points]

        for i in range(n_points):
            dists_d = np.linalg.norm(Y_d - Y_d[i], axis=1)
            dists_d[i] = np.inf

            nn_idx = np.argmin(dists_d)
            dist_d = dists_d[nn_idx]

            if dist_d > atol * np.std(self.ts):
                continue

            if nn_idx >= n_points:
                continue

            dist_d1 = np.linalg.norm(Y_d1[i] - Y_d1[nn_idx])

            if (dist_d1 - dist_d) / dist_d > rtol:
                false_count += 1

        fnn_percent = 100.0 * false_count / n_points if n_points > 0 else 100.0
        return fnn_percent

    def dimension_sweep(
        self,
        max_dim: int = 15,
        rtol: float = 15.0,
        atol: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sweep embedding dimensions and compute FNN for each."""
        # Compute max feasible dimension given data length and delay
        max_feasible_dim = (self.N - 1) // self.delay
        if max_feasible_dim < max_dim:
            print(f"    ⚠ Reducing max_dim from {max_dim} to {max_feasible_dim} (insufficient data for τ={self.delay})")
            max_dim = max_feasible_dim

        if max_dim < 1:
            print(f"    ⚠ Cannot embed with τ={self.delay} (need at least {self.delay + 1} points)")
            return np.array([]), np.array([])

        dimensions = np.arange(1, max_dim + 1)
        fnn_values = np.zeros(max_dim)

        for d in dimensions:
            try:
                fnn_values[d - 1] = self.false_nearest_neighbors(d, rtol, atol)
            except ValueError as e:
                print(f"    ⚠ Skipping d={d}: {e}")
                fnn_values[d - 1] = 100.0  # Mark as not converged

        return dimensions, fnn_values


def load_entropy_data(condition: str, data_dir: Path) -> Tuple[np.ndarray, dict]:
    """Load captured entropy data."""
    entropy_path = data_dir / f"{condition}_entropy.npy"
    metadata_path = data_dir / f"{condition}_metadata.npy"

    if not entropy_path.exists():
        raise FileNotFoundError(f"Entropy data not found: {entropy_path}")

    entropy_series = np.load(entropy_path)
    metadata = np.load(metadata_path, allow_pickle=True).item()

    return entropy_series, metadata


def compute_time_series_diagnostics(entropy: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute diagnostic statistics for time series.

    Returns dict with:
    - deltas: First differences (smooth vs jumpy indicator)
    - abs_deltas: Absolute first differences
    - plateaus: Indices where series is nearly constant
    """
    deltas = np.diff(entropy)
    abs_deltas = np.abs(deltas)

    # Plateaus: where absolute change < 0.01 * std
    plateau_threshold = 0.01 * np.std(entropy)
    plateaus = np.where(abs_deltas < plateau_threshold)[0]

    return {
        'deltas': deltas,
        'abs_deltas': abs_deltas,
        'plateaus': plateaus,
        'plateau_fraction': len(plateaus) / len(deltas) if len(deltas) > 0 else 0,
    }


def plot_comprehensive_diagnostics(
    conditions_data: Dict[str, Tuple[np.ndarray, dict]],
    tau_values: List[int],
    output_dir: Path
):
    """
    Generate comprehensive diagnostic plots.

    Per Halcyon's request:
    1. Three entropy time series
    2. Three FNN curves for each τ
    3. Three delta histograms
    """

    num_conditions = len(conditions_data)
    tau_count = len(tau_values)

    # Create figure with comprehensive layout
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

    condition_names = list(conditions_data.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    # Row 1: Time series plots
    print("\n" + "=" * 70)
    print("TIME SERIES ANALYSIS")
    print("=" * 70)

    for i, (name, (entropy, metadata)) in enumerate(conditions_data.items()):
        ax = fig.add_subplot(gs[0, i])

        ax.plot(entropy, linewidth=1.5, alpha=0.8, color=colors[i])
        ax.set_title(f'{name}\\nMean: {np.mean(entropy):.4f} ± {np.std(entropy):.4f}',
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Step', fontsize=9)
        ax.set_ylabel('Routing Entropy', fontsize=9)
        ax.grid(True, alpha=0.3)

        print(f"\n{name}:")
        print(f"  Length: {len(entropy)} steps")
        print(f"  Mean: {np.mean(entropy):.4f}")
        print(f"  Std: {np.std(entropy):.4f}")
        print(f"  Range: [{np.min(entropy):.4f}, {np.max(entropy):.4f}]")

    # Row 2: Delta histograms
    print("\n" + "=" * 70)
    print("DELTA ANALYSIS (Smooth vs Discontinuous)")
    print("=" * 70)

    for i, (name, (entropy, metadata)) in enumerate(conditions_data.items()):
        ax = fig.add_subplot(gs[1, i])

        diag = compute_time_series_diagnostics(entropy)

        ax.hist(diag['deltas'], bins=20, alpha=0.7, color=colors[i], edgecolor='black')
        ax.set_title(f'First Differences\\nPlateaus: {diag["plateau_fraction"]*100:.1f}%',
                    fontsize=10)
        ax.set_xlabel('Δ Entropy', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        print(f"\n{name}:")
        print(f"  Plateau fraction: {diag['plateau_fraction']*100:.1f}%")
        print(f"  Mean |Δ|: {np.mean(diag['abs_deltas']):.6f}")
        print(f"  Std Δ: {np.std(diag['deltas']):.6f}")
        print(f"  Max |Δ|: {np.max(diag['abs_deltas']):.6f}")

        # Interpretation
        if diag['plateau_fraction'] > 0.5:
            print(f"  → HIGH plateaus - may indicate discrete jumps")
        else:
            print(f"  → LOW plateaus - continuous evolution")

    # Row 3: FNN curves for all τ values
    print("\n" + "=" * 70)
    print("FNN ANALYSIS")
    print("=" * 70)

    ax_fnn = fig.add_subplot(gs[2, :])

    max_dim = 15
    rtol = 15.0
    atol = 2.0

    fnn_results = {}

    for tau in tau_values:
        print(f"\n{'='*70}")
        print(f"τ = {tau}")
        print(f"{'='*70}")

        for i, (name, (entropy, metadata)) in enumerate(conditions_data.items()):
            print(f"\n{name}:")

            analyzer = TakensAnalyzer(entropy, delay=tau)
            dims, fnn = analyzer.dimension_sweep(max_dim, rtol, atol)

            # Skip if no data
            if len(dims) == 0:
                print(f"  ⚠ Skipping (insufficient data)")
                continue

            # Store results
            key = f"{name} (τ={tau})"
            fnn_results[key] = (dims, fnn)

            # Plot
            linestyle = ['-', '--', '-.', ':'][tau_values.index(tau)]
            ax_fnn.plot(dims, fnn, marker='o', label=key,
                       linewidth=2, markersize=3, linestyle=linestyle,
                       color=colors[i], alpha=0.7)

            # Find optimal dimension
            optimal_idx = np.where(fnn < 1.0)[0]
            if len(optimal_idx) > 0:
                optimal_dim = dims[optimal_idx[0]]
                print(f"  ✓ Optimal dimension: {optimal_dim}")
                print(f"    FNN at optimal: {fnn[optimal_idx[0]]:.2f}%")
            else:
                print(f"  ✗ No convergence (final FNN: {fnn[-1]:.2f}%)")

    ax_fnn.axhline(y=1.0, color='red', linestyle='--', alpha=0.5,
                  label='1% threshold', linewidth=2)
    ax_fnn.set_xlabel('Embedding Dimension', fontsize=12, fontweight='bold')
    ax_fnn.set_ylabel('False Nearest Neighbors (%)', fontsize=12, fontweight='bold')
    ax_fnn.set_title('FNN vs Embedding Dimension (Multiple τ)', fontsize=14, fontweight='bold')
    ax_fnn.legend(fontsize=8, ncol=2, loc='upper right')
    ax_fnn.grid(True, alpha=0.3)
    ax_fnn.set_ylim(0, min(100, max(max(fnn) for _, fnn in fnn_results.values()) * 1.1))

    # Row 4: 2D embeddings (τ=1 only for clarity)
    print("\n" + "=" * 70)
    print("2D EMBEDDINGS (τ=1)")
    print("=" * 70)

    for i, (name, (entropy, metadata)) in enumerate(conditions_data.items()):
        ax = fig.add_subplot(gs[3, i])

        analyzer = TakensAnalyzer(entropy, delay=1)
        embedded = analyzer.embed(dimension=2)

        # Color by time
        time_colors = np.arange(len(embedded))
        scatter = ax.scatter(embedded[:, 0], embedded[:, 1],
                           c=time_colors, cmap='viridis', s=15, alpha=0.6)
        ax.set_xlabel('x(t)', fontsize=9)
        ax.set_ylabel('x(t-τ)', fontsize=9)
        ax.set_title(f'2D Embedding: {name}', fontsize=10)
        plt.colorbar(scatter, ax=ax, label='Time')

    # Save
    output_path = output_dir / "comprehensive_takens_diagnostics.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Comprehensive diagnostics saved: {output_path}")
    plt.close()

    return fnn_results


def main():
    """Run comprehensive Takens analysis."""
    print("=" * 70)
    print("COMPREHENSIVE TAKENS ANALYSIS")
    print("=" * 70)
    print()
    print("Analyzing captured routing data with full diagnostic suite.")
    print()
    print("⚠ NOTE: Current captures use pre-argmax signal for Switch")
    print("   (router_probs before selection). This shows the smoothness")
    print("   that EXISTS but is DESTROYED by argmax. If Switch FNN converges,")
    print("   it proves forced cliff, not genuine cliff.")
    print()

    # Load data
    data_dir = Path("takens_data")
    if not data_dir.exists():
        print(f"❌ Error: Data directory not found: {data_dir}")
        return

    conditions = [
        ("mixtral_chronovisor", "Mixtral+Chronovisor"),
        ("mixtral_baseline", "Mixtral Baseline"),
        ("switch_top1", "Switch (pre-argmax)"),
    ]

    # tau values per Halcyon's request
    tau_values = [1, 2, 4, 8]

    print(f"Parameters:")
    print(f"  τ values: {tau_values}")
    print(f"  Max embedding dimension: 15")
    print(f"  Relative threshold: 15.0")
    print(f"  Absolute threshold: 2.0 std")
    print()

    # Load all data
    conditions_data = {}
    for condition_id, condition_name in conditions:
        try:
            entropy, metadata = load_entropy_data(condition_id, data_dir)
            conditions_data[condition_name] = (entropy, metadata)

            if len(entropy) < 100:
                print(f"⚠ WARNING: {condition_name} has only {len(entropy)} steps")
                print(f"   Halcyon recommends 200-1000 for clean FNN curves")
                print()
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return

    # Run comprehensive diagnostics
    fnn_results = plot_comprehensive_diagnostics(conditions_data, tau_values, data_dir)

    # Final interpretation
    print("\n" + "=" * 70)
    print("FINAL INTERPRETATION")
    print("=" * 70)
    print()

    # Get final FNN for τ=1
    mixtral_chrono_fnn = fnn_results.get("Mixtral+Chronovisor (τ=1)", (None, np.array([100])))[1][-1]
    mixtral_baseline_fnn = fnn_results.get("Mixtral Baseline (τ=1)", (None, np.array([100])))[1][-1]
    switch_fnn = fnn_results.get("Switch (pre-argmax) (τ=1)", (None, np.array([100])))[1][-1]

    print(f"Final FNN (τ=1, d=15):")
    print(f"  Mixtral+Chronovisor: {mixtral_chrono_fnn:.2f}%")
    print(f"  Mixtral Baseline: {mixtral_baseline_fnn:.2f}%")
    print(f"  Switch (pre-argmax): {switch_fnn:.2f}%")
    print()

    if switch_fnn < 1.0:
        print("✓ SWITCH FNN CONVERGED (pre-argmax)")
        print()
        print("Interpretation:")
        print("  The router logits (pre-argmax) show smooth attractor structure.")
        print("  This proves the curvature EXISTS in the logits.")
        print("  Argmax DESTROYS this smoothness by forcing discrete selection.")
        print()
        print("Conclusion:")
        print("  This is quantitative evidence for the 'FORCED CLIFF' hypothesis.")
        print("  The smoothness requirement isn't arbitrary - Takens embedding")
        print("  needs it to unfold the attractor. Argmax breaks the requirement.")
        print()
        print("Next step:")
        print("  To complete the test, capture POST-argmax signal for Switch")
        print("  (routing_weights instead of router_probs). Expect FNN to NOT converge.")
    else:
        print("⚠ UNEXPECTED: Switch pre-argmax FNN did not converge")
        print()
        print("Possible causes:")
        print("  • Too few samples (need 200-1000, have ~40)")
        print("  • Wrong τ value (try sweep)")
        print("  • Observable not smooth enough")

    print()
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
