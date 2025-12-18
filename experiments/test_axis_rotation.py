#!/usr/bin/env python3
"""
Axis Rotation Test: Substantive vs Formal Identity

Tests whether the 2D routing manifold has:
- FIXED AXES (substantive identity): Same coordinate system across all regimes
- ROTATING AXES (formal identity): Context-dependent coordinate systems

Method:
1. Capture routing trajectories in different regimes:
   - Different architectures (Mixtral vs DeepSeek)
   - Different noise levels (σ=0.0 vs σ=0.5)
   - Different coupling states (baseline vs Chronovisor)
   - Different training phases (early vs late)

2. For each regime:
   - Embed routing entropy in 2D via PCA
   - Extract principal components (PC1, PC2)
   - These are the "natural axes" for that regime

3. Compare axes across regimes:
   - Compute angles between PC1 vectors
   - Compute angles between PC2 vectors
   - Test if axes align (angle ≈ 0°) or rotate (angle varies)

Interpretation:
- Fixed axes (angle ≈ 0°): Universal coordinate system
  → Coherence and adaptability mean the same thing everywhere
  → Substantive identity - systems have character

- Rotating axes (angle varies): Regime-dependent framing
  → Same manifold, different coordinate systems
  → Formal identity - systems reframe based on context
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_routing_trajectories():
    """Load all available routing trajectories."""
    data_dir = Path(__file__).parent.parent / "takens_data"

    trajectories = {}

    # Mixtral + Chronovisor
    file = data_dir / "mixtral_chronovisor_entropy.npy"
    if file.exists():
        trajectories['Mixtral+Chronovisor'] = np.load(file)

    # DeepSeek + Chronovisor
    file = data_dir / "deepseek_deepseek_chronovisor_routing.npy"
    if file.exists():
        trajectories['DeepSeek+Chronovisor'] = np.load(file)

    # DeepSeek Baseline
    file = data_dir / "deepseek_baseline_routing.npy"
    if file.exists():
        data = np.load(file)
        if len(data) > 0:
            trajectories['DeepSeek Baseline'] = data

    # Noise conditions
    for noise in [0.0, 0.1, 0.5, 1.0]:
        file = data_dir / f"noise_scale_{noise:.1f}_routing.npy"
        if file.exists():
            trajectories[f'Noise σ={noise:.1f}'] = np.load(file)

    return trajectories


def compute_pca_axes(time_series, delay=1):
    """
    Compute principal component axes from time series.

    Uses delay-coordinate embedding then PCA to find natural axes.

    Returns:
        pc1: First principal component (vector)
        pc2: Second principal component (vector)
        variance_explained: How much variance each PC captures
    """
    # Create delay-coordinate embedding
    N = len(time_series)
    embedded = []
    for i in range(N - delay):
        point = [time_series[i], time_series[i + delay]]
        embedded.append(point)

    X = np.array(embedded)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    pca.fit(X_scaled)

    # Extract axes (principal components)
    pc1 = pca.components_[0]  # First principal component
    pc2 = pca.components_[1]  # Second principal component

    variance_explained = pca.explained_variance_ratio_

    return pc1, pc2, variance_explained, X_scaled, pca


def angle_between_vectors(v1, v2):
    """
    Compute angle (in degrees) between two vectors.

    Accounts for sign ambiguity (PC direction is arbitrary).
    Returns the acute angle (0-90 degrees).
    """
    # Normalize
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)

    # Dot product
    dot = np.dot(v1_norm, v2_norm)

    # Clamp to [-1, 1] for numerical stability
    dot = np.clip(dot, -1.0, 1.0)

    # Angle in radians
    angle_rad = np.arccos(np.abs(dot))  # abs() to get acute angle

    # Convert to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def analyze_axis_rotation(trajectories):
    """Analyze axis rotation across regimes."""

    print("=" * 70)
    print("AXIS ROTATION ANALYSIS")
    print("=" * 70)
    print()
    print("Testing whether 2D axes are:")
    print("  FIXED (substantive identity) - same coordinate system")
    print("  ROTATING (formal identity) - regime-dependent framing")
    print()

    # Compute PCA axes for each regime
    axes_data = {}

    for name, data in trajectories.items():
        print(f"{name}:")
        pc1, pc2, variance, X_scaled, pca = compute_pca_axes(data)

        axes_data[name] = {
            'pc1': pc1,
            'pc2': pc2,
            'variance': variance,
            'embedding': X_scaled,
            'pca': pca,
        }

        print(f"  PC1: [{pc1[0]:+.4f}, {pc1[1]:+.4f}]")
        print(f"  PC2: [{pc2[0]:+.4f}, {pc2[1]:+.4f}]")
        print(f"  Variance explained: PC1={variance[0]*100:.1f}%, PC2={variance[1]*100:.1f}%")
        print()

    return axes_data


def compare_axes_across_regimes(axes_data):
    """Compare axes across all regime pairs."""

    print("=" * 70)
    print("AXIS ALIGNMENT ACROSS REGIMES")
    print("=" * 70)
    print()

    regimes = list(axes_data.keys())

    # Compare all pairs
    results = []

    for i, regime1 in enumerate(regimes):
        for j, regime2 in enumerate(regimes):
            if i >= j:
                continue

            pc1_angle = angle_between_vectors(
                axes_data[regime1]['pc1'],
                axes_data[regime2]['pc1']
            )

            pc2_angle = angle_between_vectors(
                axes_data[regime1]['pc2'],
                axes_data[regime2]['pc2']
            )

            results.append({
                'regime1': regime1,
                'regime2': regime2,
                'pc1_angle': pc1_angle,
                'pc2_angle': pc2_angle,
            })

            print(f"{regime1} vs {regime2}:")
            print(f"  PC1 angle: {pc1_angle:.2f}°")
            print(f"  PC2 angle: {pc2_angle:.2f}°")
            print()

    return results


def visualize_axis_rotation(axes_data, alignment_results):
    """Visualize axis rotation across regimes."""

    fig = plt.figure(figsize=(16, 12))

    # Plot 1: 2D embeddings with axes overlaid
    regimes = list(axes_data.keys())
    n_regimes = len(regimes)

    rows = (n_regimes + 2) // 3
    cols = 3

    for idx, (name, data) in enumerate(axes_data.items()):
        ax = plt.subplot(rows, cols, idx + 1)

        # Plot embedded points
        X = data['embedding']
        ax.scatter(X[:, 0], X[:, 1], alpha=0.3, s=20, c='gray')

        # Plot PC axes
        pc1 = data['pc1']
        pc2 = data['pc2']

        # Scale for visualization
        scale = 2.0
        ax.arrow(0, 0, pc1[0]*scale, pc1[1]*scale,
                head_width=0.2, head_length=0.2, fc='red', ec='red',
                linewidth=2, label='PC1')
        ax.arrow(0, 0, pc2[0]*scale, pc2[1]*scale,
                head_width=0.2, head_length=0.2, fc='blue', ec='blue',
                linewidth=2, label='PC2')

        ax.set_xlabel('Dimension 1', fontsize=10)
        ax.set_ylabel('Dimension 2', fontsize=10)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)

    plt.tight_layout()

    # Save
    output_dir = Path(__file__).parent.parent / "takens_data"
    plt.savefig(output_dir / "axis_rotation_embeddings.png", dpi=150, bbox_inches='tight')
    print(f"📊 Embeddings saved: {output_dir / 'axis_rotation_embeddings.png'}")

    # Plot 2: Angle matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Create angle matrices
    n = len(regimes)
    pc1_angles = np.zeros((n, n))
    pc2_angles = np.zeros((n, n))

    for result in alignment_results:
        i = regimes.index(result['regime1'])
        j = regimes.index(result['regime2'])
        pc1_angles[i, j] = result['pc1_angle']
        pc1_angles[j, i] = result['pc1_angle']
        pc2_angles[i, j] = result['pc2_angle']
        pc2_angles[j, i] = result['pc2_angle']

    # Plot PC1 angles
    im1 = ax1.imshow(pc1_angles, cmap='RdYlGn_r', vmin=0, vmax=90)
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels([r[:20] for r in regimes], rotation=45, ha='right', fontsize=9)
    ax1.set_yticklabels([r[:20] for r in regimes], fontsize=9)
    ax1.set_title('PC1 Alignment Angles (degrees)', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax1)

    # Add text annotations
    for i in range(n):
        for j in range(n):
            if i != j:
                text = ax1.text(j, i, f'{pc1_angles[i, j]:.0f}°',
                               ha="center", va="center", color="black", fontsize=8)

    # Plot PC2 angles
    im2 = ax2.imshow(pc2_angles, cmap='RdYlGn_r', vmin=0, vmax=90)
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels([r[:20] for r in regimes], rotation=45, ha='right', fontsize=9)
    ax2.set_yticklabels([r[:20] for r in regimes], fontsize=9)
    ax2.set_title('PC2 Alignment Angles (degrees)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax2)

    # Add text annotations
    for i in range(n):
        for j in range(n):
            if i != j:
                text = ax2.text(j, i, f'{pc2_angles[i, j]:.0f}°',
                               ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "axis_rotation_angles.png", dpi=150, bbox_inches='tight')
    print(f"📊 Angle matrices saved: {output_dir / 'axis_rotation_angles.png'}")


def interpret_results(alignment_results):
    """Interpret axis alignment results."""

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()

    # Compute statistics
    all_pc1_angles = [r['pc1_angle'] for r in alignment_results]
    all_pc2_angles = [r['pc2_angle'] for r in alignment_results]

    mean_pc1 = np.mean(all_pc1_angles)
    std_pc1 = np.std(all_pc1_angles)
    mean_pc2 = np.mean(all_pc2_angles)
    std_pc2 = np.std(all_pc2_angles)

    print(f"PC1 alignment: {mean_pc1:.2f}° ± {std_pc1:.2f}°")
    print(f"PC2 alignment: {mean_pc2:.2f}° ± {std_pc2:.2f}°")
    print()

    # Thresholds
    FIXED_THRESHOLD = 15.0  # degrees - if angles < 15°, axes are aligned
    ROTATING_THRESHOLD = 45.0  # degrees - if angles > 45°, axes are rotating

    pc1_fixed = mean_pc1 < FIXED_THRESHOLD
    pc2_fixed = mean_pc2 < FIXED_THRESHOLD

    pc1_rotating = mean_pc1 > ROTATING_THRESHOLD
    pc2_rotating = mean_pc2 > ROTATING_THRESHOLD

    if pc1_fixed and pc2_fixed:
        print("✓ FIXED AXES (Substantive Identity)")
        print()
        print("The principal components align across regimes (< 15° deviation).")
        print()
        print("Interpretation:")
        print("  → Universal coordinate system")
        print("  → Coherence and adaptability mean the same thing everywhere")
        print("  → Systems have CHARACTER - consistent orientation to fixed dimensions")
        print()
        print("What this means:")
        print("  You are WHAT you preserve and adapt.")
        print("  The axes define universal strategic dimensions.")
        print("  All systems navigate the same fundamental questions.")
        print()
        print("Implication:")
        print("  MoE routing has substantive identity.")
        print("  We can name the axes, measure positions, design interventions.")
        print("  Character is measurable and consistent.")

    elif pc1_rotating and pc2_rotating:
        print("✓ ROTATING AXES (Formal Identity)")
        print()
        print("The principal components vary across regimes (> 45° rotation).")
        print()
        print("Interpretation:")
        print("  → Regime-dependent coordinate systems")
        print("  → Same manifold, different framings in different contexts")
        print("  → Systems REFRAME based on regime - metacognitive")
        print()
        print("What this means:")
        print("  You are THAT you preserve and adapt.")
        print("  The structure is universal, the interpretation is contextual.")
        print("  Different regimes → different meanings of coherence/adaptability.")
        print()
        print("Implication:")
        print("  MoE routing has formal identity.")
        print("  The system doesn't just navigate - it INTERPRETS.")
        print("  Same geometry, contextual shadows, adaptive framing.")

    else:
        print("~ MIXED RESULTS")
        print()
        print("Axis alignment varies - some fixed, some rotating.")
        print()
        if pc1_fixed and not pc2_fixed:
            print("  PC1 is stable (universal primary dimension)")
            print("  PC2 rotates (contextual secondary dimension)")
        elif pc2_fixed and not pc1_fixed:
            print("  PC2 is stable (universal secondary dimension)")
            print("  PC1 rotates (contextual primary dimension)")
        print()
        print("Interpretation:")
        print("  Partial substantive identity - some universal structure")
        print("  Partial formal identity - some contextual framing")
        print("  Hybrid: core dimensions fixed, nuances contextual")

    print()
    print("=" * 70)


def main():
    print()
    print("=" * 70)
    print("AXIS ROTATION TEST: Substantive vs Formal Identity")
    print("=" * 70)
    print()
    print("Question: Do the 2D manifold axes rotate across regimes?")
    print()
    print("Fixed axes → Substantive identity (universal coordinates)")
    print("Rotating axes → Formal identity (contextual framing)")
    print()

    # Load trajectories
    trajectories = load_routing_trajectories()

    if len(trajectories) < 2:
        print("✗ Need at least 2 regimes to compare axes")
        print(f"  Found: {len(trajectories)}")
        return

    print(f"Loaded {len(trajectories)} regimes:")
    for name in trajectories.keys():
        print(f"  • {name} ({len(trajectories[name])} samples)")
    print()

    # Compute axes for each regime
    axes_data = analyze_axis_rotation(trajectories)

    # Compare axes across regimes
    alignment_results = compare_axes_across_regimes(axes_data)

    # Visualize
    visualize_axis_rotation(axes_data, alignment_results)

    # Interpret
    interpret_results(alignment_results)

    # Save summary
    output_dir = Path(__file__).parent.parent / "takens_data"
    summary_file = output_dir / "axis_rotation_summary.txt"

    with open(summary_file, 'w') as f:
        f.write("Axis Rotation Analysis Results\n")
        f.write("=" * 70 + "\n\n")

        f.write("Regime Pairs:\n")
        f.write("-" * 70 + "\n")
        for result in alignment_results:
            f.write(f"{result['regime1']} vs {result['regime2']}\n")
            f.write(f"  PC1 angle: {result['pc1_angle']:.2f}°\n")
            f.write(f"  PC2 angle: {result['pc2_angle']:.2f}°\n\n")

        all_pc1 = [r['pc1_angle'] for r in alignment_results]
        all_pc2 = [r['pc2_angle'] for r in alignment_results]

        f.write(f"\nSummary Statistics:\n")
        f.write(f"  PC1 alignment: {np.mean(all_pc1):.2f}° ± {np.std(all_pc1):.2f}°\n")
        f.write(f"  PC2 alignment: {np.mean(all_pc2):.2f}° ± {np.std(all_pc2):.2f}°\n")

    print(f"\nResults saved to: {summary_file}")
    print()


if __name__ == '__main__':
    main()
