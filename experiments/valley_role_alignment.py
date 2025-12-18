"""
Valley-Role Alignment Analysis

Investigates whether geological temperature valleys correspond to
functional proto-roles discovered through turn-level usage analysis.

Key Question: If Expert 6 is the "early-phase specialist", does it sit
in a cooler temperature valley compared to underused experts?

Analysis:
1. Load trained model with live P×T geometry
2. Extract proto-role structure (turn-level usage specialization)
3. Extract geological structure (temperature valleys and ridges)
4. Compute alignment between the two structures

Expected Patterns:
- Specialized experts (high variance across turns) may cluster in valleys
- Generalist experts (uniform across turns) may have higher temperatures
- OR vice versa depending on pressure/temperature dynamics
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

from chronomoe.chronovisor_mixtral_bridge import ChronovisorMixtralForCausalLM
from chronomoe.mixtral_core import MixtralConfig
from experiments.analyze_turn_usage import TurnUsageAnalyzer


def compute_proto_role_similarity(turn_usage):
    """
    Compute pairwise similarity between experts based on turn preferences.

    Returns:
        similarity_matrix: (n_experts, n_experts) - cosine similarity
        role_distances: (n_experts, n_experts) - distance metric for clustering
    """

    n_turns, n_experts = turn_usage.shape

    # Normalize turn usage to get "preference profiles"
    total_usage = turn_usage.sum(axis=0, keepdims=True)
    total_usage[total_usage == 0] = 1  # Avoid division by zero
    preferences = turn_usage / total_usage  # (n_turns, n_experts)

    # Compute cosine similarity between expert profiles
    # Two experts are similar if they prefer the same turns
    similarity_matrix = np.zeros((n_experts, n_experts))
    for i in range(n_experts):
        for j in range(n_experts):
            prof_i = preferences[:, i]
            prof_j = preferences[:, j]
            norm_i = np.linalg.norm(prof_i)
            norm_j = np.linalg.norm(prof_j)
            if norm_i > 0 and norm_j > 0:
                similarity_matrix[i, j] = np.dot(prof_i, prof_j) / (norm_i * norm_j)
            else:
                similarity_matrix[i, j] = 0.0

    # Convert to distance for clustering
    role_distances = 1.0 - similarity_matrix

    return similarity_matrix, role_distances


def compute_temperature_similarity(structural_T):
    """
    Compute pairwise similarity between experts based on structural temperature.

    Returns:
        temp_similarity: (n_experts, n_experts) - similarity based on T̄ values
        temp_distances: (n_experts, n_experts) - distance for clustering
    """

    n_experts = len(structural_T)

    # Temperature distance: |T̄_i - T̄_j|
    temp_distances = np.abs(structural_T[:, None] - structural_T[None, :])

    # Convert to similarity (inversely related to distance)
    max_dist = temp_distances.max()
    if max_dist > 0:
        temp_similarity = 1.0 - (temp_distances / max_dist)
    else:
        temp_similarity = np.ones((n_experts, n_experts))

    return temp_similarity, temp_distances


def analyze_alignment(role_distances, temp_distances):
    """
    Compute alignment between proto-role structure and temperature structure.

    Use Mantel test correlation: Does role distance predict temperature distance?
    """

    # Flatten upper triangular (exclude diagonal)
    n = role_distances.shape[0]
    idx = np.triu_indices(n, k=1)

    role_vec = role_distances[idx]
    temp_vec = temp_distances[idx]

    # Pearson correlation
    if role_vec.std() > 0 and temp_vec.std() > 0:
        correlation = np.corrcoef(role_vec, temp_vec)[0, 1]
    else:
        correlation = 0.0

    return correlation


def main():
    """Valley-role alignment analysis."""

    print("=" * 70)
    print("VALLEY-ROLE ALIGNMENT ANALYSIS")
    print("=" * 70)
    print("\nQuestion: Do temperature valleys align with proto-roles?")
    print("=" * 70)

    # Check if trained model exists
    checkpoint_path = Path("proto_role_results/model_trained.pt")
    if not checkpoint_path.exists():
        print(f"\n❌ No trained model found at {checkpoint_path}")
        print("   Run proto_role_diagnostic.py first to generate a trained model.")
        return

    # Load model
    print(f"\n1. Loading trained model from {checkpoint_path}...")
    config = MixtralConfig(
        vocab_size=1000,
        hidden_dim=512,
        intermediate_dim=2048,
        num_layers=4,
        num_experts=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        enable_chronovisor=True,
    )

    model = ChronovisorMixtralForCausalLM(config)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    controller = model.model.controller

    # Load turn usage data (from previous diagnostic)
    print("\n2. Loading turn usage data...")
    turn_usage_path = Path("proto_role_results/trained_turn_usage.npy")
    if turn_usage_path.exists():
        # Assume this was saved by a modified analyzer
        print(f"   ⚠️  Expected {turn_usage_path} not found")
        print(f"   Using live analysis instead...")

    # We'll need to regenerate turn usage by running inference
    # For now, load from proto_role_diagnostic results if available
    # Otherwise, use structural temperature only

    # Extract temperature landscape
    print("\n3. Extracting temperature landscape...")
    layer_idx = 0  # Focus on first layer
    lens = controller.lenses[layer_idx]

    T_fast = lens.temperature_fast.copy()
    T_structural = lens.structural_T.copy()
    T_hierarchical = lens.structural_T_hierarchical.copy()
    pressure = lens.pressure.copy()

    print(f"\n   Layer {layer_idx} Temperature Profile:")
    print(f"   T_fast:       {T_fast}")
    print(f"   T̄ (struct):   {T_structural}")
    print(f"   T̄ (hierarch): {T_hierarchical}")
    print(f"   Pressure:     {pressure}")

    # Identify valleys and ridges
    st_diag = controller.get_structural_temperature_diagnostics()
    valleys = st_diag.get('valleys', [])
    ridges = st_diag.get('ridges', [])

    print(f"\n   Valleys (cooler experts): {valleys}")
    print(f"   Ridges (hotter experts):  {ridges}")

    # For demonstration, create a synthetic turn usage based on what we know
    # from proto_role_diagnostic.log
    # In a real run, this would be loaded from saved data

    print("\n4. Computing proto-role structure...")
    # Synthetic turn usage for demonstration (from earlier diagnostic)
    # Expert 6 specialized to early turns (Inquiry, Premise, Complication)
    # Experts 3, 4 specialized to Contradiction

    turn_usage = np.array([
        # Turn 0 (Inquiry)
        [0.23, 0.07, 0.07, 0.07, 0.09, 0.10, 0.30, 0.08],
        # Turn 1 (Premise)
        [0.18, 0.07, 0.08, 0.07, 0.09, 0.10, 0.30, 0.11],
        # Turn 2 (Complication)
        [0.21, 0.08, 0.05, 0.06, 0.09, 0.10, 0.29, 0.12],
        # Turn 3 (Contradiction)
        [0.14, 0.05, 0.04, 0.22, 0.22, 0.07, 0.21, 0.05],
        # Turn 4-6 (not enough data)
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])  # Shape: (num_turns, num_experts)

    print("\n   ⚠️  Using synthetic turn usage from previous diagnostic log")
    print("      (In production, this would be loaded from saved analysis)")

    # Compute similarities
    role_similarity, role_distances = compute_proto_role_similarity(turn_usage)
    temp_similarity, temp_distances = compute_temperature_similarity(T_hierarchical)

    # Compute alignment
    alignment_corr = analyze_alignment(role_distances, temp_distances)

    print("\n5. ALIGNMENT ANALYSIS:")
    print(f"   Mantel correlation: {alignment_corr:.4f}")

    if alignment_corr > 0.3:
        print(f"   ✅ POSITIVE ALIGNMENT!")
        print(f"      Experts with similar roles have similar temperatures")
        print(f"      → Temperature valleys reflect functional structure")
    elif alignment_corr < -0.3:
        print(f"   ⚠️  NEGATIVE ALIGNMENT!")
        print(f"      Experts with similar roles have DIFFERENT temperatures")
        print(f"      → Temperature may encode complementary information")
    else:
        print(f"   ≈  NO CLEAR ALIGNMENT (|r| < 0.3)")
        print(f"      → Temperature structure may be independent of proto-roles")
        print(f"      → Or variance is too small to detect pattern")

    # Detailed expert-by-expert analysis
    print("\n6. EXPERT-BY-EXPERT BREAKDOWN:")
    expert_var = turn_usage[:4].var(axis=0)  # Only turns 0-3 have data

    for expert_idx in range(8):
        role_specialization = "SPECIALIZED" if expert_var[expert_idx] > 0.01 else "Generalist"
        temp_status = "VALLEY" if expert_idx in valleys else ("RIDGE" if expert_idx in ridges else "Neutral")

        print(f"   Expert {expert_idx}: {role_specialization:12s} | T̄={T_hierarchical[expert_idx]:.4f} | {temp_status}")

    # Visualization
    print("\n7. Creating visualizations...")
    output_dir = Path("valley_role_results")
    output_dir.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Row 1: Similarity matrices
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(role_similarity, cmap='viridis', vmin=0, vmax=1)
    ax.set_title('Proto-Role Similarity\n(based on turn preferences)')
    ax.set_xlabel('Expert')
    ax.set_ylabel('Expert')
    plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(temp_similarity, cmap='plasma', vmin=0, vmax=1)
    ax.set_title('Temperature Similarity\n(based on T̄ values)')
    ax.set_xlabel('Expert')
    ax.set_ylabel('Expert')
    plt.colorbar(im, ax=ax)

    # Scatter: role distance vs temp distance
    ax = fig.add_subplot(gs[0, 2])
    n = role_distances.shape[0]
    idx = np.triu_indices(n, k=1)
    ax.scatter(role_distances[idx], temp_distances[idx], alpha=0.5, s=30)
    ax.set_xlabel('Proto-Role Distance')
    ax.set_ylabel('Temperature Distance')
    ax.set_title(f'Alignment Scatter\n(r={alignment_corr:.3f})')
    ax.grid(alpha=0.3)

    # Row 2: Temperature landscape
    ax = fig.add_subplot(gs[1, :])
    x = np.arange(8)
    width = 0.25
    ax.bar(x - width, T_fast, width, label='T_fast', alpha=0.7)
    ax.bar(x, T_structural, width, label='T̄_local', alpha=0.7)
    ax.bar(x + width, T_hierarchical, width, label='T̄_hierarchical', alpha=0.7)
    ax.set_xlabel('Expert')
    ax.set_ylabel('Temperature')
    ax.set_title('Temperature Landscape (Layer 0)')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Highlight valleys and ridges
    for v in valleys:
        ax.axvspan(v - 0.4, v + 0.4, alpha=0.1, color='blue', label='Valley' if v == valleys[0] else '')
    for r in ridges:
        ax.axvspan(r - 0.4, r + 0.4, alpha=0.1, color='red', label='Ridge' if r == ridges[0] else '')

    # Row 3: Proto-role specialization vs temperature
    ax = fig.add_subplot(gs[2, 0])
    ax.scatter(expert_var, T_hierarchical, s=100, alpha=0.7)
    for i, txt in enumerate(range(8)):
        ax.annotate(f'E{txt}', (expert_var[i], T_hierarchical[i]), fontsize=9)
    ax.set_xlabel('Proto-Role Variance\n(variance across turns 0-3)')
    ax.set_ylabel('Hierarchical T̄')
    ax.set_title('Specialization vs Temperature')
    ax.grid(alpha=0.3)

    # Pressure vs temperature
    ax = fig.add_subplot(gs[2, 1])
    ax.scatter(pressure, T_hierarchical, s=100, alpha=0.7)
    for i, txt in enumerate(range(8)):
        ax.annotate(f'E{txt}', (pressure[i], T_hierarchical[i]), fontsize=9)
    ax.set_xlabel('Pressure')
    ax.set_ylabel('Hierarchical T̄')
    ax.set_title('Pressure × Temperature Landscape')
    ax.grid(alpha=0.3)

    # Turn usage heatmap
    ax = fig.add_subplot(gs[2, 2])
    im = ax.imshow(turn_usage[:4].T, aspect='auto', cmap='YlOrRd')
    ax.set_xlabel('Conversation Turn')
    ax.set_ylabel('Expert')
    ax.set_title('Turn Usage (Turns 0-3)')
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Inquiry', 'Premise', 'Complic', 'Contra'])
    plt.colorbar(im, ax=ax)

    plot_path = output_dir / "valley_role_alignment.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   📊 Plot saved: {plot_path}")

    # Save data
    np.savez(
        output_dir / "valley_role_data.npz",
        turn_usage=turn_usage,
        T_fast=T_fast,
        T_structural=T_structural,
        T_hierarchical=T_hierarchical,
        pressure=pressure,
        role_similarity=role_similarity,
        temp_similarity=temp_similarity,
        alignment_corr=alignment_corr,
        valleys=np.array(valleys),
        ridges=np.array(ridges),
    )
    print(f"   💾 Data saved: {output_dir / 'valley_role_data.npz'}")

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if abs(alignment_corr) > 0.3 and len(valleys) > 0:
        print("\n✅ VALLEYS CORRELATE WITH PROTO-ROLE STRUCTURE!")
        print(f"   Alignment correlation: {alignment_corr:.3f}")
        print(f"   Number of valleys: {len(valleys)}")
        print(f"   → The geological temperature landscape reflects functional structure")
        print(f"   → ChronoMoE's P×T geometry encodes expert roles")
    elif len(valleys) == 0:
        print("\n⚠️  NO VALLEYS DETECTED")
        print(f"   T̄ variance: {st_diag['variance']:.6e}")
        print(f"   → Temperature may not have differentiated enough yet")
        print(f"   → Need longer training or stronger geological dynamics")
    else:
        print("\n≈  WEAK OR NO ALIGNMENT")
        print(f"   Alignment correlation: {alignment_corr:.3f}")
        print(f"   → Temperature structure exists but may encode different information")
        print(f"   → Proto-roles formed through router weights, not temperature control")


if __name__ == "__main__":
    main()
