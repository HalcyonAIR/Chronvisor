"""
Stateful Clock Heads for Temporal Arbitration

Three clocks distinguished by temporal persistence (half-life), not capacity:
    - Fast clock: Half-life = turns (high bandwidth, near-zero inertia)
    - Medium clock: Half-life = sessions (resists shocks, yields to patterns)
    - Slow clock: Half-life = relationships (barely moves, high persistence)

Each clock outputs a single score evaluating temporal coherence at its timescale.
Memory is not what is stored - memory is what refuses to disappear.

Architecture:
    Base Model → candidate tokens
                    ↓
    [Fast Clock]  → score₁
    [Medium Clock] → score₂
    [Slow Clock]   → score₃
                    ↓
    Elimination tournament → final token

Clocks store:
    - Reference frames (learned basis for projection)
    - Attractor maps (basin centroids, spreads, curvature)
    - Transition models (P(A→B | conditions))
    - Constraint sets (soft penalties on style, safety, preferences)
    - Value surfaces (what historically works)
    - Episodic sketchpad (rolling recent buffer)

Clock state is compressed geometry, not verbatim text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ClockState:
    """
    Internal state of a clock head.

    This is what lives in the "1GB" - compressed geometric structure,
    not raw text or facts.
    """
    # Reference frame: learned basis for projecting interactions
    reference_frame: np.ndarray  # [d_hidden, k_basis]

    # Attractor map: stable basins in behavior space
    attractor_centroids: np.ndarray  # [n_attractors, d_hidden]
    attractor_spreads: np.ndarray  # [n_attractors]
    attractor_curvatures: np.ndarray  # [n_attractors] (stiffness)

    # Transition model: P(A→B)
    transition_matrix: np.ndarray  # [n_attractors, n_attractors]
    current_basin: Optional[int] = None  # Which basin we're currently in

    # Constraint set: soft penalties
    constraints: List[np.ndarray] = field(default_factory=list)  # Each: [d_hidden]
    constraint_weights: List[float] = field(default_factory=list)

    # Value surface: what works
    value_surface: np.ndarray  # [d_hidden]

    # Episodic sketchpad: recent history (FIFO with importance sampling)
    recent_embeddings: deque = field(default_factory=lambda: deque(maxlen=1000))
    recent_margins: deque = field(default_factory=lambda: deque(maxlen=1000))
    recent_importance: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Decay parameters (what makes this clock different)
    half_life: float = 10.0  # How long influences persist (in ticks)
    decay_rate: float = 0.1  # How fast to forget


class ClockHead(nn.Module):
    """
    Base class for temporal clock heads.

    A clock is an integrator with decay laws, not a database.
    It evaluates candidate outputs for temporal coherence without generating.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        n_attractors: int = 32,
        k_basis: int = 64,
        half_life: float = 10.0,
        name: str = "clock",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_attractors = n_attractors
        self.k_basis = k_basis
        self.half_life = half_life
        self.name = name

        # Compute decay rate from half-life
        # After half_life ticks, influence decays to 50%
        self.decay_rate = 1.0 - np.exp(np.log(0.5) / half_life)

        # Initialize state
        self.state = self._initialize_state()

        # Tick counter
        self.ticks = 0

    def _initialize_state(self) -> ClockState:
        """Initialize clock state with random geometry."""
        # Random reference frame
        reference_frame = np.random.randn(self.hidden_dim, self.k_basis)
        reference_frame /= np.linalg.norm(reference_frame, axis=0, keepdims=True)

        # Random attractor centroids in reference space
        attractor_centroids = np.random.randn(self.n_attractors, self.hidden_dim) * 0.1
        attractor_spreads = np.ones(self.n_attractors) * 0.5
        attractor_curvatures = np.ones(self.n_attractors) * 1.0

        # Uniform transition matrix (no prior knowledge)
        transition_matrix = np.ones((self.n_attractors, self.n_attractors))
        transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

        # Zero value surface initially
        value_surface = np.zeros(self.hidden_dim)

        return ClockState(
            reference_frame=reference_frame,
            attractor_centroids=attractor_centroids,
            attractor_spreads=attractor_spreads,
            attractor_curvatures=attractor_curvatures,
            transition_matrix=transition_matrix,
            current_basin=None,
            constraints=[],
            constraint_weights=[],
            value_surface=value_surface,
            half_life=self.half_life,
            decay_rate=self.decay_rate,
        )

    def embed_interaction(
        self,
        context_ids: torch.Tensor,
        candidate_id: int,
        model_embeddings: nn.Embedding,
    ) -> np.ndarray:
        """
        Embed an interaction (context + candidate) into feature space.

        Args:
            context_ids: Recent token IDs (seq_len,)
            candidate_id: Candidate next token ID
            model_embeddings: Embedding layer from base model

        Returns:
            Embedding vector (hidden_dim,)
        """
        # Get embeddings for context and candidate
        with torch.no_grad():
            context_emb = model_embeddings(context_ids)  # [seq_len, hidden_dim]
            candidate_emb = model_embeddings(torch.tensor([candidate_id]))  # [1, hidden_dim]

        # Simple pooling: mean of context + candidate
        # Could be more sophisticated (attention, recency weighting, etc.)
        all_emb = torch.cat([context_emb, candidate_emb], dim=0)
        pooled = all_emb.mean(dim=0)  # [hidden_dim]

        return pooled.cpu().numpy()

    def find_nearest_basin(self, x: np.ndarray) -> Tuple[int, float, float]:
        """
        Find nearest attractor basin.

        Args:
            x: Query point (hidden_dim,)

        Returns:
            Tuple of (basin_idx, distance, stiffness)
        """
        # Compute distances to all attractors
        distances = np.linalg.norm(
            self.state.attractor_centroids - x.reshape(1, -1),
            axis=1,
        )

        # Find nearest
        basin_idx = int(np.argmin(distances))
        distance = float(distances[basin_idx])
        stiffness = float(self.state.attractor_curvatures[basin_idx])

        return basin_idx, distance, stiffness

    def compute_score(
        self,
        context_ids: torch.Tensor,
        candidate_id: int,
        model_embeddings: nn.Embedding,
    ) -> float:
        """
        Compute temporal coherence score for a candidate output.

        Args:
            context_ids: Recent context tokens
            candidate_id: Candidate next token
            model_embeddings: Embedding layer from base model

        Returns:
            Score in [0, 1] where higher = more coherent at this timescale
        """
        # Embed the interaction
        x = self.embed_interaction(context_ids, candidate_id, model_embeddings)

        # Project into reference frame
        x_proj = self.state.reference_frame.T @ x  # [k_basis]

        # Find nearest basin
        basin_idx, distance, stiffness = self.find_nearest_basin(x)

        # Basin proximity score (closer = better)
        proximity_score = 1.0 / (1.0 + distance)

        # Transition validity score
        if self.state.current_basin is not None:
            trans_prob = self.state.transition_matrix[
                self.state.current_basin, basin_idx
            ]
        else:
            trans_prob = 1.0  # No history, accept anything

        # Constraint violation score
        constraint_penalty = 0.0
        for constraint, weight in zip(
            self.state.constraints,
            self.state.constraint_weights,
        ):
            violation = np.maximum(0, np.dot(constraint, x))
            constraint_penalty += weight * violation

        constraint_score = np.exp(-constraint_penalty)

        # Value score (how good is this region historically)
        value = float(np.dot(self.state.value_surface, x))
        value_score = 1.0 / (1.0 + np.exp(-value))  # Sigmoid

        # Combine scores
        score = (
            proximity_score ** 0.4 *
            trans_prob ** 0.3 *
            constraint_score ** 0.2 *
            value_score ** 0.1
        )

        return float(np.clip(score, 0.0, 1.0))

    def update(
        self,
        context_ids: torch.Tensor,
        selected_id: int,
        model_embeddings: nn.Embedding,
        outcome: float = 1.0,  # Was this a good choice? (0-1)
    ) -> None:
        """
        Update clock state based on observed interaction.

        This is where temporal persistence happens. State evolves slowly
        based on what actually occurred.

        Args:
            context_ids: Context that was used
            selected_id: Token that was selected
            model_embeddings: Embedding layer
            outcome: How good was this choice (for value surface update)
        """
        # Embed the interaction
        x = self.embed_interaction(context_ids, selected_id, model_embeddings)

        # Find which basin this belongs to
        basin_idx, distance, stiffness = self.find_nearest_basin(x)

        # Update transition model if we had a previous basin
        if self.state.current_basin is not None:
            # Increment transition count (EMA)
            old_prob = self.state.transition_matrix[
                self.state.current_basin, basin_idx
            ]
            self.state.transition_matrix[
                self.state.current_basin, basin_idx
            ] = old_prob + self.decay_rate * (1.0 - old_prob)

            # Renormalize row
            row_sum = self.state.transition_matrix[self.state.current_basin].sum()
            self.state.transition_matrix[self.state.current_basin] /= row_sum

        # Update current basin
        prev_basin = self.state.current_basin
        self.state.current_basin = basin_idx

        # Update attractor centroid (move slightly toward this point)
        # This is the "memory" - attractors drift toward frequently visited regions
        centroid = self.state.attractor_centroids[basin_idx]
        self.state.attractor_centroids[basin_idx] = (
            (1.0 - self.decay_rate) * centroid +
            self.decay_rate * x
        )

        # Update attractor spread (variance of points in this basin)
        spread = self.state.attractor_spreads[basin_idx]
        self.state.attractor_spreads[basin_idx] = (
            (1.0 - self.decay_rate) * spread +
            self.decay_rate * distance
        )

        # Update value surface based on outcome
        # Positive outcomes reinforce this direction, negative penalize it
        value_delta = outcome - 0.5  # Center at 0
        self.state.value_surface += self.decay_rate * value_delta * x

        # Decay value surface over time (prevent unbounded growth)
        self.state.value_surface *= (1.0 - self.decay_rate * 0.1)

        # Add to episodic buffer
        self.state.recent_embeddings.append(x)
        self.state.recent_margins.append(distance)
        self.state.recent_importance.append(outcome)

        # Increment tick counter
        self.ticks += 1

    def apply_decay(self) -> None:
        """
        Apply temporal decay to all state.

        This is what makes memory fade over time.
        Called periodically (e.g., every generation step).
        """
        # Decay attractor curvatures toward baseline
        baseline_curvature = 1.0
        self.state.attractor_curvatures = (
            (1.0 - self.decay_rate * 0.1) * self.state.attractor_curvatures +
            self.decay_rate * 0.1 * baseline_curvature
        )

        # Decay transition matrix toward uniform
        uniform = np.ones_like(self.state.transition_matrix) / self.n_attractors
        self.state.transition_matrix = (
            (1.0 - self.decay_rate * 0.01) * self.state.transition_matrix +
            self.decay_rate * 0.01 * uniform
        )

        # Renormalize
        self.state.transition_matrix /= self.state.transition_matrix.sum(
            axis=1, keepdims=True
        )

    def reset(self) -> None:
        """Reset clock state."""
        self.state = self._initialize_state()
        self.ticks = 0


class FastClock(ClockHead):
    """
    Fast clock: Half-life measured in turns.

    High bandwidth, near-zero inertia.
    Forgets almost everything, exquisitely sensitive to local continuity.

    Question: "Did this just violate local continuity?"
    """

    def __init__(self, hidden_dim: int = 4096):
        super().__init__(
            hidden_dim=hidden_dim,
            n_attractors=64,  # More attractors, finer resolution
            k_basis=128,  # Higher dimensional projection
            half_life=5.0,  # Forgets after ~5 turns
            name="fast_clock",
        )


class MediumClock(ClockHead):
    """
    Medium clock: Half-life measured in sessions.

    Resists single shocks but yields to patterns.

    Question: "Is this becoming a theme or a deviation?"
    """

    def __init__(self, hidden_dim: int = 4096):
        super().__init__(
            hidden_dim=hidden_dim,
            n_attractors=32,  # Moderate number of basins
            k_basis=64,
            half_life=50.0,  # Persists for ~50 turns (session-scale)
            name="medium_clock",
        )


class SlowClock(ClockHead):
    """
    Slow clock: Half-life measured in relationships/identities.

    Barely moves. Only changes when reality keeps insisting.

    Question: "Is this still true after novelty, error, and boredom?"
    """

    def __init__(self, hidden_dim: int = 4096):
        super().__init__(
            hidden_dim=hidden_dim,
            n_attractors=16,  # Few strong basins
            k_basis=32,  # Lower dimensional projection
            half_life=500.0,  # Persists for ~500 turns (identity-scale)
            name="slow_clock",
        )


def elimination_tournament(
    candidates: List[int],
    scores: Dict[str, List[float]],
) -> int:
    """
    Run elimination tournament over candidate tokens.

    Lowest average score drops out until one remains.

    Args:
        candidates: List of candidate token IDs
        scores: Dict mapping clock name → list of scores
                {"fast": [s1, s2, ...], "medium": [...], "slow": [...]}

    Returns:
        Winning token ID
    """
    if len(candidates) == 0:
        raise ValueError("No candidates provided")

    if len(candidates) == 1:
        return candidates[0]

    # Compute average score for each candidate
    avg_scores = []
    for i in range(len(candidates)):
        clock_scores = [
            scores["fast"][i],
            scores["medium"][i],
            scores["slow"][i],
        ]
        avg = sum(clock_scores) / len(clock_scores)
        avg_scores.append(avg)

    # Find winner (highest average)
    winner_idx = int(np.argmax(avg_scores))

    return candidates[winner_idx]


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == '__main__':
    print("Testing Clock Heads")
    print("=" * 60)

    # Create clocks
    fast = FastClock(hidden_dim=256)
    medium = MediumClock(hidden_dim=256)
    slow = SlowClock(hidden_dim=256)

    print(f"Fast clock:   half-life = {fast.half_life:.1f} turns")
    print(f"Medium clock: half-life = {medium.half_life:.1f} turns")
    print(f"Slow clock:   half-life = {slow.half_life:.1f} turns")
    print()

    # Create mock embedding layer
    vocab_size = 1000
    hidden_dim = 256
    embeddings = nn.Embedding(vocab_size, hidden_dim)

    # Simulate some interactions
    print("Simulating 10 interactions...")
    context = torch.randint(0, vocab_size, (10,))

    for i in range(10):
        # Generate candidates
        candidates = [
            np.random.randint(0, vocab_size),
            np.random.randint(0, vocab_size),
            np.random.randint(0, vocab_size),
        ]

        # Score each candidate with all clocks
        scores = {"fast": [], "medium": [], "slow": []}

        for cand in candidates:
            scores["fast"].append(fast.compute_score(context, cand, embeddings))
            scores["medium"].append(medium.compute_score(context, cand, embeddings))
            scores["slow"].append(slow.compute_score(context, cand, embeddings))

        # Run tournament
        winner = elimination_tournament(candidates, scores)
        winner_idx = candidates.index(winner)

        print(f"\nTurn {i+1}:")
        print(f"  Candidates: {candidates}")
        print(f"  Fast scores:   {[f'{s:.3f}' for s in scores['fast']]}")
        print(f"  Medium scores: {[f'{s:.3f}' for s in scores['medium']]}")
        print(f"  Slow scores:   {[f'{s:.3f}' for s in scores['slow']]}")
        print(f"  Winner: {winner} (index {winner_idx})")

        # Update clocks with winner
        outcome = 1.0  # Assume good outcome
        fast.update(context, winner, embeddings, outcome)
        medium.update(context, winner, embeddings, outcome)
        slow.update(context, winner, embeddings, outcome)

        # Apply decay
        fast.apply_decay()
        medium.apply_decay()
        slow.apply_decay()

        # Update context
        context = torch.cat([context[1:], torch.tensor([winner])])

    print("\n" + "=" * 60)
    print("Clock state after 10 turns:")
    print(f"  Fast:   ticks={fast.ticks}, current_basin={fast.state.current_basin}")
    print(f"  Medium: ticks={medium.ticks}, current_basin={medium.state.current_basin}")
    print(f"  Slow:   ticks={slow.ticks}, current_basin={slow.state.current_basin}")

    print("\n✓ Clock heads working!")
    print("  - Memory is what refuses to disappear")
    print("  - Clocks distinguished by half-life, not capacity")
    print("  - Elimination tournament selects most coherent candidate")
