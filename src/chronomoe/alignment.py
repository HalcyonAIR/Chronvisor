"""
Structural Alignment between Chronovisor and MoE experts.

V7 introduces dynamic alignment between two evolving ecosystems:
- Chronovisor experts: specialize geometrically (θ_home, k_home, s, λ, motifs)
- MoE experts: specialize organically (weight matrices, activation patterns)

The AlignmentMatrix A (N_chrono × N_moe) replaces rigid 1:1 identity mapping
with soft, learnable alignment that co-evolves as both systems specialize.

Key insight:
    A_ij = "how much Chronovisor expert i influences MoE expert j"

This enables:
- Expertise tracking across layers
- Lifelong specialization
- Merging/splitting real MoE heads
- Discovering new regions of semantic space
- Structural priors forming across layers
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def softmax(x: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax."""
    x = x / temperature
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


@dataclass
class AlignmentEvent:
    """Record of a structural alignment event."""

    tick: int
    event_type: str  # "absorb", "split", "merge", "decay"
    chrono_idx: int
    moe_idx: int
    old_value: float
    new_value: float
    reason: str


@dataclass
class AlignmentMatrix:
    """
    Soft alignment between Chronovisor experts and MoE experts.

    A (N_chrono × N_moe) matrix where:
    - A_ij = how much Chronovisor expert i influences MoE expert j
    - Rows are normalized (sum to 1)
    - Initialized as identity (1:1 mapping)
    - Learns via EMA based on confidence × specialization signals

    Pressure flows through A:
        b_j = Σ_i A_ij · b_i

    This replaces rigid assignment with dynamic co-evolution.
    """

    # Required fields (no defaults) - must come first
    n_chrono: int
    n_moe: int
    A: np.ndarray = field(repr=False)
    dead_tick_counts: np.ndarray = field(repr=False)

    # Fields with defaults
    eta_A: float = 0.05  # Learning rate for alignment updates
    temperature: float = 1.0  # Temperature for softmax in update rule
    absorb_threshold: float = 0.7  # Strong alignment triggers absorption
    dead_threshold: float = 0.01  # Weak alignment considered dead
    dead_ticks_required: int = 10  # How long before dead triggers event

    # History for analysis
    A_history: list = field(default_factory=list)
    events: list = field(default_factory=list)

    # Metrics
    total_updates: int = 0

    @classmethod
    def create(
        cls,
        n_chrono: int,
        n_moe: int,
        eta_A: float = 0.05,
        temperature: float = 1.0,
        init_mode: str = "identity",
        seed: Optional[int] = None,
    ) -> AlignmentMatrix:
        """
        Factory method to create alignment matrix.

        Args:
            n_chrono: Number of Chronovisor experts.
            n_moe: Number of MoE experts.
            eta_A: Learning rate for alignment updates.
            temperature: Softmax temperature (lower = sharper).
            init_mode: "identity" or "uniform".
            seed: Random seed for initialization.
        """
        rng = np.random.default_rng(seed)

        if init_mode == "identity":
            # Start with identity-like mapping
            if n_chrono == n_moe:
                A = np.eye(n_chrono)
            else:
                # Initialize with slight preference for diagonal
                A = np.ones((n_chrono, n_moe)) / n_moe
                for i in range(min(n_chrono, n_moe)):
                    A[i, i] = 0.5
                # Re-normalize rows
                A = A / A.sum(axis=1, keepdims=True)
        elif init_mode == "uniform":
            A = np.ones((n_chrono, n_moe)) / n_moe
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")

        # Add tiny noise for symmetry breaking
        noise = rng.uniform(0, 0.01, A.shape)
        A = A + noise
        A = A / A.sum(axis=1, keepdims=True)

        dead_tick_counts = np.zeros((n_chrono, n_moe), dtype=int)

        return cls(
            n_chrono=n_chrono,
            n_moe=n_moe,
            A=A,
            eta_A=eta_A,
            temperature=temperature,
            dead_tick_counts=dead_tick_counts,
        )

    def update(
        self,
        chrono_confidence: np.ndarray,
        moe_specialization: np.ndarray,
        current_tick: int = 0,
    ) -> dict:
        """
        Update alignment matrix based on confidence × specialization.

        Update rule:
            target_ij = softmax_j(C_i · S_j)
            A_ij ← (1 − η_A) A_ij + η_A · target_ij

        Args:
            chrono_confidence: (N_chrono,) confidence/trust per Chronovisor expert.
            moe_specialization: (N_moe,) specialization/usefulness per MoE expert.
            current_tick: Current tick for event logging.

        Returns:
            Update diagnostics.
        """
        assert chrono_confidence.shape == (self.n_chrono,)
        assert moe_specialization.shape == (self.n_moe,)

        # Store old A for comparison
        A_old = self.A.copy()

        # Compute update target for each Chronovisor expert
        # target_ij = softmax_j(C_i · S_j)
        # Outer product gives C_i · S_j matrix
        CS = np.outer(chrono_confidence, moe_specialization)

        # Apply softmax per row (per Chronovisor expert)
        target = softmax(CS, axis=1, temperature=self.temperature)

        # EMA update
        self.A = (1 - self.eta_A) * self.A + self.eta_A * target

        # Re-normalize rows to ensure sum to 1
        self.A = self.A / self.A.sum(axis=1, keepdims=True)

        # Track dead alignments
        self._update_dead_tracking(current_tick)

        # Check for structural events
        events = self._check_structural_events(A_old, current_tick)

        # Update history
        self.A_history.append(self.A.copy())
        self.total_updates += 1

        return {
            "delta_A_norm": float(np.linalg.norm(self.A - A_old)),
            "max_alignment": float(self.A.max()),
            "min_alignment": float(self.A.min()),
            "entropy": self.alignment_entropy(),
            "events": events,
        }

    def update_with_knob(
        self,
        chrono_confidence: np.ndarray,
        moe_specialization: np.ndarray,
        alignment_lr_mul: float = 1.0,
        current_tick: int = 0,
    ) -> dict:
        """
        Update alignment matrix with meta-knob modulation.

        This extends update() with knob-controlled learning rate:
        - alignment_lr_mul > 1 → faster adaptation (κ > 0)
        - alignment_lr_mul < 1 → slower/frozen adaptation (κ < 0)
        - alignment_lr_mul = 0 → completely frozen

        Args:
            chrono_confidence: (N_chrono,) confidence per Chronovisor expert.
            moe_specialization: (N_moe,) specialization per MoE expert.
            alignment_lr_mul: Multiplier for learning rate (from MetaKnob).
            current_tick: Current tick for event logging.

        Returns:
            Update diagnostics.
        """
        assert chrono_confidence.shape == (self.n_chrono,)
        assert moe_specialization.shape == (self.n_moe,)

        # Store old A for comparison
        A_old = self.A.copy()

        # Compute effective learning rate
        effective_eta = self.eta_A * alignment_lr_mul

        # If learning rate is effectively zero, skip update
        if effective_eta < 1e-10:
            return {
                "delta_A_norm": 0.0,
                "max_alignment": float(self.A.max()),
                "min_alignment": float(self.A.min()),
                "entropy": self.alignment_entropy(),
                "events": [],
                "frozen": True,
            }

        # Compute update target
        CS = np.outer(chrono_confidence, moe_specialization)
        target = softmax(CS, axis=1, temperature=self.temperature)

        # EMA update with modulated learning rate
        self.A = (1 - effective_eta) * self.A + effective_eta * target

        # Re-normalize rows
        self.A = self.A / self.A.sum(axis=1, keepdims=True)

        # Track dead alignments
        self._update_dead_tracking(current_tick)

        # Check for structural events
        events = self._check_structural_events(A_old, current_tick)

        # Update history
        self.A_history.append(self.A.copy())
        self.total_updates += 1

        return {
            "delta_A_norm": float(np.linalg.norm(self.A - A_old)),
            "max_alignment": float(self.A.max()),
            "min_alignment": float(self.A.min()),
            "entropy": self.alignment_entropy(),
            "events": events,
            "effective_eta": effective_eta,
            "frozen": False,
        }

    def _update_dead_tracking(self, current_tick: int) -> None:
        """Track how long each alignment has been below dead threshold."""
        dead_mask = self.A < self.dead_threshold
        # Increment dead ticks where below threshold
        self.dead_tick_counts = np.where(
            dead_mask,
            self.dead_tick_counts + 1,
            0,  # Reset if above threshold
        )

    def _check_structural_events(
        self,
        A_old: np.ndarray,
        current_tick: int,
    ) -> list[AlignmentEvent]:
        """Check for and log structural events (absorb, decay, etc.)."""
        events = []

        # Check for absorption (strong alignment)
        for i in range(self.n_chrono):
            for j in range(self.n_moe):
                # Absorption: alignment crossed above threshold
                if self.A[i, j] >= self.absorb_threshold and A_old[i, j] < self.absorb_threshold:
                    event = AlignmentEvent(
                        tick=current_tick,
                        event_type="absorb",
                        chrono_idx=i,
                        moe_idx=j,
                        old_value=float(A_old[i, j]),
                        new_value=float(self.A[i, j]),
                        reason="strong_alignment",
                    )
                    events.append(event)
                    self.events.append(event)

                # Decay: alignment dead for too long
                if self.dead_tick_counts[i, j] >= self.dead_ticks_required:
                    if A_old[i, j] > 0:  # Only log if there was something there
                        event = AlignmentEvent(
                            tick=current_tick,
                            event_type="decay",
                            chrono_idx=i,
                            moe_idx=j,
                            old_value=float(A_old[i, j]),
                            new_value=float(self.A[i, j]),
                            reason="prolonged_dead",
                        )
                        events.append(event)
                        self.events.append(event)
                    # Reset counter
                    self.dead_tick_counts[i, j] = 0

        return events

    def apply_pressure(self, chrono_pressure: np.ndarray) -> np.ndarray:
        """
        Transform Chronovisor pressure through alignment matrix.

        b_j = Σ_i A_ij · b_i

        Args:
            chrono_pressure: (N_chrono,) pressure from Chronovisor experts.

        Returns:
            (N_moe,) aligned pressure for MoE experts.
        """
        assert chrono_pressure.shape == (self.n_chrono,)

        # Matrix-vector product: A^T @ b gives aligned pressure
        # A is (N_chrono, N_moe), so A.T @ chrono_pressure gives (N_moe,)
        moe_pressure = self.A.T @ chrono_pressure

        return moe_pressure

    def alignment_entropy(self) -> float:
        """
        Compute entropy of alignment matrix.

        Higher entropy = more diffuse alignment (less specialized).
        Lower entropy = sharper alignment (more specialized mapping).

        Returns:
            Average row entropy.
        """
        epsilon = 1e-10
        A_safe = np.clip(self.A, epsilon, 1.0)

        # Entropy per row
        row_entropies = -np.sum(A_safe * np.log(A_safe), axis=1)

        return float(np.mean(row_entropies))

    def max_entropy(self) -> float:
        """Maximum possible entropy (uniform distribution)."""
        return math.log(self.n_moe)

    def normalized_entropy(self) -> float:
        """Entropy normalized to [0, 1]."""
        max_ent = self.max_entropy()
        if max_ent == 0:
            return 0.0
        return self.alignment_entropy() / max_ent

    def sparsity(self, threshold: float = 0.1) -> float:
        """
        Fraction of alignments above threshold.

        Lower sparsity = sparser, more specialized mapping.
        """
        return float((self.A > threshold).mean())

    def dominant_mapping(self) -> np.ndarray:
        """
        Get the dominant MoE expert for each Chronovisor expert.

        Returns:
            (N_chrono,) array of MoE indices.
        """
        return np.argmax(self.A, axis=1)

    def reverse_mapping(self) -> np.ndarray:
        """
        Get the dominant Chronovisor expert for each MoE expert.

        Returns:
            (N_moe,) array of Chronovisor indices.
        """
        return np.argmax(self.A, axis=0)

    def get_alignment_strength(self, chrono_idx: int, moe_idx: int) -> float:
        """Get alignment strength between specific experts."""
        return float(self.A[chrono_idx, moe_idx])

    def get_chrono_influence(self, chrono_idx: int) -> np.ndarray:
        """Get influence distribution of a Chronovisor expert over MoE experts."""
        return self.A[chrono_idx].copy()

    def get_moe_sources(self, moe_idx: int) -> np.ndarray:
        """Get which Chronovisor experts influence a MoE expert."""
        return self.A[:, moe_idx].copy()

    def reset(self, init_mode: str = "identity", seed: Optional[int] = None) -> None:
        """Reset alignment matrix to initial state."""
        rng = np.random.default_rng(seed)

        if init_mode == "identity":
            if self.n_chrono == self.n_moe:
                self.A = np.eye(self.n_chrono)
            else:
                self.A = np.ones((self.n_chrono, self.n_moe)) / self.n_moe
                for i in range(min(self.n_chrono, self.n_moe)):
                    self.A[i, i] = 0.5
                self.A = self.A / self.A.sum(axis=1, keepdims=True)
        else:
            self.A = np.ones((self.n_chrono, self.n_moe)) / self.n_moe

        # Add tiny noise
        noise = rng.uniform(0, 0.01, self.A.shape)
        self.A = self.A + noise
        self.A = self.A / self.A.sum(axis=1, keepdims=True)

        self.dead_tick_counts = np.zeros((self.n_chrono, self.n_moe), dtype=int)
        self.A_history.clear()
        self.events.clear()
        self.total_updates = 0


@dataclass
class StructuralAligner:
    """
    Manages co-evolution between Chronovisor and MoE expert structures.

    Handles:
    - Alignment matrix updates
    - Absorption (Chronovisor absorbs MoE properties)
    - Stabilization (MoE gets stabilizing effects)
    - Splitting/merging suggestions

    This is the "feedback loop that aligns both structures as they evolve."
    """

    alignment: AlignmentMatrix

    # Absorption parameters
    absorption_rate: float = 0.1  # How fast Chronovisor absorbs MoE properties
    stabilization_rate: float = 0.05  # How much stability MoE gets from alignment

    # Tracking
    absorptions: list = field(default_factory=list)
    stabilizations: list = field(default_factory=list)

    @classmethod
    def create(
        cls,
        n_chrono: int,
        n_moe: int,
        eta_A: float = 0.05,
        seed: Optional[int] = None,
    ) -> StructuralAligner:
        """Create a structural aligner with fresh alignment matrix."""
        alignment = AlignmentMatrix.create(
            n_chrono=n_chrono,
            n_moe=n_moe,
            eta_A=eta_A,
            seed=seed,
        )
        return cls(alignment=alignment)

    def update_alignment(
        self,
        chrono_confidence: np.ndarray,
        moe_specialization: np.ndarray,
        current_tick: int = 0,
    ) -> dict:
        """
        Update alignment matrix and process structural events.

        Args:
            chrono_confidence: (N_chrono,) confidence per Chronovisor expert.
            moe_specialization: (N_moe,) specialization per MoE expert.
            current_tick: Current tick.

        Returns:
            Update diagnostics including any structural events.
        """
        return self.alignment.update(
            chrono_confidence=chrono_confidence,
            moe_specialization=moe_specialization,
            current_tick=current_tick,
        )

    def compute_aligned_pressure(self, chrono_pressure: np.ndarray) -> np.ndarray:
        """
        Transform Chronovisor pressure through alignment matrix.

        b_j = Σ_i A_ij · b_i

        Args:
            chrono_pressure: (N_chrono,) pressure from Chronovisor.

        Returns:
            (N_moe,) aligned pressure for MoE router.
        """
        return self.alignment.apply_pressure(chrono_pressure)

    def compute_absorption_deltas(
        self,
        chrono_states: list[dict],
        moe_specializations: np.ndarray,
    ) -> list[dict]:
        """
        Compute how Chronovisor experts should absorb MoE properties.

        When A_ij is high, Chronovisor expert i should:
        - Shift θ_home toward MoE expert j's "semantic center"
        - Adjust k_home based on MoE expert j's stability

        Args:
            chrono_states: List of Chronovisor expert state dicts.
            moe_specializations: (N_moe,) specialization scores.

        Returns:
            List of absorption delta dicts per Chronovisor expert.
        """
        deltas = []

        for i, state in enumerate(chrono_states):
            # Get this Chronovisor expert's alignment row
            alignment_row = self.alignment.A[i]

            # Weighted average of MoE specializations
            weighted_spec = float(np.sum(alignment_row * moe_specializations))

            # Compute delta for θ_home absorption
            # High alignment + high MoE specialization → absorb
            max_alignment = float(alignment_row.max())
            dominant_moe = int(np.argmax(alignment_row))

            # Delta proportional to alignment strength and specialization
            delta_theta_home = 0.0
            delta_k_home = 0.0

            if max_alignment > self.alignment.absorb_threshold:
                # Strong alignment: absorb MoE properties
                moe_spec = moe_specializations[dominant_moe]

                # Absorb toward more specialized behavior
                # (In real system, would use actual MoE semantic position)
                delta_theta_home = self.absorption_rate * max_alignment * moe_spec

                # Adjust adventurousness based on specialization
                # High specialization → more focused (lower k_home)
                delta_k_home = -self.absorption_rate * 0.1 * max_alignment * moe_spec

            deltas.append({
                "chrono_idx": i,
                "dominant_moe": dominant_moe,
                "max_alignment": max_alignment,
                "weighted_spec": weighted_spec,
                "delta_theta_home": delta_theta_home,
                "delta_k_home": delta_k_home,
            })

        return deltas

    def compute_stabilization_factors(self) -> np.ndarray:
        """
        Compute stabilization factors for MoE experts.

        MoE experts with strong Chronovisor alignment get stabilized
        (avoid thrashing, more consistent behavior).

        Returns:
            (N_moe,) stabilization factors in [0, 1].
        """
        # Sum of alignment strengths pointing to each MoE expert
        # A is (N_chrono, N_moe), so sum over axis 0
        total_alignment = self.alignment.A.sum(axis=0)

        # Normalize to [0, 1]
        max_total = total_alignment.max()
        if max_total > 0:
            stabilization = total_alignment / max_total * self.stabilization_rate
        else:
            stabilization = np.zeros(self.alignment.n_moe)

        return stabilization

    def suggest_splits(self, threshold: float = 0.3) -> list[dict]:
        """
        Suggest Chronovisor expert splits based on multi-modal alignment.

        If a Chronovisor expert has strong alignment with multiple
        distant MoE experts, it may benefit from splitting.

        Returns:
            List of split suggestions.
        """
        suggestions = []

        for i in range(self.alignment.n_chrono):
            row = self.alignment.A[i]

            # Find MoE experts with significant alignment
            significant = np.where(row > threshold)[0]

            if len(significant) >= 2:
                # Multiple significant alignments - potential split
                suggestions.append({
                    "chrono_idx": i,
                    "aligned_moe_experts": significant.tolist(),
                    "alignment_values": row[significant].tolist(),
                    "reason": "multi_modal_alignment",
                })

        return suggestions

    def suggest_merges(self, similarity_threshold: float = 0.9) -> list[dict]:
        """
        Suggest Chronovisor expert merges based on alignment similarity.

        If two Chronovisor experts have very similar alignment patterns,
        they may be redundant and could merge.

        Returns:
            List of merge suggestions.
        """
        suggestions = []

        for i in range(self.alignment.n_chrono):
            for j in range(i + 1, self.alignment.n_chrono):
                # Cosine similarity of alignment rows
                row_i = self.alignment.A[i]
                row_j = self.alignment.A[j]

                dot = np.dot(row_i, row_j)
                norm_i = np.linalg.norm(row_i)
                norm_j = np.linalg.norm(row_j)

                if norm_i > 0 and norm_j > 0:
                    similarity = dot / (norm_i * norm_j)

                    if similarity > similarity_threshold:
                        suggestions.append({
                            "chrono_idx_1": i,
                            "chrono_idx_2": j,
                            "similarity": float(similarity),
                            "reason": "redundant_alignment",
                        })

        return suggestions

    def get_diagnostics(self) -> dict:
        """Get comprehensive diagnostics about structural alignment."""
        return {
            "alignment_entropy": self.alignment.alignment_entropy(),
            "normalized_entropy": self.alignment.normalized_entropy(),
            "sparsity": self.alignment.sparsity(),
            "total_updates": self.alignment.total_updates,
            "num_events": len(self.alignment.events),
            "dominant_mapping": self.alignment.dominant_mapping().tolist(),
            "reverse_mapping": self.alignment.reverse_mapping().tolist(),
            "max_alignment": float(self.alignment.A.max()),
            "min_alignment": float(self.alignment.A.min()),
            "mean_alignment": float(self.alignment.A.mean()),
        }

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset aligner to initial state."""
        self.alignment.reset(seed=seed)
        self.absorptions.clear()
        self.stabilizations.clear()
