"""
Simulated Mixture-of-Experts layer.

A minimal, fully observable MoE for experimentation:
- N experts (default 8-16)
- Simple linear transforms
- Full logging of activations and outputs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Expert:
    """
    A single expert: simple linear transform.

    y = Wx + b

    Attributes:
        expert_id: Unique identifier for this expert.
        input_dim: Dimension of input vectors.
        output_dim: Dimension of output vectors.
        W: Weight matrix (output_dim x input_dim).
        b: Bias vector (output_dim,).
        cluster_affinity: Which token clusters this expert is "good" at.
            Maps cluster_id -> affinity score (higher = better match).
    """

    expert_id: int
    input_dim: int
    output_dim: int
    W: np.ndarray = field(repr=False)
    b: np.ndarray = field(repr=False)
    cluster_affinity: dict[int, float] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        expert_id: int,
        input_dim: int,
        output_dim: int,
        cluster_affinity: Optional[dict[int, float]] = None,
        seed: Optional[int] = None,
    ) -> Expert:
        """
        Factory method to create an expert with random initialization.

        Args:
            expert_id: Unique identifier.
            input_dim: Input dimension.
            output_dim: Output dimension.
            cluster_affinity: Optional affinity map {cluster_id: score}.
            seed: Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)

        # Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        W = rng.normal(0, scale, (output_dim, input_dim))
        b = np.zeros(output_dim)

        return cls(
            expert_id=expert_id,
            input_dim=input_dim,
            output_dim=output_dim,
            W=W,
            b=b,
            cluster_affinity=cluster_affinity or {},
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = Wx + b.

        Args:
            x: Input vector of shape (input_dim,) or (batch, input_dim).

        Returns:
            Output vector of shape (output_dim,) or (batch, output_dim).
        """
        return x @ self.W.T + self.b

    def get_affinity(self, cluster_id: int) -> float:
        """Get this expert's affinity for a given cluster."""
        return self.cluster_affinity.get(cluster_id, 0.0)


@dataclass
class MoEOutput:
    """
    Output from a single MoE forward pass.

    Attributes:
        output: The final gated output (batch, output_dim).
        expert_outputs: Individual expert outputs (n_experts, batch, output_dim).
        gate_weights: Router gate weights (batch, n_experts).
        chosen_experts: Top-k expert indices per token (batch, k).
    """

    output: np.ndarray
    expert_outputs: np.ndarray
    gate_weights: np.ndarray
    chosen_experts: np.ndarray


class MoE:
    """
    Simulated Mixture-of-Experts layer.

    token -> hidden state h
    router: logits_k(h) -> gate weights -> experts
    output: sum_k gate_k * expert_k(h)
    """

    def __init__(
        self,
        n_experts: int = 8,
        input_dim: int = 64,
        output_dim: int = 64,
        top_k: int = 2,
        cluster_affinities: Optional[dict[int, dict[int, float]]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize MoE layer.

        Args:
            n_experts: Number of experts (default 8).
            input_dim: Input dimension.
            output_dim: Output dimension.
            top_k: Number of experts to route to (default 2 for top-2 routing).
            cluster_affinities: Optional mapping {expert_id: {cluster_id: affinity}}.
            seed: Random seed for reproducibility.
        """
        self.n_experts = n_experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.top_k = top_k

        # Create experts
        self.experts: list[Expert] = []
        for i in range(n_experts):
            affinity = cluster_affinities.get(i, {}) if cluster_affinities else {}
            expert_seed = seed + i if seed is not None else None
            expert = Expert.create(
                expert_id=i,
                input_dim=input_dim,
                output_dim=output_dim,
                cluster_affinity=affinity,
                seed=expert_seed,
            )
            self.experts.append(expert)

    def forward(
        self,
        x: np.ndarray,
        gate_weights: np.ndarray,
    ) -> MoEOutput:
        """
        Forward pass through MoE given pre-computed gate weights.

        The router is separate (see router.py) so we can inject pressure.

        Args:
            x: Input tensor of shape (batch, input_dim).
            gate_weights: Gate weights from router, shape (batch, n_experts).

        Returns:
            MoEOutput with all intermediate values for analysis.
        """
        batch_size = x.shape[0]

        # Compute all expert outputs
        expert_outputs = np.zeros((self.n_experts, batch_size, self.output_dim))
        for i, expert in enumerate(self.experts):
            expert_outputs[i] = expert.forward(x)

        # Get top-k experts per token
        chosen_experts = np.argsort(gate_weights, axis=1)[:, -self.top_k :]

        # Create sparse gate mask (only top-k experts contribute)
        sparse_gates = np.zeros_like(gate_weights)
        for b in range(batch_size):
            sparse_gates[b, chosen_experts[b]] = gate_weights[b, chosen_experts[b]]

        # Renormalize sparse gates
        gate_sums = sparse_gates.sum(axis=1, keepdims=True)
        gate_sums = np.where(gate_sums > 0, gate_sums, 1.0)  # Avoid division by zero
        sparse_gates = sparse_gates / gate_sums

        # Compute gated output: sum_k gate_k * expert_k(x)
        output = np.zeros((batch_size, self.output_dim))
        for k in range(self.n_experts):
            output += sparse_gates[:, k : k + 1] * expert_outputs[k]

        return MoEOutput(
            output=output,
            expert_outputs=expert_outputs,
            gate_weights=gate_weights,
            chosen_experts=chosen_experts,
        )

    def get_expert(self, expert_id: int) -> Expert:
        """Get expert by ID."""
        return self.experts[expert_id]

    def get_ground_truth_expert(self, cluster_id: int) -> int:
        """
        Get the "best" expert for a given cluster based on affinity.

        Used for computing synthetic loss.
        """
        best_expert = 0
        best_affinity = -float("inf")
        for expert in self.experts:
            affinity = expert.get_affinity(cluster_id)
            if affinity > best_affinity:
                best_affinity = affinity
                best_expert = expert.expert_id
        return best_expert
