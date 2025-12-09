"""
MoE Router with pressure injection hook.

Router architecture: linear -> ReLU -> linear -> softmax
Pressure injection: logits'_k = logits_k + b_k

The pressure bias b_k comes from Chronovisor and should remain in
the small-signal regime. Pressure is a wind, not a hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RoutingDecision:
    """
    A single routing decision for one token.

    Attributes:
        token_idx: Index of the token in the sequence.
        layer_idx: Index of the MoE layer.
        logits: Raw router logits (n_experts,).
        pressure_bias: Pressure bias from Chronovisor (n_experts,).
        adjusted_logits: logits + pressure_bias (n_experts,).
        gate_weights: Post-softmax gate weights (n_experts,).
        top_k_experts: Indices of chosen experts.
    """

    token_idx: int
    layer_idx: int
    logits: np.ndarray
    pressure_bias: np.ndarray
    adjusted_logits: np.ndarray
    gate_weights: np.ndarray
    top_k_experts: np.ndarray


@dataclass
class RoutingLog:
    """
    Log of routing decisions over a batch or sequence.

    Used to analyze decision trees and measure pressure effects.
    """

    decisions: list[RoutingDecision] = field(default_factory=list)

    def add(self, decision: RoutingDecision) -> None:
        """Add a routing decision to the log."""
        self.decisions.append(decision)

    def clear(self) -> None:
        """Clear all logged decisions."""
        self.decisions.clear()

    def get_expert_usage(self) -> np.ndarray:
        """
        Compute expert usage counts from logged decisions.

        Returns:
            Array of shape (n_experts,) with usage counts.
        """
        if not self.decisions:
            return np.array([])

        n_experts = len(self.decisions[0].logits)
        usage = np.zeros(n_experts)

        for decision in self.decisions:
            for expert_idx in decision.top_k_experts:
                usage[expert_idx] += 1

        return usage

    def get_expert_distribution(self) -> np.ndarray:
        """
        Compute normalized expert usage distribution.

        Returns:
            Probability distribution over experts.
        """
        usage = self.get_expert_usage()
        total = usage.sum()
        if total == 0:
            return usage
        return usage / total

    def get_mean_gate_weights(self) -> np.ndarray:
        """
        Compute mean gate weights per expert.

        Returns:
            Array of shape (n_experts,) with mean gate weights.
        """
        if not self.decisions:
            return np.array([])

        n_experts = len(self.decisions[0].logits)
        total_weights = np.zeros(n_experts)

        for decision in self.decisions:
            total_weights += decision.gate_weights

        return total_weights / len(self.decisions)

    def get_transition_matrix(self) -> np.ndarray:
        """
        Compute expert transition matrix.

        p(expert_j at t+1 | expert_i at t)

        Returns:
            Matrix of shape (n_experts, n_experts).
        """
        if len(self.decisions) < 2:
            return np.array([])

        n_experts = len(self.decisions[0].logits)
        transitions = np.zeros((n_experts, n_experts))

        for i in range(len(self.decisions) - 1):
            curr_top = self.decisions[i].top_k_experts[0]  # Top-1 expert
            next_top = self.decisions[i + 1].top_k_experts[0]
            transitions[curr_top, next_top] += 1

        # Normalize rows
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        return transitions / row_sums


class Router:
    """
    MoE router with pressure injection.

    Architecture: linear -> ReLU -> linear -> softmax
    Pressure: logits'_k = logits_k + b_k
    """

    def __init__(
        self,
        input_dim: int,
        n_experts: int,
        hidden_dim: Optional[int] = None,
        temperature: float = 1.0,
        layer_idx: int = 0,
        seed: Optional[int] = None,
    ):
        """
        Initialize router.

        Args:
            input_dim: Dimension of input hidden states.
            n_experts: Number of experts to route to.
            hidden_dim: Hidden dimension (default: input_dim // 2).
            temperature: Softmax temperature (lower = sharper routing).
            layer_idx: Index of this layer (for logging).
            seed: Random seed for reproducibility.
        """
        self.input_dim = input_dim
        self.n_experts = n_experts
        self.hidden_dim = hidden_dim or input_dim // 2
        self.temperature = temperature
        self.layer_idx = layer_idx

        rng = np.random.default_rng(seed)

        # Two-layer router: input -> hidden -> logits
        scale1 = np.sqrt(2.0 / (input_dim + self.hidden_dim))
        self.W1 = rng.normal(0, scale1, (self.hidden_dim, input_dim))
        self.b1 = np.zeros(self.hidden_dim)

        scale2 = np.sqrt(2.0 / (self.hidden_dim + n_experts))
        self.W2 = rng.normal(0, scale2, (n_experts, self.hidden_dim))
        self.b2 = np.zeros(n_experts)

        # Logging
        self.log = RoutingLog()
        self.logging_enabled = True

        # Current pressure bias (set externally via inject_pressure)
        self._pressure_bias: Optional[np.ndarray] = None

    def inject_pressure(self, b_k: np.ndarray) -> None:
        """
        Inject pressure bias from Chronovisor.

        Args:
            b_k: Pressure bias array of shape (n_experts,).
                Should be small (e.g., |b_k| < 1.0).
        """
        if b_k.shape != (self.n_experts,):
            raise ValueError(
                f"Pressure bias shape {b_k.shape} doesn't match n_experts {self.n_experts}"
            )
        self._pressure_bias = b_k.copy()

    def clear_pressure(self) -> None:
        """Remove pressure bias (return to baseline routing)."""
        self._pressure_bias = None

    def get_pressure(self) -> np.ndarray:
        """Get current pressure bias (zeros if none injected)."""
        if self._pressure_bias is None:
            return np.zeros(self.n_experts)
        return self._pressure_bias.copy()

    def compute_logits(self, x: np.ndarray) -> np.ndarray:
        """
        Compute raw router logits (before pressure).

        Args:
            x: Input hidden states of shape (batch, input_dim).

        Returns:
            Logits of shape (batch, n_experts).
        """
        # Layer 1: linear + ReLU
        h = x @ self.W1.T + self.b1
        h = np.maximum(h, 0)  # ReLU

        # Layer 2: linear
        logits = h @ self.W2.T + self.b2

        return logits

    def forward(
        self,
        x: np.ndarray,
        token_indices: Optional[np.ndarray] = None,
        top_k: int = 2,
    ) -> np.ndarray:
        """
        Forward pass: compute gate weights with pressure injection.

        Args:
            x: Input hidden states of shape (batch, input_dim).
            token_indices: Optional token indices for logging.
            top_k: Number of experts to select.

        Returns:
            Gate weights of shape (batch, n_experts).
        """
        batch_size = x.shape[0]

        # Compute raw logits
        logits = self.compute_logits(x)

        # Get pressure bias
        pressure = self.get_pressure()

        # Inject pressure: logits' = logits + b_k
        adjusted_logits = logits + pressure[np.newaxis, :]

        # Softmax with temperature
        scaled_logits = adjusted_logits / self.temperature
        exp_logits = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
        gate_weights = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # Get top-k experts
        top_k_experts = np.argsort(gate_weights, axis=1)[:, -top_k:]

        # Log decisions
        if self.logging_enabled:
            if token_indices is None:
                token_indices = np.arange(batch_size)

            for i in range(batch_size):
                decision = RoutingDecision(
                    token_idx=int(token_indices[i]),
                    layer_idx=self.layer_idx,
                    logits=logits[i].copy(),
                    pressure_bias=pressure.copy(),
                    adjusted_logits=adjusted_logits[i].copy(),
                    gate_weights=gate_weights[i].copy(),
                    top_k_experts=top_k_experts[i].copy(),
                )
                self.log.add(decision)

        return gate_weights

    def get_routing_stats(self) -> dict:
        """
        Get summary statistics from logged routing decisions.

        Returns:
            Dictionary with usage counts, mean gates, transitions.
        """
        return {
            "usage": self.log.get_expert_usage(),
            "distribution": self.log.get_expert_distribution(),
            "mean_gates": self.log.get_mean_gate_weights(),
            "transitions": self.log.get_transition_matrix(),
            "n_decisions": len(self.log.decisions),
        }

    def reset_log(self) -> None:
        """Clear the routing log."""
        self.log.clear()

    def forward_with_knob(
        self,
        x: np.ndarray,
        pressure_scale: float = 1.0,
        explore_bias: float = 1.0,
        token_indices: Optional[np.ndarray] = None,
        top_k: int = 2,
    ) -> np.ndarray:
        """
        Forward pass with meta-knob modulation.

        This extends the basic forward() with knob factors:
        - pressure_scale: Multiplies the pressure bias (κ > 0 → stronger pressure)
        - explore_bias: Multiplies softmax temperature (κ > 0 → softer distribution)

        Args:
            x: Input hidden states of shape (batch, input_dim).
            pressure_scale: Multiplier for pressure bias (from MetaKnob).
            explore_bias: Multiplier for softmax temperature (from MetaKnob).
            token_indices: Optional token indices for logging.
            top_k: Number of experts to select.

        Returns:
            Gate weights of shape (batch, n_experts).
        """
        batch_size = x.shape[0]

        # Compute raw logits
        logits = self.compute_logits(x)

        # Get pressure bias and apply scale
        pressure = self.get_pressure()
        scaled_pressure = pressure * pressure_scale

        # Inject scaled pressure: logits' = logits + scale * b_k
        adjusted_logits = logits + scaled_pressure[np.newaxis, :]

        # Softmax with modulated temperature
        # explore_bias > 1 → softer (more exploration)
        # explore_bias < 1 → sharper (more exploitation)
        effective_temperature = self.temperature * explore_bias
        scaled_logits = adjusted_logits / effective_temperature
        exp_logits = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
        gate_weights = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # Get top-k experts
        top_k_experts = np.argsort(gate_weights, axis=1)[:, -top_k:]

        # Log decisions
        if self.logging_enabled:
            if token_indices is None:
                token_indices = np.arange(batch_size)

            for i in range(batch_size):
                decision = RoutingDecision(
                    token_idx=int(token_indices[i]),
                    layer_idx=self.layer_idx,
                    logits=logits[i].copy(),
                    pressure_bias=scaled_pressure.copy(),  # Log scaled pressure
                    adjusted_logits=adjusted_logits[i].copy(),
                    gate_weights=gate_weights[i].copy(),
                    top_k_experts=top_k_experts[i].copy(),
                )
                self.log.add(decision)

        return gate_weights

    def forward_with_temperature(
        self,
        x: np.ndarray,
        temperatures: np.ndarray,
        pressure_scale: float = 1.0,
        temp_scale: float = 1.0,
        token_indices: Optional[np.ndarray] = None,
        top_k: int = 2,
    ) -> np.ndarray:
        """
        Forward pass with per-expert temperature warping.

        This creates a 2-field routing environment:
        - Pressure (b_k): force field that pushes toward/away from experts
        - Temperature (T_k): permeability that controls how slippery each region is

        The combined formula:
            logits'_k = (logits_k + pressure_scale * b_k) / (temp_scale * T_k)
            probs = softmax(logits')

        High temperature = diffuse routing (exploratory)
        Low temperature = sharp routing (exploitative)

        Args:
            x: Input hidden states of shape (batch, input_dim).
            temperatures: Per-expert temperature vector of shape (n_experts,).
            pressure_scale: Multiplier for pressure bias (from MetaKnob).
            temp_scale: Global multiplier for temperatures (κ > 0 → higher temps).
            token_indices: Optional token indices for logging.
            top_k: Number of experts to select.

        Returns:
            Gate weights of shape (batch, n_experts).
        """
        batch_size = x.shape[0]

        # Validate temperatures
        if temperatures.shape != (self.n_experts,):
            raise ValueError(
                f"Temperature shape {temperatures.shape} doesn't match n_experts {self.n_experts}"
            )

        # Compute raw logits
        logits = self.compute_logits(x)

        # Get pressure bias and apply scale
        pressure = self.get_pressure()
        scaled_pressure = pressure * pressure_scale

        # Inject pressure
        adjusted_logits = logits + scaled_pressure[np.newaxis, :]

        # Apply per-expert temperature scaling
        # T_k modulated by global temp_scale from knob
        effective_temps = temperatures * temp_scale

        # Clamp to prevent division issues
        effective_temps = np.clip(effective_temps, 0.1, 10.0)

        # Temperature-warped routing: divide each expert's logits by its temperature
        # logits'_k = adjusted_logits_k / T_k
        warped_logits = adjusted_logits / effective_temps[np.newaxis, :]

        # Softmax (no additional temperature - already applied per-expert)
        exp_logits = np.exp(warped_logits - warped_logits.max(axis=1, keepdims=True))
        gate_weights = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # Get top-k experts
        top_k_experts = np.argsort(gate_weights, axis=1)[:, -top_k:]

        # Log decisions
        if self.logging_enabled:
            if token_indices is None:
                token_indices = np.arange(batch_size)

            for i in range(batch_size):
                decision = RoutingDecision(
                    token_idx=int(token_indices[i]),
                    layer_idx=self.layer_idx,
                    logits=logits[i].copy(),
                    pressure_bias=scaled_pressure.copy(),
                    adjusted_logits=warped_logits[i].copy(),  # Log warped logits
                    gate_weights=gate_weights[i].copy(),
                    top_k_experts=top_k_experts[i].copy(),
                )
                self.log.add(decision)

        return gate_weights
