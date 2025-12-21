"""
Observer: Reads ChronoMoE state, no writes.

Provides read-only access to ChronoMoE internal state for telemetry.
Supports both real ChronoMoE models and proxy/synthetic data.
"""

from typing import Optional, Dict, Any
import numpy as np


class ChronoMoEObserver:
    """
    Read-only observer for ChronoMoE state.

    Extracts routing statistics, expert usage, and dynamics
    without modifying model state.
    """

    def __init__(self):
        """Initialize observer."""
        pass

    def get_expert_usage(self, chrono_state, layer_idx: int = 0) -> Optional[np.ndarray]:
        """
        Get expert usage from chronovisor state.

        Args:
            chrono_state: ChronovisorState from model
            layer_idx: Which layer to observe (default: 0)

        Returns:
            Expert usage array, or None if unavailable
        """
        if chrono_state is None or not hasattr(chrono_state, 'expert_usage'):
            return None

        expert_usage = chrono_state.expert_usage
        if not expert_usage or layer_idx not in expert_usage:
            return None

        return expert_usage[layer_idx]

    def get_routing_weights(self, chrono_state, layer_idx: int = 0) -> Optional[np.ndarray]:
        """
        Get routing weights (probability distribution over experts).

        Args:
            chrono_state: ChronovisorState from model
            layer_idx: Which layer to observe

        Returns:
            Normalized routing weights, or None if unavailable
        """
        usage = self.get_expert_usage(chrono_state, layer_idx)
        if usage is None:
            return None

        # Normalize to probability distribution
        normalized = usage / (usage.sum() + 1e-10)
        return normalized

    def get_coherence(self, chrono_state) -> Optional[float]:
        """Get current coherence value."""
        if chrono_state is None or not hasattr(chrono_state, 'coherence'):
            return None
        return float(chrono_state.coherence)

    def get_delta_coherence(self, chrono_state) -> Optional[float]:
        """Get delta coherence (rate of change)."""
        if chrono_state is None or not hasattr(chrono_state, 'delta_coherence'):
            return None
        return float(chrono_state.delta_coherence)


class ProxyObserver:
    """
    Proxy observer for testing without real ChronoMoE.

    Allows telemetry system to be tested with synthetic data,
    completely decoupled from actual model implementation.
    """

    def __init__(
        self,
        num_experts: int = 64,
        num_layers: int = 2,
    ):
        """
        Args:
            num_experts: Number of experts to simulate
            num_layers: Number of layers to simulate
        """
        self.num_experts = num_experts
        self.num_layers = num_layers

        # Synthetic state
        self.expert_usage: Dict[int, np.ndarray] = {}
        self.coherence: float = 0.5
        self.delta_coherence: float = 0.0

        # Initialize with random routing
        self._reset_to_random()

    def _reset_to_random(self):
        """Reset to random routing state."""
        for layer_idx in range(self.num_layers):
            # Random expert usage
            usage = np.random.dirichlet(np.ones(self.num_experts))
            self.expert_usage[layer_idx] = usage

    def set_expert_usage(self, layer_idx: int, usage: np.ndarray):
        """Manually set expert usage for testing."""
        self.expert_usage[layer_idx] = usage

    def set_coherence(self, coherence: float, delta: float = 0.0):
        """Manually set coherence for testing."""
        self.coherence = coherence
        self.delta_coherence = delta

    def get_expert_usage(self, layer_idx: int = 0) -> Optional[np.ndarray]:
        """Get expert usage (proxy implementation)."""
        return self.expert_usage.get(layer_idx, None)

    def get_routing_weights(self, layer_idx: int = 0) -> Optional[np.ndarray]:
        """Get normalized routing weights (proxy implementation)."""
        usage = self.get_expert_usage(layer_idx)
        if usage is None:
            return None

        normalized = usage / (usage.sum() + 1e-10)
        return normalized

    def get_coherence(self) -> Optional[float]:
        """Get coherence (proxy implementation)."""
        return self.coherence

    def get_delta_coherence(self) -> Optional[float]:
        """Get delta coherence (proxy implementation)."""
        return self.delta_coherence

    def step(self, noise_scale: float = 0.01):
        """
        Simulate one step of dynamics for testing.

        Args:
            noise_scale: Amount of random perturbation to add
        """
        # Add small random perturbation to routing
        for layer_idx in self.expert_usage:
            noise = np.random.randn(self.num_experts) * noise_scale
            perturbed = self.expert_usage[layer_idx] + noise
            perturbed = np.maximum(perturbed, 0)  # Keep non-negative
            self.expert_usage[layer_idx] = perturbed / (perturbed.sum() + 1e-10)

        # Drift coherence slightly
        self.delta_coherence = np.random.randn() * 0.01
        self.coherence = np.clip(self.coherence + self.delta_coherence, 0, 1)
