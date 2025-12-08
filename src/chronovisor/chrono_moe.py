"""
ChronoMoE V1: Wiring Chronovisor pressure into MoE routing.

This module bridges the Chronovisor "brain" (slow global pressures) with
MoE routing "body" (fast local decisions), making decision tree changes
measurable and reversible.

Core idea:
    MoE router makes a local decision; Chronovisor adds a slow, global
    "pressure" term that bends those decisions over time.

Key components:
    - ChronoMoEController: Adapter between Chronovisor and MoE
    - RoutingLogger: Tracks routing decisions for analysis
    - MockMoERouter: Simple router for testing without real MoE
    - Metrics: KL divergence, entropy, transition matrices
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json

from chronovisor.simulation_v6 import (
    CulturalExpert,
    CulturalEvolutionaryController,
    CulturalController,
    Governor,
    Lens,
    compute_kuramoto_R_and_psi,
    reset_id_counters,
    _next_expert_id,
    sigmoid,
)


# =============================================================================
# ChronoMoE Controller: The Bridge
# =============================================================================

@dataclass
class ChronoMoEController:
    """
    Adapter that bridges Chronovisor dynamics to MoE routing.

    Responsibilities:
    1. Maintain Chronovisor experts corresponding to MoE experts
    2. Translate MoE routing stats into Chronovisor alignment signals
    3. Compute pressure biases (T, P, C) for router injection
    4. Run Chronovisor ticks at appropriate intervals

    The pressure bias formula:
        b_k = α_T · T_k + α_P · P_k + α_C · C_k

    Where:
        T_k = log(σ(β_s · s_k))  (trust from reliability)
        P_k = g_lens_k - 1       (lens pressure, zero-centered)
        C_k = cultural_capital_k (cultural bonus)
    """

    num_experts: int

    # Chronovisor internals
    controller: CulturalEvolutionaryController = field(init=False)

    # Pressure coefficients
    alpha_T: float = 0.3   # Trust weight
    alpha_P: float = 0.1   # Lens pressure weight
    alpha_C: float = 0.05  # Cultural weight

    # Signal translation parameters
    beta_s: float = 3.0    # Sigmoid steepness for trust
    baseline_loss: float = 1.0  # For normalizing loss

    # State tracking
    batch_count: int = 0
    pressure_history: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        """Initialize Chronovisor controller with matching expert count."""
        reset_id_counters()

        # Create Chronovisor experts matching MoE expert count
        experts = [
            CulturalExpert(
                name=f"MoE_{k}",
                expert_id=_next_expert_id(),
                phi=random.uniform(0, 2 * math.pi),
                omega=0.1 + 0.02 * k,  # Slight frequency variation
                theta=0.0,
                theta_home=0.0,
                v=0.0,
                lambd=0.05,
                generation=0,
                birth_tick=0,
            )
            for k in range(self.num_experts)
        ]

        # Create the Chronovisor controller
        self.controller = CulturalEvolutionaryController(
            experts=experts,
            governor=Governor(max_population=self.num_experts * 2, min_population=self.num_experts),
            cultural=CulturalController(),
            lens=Lens(),
            micro_period=5,
            macro_period=4,
            cultural_period=3,
        )

    def update_from_routing(
        self,
        expert_usage: List[float],
        batch_loss: float,
        per_expert_loss: Optional[List[float]] = None,
    ) -> Dict:
        """
        Translate MoE routing stats into Chronovisor alignment signals.

        This is the key translation: MoE performance -> Chronovisor dynamics.

        Args:
            expert_usage: Count or fraction of tokens routed to each expert
            batch_loss: Overall batch loss (lower is better)
            per_expert_loss: Optional per-expert loss (if available)

        Returns:
            Dict with update diagnostics
        """
        self.batch_count += 1

        # Normalize usage to get routing distribution
        total_usage = sum(expert_usage) + 1e-8
        usage_dist = [u / total_usage for u in expert_usage]

        # Compute alignment signal for each expert
        # a_k ∝ usage_k × (−normalized_loss)
        # Higher usage + lower loss = better alignment

        if per_expert_loss is not None:
            # Use per-expert loss if available
            for k, expert in enumerate(self.controller.experts):
                # Invert and normalize loss (higher is better)
                quality = -per_expert_loss[k] / (self.baseline_loss + 1e-8)
                # Alignment = usage × quality
                alignment = usage_dist[k] * (1.0 + quality)
                # Inject into expert's alignment tracking
                expert.macro_align_sum += alignment
                expert.macro_align_count += 1
        else:
            # Use global loss for all experts, weighted by usage
            quality = -batch_loss / (self.baseline_loss + 1e-8)
            for k, expert in enumerate(self.controller.experts):
                alignment = usage_dist[k] * (1.0 + quality)
                expert.macro_align_sum += alignment
                expert.macro_align_count += 1

        return {
            "batch": self.batch_count,
            "usage_dist": usage_dist,
            "batch_loss": batch_loss,
        }

    def step(self, n_ticks: int = 20) -> Dict:
        """
        Run Chronovisor ticks and update internal state.

        Call this after update_from_routing() to let Chronovisor process
        the accumulated signals.

        Args:
            n_ticks: Number of fast ticks to run (default 20 = 1 macro tick)

        Returns:
            Dict with Chronovisor state after ticks
        """
        last_info = {}
        for _ in range(n_ticks):
            last_info = self.controller.tick()

        return {
            "R": last_info.get("R", 0.0),
            "lens_L": last_info.get("lens_L", 0.0),
            "avg_g_lens": last_info.get("avg_g_lens", 1.0),
            "avg_delta_s": last_info.get("avg_delta_s", 0.0),
            "num_motifs": last_info.get("num_motifs", 0),
        }

    def get_pressure_biases(self) -> List[float]:
        """
        Compute pressure biases for MoE router injection.

        Returns b_k for each expert:
            b_k = α_T · T_k + α_P · P_k + α_C · C_k

        These should be added to router logits:
            logits'_k = logits_k + b_k
        """
        biases = []

        for expert in self.controller.experts:
            # T_k: Trust signal from reliability
            # T_k = log(σ(β·s)) ∈ (-∞, 0]
            w_k = sigmoid(self.beta_s * expert.s)
            T_k = math.log(w_k + 1e-8)

            # P_k: Lens pressure (zero-centered)
            # P_k = g_lens - 1 ∈ [-0.5, 0.5]
            P_k = expert.g_lens_ema - 1.0

            # C_k: Cultural bonus
            C_k = expert.cultural_capital + expert.diversity_bonus

            # Compute total bias
            b_k = self.alpha_T * T_k + self.alpha_P * P_k + self.alpha_C * C_k
            biases.append(b_k)

        # Record for analysis
        self.pressure_history.append({
            "batch": self.batch_count,
            "biases": biases.copy(),
            "T": [sigmoid(self.beta_s * e.s) for e in self.controller.experts],
            "P": [e.g_lens_ema - 1.0 for e in self.controller.experts],
            "C": [e.cultural_capital + e.diversity_bonus for e in self.controller.experts],
        })

        return biases

    def get_expert_states(self) -> List[Dict]:
        """Return detailed state for each Chronovisor expert."""
        return [
            {
                "id": k,
                "name": e.name,
                "s": e.s,
                "lambda": e.lambd,
                "theta": e.theta,
                "theta_home": e.theta_home,
                "g_lens_ema": e.g_lens_ema,
                "R_ema": e.R_ema,
                "delta_s_ema": e.delta_s_ema,
                "cultural_capital": e.cultural_capital,
                "diversity_bonus": e.diversity_bonus,
            }
            for k, e in enumerate(self.controller.experts)
        ]


# =============================================================================
# Routing Logger: Decision Tree Tracking
# =============================================================================

@dataclass
class RoutingLogger:
    """
    Logs MoE routing decisions for decision tree analysis.

    Tracks:
    - Per-token routing choices
    - Expert usage distributions over time
    - Transition probabilities between experts
    """

    num_experts: int

    # Raw logs
    routing_log: List[Dict] = field(default_factory=list)

    # Aggregated stats
    usage_over_time: List[List[float]] = field(default_factory=list)

    def log_routing(
        self,
        token_idx: int,
        layer_idx: int,
        logits: List[float],
        gates: List[float],
        chosen_expert: int,
        batch_idx: int = 0,
    ):
        """Log a single routing decision."""
        self.routing_log.append({
            "batch": batch_idx,
            "token": token_idx,
            "layer": layer_idx,
            "logits": logits.copy() if isinstance(logits, list) else list(logits),
            "gates": gates.copy() if isinstance(gates, list) else list(gates),
            "chosen": chosen_expert,
        })

    def log_batch_usage(self, usage: List[float], batch_idx: int):
        """Log expert usage for a batch."""
        self.usage_over_time.append({
            "batch": batch_idx,
            "usage": usage.copy(),
        })

    def get_usage_distribution(self, start_batch: int = 0, end_batch: int = -1) -> List[float]:
        """Compute average usage distribution over a range of batches."""
        if not self.usage_over_time:
            return [1.0 / self.num_experts] * self.num_experts

        if end_batch < 0:
            end_batch = len(self.usage_over_time)

        relevant = [u["usage"] for u in self.usage_over_time[start_batch:end_batch]]
        if not relevant:
            return [1.0 / self.num_experts] * self.num_experts

        avg = [0.0] * self.num_experts
        for usage in relevant:
            for k in range(self.num_experts):
                avg[k] += usage[k]

        total = sum(avg) + 1e-8
        return [a / total for a in avg]

    def get_transition_matrix(self, start_batch: int = 0, end_batch: int = -1) -> List[List[float]]:
        """
        Compute transition probabilities between consecutive routing decisions.

        Returns P[i][j] = P(expert_j at t+1 | expert_i at t)
        """
        # Filter relevant logs
        if end_batch < 0:
            relevant = [r for r in self.routing_log if r["batch"] >= start_batch]
        else:
            relevant = [r for r in self.routing_log
                       if start_batch <= r["batch"] < end_batch]

        if len(relevant) < 2:
            # Return uniform
            return [[1.0 / self.num_experts] * self.num_experts
                    for _ in range(self.num_experts)]

        # Count transitions
        counts = [[0.0] * self.num_experts for _ in range(self.num_experts)]

        for i in range(len(relevant) - 1):
            curr = relevant[i]["chosen"]
            next_ = relevant[i + 1]["chosen"]
            counts[curr][next_] += 1

        # Normalize rows
        matrix = []
        for row in counts:
            total = sum(row) + 1e-8
            matrix.append([c / total for c in row])

        return matrix

    def clear(self):
        """Clear all logs."""
        self.routing_log.clear()
        self.usage_over_time.clear()


# =============================================================================
# Mock MoE Router: For Testing
# =============================================================================

@dataclass
class MockMoERouter:
    """
    Simple mock MoE router for testing ChronoMoE without a real model.

    Simulates:
    - Token -> hidden state (random)
    - Router: logits -> softmax -> top-k selection
    - Pressure bias injection
    """

    num_experts: int
    top_k: int = 1
    temperature: float = 1.0

    # Internal state
    base_logits: List[float] = field(default_factory=list)

    def __post_init__(self):
        """Initialize with uniform base logits."""
        self.base_logits = [0.0] * self.num_experts

    def set_base_logits(self, logits: List[float]):
        """Set base logits (before pressure)."""
        self.base_logits = logits.copy()

    def route(
        self,
        pressure_biases: Optional[List[float]] = None,
        noise_std: float = 0.1,
    ) -> Tuple[List[float], List[float], int]:
        """
        Route a single token (simulated).

        Args:
            pressure_biases: b_k from ChronoMoE (or None for baseline)
            noise_std: Add noise to simulate token variation

        Returns:
            (logits, gates, chosen_expert)
        """
        # Start with base logits + noise
        logits = [
            self.base_logits[k] + random.gauss(0, noise_std)
            for k in range(self.num_experts)
        ]

        # Add pressure biases if provided
        if pressure_biases is not None:
            logits = [logits[k] + pressure_biases[k] for k in range(self.num_experts)]

        # Softmax
        max_logit = max(logits)
        exp_logits = [math.exp((l - max_logit) / self.temperature) for l in logits]
        sum_exp = sum(exp_logits)
        gates = [e / sum_exp for e in exp_logits]

        # Top-k selection (for now just top-1)
        chosen = max(range(self.num_experts), key=lambda k: gates[k])

        return logits, gates, chosen

    def route_batch(
        self,
        batch_size: int,
        pressure_biases: Optional[List[float]] = None,
        noise_std: float = 0.1,
    ) -> Tuple[List[int], List[float]]:
        """
        Route a batch of tokens.

        Returns:
            (chosen_experts, usage_counts)
        """
        usage = [0] * self.num_experts
        chosen_list = []

        for _ in range(batch_size):
            _, _, chosen = self.route(pressure_biases, noise_std)
            chosen_list.append(chosen)
            usage[chosen] += 1

        return chosen_list, usage


# =============================================================================
# Metrics: Decision Tree Analysis
# =============================================================================

def kl_divergence(p: List[float], q: List[float]) -> float:
    """
    Compute KL(p || q) = Σ p_i log(p_i / q_i)

    Args:
        p: "True" distribution
        q: "Approximate" distribution
    """
    kl = 0.0
    for pi, qi in zip(p, q):
        if pi > 1e-10:
            kl += pi * math.log(pi / (qi + 1e-10))
    return kl


def entropy(p: List[float]) -> float:
    """
    Compute entropy H(p) = -Σ p_i log(p_i)
    """
    h = 0.0
    for pi in p:
        if pi > 1e-10:
            h -= pi * math.log(pi)
    return h


def matrix_frobenius_distance(A: List[List[float]], B: List[List[float]]) -> float:
    """
    Compute Frobenius norm of (A - B): ||A - B||_F
    """
    total = 0.0
    for i in range(len(A)):
        for j in range(len(A[0])):
            diff = A[i][j] - B[i][j]
            total += diff * diff
    return math.sqrt(total)


def compute_routing_metrics(
    logger_before: RoutingLogger,
    logger_after: RoutingLogger,
) -> Dict:
    """
    Compare routing patterns before and after pressure.

    Returns metrics showing how decision trees changed.
    """
    # Usage distributions
    p_before = logger_before.get_usage_distribution()
    p_after = logger_after.get_usage_distribution()

    # Transition matrices
    trans_before = logger_before.get_transition_matrix()
    trans_after = logger_after.get_transition_matrix()

    return {
        "usage_before": p_before,
        "usage_after": p_after,
        "kl_before_after": kl_divergence(p_before, p_after),
        "kl_after_before": kl_divergence(p_after, p_before),
        "entropy_before": entropy(p_before),
        "entropy_after": entropy(p_after),
        "entropy_change": entropy(p_after) - entropy(p_before),
        "transition_distance": matrix_frobenius_distance(trans_before, trans_after),
    }


# =============================================================================
# Experiment Runner
# =============================================================================

def run_chrono_moe_experiment(
    num_experts: int = 8,
    num_batches: int = 50,
    batch_size: int = 64,
    seed: int = 42,
    alpha_T: float = 0.3,
    alpha_P: float = 0.1,
    alpha_C: float = 0.05,
) -> Dict:
    """
    Run a baseline vs pressure experiment.

    1. Run baseline (no pressure) and log routing
    2. Run with ChronoMoE pressure and log routing
    3. Compare decision trees

    Returns metrics and logs for analysis.
    """
    random.seed(seed)

    # === Phase 1: Baseline (no pressure) ===
    router_baseline = MockMoERouter(num_experts=num_experts)
    logger_baseline = RoutingLogger(num_experts=num_experts)

    # Create some expert "specialization" via base logits
    # Expert 0-3 prefer "early" tokens, 4-7 prefer "late" tokens
    base_logits = [0.0] * num_experts

    for batch_idx in range(num_batches):
        # Simulate token variation: early vs late
        for token_idx in range(batch_size):
            # Shift logits based on token position
            token_bias = (token_idx / batch_size - 0.5) * 2  # [-1, 1]

            # Early experts (0-3) prefer negative bias, late (4-7) prefer positive
            temp_logits = base_logits.copy()
            for k in range(num_experts):
                if k < num_experts // 2:
                    temp_logits[k] += -token_bias * 0.5
                else:
                    temp_logits[k] += token_bias * 0.5

            router_baseline.set_base_logits(temp_logits)
            logits, gates, chosen = router_baseline.route(pressure_biases=None)

            logger_baseline.log_routing(
                token_idx=token_idx,
                layer_idx=0,
                logits=logits,
                gates=gates,
                chosen_expert=chosen,
                batch_idx=batch_idx,
            )

        # Log batch usage
        _, usage = router_baseline.route_batch(batch_size, pressure_biases=None)
        logger_baseline.log_batch_usage(usage, batch_idx)

    # === Phase 2: With ChronoMoE Pressure ===
    random.seed(seed)  # Reset for fair comparison

    chrono = ChronoMoEController(
        num_experts=num_experts,
        alpha_T=alpha_T,
        alpha_P=alpha_P,
        alpha_C=alpha_C,
    )
    router_pressure = MockMoERouter(num_experts=num_experts)
    logger_pressure = RoutingLogger(num_experts=num_experts)

    for batch_idx in range(num_batches):
        batch_usage = [0] * num_experts

        for token_idx in range(batch_size):
            token_bias = (token_idx / batch_size - 0.5) * 2

            temp_logits = base_logits.copy()
            for k in range(num_experts):
                if k < num_experts // 2:
                    temp_logits[k] += -token_bias * 0.5
                else:
                    temp_logits[k] += token_bias * 0.5

            router_pressure.set_base_logits(temp_logits)

            # Get pressure biases from ChronoMoE
            biases = chrono.get_pressure_biases()

            logits, gates, chosen = router_pressure.route(pressure_biases=biases)
            batch_usage[chosen] += 1

            logger_pressure.log_routing(
                token_idx=token_idx,
                layer_idx=0,
                logits=logits,
                gates=gates,
                chosen_expert=chosen,
                batch_idx=batch_idx,
            )

        logger_pressure.log_batch_usage(batch_usage, batch_idx)

        # Update ChronoMoE with routing stats
        # Simulate loss: experts with more balanced usage = lower loss
        usage_entropy = entropy([u / sum(batch_usage) for u in batch_usage])
        batch_loss = 1.0 - usage_entropy / math.log(num_experts)  # Lower entropy = higher loss

        chrono.update_from_routing(batch_usage, batch_loss)
        chrono.step(n_ticks=20)  # One macro tick

    # === Phase 3: Compare ===
    metrics = compute_routing_metrics(logger_baseline, logger_pressure)

    # Add ChronoMoE final state
    metrics["chrono_final_state"] = chrono.get_expert_states()
    metrics["chrono_R"] = chrono.controller.lens.L

    return {
        "metrics": metrics,
        "baseline_usage_history": [u["usage"] for u in logger_baseline.usage_over_time],
        "pressure_usage_history": [u["usage"] for u in logger_pressure.usage_over_time],
        "pressure_bias_history": chrono.pressure_history,
    }


def print_experiment_results(results: Dict):
    """Pretty-print experiment results."""
    metrics = results["metrics"]

    print("=" * 80)
    print("ChronoMoE V1 Experiment Results")
    print("=" * 80)

    print("\n--- Usage Distribution ---")
    print(f"Before: {[f'{x:.3f}' for x in metrics['usage_before']]}")
    print(f"After:  {[f'{x:.3f}' for x in metrics['usage_after']]}")

    print("\n--- Entropy (Higher = More Distributed) ---")
    print(f"Before: {metrics['entropy_before']:.4f}")
    print(f"After:  {metrics['entropy_after']:.4f}")
    print(f"Change: {metrics['entropy_change']:+.4f}")

    print("\n--- KL Divergence (Measures Distribution Shift) ---")
    print(f"KL(before || after): {metrics['kl_before_after']:.4f}")
    print(f"KL(after || before): {metrics['kl_after_before']:.4f}")

    print("\n--- Transition Matrix Distance ---")
    print(f"||P_before - P_after||_F: {metrics['transition_distance']:.4f}")

    print("\n--- ChronoMoE Expert States ---")
    for state in metrics["chrono_final_state"][:4]:  # Show first 4
        print(f"  {state['name']}: s={state['s']:.3f}, g_lens={state['g_lens_ema']:.3f}, "
              f"δs={state['delta_s_ema']:.3f}")
    print("  ...")

    print("=" * 80)


if __name__ == "__main__":
    results = run_chrono_moe_experiment(
        num_experts=8,
        num_batches=50,
        batch_size=64,
        seed=42,
    )
    print_experiment_results(results)
