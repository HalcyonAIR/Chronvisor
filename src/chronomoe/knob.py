"""
LLM-Controlled Meta-Knob for ChronoMoE.

The meta-knob κ ∈ [-1, +1] is a single scalar that the outer LLM controller
can nudge to modulate the entire pressure/alignment system.

Core principle:
    κ > 0 → more pressure, more exploration, faster alignment learning
    κ < 0 → less pressure, more exploitation, slower/frozen alignment
    κ = 0 → baseline behavior (no modulation)

The knob converts to three multiplicative factors:
    pressure_scale   = exp(β_p × κ)  # Scales pressure bias magnitude
    explore_bias     = exp(β_e × κ)  # Softens/sharpens routing distribution
    alignment_lr_mul = exp(β_A × κ)  # Modulates alignment learning rate

This creates a "learned thermostat" over the pressure system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class KnobFactors:
    """
    Factors derived from the meta-knob κ.

    All factors are multiplicative and centered at 1.0 when κ = 0.
    """

    kappa: float  # Raw κ value

    pressure_scale: float  # Multiplier for pressure bias
    explore_bias: float  # Softmax temperature multiplier (deprecated, use temp_scale)
    alignment_lr_mul: float  # Alignment learning rate multiplier
    temp_scale: float = 1.0  # Per-expert temperature field multiplier

    # Derived intent (optional, from LLM)
    intent: str = ""


@dataclass
class KnobState:
    """
    Compact state summary for LLM decision-making.

    This is what the outer LLM sees before choosing κ.
    """

    loss: float
    loss_trend: float  # Recent change in loss
    routing_entropy: float
    alignment_entropy: float
    drift_correlation: float
    coherence_R: float
    population: int

    # Optional extended state
    specialization: Optional[float] = None
    avg_pressure_magnitude: Optional[float] = None
    mode: Optional[str] = None  # e.g., "normal", "outside_box"

    def to_dict(self) -> dict:
        """Convert to dictionary for LLM consumption."""
        d = {
            "loss": round(self.loss, 4),
            "loss_trend": round(self.loss_trend, 4),
            "routing_entropy": round(self.routing_entropy, 4),
            "alignment_entropy": round(self.alignment_entropy, 4),
            "drift_correlation": round(self.drift_correlation, 4),
            "coherence_R": round(self.coherence_R, 4),
            "population": self.population,
        }
        if self.specialization is not None:
            d["specialization"] = round(self.specialization, 4)
        if self.avg_pressure_magnitude is not None:
            d["avg_pressure_magnitude"] = round(self.avg_pressure_magnitude, 4)
        if self.mode is not None:
            d["mode"] = self.mode
        return d

    def to_prompt(self) -> str:
        """Convert to natural language prompt for LLM."""
        lines = [
            f"Current loss: {self.loss:.4f} (trend: {self.loss_trend:+.4f})",
            f"Routing entropy: {self.routing_entropy:.4f}",
            f"Alignment entropy: {self.alignment_entropy:.4f}",
            f"Drift correlation: {self.drift_correlation:.4f}",
            f"Coherence R: {self.coherence_R:.4f}",
            f"Population: {self.population}",
        ]
        if self.specialization is not None:
            lines.append(f"Specialization: {self.specialization:.4f}")
        if self.mode is not None:
            lines.append(f"Mode: {self.mode}")
        return "\n".join(lines)


@dataclass
class KnobDecision:
    """
    Decision from the LLM (or rule-based controller).
    """

    kappa: float  # The meta-knob value
    intent: str = ""  # Optional intent description
    reasoning: str = ""  # Optional reasoning


@dataclass
class MetaKnob:
    """
    LLM-controlled meta-knob for ChronoMoE pressure modulation.

    The knob κ ∈ [-1, +1] modulates:
    - Pressure strength (how much the router is influenced)
    - Exploration bias (how soft/sharp the routing distribution is)
    - Alignment learning rate (how fast the alignment matrix adapts)

    This is the interface between an outer LLM controller and the
    ChronoMoE pressure system.
    """

    # Sensitivity parameters (how much κ affects each factor)
    beta_pressure: float = 0.5  # exp(0.5 × ±1) ≈ 0.6 to 1.6
    beta_explore: float = 0.3  # exp(0.3 × ±1) ≈ 0.7 to 1.3 (deprecated)
    beta_alignment: float = 0.7  # exp(0.7 × ±1) ≈ 0.5 to 2.0
    beta_temperature: float = 0.4  # exp(0.4 × ±1) ≈ 0.67 to 1.5

    # Bounds for κ
    kappa_min: float = -1.0
    kappa_max: float = 1.0

    # Current state
    current_kappa: float = 0.0
    current_intent: str = ""

    # History for analysis
    kappa_history: list = field(default_factory=list)
    decision_history: list = field(default_factory=list)

    # Smoothing (optional EMA for gradual transitions)
    use_smoothing: bool = False
    eta_smooth: float = 0.3  # EMA rate for κ transitions

    def set_kappa(self, kappa: float, intent: str = "") -> KnobFactors:
        """
        Set the meta-knob value.

        Args:
            kappa: Value in [-1, +1].
            intent: Optional intent description.

        Returns:
            KnobFactors derived from κ.
        """
        # Clamp to bounds
        kappa = max(self.kappa_min, min(self.kappa_max, kappa))

        # Optional smoothing
        if self.use_smoothing:
            kappa = (1 - self.eta_smooth) * self.current_kappa + self.eta_smooth * kappa

        self.current_kappa = kappa
        self.current_intent = intent
        self.kappa_history.append(kappa)

        return self.get_factors()

    def get_factors(self) -> KnobFactors:
        """
        Get current knob factors.

        Returns:
            KnobFactors with all multiplicative factors.
        """
        kappa = self.current_kappa

        return KnobFactors(
            kappa=kappa,
            pressure_scale=math.exp(self.beta_pressure * kappa),
            explore_bias=math.exp(self.beta_explore * kappa),
            alignment_lr_mul=math.exp(self.beta_alignment * kappa),
            temp_scale=math.exp(self.beta_temperature * kappa),
            intent=self.current_intent,
        )

    def apply_decision(self, decision: KnobDecision) -> KnobFactors:
        """
        Apply a knob decision from the LLM.

        Args:
            decision: KnobDecision with κ and intent.

        Returns:
            KnobFactors derived from the decision.
        """
        self.decision_history.append(decision)
        return self.set_kappa(decision.kappa, decision.intent)

    def reset(self) -> None:
        """Reset knob to neutral state."""
        self.current_kappa = 0.0
        self.current_intent = ""
        self.kappa_history.clear()
        self.decision_history.clear()

    def get_diagnostics(self) -> dict:
        """Get diagnostic information about knob state."""
        factors = self.get_factors()
        return {
            "kappa": self.current_kappa,
            "intent": self.current_intent,
            "pressure_scale": factors.pressure_scale,
            "explore_bias": factors.explore_bias,
            "alignment_lr_mul": factors.alignment_lr_mul,
            "temp_scale": factors.temp_scale,
            "history_len": len(self.kappa_history),
            "kappa_mean": np.mean(self.kappa_history) if self.kappa_history else 0.0,
            "kappa_std": np.std(self.kappa_history) if self.kappa_history else 0.0,
        }


class RuleBasedKnobController:
    """
    Simple rule-based controller for the meta-knob.

    This can be used as a baseline or when an LLM is not available.
    Implements heuristic rules based on system state.
    """

    def __init__(
        self,
        loss_target: float = 0.5,
        entropy_target: float = 1.5,
        explore_when_stuck: bool = True,
        stabilize_when_good: bool = True,
    ):
        """
        Initialize rule-based controller.

        Args:
            loss_target: Target loss value.
            entropy_target: Target routing entropy.
            explore_when_stuck: Increase κ when loss is flat/increasing.
            stabilize_when_good: Decrease κ when loss is decreasing.
        """
        self.loss_target = loss_target
        self.entropy_target = entropy_target
        self.explore_when_stuck = explore_when_stuck
        self.stabilize_when_good = stabilize_when_good

        self.prev_loss = None
        self.stuck_count = 0

    def decide(self, state: KnobState) -> KnobDecision:
        """
        Make a knob decision based on current state.

        Args:
            state: Current system state.

        Returns:
            KnobDecision with κ and intent.
        """
        kappa = 0.0
        intent = "neutral"
        reasoning = ""

        # Track loss trend
        if self.prev_loss is not None:
            loss_delta = state.loss - self.prev_loss
        else:
            loss_delta = 0.0
        self.prev_loss = state.loss

        # Rule 1: Explore when stuck (loss not improving)
        if self.explore_when_stuck:
            if loss_delta >= 0:
                self.stuck_count += 1
            else:
                self.stuck_count = max(0, self.stuck_count - 1)

            if self.stuck_count > 3:
                kappa += 0.3
                intent = "explore_stuck"
                reasoning = f"Loss stuck for {self.stuck_count} steps"

        # Rule 2: Stabilize when improving
        if self.stabilize_when_good:
            if loss_delta < -0.01:
                kappa -= 0.2
                intent = "stabilize_improving"
                reasoning = f"Loss improving by {-loss_delta:.4f}"

        # Rule 3: Entropy balance
        entropy_delta = state.routing_entropy - self.entropy_target
        if abs(entropy_delta) > 0.5:
            # Entropy too far from target
            if entropy_delta > 0:
                # Too much entropy → exploit more
                kappa -= 0.1
                if intent == "neutral":
                    intent = "reduce_entropy"
            else:
                # Too little entropy → explore more
                kappa += 0.1
                if intent == "neutral":
                    intent = "increase_entropy"

        # Rule 4: Low coherence → explore
        if state.coherence_R < 0.3:
            kappa += 0.2
            if intent == "neutral":
                intent = "explore_low_coherence"

        # Rule 5: High drift correlation + high alignment entropy → stabilize
        if state.drift_correlation > 0.9 and state.alignment_entropy > 1.5:
            kappa -= 0.1
            if intent == "neutral":
                intent = "stabilize_aligned"

        # Clamp final κ
        kappa = max(-1.0, min(1.0, kappa))

        return KnobDecision(
            kappa=kappa,
            intent=intent,
            reasoning=reasoning,
        )


class LLMKnobInterface:
    """
    Interface for LLM-based knob control.

    This class handles the formatting of state → prompt and
    parsing of LLM response → KnobDecision.
    """

    SYSTEM_PROMPT = """You are a meta-controller for a neural network routing system.

You control a single scalar κ ∈ [-1, +1] that modulates:
- κ > 0: More pressure (stronger routing bias), more exploration, faster adaptation
- κ < 0: Less pressure (weaker routing bias), more exploitation, slower/stable adaptation
- κ = 0: Neutral, baseline behavior

Based on the current system state, output a JSON response:
{"kappa": <float>, "intent": "<short description>"}

Guidelines:
- If loss is stuck or increasing, try κ > 0 to explore
- If loss is improving, try κ < 0 to stabilize
- If routing entropy is too low, try κ > 0 to diversify
- If routing entropy is too high, try κ < 0 to focus
- Keep |κ| small (0.1-0.3) for gentle adjustments
- Only use |κ| > 0.5 for significant interventions"""

    def format_state(self, state: KnobState) -> str:
        """Format state for LLM prompt."""
        return f"""Current system state:
{state.to_prompt()}

What should κ be? Respond with JSON: {{"kappa": <float>, "intent": "<description>"}}"""

    def parse_response(self, response: str) -> KnobDecision:
        """
        Parse LLM response to KnobDecision.

        Args:
            response: Raw LLM response text.

        Returns:
            KnobDecision parsed from response.
        """
        import json
        import re

        # Try to extract JSON from response
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                kappa = float(data.get("kappa", 0.0))
                intent = str(data.get("intent", ""))
                return KnobDecision(kappa=kappa, intent=intent, reasoning=response)
            except (json.JSONDecodeError, ValueError):
                pass

        # Try to extract just the number after "kappa" or κ
        kappa_match = re.search(r'[κkappa:=\s]+([+-]?\d*\.?\d+)', response, re.IGNORECASE)
        if kappa_match:
            try:
                kappa = float(kappa_match.group(1))
                return KnobDecision(kappa=kappa, intent="parsed", reasoning=response)
            except ValueError:
                pass

        # Default to neutral
        return KnobDecision(kappa=0.0, intent="parse_failed", reasoning=response)

    def create_prompt(self, state: KnobState) -> tuple[str, str]:
        """
        Create full prompt for LLM.

        Returns:
            (system_prompt, user_prompt)
        """
        return self.SYSTEM_PROMPT, self.format_state(state)
