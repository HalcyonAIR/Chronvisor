"""
ChronoMoE Training Infrastructure

Training system designed around the P×T geometric control layer.

Key innovations:
1. Coherence-Gated Training: Kuramoto R modulates gradient flow
2. Geological Curriculum: Let structural T̄ landscape form naturally
3. Meta-Knob Scheduling: κ anneals from exploration to exploitation
4. Valley-Aware Loss: Penalize bad valleys, reward healthy ones
5. Chronovisor Checkpointing: Preserve geological memory across runs

The training loop is designed to let the geometry emerge organically
rather than forcing it through explicit regularization.
"""

from __future__ import annotations

import math
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from chronomoe.chronovisor_mixtral_bridge import (
    ChronovisorMixtralForCausalLM,
    ChronovisorMixtralController,
    ChronovisorMixtralState,
)
from chronomoe.mixtral_core import MixtralConfig
from chronomoe.knob import MetaKnob, KnobState, RuleBasedKnobController


# =============================================================================
# Loss Functions
# =============================================================================

@dataclass
class ChronoMoELoss:
    """
    Combined loss for ChronoMoE training.

    Components:
        1. Language modeling loss (cross-entropy)
        2. Load balancing loss (encourage even expert usage)
        3. Coherence regularization (reward high Kuramoto R)
        4. Valley health penalty (discourage bad valleys)

    The total loss is:
        L = L_lm + λ_balance * L_balance + λ_coherence * L_coherence + λ_valley * L_valley
    """

    # Component weights
    lambda_balance: float = 0.01  # Load balancing weight
    lambda_coherence: float = 0.001  # Coherence regularization
    lambda_valley: float = 0.0001  # Valley health penalty

    # Tracking
    history: List[Dict[str, float]] = field(default_factory=list)

    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        chrono_state: Optional[ChronovisorMixtralState] = None,
        controller: Optional[ChronovisorMixtralController] = None,
        ignore_index: int = -100,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the combined ChronoMoE loss.

        Args:
            logits: Model output (batch, seq_len, vocab_size)
            labels: Target tokens (batch, seq_len)
            chrono_state: Optional Chronovisor state for regularization
            controller: Optional controller for valley health
            ignore_index: Token ID to ignore in loss computation

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        batch_size, seq_len, vocab_size = logits.shape

        # 1. Language modeling loss
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        lm_loss = F.cross_entropy(
            logits_flat, labels_flat,
            ignore_index=ignore_index,
            reduction='mean'
        )

        components = {"lm_loss": lm_loss.item()}
        total_loss = lm_loss

        # 2. Load balancing loss (if we have routing stats)
        if chrono_state is not None and chrono_state.expert_usage:
            balance_loss = self._compute_balance_loss(chrono_state)
            total_loss = total_loss + self.lambda_balance * balance_loss
            components["balance_loss"] = balance_loss.item()

        # 3. Coherence regularization (reward high R)
        if chrono_state is not None:
            coherence_loss = self._compute_coherence_loss(chrono_state)
            total_loss = total_loss + self.lambda_coherence * coherence_loss
            components["coherence_loss"] = coherence_loss.item()
            components["kuramoto_R"] = chrono_state.coherence

        # 4. Valley health penalty
        if controller is not None:
            valley_loss = self._compute_valley_loss(controller)
            total_loss = total_loss + self.lambda_valley * valley_loss
            components["valley_loss"] = valley_loss.item()

        components["total_loss"] = total_loss.item()
        self.history.append(components)

        return total_loss, components

    def _compute_balance_loss(self, chrono_state: ChronovisorMixtralState) -> torch.Tensor:
        """
        Compute load balancing loss to encourage even expert usage.

        Uses the coefficient of variation (std/mean) of expert usage.
        CV = 0 means perfectly balanced, CV > 1 means highly imbalanced.
        """
        all_usage = []
        for layer_idx, usage in chrono_state.expert_usage.items():
            if isinstance(usage, np.ndarray):
                all_usage.append(torch.from_numpy(usage).float())
            else:
                all_usage.append(usage.float())

        if not all_usage:
            return torch.tensor(0.0)

        # Aggregate usage across layers
        total_usage = torch.stack(all_usage).sum(dim=0)

        # Coefficient of variation
        mean_usage = total_usage.mean()
        if mean_usage > 0:
            cv = total_usage.std() / mean_usage
        else:
            cv = torch.tensor(0.0)

        return cv

    def _compute_coherence_loss(self, chrono_state: ChronovisorMixtralState) -> torch.Tensor:
        """
        Compute coherence regularization loss.

        We want to encourage high Kuramoto R (coherent routing).
        Loss = 1 - R (so minimizing loss maximizes coherence).
        """
        R = chrono_state.coherence
        # Smooth penalty: higher weight when R is low
        return torch.tensor(1.0 - R)

    def _compute_valley_loss(self, controller: ChronovisorMixtralController) -> torch.Tensor:
        """
        Compute valley health penalty.

        Penalizes unhealthy valleys (low T̄ + low reliability).
        This encourages the self-correction mechanism to work.
        """
        diag = controller.get_valley_health_diagnostics()

        # Penalty proportional to number of unhealthy valleys
        num_unhealthy = len(diag.get("unhealthy_valleys", []))
        num_experts = controller.config.num_experts

        # Normalized penalty
        penalty = num_unhealthy / num_experts

        return torch.tensor(penalty)

    def get_summary(self, last_n: int = 100) -> Dict[str, float]:
        """Get summary statistics of recent losses."""
        if not self.history:
            return {}

        recent = self.history[-last_n:]
        summary = {}

        for key in recent[0].keys():
            values = [h[key] for h in recent if key in h]
            if values:
                summary[f"{key}_mean"] = np.mean(values)
                summary[f"{key}_std"] = np.std(values)

        return summary


# =============================================================================
# Meta-Knob Scheduling
# =============================================================================

class KnobScheduler:
    """
    Scheduler for meta-knob κ during training.

    Implements curriculum learning through the geometric control:
    - Start with high κ (exploration phase)
    - Anneal to low κ (exploitation phase)
    - Optionally react to training dynamics

    The idea is that early training should explore the expert space,
    while later training should exploit discovered specializations.
    """

    def __init__(
        self,
        initial_kappa: float = 0.5,
        final_kappa: float = -0.3,
        warmup_steps: int = 1000,
        total_steps: int = 10000,
        schedule: str = "cosine",  # "linear", "cosine", "step", "adaptive"
        adaptive_controller: Optional[RuleBasedKnobController] = None,
    ):
        """
        Initialize knob scheduler.

        Args:
            initial_kappa: Starting κ (usually positive for exploration)
            final_kappa: Ending κ (usually negative for exploitation)
            warmup_steps: Steps to hold at initial κ
            total_steps: Total training steps
            schedule: Scheduling strategy
            adaptive_controller: Optional rule-based controller for adaptive mode
        """
        self.initial_kappa = initial_kappa
        self.final_kappa = final_kappa
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.schedule = schedule
        self.adaptive_controller = adaptive_controller or RuleBasedKnobController()

        self.current_step = 0
        self.history: List[Tuple[int, float, str]] = []

    def step(self, knob_state: Optional[KnobState] = None) -> Tuple[float, str]:
        """
        Compute κ for current training step.

        Args:
            knob_state: Optional current system state (for adaptive mode)

        Returns:
            Tuple of (kappa, intent_string)
        """
        self.current_step += 1

        # Warmup: hold at initial κ
        if self.current_step <= self.warmup_steps:
            kappa = self.initial_kappa
            intent = "warmup_explore"

        # Adaptive: use rule-based controller
        elif self.schedule == "adaptive" and knob_state is not None:
            decision = self.adaptive_controller.decide(knob_state)
            kappa = decision.kappa
            intent = decision.intent

        # Scheduled annealing
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, progress))

            if self.schedule == "linear":
                kappa = self.initial_kappa + (self.final_kappa - self.initial_kappa) * progress
                intent = "linear_anneal"

            elif self.schedule == "cosine":
                # Cosine annealing (smoother)
                cosine_progress = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
                kappa = self.final_kappa + (self.initial_kappa - self.final_kappa) * cosine_progress
                intent = "cosine_anneal"

            elif self.schedule == "step":
                # Step schedule: switch at 1/3 and 2/3
                if progress < 0.33:
                    kappa = self.initial_kappa
                    intent = "step_explore"
                elif progress < 0.67:
                    kappa = (self.initial_kappa + self.final_kappa) / 2
                    intent = "step_transition"
                else:
                    kappa = self.final_kappa
                    intent = "step_exploit"

            else:
                kappa = self.initial_kappa
                intent = "unknown_schedule"

        self.history.append((self.current_step, kappa, intent))
        return kappa, intent

    def reset(self):
        """Reset scheduler to initial state."""
        self.current_step = 0
        self.history.clear()


# =============================================================================
# Coherence-Gated Optimizer
# =============================================================================

class CoherenceGatedOptimizer:
    """
    Optimizer wrapper that gates gradient updates based on Kuramoto coherence.

    When coherence R is high, gradients flow normally.
    When coherence R is low, gradients are scaled down.

    This implements the principle that we should trust updates more
    when the expert ensemble is coherent (aligned).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        min_coherence: float = 0.2,
        gating_strength: float = 0.5,
    ):
        """
        Initialize coherence-gated optimizer.

        Args:
            optimizer: Base optimizer (e.g., AdamW)
            min_coherence: Coherence below this fully gates gradients
            gating_strength: How strongly coherence affects gradient scale (0-1)
        """
        self.optimizer = optimizer
        self.min_coherence = min_coherence
        self.gating_strength = gating_strength

        self.last_coherence = 1.0
        self.last_gate_value = 1.0

    def step(self, coherence: float = 1.0):
        """
        Perform optimizer step with coherence gating.

        Args:
            coherence: Current Kuramoto R value (0-1)
        """
        self.last_coherence = coherence

        # Compute gate value
        if coherence < self.min_coherence:
            gate = 0.0
        else:
            # Linear interpolation from min_coherence to 1.0
            gate = (coherence - self.min_coherence) / (1.0 - self.min_coherence)

        # Blend with gating strength
        effective_gate = 1.0 - self.gating_strength * (1.0 - gate)
        self.last_gate_value = effective_gate

        # Scale gradients
        if effective_gate < 1.0:
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.data.mul_(effective_gate)

        # Perform optimizer step
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups


# =============================================================================
# Geological Checkpointing
# =============================================================================

@dataclass
class ChronovisorCheckpoint:
    """
    Checkpoint state for Chronovisor controller.

    Preserves the geological memory (structural temperatures) so training
    can resume with the evolved landscape intact.
    """

    # Global state
    structural_T_global: np.ndarray
    coherence_history: List[float]
    fast_clock: int
    micro_clock: int
    macro_clock: int

    # Per-layer lens states
    lens_states: Dict[int, Dict[str, np.ndarray]]

    # Meta-knob state
    meta_knob_kappa: float
    meta_knob_history: List[float]

    # Training metadata
    step: int
    epoch: int
    loss_history: List[Dict[str, float]]

    def save(self, path: Path) -> None:
        """Save checkpoint to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        data = {
            "structural_T_global": self.structural_T_global.tolist(),
            "coherence_history": self.coherence_history,
            "fast_clock": self.fast_clock,
            "micro_clock": self.micro_clock,
            "macro_clock": self.macro_clock,
            "lens_states": {
                str(k): {kk: vv.tolist() for kk, vv in v.items()}
                for k, v in self.lens_states.items()
            },
            "meta_knob_kappa": self.meta_knob_kappa,
            "meta_knob_history": self.meta_knob_history,
            "step": self.step,
            "epoch": self.epoch,
            "loss_history": self.loss_history[-1000:],  # Keep last 1000
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ChronovisorCheckpoint":
        """Load checkpoint from disk."""
        with open(path, 'r') as f:
            data = json.load(f)

        return cls(
            structural_T_global=np.array(data["structural_T_global"]),
            coherence_history=data["coherence_history"],
            fast_clock=data["fast_clock"],
            micro_clock=data["micro_clock"],
            macro_clock=data["macro_clock"],
            lens_states={
                int(k): {kk: np.array(vv) for kk, vv in v.items()}
                for k, v in data["lens_states"].items()
            },
            meta_knob_kappa=data["meta_knob_kappa"],
            meta_knob_history=data["meta_knob_history"],
            step=data["step"],
            epoch=data["epoch"],
            loss_history=data["loss_history"],
        )

    @classmethod
    def from_controller(
        cls,
        controller: ChronovisorMixtralController,
        step: int,
        epoch: int,
        loss_history: List[Dict[str, float]],
    ) -> "ChronovisorCheckpoint":
        """Create checkpoint from current controller state."""
        lens_states = {}
        for layer_idx, lens in controller.lenses.items():
            lens_states[layer_idx] = lens.get_state()

        meta_knob_kappa = 0.0
        meta_knob_history = []
        if controller.meta_knob is not None:
            meta_knob_kappa = controller.meta_knob.current_kappa
            meta_knob_history = controller.meta_knob.kappa_history.copy()

        return cls(
            structural_T_global=controller.structural_T_global.copy(),
            coherence_history=controller.coherence_history.copy(),
            fast_clock=controller.fast_clock,
            micro_clock=controller.micro_clock,
            macro_clock=controller.macro_clock,
            lens_states=lens_states,
            meta_knob_kappa=meta_knob_kappa,
            meta_knob_history=meta_knob_history,
            step=step,
            epoch=epoch,
            loss_history=loss_history,
        )

    def restore_to_controller(self, controller: ChronovisorMixtralController) -> None:
        """Restore checkpoint state to controller."""
        controller.structural_T_global = self.structural_T_global.copy()
        controller.coherence_history = self.coherence_history.copy()
        controller.fast_clock = self.fast_clock
        controller.micro_clock = self.micro_clock
        controller.macro_clock = self.macro_clock

        for layer_idx, state in self.lens_states.items():
            if layer_idx in controller.lenses:
                lens = controller.lenses[layer_idx]
                for key, value in state.items():
                    if hasattr(lens, key):
                        setattr(lens, key, value.copy())

        if controller.meta_knob is not None:
            controller.meta_knob.current_kappa = self.meta_knob_kappa
            controller.meta_knob.kappa_history = self.meta_knob_history.copy()


# =============================================================================
# Training Loop
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for ChronoMoE training."""

    # Basic training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 10
    max_steps: Optional[int] = None
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Loss weights
    lambda_balance: float = 0.01
    lambda_coherence: float = 0.001
    lambda_valley: float = 0.0001

    # Meta-knob scheduling
    initial_kappa: float = 0.5
    final_kappa: float = -0.3
    kappa_warmup_steps: int = 1000
    kappa_schedule: str = "cosine"

    # Coherence gating
    use_coherence_gating: bool = True
    min_coherence: float = 0.2
    gating_strength: float = 0.5

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 1000
    save_chronovisor_state: bool = True

    # Geological dynamics
    fast_geology: bool = False  # If True, speed up structural T evolution 10x for debugging

    # Logging
    log_every_n_steps: int = 100
    eval_every_n_steps: int = 500


class ChronoMoETrainer:
    """
    Trainer for ChronoMoE models.

    Integrates:
    - ChronoMoE loss with coherence regularization
    - Meta-knob scheduling for curriculum learning
    - Coherence-gated gradient updates
    - Geological checkpointing
    """

    def __init__(
        self,
        model: ChronovisorMixtralForCausalLM,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: ChronovisorMixtralForCausalLM model
            config: Training configuration
            train_dataloader: Training data loader
            eval_dataloader: Optional evaluation data loader
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Get controller reference
        self.controller = model.model.controller

        # Apply fast geology if requested (10x speedup for debugging)
        if config.fast_geology:
            self.controller.eta_structural_T_global = 0.05  # 10x faster (was 0.005)
            for lens in self.controller.lenses.values():
                lens.eta_structural_T = 0.1  # 10x faster (was 0.01)
            print("=" * 70)
            print("⚠️  FAST GEOLOGY MODE ENABLED (DEBUG ONLY)")
            print("   η_global = 0.05 (10x faster than production 0.005)")
            print("   η_local = 0.1 (10x faster than production 0.01)")
            print("   Valleys form in ~100 steps instead of ~1000")
            print("   DO NOT USE FOR PRODUCTION TRAINING OR PUBLISHED RESULTS")
            print("=" * 70)

        # Initialize loss function
        self.loss_fn = ChronoMoELoss(
            lambda_balance=config.lambda_balance,
            lambda_coherence=config.lambda_coherence,
            lambda_valley=config.lambda_valley,
        )

        # Initialize optimizer
        base_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        if config.use_coherence_gating:
            self.optimizer = CoherenceGatedOptimizer(
                base_optimizer,
                min_coherence=config.min_coherence,
                gating_strength=config.gating_strength,
            )
        else:
            self.optimizer = base_optimizer

        # Estimate total steps
        if config.max_steps is not None:
            total_steps = config.max_steps
        else:
            steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
            total_steps = steps_per_epoch * config.max_epochs

        # Initialize knob scheduler
        self.knob_scheduler = KnobScheduler(
            initial_kappa=config.initial_kappa,
            final_kappa=config.final_kappa,
            warmup_steps=config.kappa_warmup_steps,
            total_steps=total_steps,
            schedule=config.kappa_schedule,
        )

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> Dict[str, Any]:
        """
        Run training loop.

        Returns:
            Dictionary with training results and metrics
        """
        self.model.train()

        total_loss = 0.0
        accumulation_counter = 0

        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            epoch_start = time.time()

            for batch_idx, batch in enumerate(self.train_dataloader):
                # Check max steps
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    break

                # Forward pass
                loss, chrono_state = self._training_step(batch)

                # Accumulate loss
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                total_loss += loss.item()
                accumulation_counter += 1

                # Optimizer step
                if accumulation_counter >= self.config.gradient_accumulation_steps:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )

                    # Coherence-gated step
                    if isinstance(self.optimizer, CoherenceGatedOptimizer):
                        coherence = chrono_state.coherence if chrono_state else 1.0
                        self.optimizer.step(coherence)
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad()

                    # Update meta-knob
                    knob_state = self.controller.get_knob_state(loss=total_loss)
                    kappa, intent = self.knob_scheduler.step(knob_state)
                    self.controller.set_knob(kappa, intent)

                    self.global_step += 1
                    total_loss = 0.0
                    accumulation_counter = 0

                    # Logging
                    if self.global_step % self.config.log_every_n_steps == 0:
                        self._log_step(chrono_state, kappa, intent)

                    # Checkpointing
                    if self.global_step % self.config.save_every_n_steps == 0:
                        self._save_checkpoint()

                    # Evaluation
                    if (self.eval_dataloader is not None and
                        self.global_step % self.config.eval_every_n_steps == 0):
                        self._evaluate()

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")

            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break

        # Final checkpoint
        self._save_checkpoint(is_final=True)

        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "loss_summary": self.loss_fn.get_summary(),
            "knob_history": self.knob_scheduler.history,
        }

    def _training_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[ChronovisorMixtralState]]:
        """Execute single training step."""
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids.clone())
        attention_mask = batch.get("attention_mask", None)

        # Move to device
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Forward pass
        logits, chrono_state = self.model(
            input_ids,
            attention_mask=attention_mask,
            update_chronovisor=True,
        )

        # Compute loss
        loss, components = self.loss_fn.compute(
            logits,
            labels,
            chrono_state=chrono_state,
            controller=self.controller,
        )

        return loss, chrono_state

    def _log_step(
        self,
        chrono_state: Optional[ChronovisorMixtralState],
        kappa: float,
        intent: str,
    ) -> None:
        """Log training progress."""
        summary = self.loss_fn.get_summary(last_n=self.config.log_every_n_steps)

        log_msg = f"Step {self.global_step}"
        log_msg += f" | Loss: {summary.get('total_loss_mean', 0):.4f}"
        log_msg += f" | κ: {kappa:.3f} ({intent})"

        if chrono_state:
            log_msg += f" | R: {chrono_state.coherence:.3f}"

        if isinstance(self.optimizer, CoherenceGatedOptimizer):
            log_msg += f" | Gate: {self.optimizer.last_gate_value:.3f}"

        print(log_msg)

        # Log structural temperature landscape periodically
        if self.global_step % (self.config.log_every_n_steps * 10) == 0:
            st_diag = self.controller.get_structural_temperature_diagnostics()
            print(f"  Landscape: var={st_diag['variance']:.4f}, "
                  f"valleys={st_diag['valleys']}, ridges={st_diag['ridges']}")

    def _evaluate(self) -> float:
        """Run evaluation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                loss, _ = self._training_step(batch)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Eval Loss: {avg_loss:.4f}")

        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self._save_checkpoint(is_best=True)

        self.model.train()
        return avg_loss

    def _save_checkpoint(self, is_final: bool = False, is_best: bool = False) -> None:
        """Save model and Chronovisor checkpoint."""
        # Determine checkpoint name
        if is_final:
            name = "final"
        elif is_best:
            name = "best"
        else:
            name = f"step_{self.global_step}"

        # Save model weights
        model_path = self.checkpoint_dir / f"model_{name}.pt"
        torch.save(self.model.state_dict(), model_path)

        # Save Chronovisor state (geological memory)
        if self.config.save_chronovisor_state:
            chrono_path = self.checkpoint_dir / f"chronovisor_{name}.json"
            chrono_ckpt = ChronovisorCheckpoint.from_controller(
                self.controller,
                step=self.global_step,
                epoch=self.epoch,
                loss_history=self.loss_fn.history,
            )
            chrono_ckpt.save(chrono_path)

        print(f"Saved checkpoint: {name}")

    def load_checkpoint(self, name: str) -> None:
        """Load model and Chronovisor checkpoint."""
        # Load model weights
        model_path = self.checkpoint_dir / f"model_{name}.pt"
        self.model.load_state_dict(torch.load(model_path))

        # Load Chronovisor state
        if self.config.save_chronovisor_state:
            chrono_path = self.checkpoint_dir / f"chronovisor_{name}.json"
            if chrono_path.exists():
                chrono_ckpt = ChronovisorCheckpoint.load(chrono_path)
                chrono_ckpt.restore_to_controller(self.controller)
                self.global_step = chrono_ckpt.step
                self.epoch = chrono_ckpt.epoch
                self.loss_fn.history = chrono_ckpt.loss_history

        print(f"Loaded checkpoint: {name}")


# =============================================================================
# Simple Dataset for Testing
# =============================================================================

class DummyDataset(Dataset):
    """Dummy dataset for testing the training loop."""

    def __init__(self, vocab_size: int = 1000, seq_len: int = 128, size: int = 1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": input_ids, "labels": input_ids}


# =============================================================================
# Demo
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("ChronoMoE Training Infrastructure Demo")
    print("=" * 60)

    # Create small model for testing
    config = MixtralConfig(
        vocab_size=1000,
        hidden_dim=128,
        intermediate_dim=256,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=4,
        num_experts_per_token=2,
        max_seq_length=128,
        enable_chronovisor=True,
    )

    model = ChronovisorMixtralForCausalLM(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy dataset
    dataset = DummyDataset(vocab_size=config.vocab_size, seq_len=64, size=100)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Training config
    train_config = TrainingConfig(
        learning_rate=1e-4,
        max_epochs=1,
        max_steps=20,  # Just a few steps for demo
        batch_size=4,
        gradient_accumulation_steps=1,
        initial_kappa=0.5,
        final_kappa=-0.3,
        kappa_warmup_steps=5,
        kappa_schedule="cosine",
        use_coherence_gating=True,
        log_every_n_steps=5,
        save_every_n_steps=10,
        checkpoint_dir="demo_checkpoints",
    )

    print("\nTraining Configuration:")
    print(f"  Initial κ: {train_config.initial_kappa}")
    print(f"  Final κ: {train_config.final_kappa}")
    print(f"  Schedule: {train_config.kappa_schedule}")
    print(f"  Coherence gating: {train_config.use_coherence_gating}")

    # Create trainer
    trainer = ChronoMoETrainer(
        model=model,
        config=train_config,
        train_dataloader=dataloader,
    )

    print("\n" + "-" * 60)
    print("Starting training...")
    print("-" * 60 + "\n")

    # Run training
    results = trainer.train()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Steps: {results['global_step']}")
    print(f"Final loss: {results['loss_summary'].get('total_loss_mean', 'N/A'):.4f}")

    # Show final geological landscape
    controller = model.model.controller
    st_diag = controller.get_structural_temperature_diagnostics()
    vh_diag = controller.get_valley_health_diagnostics()

    print("\nFinal Geological Landscape:")
    print(f"  Structural T variance: {st_diag['variance']:.6f}")
    print(f"  Landscape formed: {st_diag['landscape_formed']}")
    print(f"  Valleys: {st_diag['valleys']}")
    print(f"  Ridges: {st_diag['ridges']}")
    print(f"  Valley health: {vh_diag['mean_valley_health']:.3f}")
    print(f"  Self-correction working: {vh_diag['self_correction_working']}")

    print("\n✓ Training infrastructure complete!")
    print("  - ChronoMoELoss: LM + balance + coherence + valley health")
    print("  - KnobScheduler: κ annealing (linear/cosine/step/adaptive)")
    print("  - CoherenceGatedOptimizer: R-gated gradient updates")
    print("  - ChronovisorCheckpoint: Geological memory preservation")
    print("  - ChronoMoETrainer: Full training loop integration")
