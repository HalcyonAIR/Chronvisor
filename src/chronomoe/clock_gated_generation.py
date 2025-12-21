"""
Clock-Gated Generation for Mixtral

Extends ChronovisorMixtralForCausalLM with temporal arbitration via clock heads.

Architecture:
    Base Model (stateless) → candidate tokens
                                ↓
    Clock Heads (stateful) → temporal scores
                                ↓
    Elimination Tournament → final token

The base model's forward pass is never touched. Clocks sit outside,
judging outputs after generation.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from chronomoe.chronovisor_mixtral_bridge import (
    ChronovisorMixtralForCausalLM,
    ChronovisorMixtralState,
    MixtralConfig,
)
from chronomoe.clock_heads import (
    FastClock,
    MediumClock,
    SlowClock,
    elimination_tournament,
)


class ClockGatedMixtralForCausalLM(ChronovisorMixtralForCausalLM):
    """
    Mixtral with clock-gated generation.

    Extends base model with three stateful clock heads that arbitrate
    token selection based on temporal coherence at different timescales.

    The base model generates candidates. The clocks judge them.
    """

    def __init__(self, config: MixtralConfig):
        super().__init__(config)

        # Create clock heads
        self.fast_clock = FastClock(hidden_dim=config.hidden_dim)
        self.medium_clock = MediumClock(hidden_dim=config.hidden_dim)
        self.slow_clock = SlowClock(hidden_dim=config.hidden_dim)

        # Config for gated generation
        self.num_candidates = 5  # How many candidates to score
        self.sampling_temperature = 1.0  # Temperature for candidate sampling

    def generate_with_clocks(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        top_k_candidates: int = 5,
        sampling_temperature: float = 1.0,
        update_chronovisor: bool = True,
        context_window: int = 50,  # How much context to pass to clocks
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, List[ChronovisorMixtralState], dict]:
        """
        Generate with clock-gated arbitration.

        For each generation step:
            1. Base model generates logits (stateless)
            2. Sample top-k candidates from logits
            3. Each clock scores each candidate
            4. Elimination tournament selects winner
            5. Clocks update their state (temporal persistence)

        Args:
            input_ids: Initial token IDs [batch, seq_len]
            max_length: Maximum total sequence length
            top_k_candidates: How many candidates to score per step
            sampling_temperature: Temperature for sampling candidates
            update_chronovisor: Whether to update Chronovisor controller
            context_window: How much recent context to pass to clocks
            verbose: Print debug info

        Returns:
            Tuple of:
                - generated_ids: [batch, max_length]
                - chronovisor_states: List of states from base model
                - clock_stats: Dict with clock arbitration statistics
        """
        batch_size = input_ids.shape[0]
        if batch_size > 1:
            raise NotImplementedError("Clock-gated generation only supports batch_size=1 for now")

        current_ids = input_ids.clone()
        chronovisor_states = []

        # Statistics tracking
        clock_stats = {
            "fast_wins": 0,
            "medium_wins": 0,
            "slow_wins": 0,
            "fast_scores": [],
            "medium_scores": [],
            "slow_scores": [],
            "decisions": [],  # Which clock vetoed which candidates
        }

        if verbose:
            print("=" * 60)
            print("CLOCK-GATED GENERATION")
            print("=" * 60)
            print(f"Max length: {max_length}")
            print(f"Candidates per step: {top_k_candidates}")
            print(f"Context window: {context_window}")
            print()

        for step in range(max_length - input_ids.shape[1]):
            # ================================================================
            # STEP 1: Base model generates logits (stateless)
            # ================================================================
            with torch.no_grad():
                logits, chrono_state = self.forward(
                    current_ids,
                    update_chronovisor=update_chronovisor,
                )
            chronovisor_states.append(chrono_state)

            # Get logits for next token
            next_token_logits = logits[:, -1, :] / sampling_temperature

            # ================================================================
            # STEP 2: Sample top-k candidates
            # ================================================================
            # Get top-k tokens by probability
            probs = F.softmax(next_token_logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, top_k_candidates, dim=-1)

            candidates = top_k_indices[0].tolist()  # [k]
            candidate_probs = top_k_probs[0].tolist()

            if verbose:
                print(f"\nStep {step + 1}:")
                print(f"  Candidates: {candidates}")
                print(f"  Base probs: {[f'{p:.4f}' for p in candidate_probs]}")

            # ================================================================
            # STEP 3: Clocks score each candidate
            # ================================================================
            # Get recent context for clocks
            context_start = max(0, current_ids.shape[1] - context_window)
            context = current_ids[0, context_start:]  # [context_len]

            scores = {
                "fast": [],
                "medium": [],
                "slow": [],
            }

            for cand in candidates:
                fast_score = self.fast_clock.compute_score(
                    context, cand, self.embed_tokens
                )
                medium_score = self.medium_clock.compute_score(
                    context, cand, self.embed_tokens
                )
                slow_score = self.slow_clock.compute_score(
                    context, cand, self.embed_tokens
                )

                scores["fast"].append(fast_score)
                scores["medium"].append(medium_score)
                scores["slow"].append(slow_score)

            # Track scores
            clock_stats["fast_scores"].append(scores["fast"])
            clock_stats["medium_scores"].append(scores["medium"])
            clock_stats["slow_scores"].append(scores["slow"])

            if verbose:
                print(f"  Fast scores:   {[f'{s:.4f}' for s in scores['fast']]}")
                print(f"  Medium scores: {[f'{s:.4f}' for s in scores['medium']]}")
                print(f"  Slow scores:   {[f'{s:.4f}' for s in scores['slow']]}")

            # ================================================================
            # STEP 4: Elimination tournament
            # ================================================================
            winner = elimination_tournament(candidates, scores)
            winner_idx = candidates.index(winner)

            # Identify which clock would have vetoed which candidates
            # (lowest score for each candidate)
            decision_info = {
                "winner": winner,
                "winner_idx": winner_idx,
                "vetoes": [],
            }

            for i, cand in enumerate(candidates):
                min_score_clock = min(
                    [("fast", scores["fast"][i]),
                     ("medium", scores["medium"][i]),
                     ("slow", scores["slow"][i])],
                    key=lambda x: x[1],
                )[0]
                decision_info["vetoes"].append((cand, min_score_clock))

            clock_stats["decisions"].append(decision_info)

            if verbose:
                print(f"  Winner: {winner} (index {winner_idx})")
                print(f"  Avg score: {(scores['fast'][winner_idx] + scores['medium'][winner_idx] + scores['slow'][winner_idx]) / 3:.4f}")

            # Track which clock's veto mattered most
            # (which clock had the lowest score for the runner-up)
            if len(candidates) > 1:
                # Find runner-up (highest avg score excluding winner)
                avg_scores = [
                    (scores["fast"][i] + scores["medium"][i] + scores["slow"][i]) / 3
                    for i in range(len(candidates))
                ]
                avg_scores[winner_idx] = -1  # Exclude winner
                runner_up_idx = max(range(len(avg_scores)), key=lambda i: avg_scores[i])

                # Which clock had lowest score for runner-up?
                runner_up_scores = {
                    "fast": scores["fast"][runner_up_idx],
                    "medium": scores["medium"][runner_up_idx],
                    "slow": scores["slow"][runner_up_idx],
                }
                veto_clock = min(runner_up_scores.items(), key=lambda x: x[1])[0]

                clock_stats[f"{veto_clock}_wins"] += 1

            # ================================================================
            # STEP 5: Update clock states (temporal persistence)
            # ================================================================
            # Outcome = 1.0 (assume good for now; could use loss or reward)
            outcome = 1.0

            self.fast_clock.update(context, winner, self.embed_tokens, outcome)
            self.medium_clock.update(context, winner, self.embed_tokens, outcome)
            self.slow_clock.update(context, winner, self.embed_tokens, outcome)

            # Apply decay
            self.fast_clock.apply_decay()
            self.medium_clock.apply_decay()
            self.slow_clock.apply_decay()

            # ================================================================
            # STEP 6: Append winner to sequence
            # ================================================================
            next_token = torch.tensor([[winner]], dtype=current_ids.dtype, device=current_ids.device)
            current_ids = torch.cat([current_ids, next_token], dim=-1)

        if verbose:
            print("\n" + "=" * 60)
            print("GENERATION COMPLETE")
            print("=" * 60)
            print(f"Fast clock wins:   {clock_stats['fast_wins']}")
            print(f"Medium clock wins: {clock_stats['medium_wins']}")
            print(f"Slow clock wins:   {clock_stats['slow_wins']}")
            print()

        return current_ids, chronovisor_states, clock_stats

    def reset_clocks(self) -> None:
        """Reset all clock heads to initial state."""
        self.fast_clock.reset()
        self.medium_clock.reset()
        self.slow_clock.reset()

    def get_clock_diagnostics(self) -> dict:
        """Get diagnostic information about clock states."""
        return {
            "fast": {
                "ticks": self.fast_clock.ticks,
                "current_basin": self.fast_clock.state.current_basin,
                "half_life": self.fast_clock.half_life,
                "n_attractors": self.fast_clock.n_attractors,
            },
            "medium": {
                "ticks": self.medium_clock.ticks,
                "current_basin": self.medium_clock.state.current_basin,
                "half_life": self.medium_clock.half_life,
                "n_attractors": self.medium_clock.n_attractors,
            },
            "slow": {
                "ticks": self.slow_clock.ticks,
                "current_basin": self.slow_clock.state.current_basin,
                "half_life": self.slow_clock.half_life,
                "n_attractors": self.slow_clock.n_attractors,
            },
        }


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == '__main__':
    print("Testing Clock-Gated Mixtral Generation")
    print("=" * 60)
    print()

    # Create small config for testing
    config = MixtralConfig(
        vocab_size=1000,
        hidden_dim=256,
        intermediate_dim=512,
        num_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=4,
        num_experts_per_token=2,
        max_seq_length=512,
        enable_chronovisor=True,
    )

    # Create model
    model = ClockGatedMixtralForCausalLM(config)

    # Generate
    input_ids = torch.randint(0, config.vocab_size, (1, 10))

    print("Generating with clock arbitration...")
    print()

    generated_ids, chrono_states, clock_stats = model.generate_with_clocks(
        input_ids,
        max_length=20,
        top_k_candidates=5,
        sampling_temperature=1.0,
        verbose=True,
    )

    print("\nGeneration Results:")
    print(f"  Input length: {input_ids.shape[1]}")
    print(f"  Output length: {generated_ids.shape[1]}")
    print(f"  Tokens generated: {generated_ids.shape[1] - input_ids.shape[1]}")

    print("\nClock Diagnostics:")
    diag = model.get_clock_diagnostics()
    for clock_name, clock_diag in diag.items():
        print(f"  {clock_name}:")
        print(f"    ticks: {clock_diag['ticks']}")
        print(f"    current_basin: {clock_diag['current_basin']}")
        print(f"    half_life: {clock_diag['half_life']}")

    print("\nChronovisor State (final):")
    final_state = chrono_states[-1]
    if final_state:
        print(f"  Coherence R: {final_state.coherence:.4f}")
        print(f"  Δ Coherence: {final_state.delta_coherence:.4f}")
        print(f"  Fast clock: {final_state.fast_clock}")

    print("\n✓ Clock-gated generation complete!")
    print("  Base model: Stateless generation")
    print("  Clocks: Temporal arbitration (fast/medium/slow)")
    print("  Tournament: Elimination by lowest score")
