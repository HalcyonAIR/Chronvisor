"""
External Mixtral Model Adapter

Wraps HuggingFace Mixtral models to work with Chronovisor pressure system.

This adapter:
1. Loads external Mixtral (Mixtral-8x7B, Mixtral-8x22B, etc.)
2. Extracts expert routing signals during generation
3. Integrates with ChronovisorMixtralController
4. Enables pressure-based multistep generation

Key difference from toy model:
- Uses real pre-trained Mixtral weights
- Real routing behavior (not random)
- Real semantic content (not random tokens)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Chronovisor integration
from chronomoe.chronovisor_mixtral_bridge import ChronovisorMixtralController
from chronomoe.session_controller import SessionController, SessionMode
from chronomoe.pressure import PressureSignals
from chronomoe.clock_heads_corrected import ClockHead


@dataclass
class ExternalMixtralConfig:
    """Configuration for external Mixtral adapter."""

    # Model identifier
    model_name: str = "mistralai/Mixtral-8x7B-v0.1"

    # Chronovisor settings
    enable_chronovisor: bool = True
    chronovisor_config: Optional[dict] = None

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    load_in_8bit: bool = False  # For memory efficiency
    load_in_4bit: bool = False

    # Clock heads
    enable_clock_heads: bool = True
    clock_head_dim: int = 128  # Projection dimension for clock space


class ExternalMixtralAdapter(nn.Module):
    """
    Adapter for external Mixtral models.

    Wraps HuggingFace Mixtral and adds:
    - Expert routing signal extraction
    - Chronovisor integration
    - Clock heads
    - Pressure-based session control
    """

    def __init__(self, config: ExternalMixtralConfig):
        super().__init__()
        self.config = config

        # Load external Mixtral model
        print(f"Loading {config.model_name}...")
        self._load_model()

        # Extract model config
        self.vocab_size = self.model.config.vocab_size
        self.hidden_dim = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers
        self.num_experts = self.model.config.num_local_experts
        self.num_experts_per_token = self.model.config.num_experts_per_tok

        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Hidden dim: {self.hidden_dim}")
        print(f"  Layers: {self.num_layers}")
        print(f"  Experts: {self.num_experts}")
        print(f"  Experts per token: {self.num_experts_per_token}")

        # Initialize Chronovisor controller
        if config.enable_chronovisor:
            self.chronovisor = ChronovisorMixtralController(
                num_layers=self.num_layers,
                num_experts=self.num_experts,
                **(config.chronovisor_config or {}),
            )
        else:
            self.chronovisor = None

        # Initialize clock heads
        if config.enable_clock_heads:
            self._init_clock_heads()
        else:
            self.fast_clock = None
            self.medium_clock = None
            self.slow_clock = None

        # Session controller
        self.session = None

    def _load_model(self):
        """Load external Mixtral model from HuggingFace."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            load_kwargs = {
                "device_map": "auto" if self.config.device == "cuda" else None,
                "torch_dtype": torch.float16 if self.config.device == "cuda" else torch.float32,
            }

            if self.config.load_in_8bit:
                load_kwargs["load_in_8bit"] = True
            elif self.config.load_in_4bit:
                load_kwargs["load_in_4bit"] = True

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, **load_kwargs
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f"✓ Model loaded: {self.config.model_name}")

        except ImportError:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _init_clock_heads(self):
        """Initialize clock heads for temporal control."""
        # Project from model hidden dim to clock space
        clock_dim = self.config.clock_head_dim

        # Fast clock (stability, half-life ~5)
        self.fast_clock = ClockHead(
            d_model=self.hidden_dim,
            d_proj=clock_dim,
            N_micro=8,
            N_meso=4,
            N_macro=2,
            half_life_turns=5,
            name="fast",
        )

        # Medium clock (trajectory, half-life ~50)
        self.medium_clock = ClockHead(
            d_model=self.hidden_dim,
            d_proj=clock_dim,
            N_micro=16,
            N_meso=8,
            N_macro=4,
            half_life_turns=50,
            name="medium",
        )

        # Slow clock (identity, half-life ~500)
        self.slow_clock = ClockHead(
            d_model=self.hidden_dim,
            d_proj=clock_dim,
            N_micro=8,
            N_meso=16,
            N_macro=8,
            half_life_turns=500,
            name="slow",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_chronovisor: bool = True,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass with expert routing signal extraction.

        Returns:
            logits: [batch, seq_len, vocab_size]
            chrono_state: Dict with routing stats and coherence
        """
        # Forward through Mixtral
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_router_logits=True,  # Get routing information
        )

        logits = outputs.logits

        # Extract expert routing signals
        chrono_state = None
        if self.chronovisor and update_chronovisor:
            chrono_state = self._extract_routing_signals(outputs)

        return logits, chrono_state

    def _extract_routing_signals(self, outputs) -> dict:
        """
        Extract expert routing signals from model outputs.

        HuggingFace Mixtral returns router_logits for each MoE layer.
        We use these to compute:
        - Expert usage distribution per layer
        - Routing entropy
        - Top-k selections
        """
        router_logits = outputs.router_logits  # List of [batch, seq_len, num_experts]

        # Initialize storage
        expert_usage = {}
        routing_entropy = {}

        for layer_idx, router_logit in enumerate(router_logits):
            # Get last token's routing (for generation)
            last_token_logits = router_logit[:, -1, :]  # [batch, num_experts]

            # Compute routing probabilities
            probs = torch.softmax(last_token_logits, dim=-1)  # [batch, num_experts]

            # Get expert usage (mean over batch)
            usage = probs.mean(dim=0).detach().cpu().numpy()  # [num_experts]
            expert_usage[layer_idx] = usage

            # Compute entropy
            entropy = -np.sum(usage * np.log(usage + 1e-10))
            normalized_entropy = entropy / np.log(len(usage))
            routing_entropy[layer_idx] = normalized_entropy

        # Update Chronovisor controller
        if self.chronovisor:
            chrono_state = self.chronovisor.tick(
                expert_usage=expert_usage, routing_entropy=routing_entropy
            )
        else:
            chrono_state = {
                "expert_usage": expert_usage,
                "routing_entropy": routing_entropy,
                "coherence": 0.5,
                "delta_coherence": 0.0,
            }

        return chrono_state

    def generate_multistep(
        self,
        input_ids: torch.Tensor,
        mode: SessionMode = SessionMode.MULTISTEP,
        chunk_size: int = 50,
        max_chunks: int = 10,
        verbose: bool = False,
    ):
        """
        Generate with pressure-based pausing.

        Same interface as ClockGatedMultistepModel for compatibility.
        """
        # Initialize session if needed
        if self.session is None:
            self.session = SessionController(mode, chunk_size, max_chunks, verbose)

        current_ids = input_ids.clone()

        for chunk_idx in range(max_chunks):
            chunk_tokens = 0

            # Generate chunk token by token
            for _ in range(chunk_size):
                # Forward pass
                logits, chrono_state = self.forward(current_ids, update_chronovisor=True)

                # Sample next token
                next_token_logits = logits[:, -1, :]
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append token
                current_ids = torch.cat([current_ids, next_token], dim=-1)
                chunk_tokens += 1

                # Update clock heads (if enabled)
                if self.fast_clock:
                    # Extract signals for clock update
                    final_hidden = outputs.hidden_states[-1][:, -1, :].squeeze(0)
                    # (Clock update logic would go here)
                    pass

            # End of chunk: compute pressure and decide
            # Extract router stats from last generation
            router_stats = self._compute_router_stats(chrono_state, layer_idx=self.num_layers - 1)

            # Get clock signals (placeholder for now)
            clock_signals = {
                "fast_confidence": 0.5,
                "mid_proximity_meso": 0.5,
                "mid_transition_prob": 0.5,
                "slow_confidence": 0.5,
                "slow_proximity_macro": 0.5,
                "slow_constraint_penalty": 0.0,
            }

            pressure_signals = PressureSignals(
                router_entropy=router_stats["entropy"],
                router_margin=router_stats["margin"],
                coherence_R=chrono_state.get("coherence", 0.5),
                delta_R=chrono_state.get("delta_coherence", 0.0),
                margin=1.0,  # Placeholder
                fast_confidence=clock_signals["fast_confidence"],
                mid_proximity_meso=clock_signals["mid_proximity_meso"],
                mid_transition_prob=clock_signals["mid_transition_prob"],
                mid_residual_intent=self.session.residual_intent,
                slow_confidence=clock_signals["slow_confidence"],
                slow_proximity_macro=clock_signals["slow_proximity_macro"],
                slow_constraint_penalty=clock_signals["slow_constraint_penalty"],
            )

            should_pause, pause_reason, telemetry = self.session.decide_and_log(
                pressure_signals, chunk_tokens
            )

            if verbose:
                print(f"\nChunk {chunk_idx + 1} complete:")
                print(f"  Tokens: {chunk_tokens}")
                print(f"  Entropy: {router_stats['entropy']:.4f}")
                print(f"  Net pressure: {telemetry.net_pressure:+.4f}")
                if should_pause:
                    print(f"  Decision: PAUSE ({pause_reason})")

            # Check if should pause
            if should_pause:
                if mode == SessionMode.MULTISTEP:
                    break
                elif pause_reason in ["fast_instability", "negative_pressure"]:
                    break

        session_telemetry = self.session.get_session_telemetry()
        return current_ids, session_telemetry

    def _compute_router_stats(self, chrono_state, layer_idx):
        """Compute router statistics from chrono state."""
        if chrono_state is None:
            return {"entropy": 0.5, "margin": 0.5}

        usage = chrono_state.get("expert_usage", {}).get(layer_idx)
        if usage is None or len(usage) < 2:
            return {"entropy": 0.5, "margin": 0.5}

        # Normalize
        usage = np.array(usage)
        usage = usage / (usage.sum() + 1e-10)

        # Entropy
        entropy = -np.sum(usage * np.log(usage + 1e-10))
        normalized_entropy = entropy / (np.log(len(usage)) + 1e-10)

        # Margin (top1 - top2)
        sorted_usage = np.sort(usage)[::-1]
        margin = float(sorted_usage[0] - sorted_usage[1])

        return {
            "entropy": float(np.clip(normalized_entropy, 0.0, 1.0)),
            "margin": float(np.clip(margin, 0.0, 1.0)),
        }

    def reset_session(self):
        """Reset session and clock states."""
        if self.session:
            self.session.reset()
        if self.fast_clock:
            self.fast_clock.reset()
        if self.medium_clock:
            self.medium_clock.reset()
        if self.slow_clock:
            self.slow_clock.reset()

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text input."""
        tokens = self.tokenizer(text, return_tensors="pt").input_ids
        return tokens.to(self.config.device)

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids[0], skip_special_tokens=True)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("External Mixtral Adapter - Example Usage")
    print("=" * 70)
    print()

    # Create config
    config = ExternalMixtralConfig(
        model_name="mistralai/Mixtral-8x7B-v0.1",
        enable_chronovisor=True,
        enable_clock_heads=True,
        load_in_8bit=True,  # For memory efficiency
    )

    # Load model
    try:
        adapter = ExternalMixtralAdapter(config)
        print()
        print("✓ Adapter initialized successfully")
        print()

        # Test prompt
        prompt = "Explain how transformers work in machine learning:"
        print(f"Prompt: {prompt}")
        print()

        input_ids = adapter.tokenize(prompt)
        print(f"Input tokens: {input_ids.shape[1]}")
        print()

        # Generate with pressure-based multistep
        print("Generating with multistep mode...")
        generated, telemetry = adapter.generate_multistep(
            input_ids, mode=SessionMode.MULTISTEP, chunk_size=20, max_chunks=3, verbose=True
        )

        print()
        print("Generated text:")
        print("-" * 70)
        print(adapter.decode(generated))
        print("-" * 70)
        print()

        print("Telemetry:")
        print(f"  Total chunks: {telemetry.total_chunks}")
        print(f"  Total tokens: {telemetry.total_tokens}")

    except Exception as e:
        print(f"Error: {e}")
        print()
        print("Note: This requires:")
        print("  1. pip install transformers")
        print("  2. Sufficient GPU memory (or use load_in_8bit=True)")
        print("  3. HuggingFace model access")
