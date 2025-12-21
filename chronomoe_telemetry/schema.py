"""
Schema: JSON schema + gloss strings.

Defines the telemetry data format and human-readable interpretations.
"""

from dataclasses import dataclass, asdict
from typing import Dict
import json


@dataclass
class TelemetrySnapshot:
    """
    Single telemetry snapshot at a pause point.

    Contains all measurements needed to assess whether the system
    has reached meaningful completion vs premature collapse.
    """
    session_id: str
    pause_index: int
    pause_reason: str  # "candidate_stillness" | "max_steps" | "user_interrupt"

    # Position in strategy space
    pc1: float  # Momentum / continuation strength
    pc2: float  # Exploration margin

    # Dynamics (velocity)
    d_pc1: float  # Change in momentum
    d_pc2: float  # Change in exploration

    # Routing statistics
    routing_entropy: float  # Dispersion of expert votes
    expert_switch_rate: float  # Coalition stability

    # Stillness detection
    stillness_flags: Dict[str, bool]

    # Human-readable interpretations
    gloss: Dict[str, str]

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON with glossary included."""
        data = asdict(self)
        return json.dumps(data, indent=indent)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'TelemetrySnapshot':
        """Construct from dictionary."""
        return cls(**data)


def create_gloss_reference() -> Dict[str, str]:
    """
    Create reference glossary for telemetry fields.

    This can be shown to observers (human or model) to explain
    what each field measures.
    """
    return {
        'pc1': 'Momentum / continuation strength (higher = stronger push to continue current trajectory)',
        'pc2': 'Exploration margin (higher = more viable alternative routes under consideration)',
        'd_pc1': 'Change in momentum since last pause (near zero = momentum exhausted)',
        'd_pc2': 'Change in exploration (near zero = alternatives exhausted)',
        'routing_entropy': 'Dispersion of expert votes (lower = coalition consensus)',
        'expert_switch_rate': 'How often expert coalition changes (lower = stabilized routing)',
        'stillness_flags': {
            'fast': 'Token-scale motion has flattened',
            'medium': 'Arc-scale progress has stalled',
            'slow': 'Long-horizon regime frozen (identity-level stillness)',
        },
        'pause_reason': {
            'candidate_stillness': 'System detected convergence (pause for review)',
            'candidate_stillness_all': 'All timescales show stillness (strong signal)',
            'max_steps': 'Hard limit reached (forced pause)',
            'user_interrupt': 'Human-initiated pause',
        }
    }


def create_phenomenology_template() -> dict:
    """
    Create template for phenomenology reports.

    This structure can be filled out by human or model observers
    to provide execution-phase experience reports.
    """
    return {
        'snapshot_id': None,  # Which snapshot this refers to
        'felt_state': None,   # "complete" | "premature" | "stuck"
        'confidence': None,   # 0.0 to 1.0
        'notes': '',          # Free text explanation
    }
