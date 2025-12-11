"""
Conversational Synthetic Dataset for ChronoMoE

Creates multi-turn dialogues with internal phase shifts:
- inquiry → clarification → complication → contradiction → exception → concession → synthesis

Each sequence contains structural tension that forces expert divergence.
"""

import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class ConversationalDomain:
    """A domain with conversation templates and phase-specific vocab."""
    name: str
    color: str

    # Vocab for different phases
    opening_vocab: List[str]
    premise_vocab: List[str]
    complication_vocab: List[str]
    contradiction_vocab: List[str]
    exception_vocab: List[str]
    concession_vocab: List[str]
    synthesis_vocab: List[str]

    # Templates for each phase
    opening_templates: List[str]
    premise_templates: List[str]
    complication_templates: List[str]
    contradiction_templates: List[str]
    exception_templates: List[str]
    concession_templates: List[str]
    synthesis_templates: List[str]


# Legal Domain - clause-based reasoning
LEGAL_DOMAIN = ConversationalDomain(
    name="legal",
    color="blue",
    opening_vocab=["contract", "statute", "warranty", "dispute", "jurisdiction", "clause"],
    premise_vocab=["plaintiff", "defendant", "liability", "negligence", "good faith", "defect"],
    complication_vocab=["precedent", "exception", "withheld", "logs", "records", "evidence"],
    contradiction_vocab=["automatically", "imply", "assumption", "breaks", "ruling", "court"],
    exception_vocab=["unless", "directly", "relate", "failure", "threshold", "burden"],
    concession_vocab=["fair", "credible", "might", "survive", "partial", "concede"],
    synthesis_vocab=["factor", "argument", "synthesis", "conclude", "resolution", "outcome"],

    opening_templates=[
        "So the {open1} dispute hinges on whether the {open2} still applies, right?",
        "The {open1} question centers on {open2} interpretation under this {open3}.",
        "Does the {open1} cover situations where {open2} was breached by the {open3}?",
    ],
    premise_templates=[
        "Well, the statute covers {premise1} discovered within ninety days, but only if the {premise2} acted in {premise3}.",
        "The {premise1} bears the burden of proving {premise2} once {premise3} is established.",
        "Under common law, {premise1} requires showing {premise2} and {premise3} simultaneously.",
    ],
    complication_templates=[
        "But in this case the assumption gets weird because the seller {complic1} the {complic2}.",
        "The problem is that {complic1} from prior cases suggests {complic2} may not apply here.",
        "However, the {complic1} in the discovery phase revealed {complic2} that changes everything.",
    ],
    contradiction_templates=[
        "That's the thing—there's {contra1} that says {contra2} doesn't {contra3} imply negligence.",
        "The {contra1} court explicitly rejected that {contra2} in similar circumstances.",
        "Actually, the {contra1} shows that the {contra2} breaks the {contra3} entirely.",
    ],
    exception_templates=[
        "{except1} the logs {except2} relate to the {except3}, which changes the {except4}.",
        "{except1} there's a statutory {except2} that raises the {except3} for the {except4}.",
        "{except1} the plaintiff can show {except2}, the {except3} defense fails on the {except4}.",
    ],
    concession_templates=[
        "Mmm... {concede1}. So the warranty might {concede2}. But the threshold is {concede3} now.",
        "Fair point. The {concede1} has a {concede2} argument once you factor the {concede3}.",
        "I'll {concede1} that. The {concede2} position is stronger if we assume {concede3}.",
    ],
    synthesis_templates=[
        "Exactly. It's not a slam dunk, but the plaintiff has a {synth1} argument once you factor in the {synth2}.",
        "So the resolution hinges on proving {synth1} which affects {synth2} materially.",
        "Bottom line: {synth1} survives if and only if {synth2} can be established.",
    ],
)

# Physics Domain - hypothesis testing with edge cases
PHYSICS_DOMAIN = ConversationalDomain(
    name="physics",
    color="green",
    opening_vocab=["acceleration", "anomaly", "chamber", "pressure", "particle", "field"],
    premise_vocab=["radiation", "artefact", "ion", "pump", "measurement", "gradient"],
    complication_vocab=["active", "run", "showed", "reading", "thermal", "cooldown"],
    contradiction_vocab=["unlikely", "phantom", "drive", "invert", "sign", "boundary"],
    exception_vocab=["unless", "condition", "inverted", "flip", "curve", "exotic"],
    concession_vocab=["tracks", "fits", "run", "model", "terms", "matches"],
    synthesis_vocab=["boundary", "effect", "constraint", "phenomenon", "mechanism", "prediction"],

    opening_templates=[
        "So the anomalous {open1} shows up only when the {open2} dips below threshold, right?",
        "The {open1} readings suggest {open2} is affecting the {open3} somehow.",
        "Does the {open1} correlate with changes in {open2} or is it independent of {open3}?",
    ],
    premise_templates=[
        "That's what it looks like. Could be a {premise1} artefact or something with the {premise2}.",
        "The {premise1} shows a clear dependence on {premise2} when we control for {premise3}.",
        "Theory suggests {premise1} should produce {premise2} under these {premise3} conditions.",
    ],
    complication_templates=[
        "But the {complic1} wasn't {complic2} during the second {complic3}, and the anomaly still showed.",
        "The problem is the {complic1} behaved differently when {complic2} changed during {complic3}.",
        "However, {complic1} measurements from the {complic2} run contradict the {complic3} prediction.",
    ],
    contradiction_templates=[
        "{contra1} the thermal {contra2} drove a {contra3} reading—though that seems unlikely.",
        "The {contra1} interpretation fails because {contra2} can't explain the {contra3} dependence.",
        "Actually, {contra1} conditions would {contra2} the sign, making {contra3} impossible.",
    ],
    exception_templates=[
        "Wait, {except1} the gradient {except2} during cooldown. That could {except3} the {except4} measurement.",
        "{except1} there's a {except2} effect that inverts the {except3} under {except4} conditions.",
        "{except1} we account for {except2}, the {except3} should flip at the {except4} boundary.",
    ],
    concession_templates=[
        "Alright, that {concede1}. Let's run the {concede2} with inverted {concede3}.",
        "Fair enough. The {concede1} model does fit if we include {concede2} as a {concede3}.",
        "I'll concede that. {concede1} explains the {concede2} once we factor in {concede3}.",
    ],
    synthesis_templates=[
        "If the curve {synth1}, we're looking at a {synth2} effect, not exotic physics.",
        "So the phenomenon reduces to {synth1} coupled with {synth2} at the boundary.",
        "Bottom line: {synth1} is a second-order {synth2} effect, fully explained.",
    ],
)

# Poetry Domain - emotional arcs and metaphor
POETRY_DOMAIN = ConversationalDomain(
    name="poetry",
    color="red",
    opening_vocab=["quiet", "room", "silence", "moment", "air", "space"],
    premise_vocab=["remembers", "arguments", "weight", "echoes", "shadows", "traces"],
    complication_vocab=["ourselves", "maybe", "heavy", "lighter", "carries", "holds"],
    contradiction_vocab=["sometimes", "lighter", "forgives", "heavier", "refuses", "remembers"],
    exception_vocab=["unless", "refusing", "forgiven", "trying", "another", "keeps"],
    concession_vocab=["maybe", "could", "perhaps", "might", "seems", "feels"],
    synthesis_vocab=["either way", "listening", "waiting", "becoming", "remains", "transforms"],

    opening_templates=[
        "You ever notice how a {open1} {open2} feels heavier after someone leaves?",
        "There's something about {open1} that makes the {open2} feel different.",
        "Does {open1} change the {open2}, or do we change it ourselves?",
    ],
    premise_templates=[
        "It's like the {premise1} {premise2} the {premise3} we didn't finish.",
        "The {premise1} carries {premise2} from all the {premise3} we left behind.",
        "{premise1} becomes {premise2} when {premise3} settle into the walls.",
    ],
    complication_templates=[
        "Yeah... or maybe we {complic1} that {complic2} there {complic3}.",
        "But {complic1} the {complic2} feels {complic3}, like it never happened.",
        "Though {complic1}, the {complic2} seems {complic3} than it was.",
    ],
    contradiction_templates=[
        "Funny thing, though—{contra1} the {contra2} feels {contra3}, like it forgives us.",
        "Yet {contra1} I think the {contra2} becomes {contra3} with distance.",
        "Strange how {contra1} the {contra2} turns {contra3} when we're not looking.",
    ],
    exception_templates=[
        "{except1} we're the ones {except2} to be {except3}.",
        "{except1} {except2} is just {except3} name for trying again.",
        "{except1} we {except2} that what {except3} becomes what waits.",
    ],
    concession_templates=[
        "{concede1}. Or {concede2} forgiveness is just {concede3} name for trying again.",
        "Fair. The {concede1} probably {concede2} differently once {concede3} settles.",
        "I'll admit that. {concede1} and {concede2} become {concede3} eventually.",
    ],
    synthesis_templates=[
        "Could be. Either way, the {synth1} keeps {synth2}.",
        "So {synth1} becomes {synth2}, and that's what remains.",
        "In the end, {synth1} transforms into {synth2}, always.",
    ],
)

ALL_CONVERSATIONAL_DOMAINS = [LEGAL_DOMAIN, PHYSICS_DOMAIN, POETRY_DOMAIN]


class ConversationalDataset:
    """
    Generator for conversational synthetic dataset.

    Creates multi-turn dialogues with phase shifts to force expert divergence.
    """

    def __init__(self, seq_length: int = 128, vocab_size: int = 5000):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.vocab = self._build_vocabulary()
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}

    def _build_vocabulary(self) -> List[str]:
        """Build vocabulary from all domains plus special tokens."""
        vocab = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<TURN>"]

        # Domain markers
        vocab.extend(["<LEGAL>", "<PHYSICS>", "<POETRY>"])

        # Speaker markers
        vocab.extend(["<A>", "<B>"])

        # Collect all domain vocab
        for domain in ALL_CONVERSATIONAL_DOMAINS:
            vocab.extend(domain.opening_vocab)
            vocab.extend(domain.premise_vocab)
            vocab.extend(domain.complication_vocab)
            vocab.extend(domain.contradiction_vocab)
            vocab.extend(domain.exception_vocab)
            vocab.extend(domain.concession_vocab)
            vocab.extend(domain.synthesis_vocab)

        # Common conversation words
        common = [
            "the", "a", "an", "of", "in", "to", "and", "is", "that", "for",
            "with", "as", "by", "on", "at", "from", "this", "which", "are", "be",
            "has", "have", "can", "will", "through", "when", "where", "how", "why",
            "so", "but", "if", "or", "not", "what", "about", "up", "out", "into",
            "right", "well", "yeah", "hmm", "okay", "sure", "fair", "exactly",
            "wait", "unless", "though", "thing", "way", "case", "point", "seem",
        ]
        vocab.extend(common)

        # Pad to vocab_size
        while len(vocab) < self.vocab_size:
            vocab.append(f"<TOKEN_{len(vocab)}>")

        return vocab[:self.vocab_size]

    def _fill_template(self, template: str, vocab_pool: List[str], domain_name: str) -> str:
        """Fill a template with random vocabulary."""
        # Extract placeholders
        import re
        placeholders = re.findall(r'\{(\w+)\}', template)

        # Sample vocab for each unique placeholder prefix
        replacements = {}
        for ph in set(placeholders):
            replacements[ph] = random.choice(vocab_pool)

        # Fill template
        result = template
        for ph, word in replacements.items():
            result = result.replace(f"{{{ph}}}", word)

        return result

    def generate_conversation(self, domain: ConversationalDomain) -> Tuple[List[int], str, str, List[int]]:
        """
        Generate a full 7-turn conversation with phase shifts.

        Structure:
        A: opening inquiry
        B: premise clarification
        A: complication
        B: contradiction
        A: exception
        B: concession
        A: synthesis

        Returns:
            token_ids: List of token IDs
            text: Original text
            domain_name: Domain label
            turn_boundaries: List of token indices where each turn starts
        """

        turns = []

        # Turn 1: A opens with inquiry
        opening = self._fill_template(
            random.choice(domain.opening_templates),
            domain.opening_vocab,
            domain.name
        )
        turns.append(f"<A> {opening}")

        # Turn 2: B responds with premise
        premise = self._fill_template(
            random.choice(domain.premise_templates),
            domain.premise_vocab,
            domain.name
        )
        turns.append(f"<B> {premise}")

        # Turn 3: A introduces complication
        complication = self._fill_template(
            random.choice(domain.complication_templates),
            domain.complication_vocab,
            domain.name
        )
        turns.append(f"<A> {complication}")

        # Turn 4: B pushes contradiction
        contradiction = self._fill_template(
            random.choice(domain.contradiction_templates),
            domain.contradiction_vocab,
            domain.name
        )
        turns.append(f"<B> {contradiction}")

        # Turn 5: A escalates to exception
        exception = self._fill_template(
            random.choice(domain.exception_templates),
            domain.exception_vocab,
            domain.name
        )
        turns.append(f"<A> {exception}")

        # Turn 6: B concedes partially
        concession = self._fill_template(
            random.choice(domain.concession_templates),
            domain.concession_vocab,
            domain.name
        )
        turns.append(f"<B> {concession}")

        # Turn 7: A synthesizes
        synthesis = self._fill_template(
            random.choice(domain.synthesis_templates),
            domain.synthesis_vocab,
            domain.name
        )
        turns.append(f"<A> {synthesis}")

        # Join with turn markers and track boundaries
        # Build text incrementally to track where each turn starts
        text_parts = []
        turn_boundaries = []

        # Domain marker
        text_parts.append(f"<{domain.name.upper()}>")

        # Track each turn
        for i, turn in enumerate(turns):
            if i > 0:
                text_parts.append("<TURN>")

            # Record where this turn starts (in token indices)
            current_tokens = " ".join(text_parts).split()
            turn_boundaries.append(len(current_tokens))

            # Add the turn content
            text_parts.append(turn)

        text = " ".join(text_parts)

        # Tokenize
        tokens = text.split()
        token_ids = [self.token_to_id.get(t, self.token_to_id["<UNK>"]) for t in tokens]

        # Store original boundaries before padding
        original_boundaries = turn_boundaries.copy()

        # Pad or truncate
        if len(token_ids) < self.seq_length:
            token_ids.extend([self.token_to_id["<PAD>"]] * (self.seq_length - len(token_ids)))
        else:
            token_ids = token_ids[:self.seq_length]
            # Clip boundaries that exceed truncated length
            original_boundaries = [b for b in original_boundaries if b < self.seq_length]

        return token_ids, text, domain.name, original_boundaries

    def generate_dataset(
        self,
        num_sequences: int,
        output_path: str = None,
        balanced: bool = True
    ) -> Dict:
        """Generate full conversational dataset."""

        dataset = {
            "sequences": [],
            "domain_labels": [],
            "metadata": {
                "num_sequences": num_sequences,
                "seq_length": self.seq_length,
                "vocab_size": self.vocab_size,
                "domains": [d.name for d in ALL_CONVERSATIONAL_DOMAINS],
                "balanced": balanced,
                "structure": "7-turn conversation with phase shifts"
            }
        }

        if balanced:
            per_domain = num_sequences // len(ALL_CONVERSATIONAL_DOMAINS)
            for domain in ALL_CONVERSATIONAL_DOMAINS:
                for _ in range(per_domain):
                    token_ids, text, domain_label, turn_boundaries = self.generate_conversation(domain)
                    dataset["sequences"].append({
                        "token_ids": token_ids,
                        "text": text,
                        "domain": domain_label,
                        "turn_boundaries": turn_boundaries
                    })
                    dataset["domain_labels"].append(domain_label)
        else:
            for _ in range(num_sequences):
                domain = random.choice(ALL_CONVERSATIONAL_DOMAINS)
                token_ids, text, domain_label, turn_boundaries = self.generate_conversation(domain)
                dataset["sequences"].append({
                    "token_ids": token_ids,
                    "text": text,
                    "domain": domain_label,
                    "turn_boundaries": turn_boundaries
                })
                dataset["domain_labels"].append(domain_label)

        # Shuffle
        combined = list(zip(dataset["sequences"], dataset["domain_labels"]))
        random.shuffle(combined)
        dataset["sequences"], dataset["domain_labels"] = zip(*combined)
        dataset["sequences"] = list(dataset["sequences"])
        dataset["domain_labels"] = list(dataset["domain_labels"])

        # Save if requested
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(dataset, f, indent=2)
            print(f"Dataset saved to {output_path}")

        return dataset

    def print_examples(self, num_examples: int = 2):
        """Print example conversations."""
        print("=" * 70)
        print("CONVERSATIONAL SYNTHETIC DATASET EXAMPLES")
        print("=" * 70)

        for domain in ALL_CONVERSATIONAL_DOMAINS:
            print(f"\n{domain.name.upper()} Domain:")
            print("-" * 70)

            for i in range(num_examples):
                _, text, _, boundaries = self.generate_conversation(domain)
                print(f"\nExample {i+1}:")
                print(f"  Turn boundaries: {boundaries}")
                # Pretty print the conversation
                turns = text.split("<TURN>")
                for turn in turns:
                    turn = turn.strip()
                    if turn:
                        print(f"  {turn}")


if __name__ == "__main__":
    # Create dataset generator
    dataset_gen = ConversationalDataset(seq_length=128, vocab_size=1000)

    # Print examples
    dataset_gen.print_examples(num_examples=2)

    # Generate dataset
    print("\n\nGenerating dataset...")
    dataset = dataset_gen.generate_dataset(
        num_sequences=900,
        output_path="data/conversational_dataset.json",
        balanced=True
    )

    print(f"\nDataset created:")
    print(f"  Total: {len(dataset['sequences'])} sequences")
    print(f"  Domains: {dataset['metadata']['domains']}")
    print(f"  Vocab size: {dataset['metadata']['vocab_size']}")
    print(f"  Sequence length: {dataset['metadata']['seq_length']}")
