"""
3-Domain Synthetic Dataset for ChronoMoE Valley Formation

Generates clean, interpretable sequences across three domains:
- Legal: precedent, jurisdiction, tort, statute
- Physics: momentum, entanglement, quantum, relativity
- Poetry: sonnet, metaphor, verse, rhyme

Each domain has distinct vocabulary and sentence patterns.
This makes valley formation interpretable: we can verify that
legal experts ≠ physics experts ≠ poetry experts.
"""

import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class Domain:
    """A semantic domain with vocabulary and templates."""
    name: str
    vocab: List[str]
    templates: List[str]
    color: str  # For visualization


# Define the three domains
LEGAL_DOMAIN = Domain(
    name="legal",
    vocab=[
        "precedent", "jurisdiction", "tort", "statute", "plaintiff", "defendant",
        "liability", "negligence", "contract", "remedy", "damages", "court",
        "appeal", "ruling", "evidence", "testimony", "judge", "jury",
        "prosecution", "defense", "verdict", "sentence", "law", "legal",
        "doctrine", "principle", "case", "decision", "opinion", "brief"
    ],
    templates=[
        "The {legal1} establishes that {legal2} requires {legal3} in cases of {legal4}.",
        "Legal {legal1} indicates that {legal2} must prove {legal3} before {legal4}.",
        "Under {legal1}, the {legal2} bears responsibility for {legal3} and {legal4}.",
        "The court held that {legal1} supersedes {legal2} when {legal3} is evident.",
        "Judicial {legal1} requires examination of {legal2} alongside {legal3} and {legal4}.",
        "In matters of {legal1}, the {legal2} must demonstrate {legal3} beyond {legal4}.",
        "The {legal1} argued that {legal2} violated {legal3} through {legal4}.",
        "Legal {legal1} establishes clear {legal2} when {legal3} affects {legal4}."
    ],
    color="blue"
)

PHYSICS_DOMAIN = Domain(
    name="physics",
    vocab=[
        "momentum", "entanglement", "quantum", "relativity", "energy", "mass",
        "velocity", "acceleration", "force", "field", "particle", "wave",
        "photon", "electron", "proton", "atom", "nucleus", "radiation",
        "gravity", "spacetime", "dimension", "symmetry", "conservation", "entropy",
        "thermodynamics", "mechanics", "electromagnetic", "wavelength", "frequency", "amplitude"
    ],
    templates=[
        "The {physics1} of {physics2} demonstrates that {physics3} depends on {physics4}.",
        "Quantum {physics1} shows that {physics2} influences {physics3} through {physics4}.",
        "Conservation of {physics1} implies that {physics2} equals {physics3} plus {physics4}.",
        "The {physics1} field interacts with {physics2} to produce {physics3} and {physics4}.",
        "Relativistic {physics1} requires that {physics2} transforms according to {physics3} in {physics4}.",
        "The {physics1} exhibits {physics2} behavior when {physics3} exceeds {physics4}.",
        "Electromagnetic {physics1} propagates through {physics2} with {physics3} determined by {physics4}.",
        "The {physics1} principle states that {physics2} remains constant despite {physics3} and {physics4}."
    ],
    color="green"
)

POETRY_DOMAIN = Domain(
    name="poetry",
    vocab=[
        "sonnet", "metaphor", "verse", "rhyme", "stanza", "imagery",
        "rhythm", "meter", "alliteration", "assonance", "simile", "symbol",
        "lyric", "ode", "elegy", "ballad", "haiku", "epic",
        "prose", "narrative", "theme", "tone", "mood", "diction",
        "syntax", "form", "structure", "voice", "perspective", "expression"
    ],
    templates=[
        "The {poetry1} captures {poetry2} through {poetry3} and {poetry4}.",
        "Poetic {poetry1} evokes {poetry2} by weaving {poetry3} with {poetry4}.",
        "In this {poetry1}, the {poetry2} reflects {poetry3} against {poetry4}.",
        "The {poetry1} employs {poetry2} to convey {poetry3} through {poetry4}.",
        "Literary {poetry1} intertwines {poetry2} and {poetry3} within {poetry4}.",
        "The {poetry1} resonates with {poetry2} while {poetry3} echoes {poetry4}.",
        "Through {poetry1}, the poet expresses {poetry2} using {poetry3} and {poetry4}.",
        "The {poetry1} reveals {poetry2} as {poetry3} transforms into {poetry4}."
    ],
    color="red"
)

ALL_DOMAINS = [LEGAL_DOMAIN, PHYSICS_DOMAIN, POETRY_DOMAIN]


class ThreeDomainDataset:
    """
    Generator for 3-domain synthetic dataset.

    Creates sequences with clear domain labels and vocabulary.
    """

    def __init__(self, seq_length: int = 32, vocab_size: int = 5000):
        """
        Initialize dataset generator.

        Args:
            seq_length: Number of tokens per sequence
            vocab_size: Total vocabulary size
        """
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        # Build vocabulary
        self.vocab = self._build_vocabulary()
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}

        # Domain markers
        self.domain_markers = {
            "legal": "<LEGAL>",
            "physics": "<PHYSICS>",
            "poetry": "<POETRY>"
        }

    def _build_vocabulary(self) -> List[str]:
        """Build complete vocabulary from all domains plus special tokens."""
        vocab = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

        # Add domain markers
        vocab.extend(["<LEGAL>", "<PHYSICS>", "<POETRY>"])

        # Add domain-specific vocab
        for domain in ALL_DOMAINS:
            vocab.extend(domain.vocab)

        # Add common words
        common = [
            "the", "a", "an", "of", "in", "to", "and", "is", "that", "for",
            "with", "as", "by", "on", "at", "from", "this", "which", "are", "be",
            "has", "have", "can", "will", "through", "when", "where", "how", "why"
        ]
        vocab.extend(common)

        # Pad to vocab_size with generated tokens
        while len(vocab) < self.vocab_size:
            vocab.append(f"<TOKEN_{len(vocab)}>")

        return vocab[:self.vocab_size]

    def generate_sequence(self, domain: Domain) -> Tuple[List[int], str, str]:
        """
        Generate a single sequence from a domain.

        Args:
            domain: Which domain to generate from

        Returns:
            Tuple of (token_ids, text, domain_name)
        """
        # Start with domain marker
        text = f"{self.domain_markers[domain.name]} "

        # Choose a random template
        template = random.choice(domain.templates)

        # Fill in the template with random vocab
        vocab_subset = random.sample(domain.vocab, min(4, len(domain.vocab)))
        filled = template.format(
            **{f"{domain.name}1": vocab_subset[0],
               f"{domain.name}2": vocab_subset[1],
               f"{domain.name}3": vocab_subset[2],
               f"{domain.name}4": vocab_subset[3]}
        )

        text += filled

        # Tokenize
        tokens = text.split()
        token_ids = [self.token_to_id.get(t, self.token_to_id["<UNK>"]) for t in tokens]

        # Pad or truncate to seq_length
        if len(token_ids) < self.seq_length:
            token_ids.extend([self.token_to_id["<PAD>"]] * (self.seq_length - len(token_ids)))
        else:
            token_ids = token_ids[:self.seq_length]

        return token_ids, text, domain.name

    def generate_batch(
        self,
        batch_size: int,
        domain: str = None
    ) -> Tuple[List[List[int]], List[str], List[str]]:
        """
        Generate a batch of sequences.

        Args:
            batch_size: Number of sequences
            domain: Specific domain, or None for mixed

        Returns:
            Tuple of (token_ids_batch, texts, domain_labels)
        """
        token_ids_batch = []
        texts = []
        domain_labels = []

        for _ in range(batch_size):
            if domain is None:
                # Mixed batch - choose random domain
                chosen_domain = random.choice(ALL_DOMAINS)
            else:
                # Specific domain
                chosen_domain = next(d for d in ALL_DOMAINS if d.name == domain)

            token_ids, text, domain_label = self.generate_sequence(chosen_domain)
            token_ids_batch.append(token_ids)
            texts.append(text)
            domain_labels.append(domain_label)

        return token_ids_batch, texts, domain_labels

    def generate_dataset(
        self,
        num_sequences: int,
        output_path: str = None,
        balanced: bool = True
    ) -> Dict:
        """
        Generate full dataset.

        Args:
            num_sequences: Total number of sequences
            output_path: Where to save (optional)
            balanced: If True, equal samples per domain

        Returns:
            Dataset dictionary
        """
        dataset = {
            "sequences": [],
            "domain_labels": [],
            "metadata": {
                "num_sequences": num_sequences,
                "seq_length": self.seq_length,
                "vocab_size": self.vocab_size,
                "domains": [d.name for d in ALL_DOMAINS],
                "balanced": balanced
            }
        }

        if balanced:
            # Equal samples per domain
            per_domain = num_sequences // len(ALL_DOMAINS)
            for domain in ALL_DOMAINS:
                for _ in range(per_domain):
                    token_ids, text, domain_label = self.generate_sequence(domain)
                    dataset["sequences"].append({
                        "token_ids": token_ids,
                        "text": text,
                        "domain": domain_label
                    })
                    dataset["domain_labels"].append(domain_label)
        else:
            # Random sampling
            for _ in range(num_sequences):
                domain = random.choice(ALL_DOMAINS)
                token_ids, text, domain_label = self.generate_sequence(domain)
                dataset["sequences"].append({
                    "token_ids": token_ids,
                    "text": text,
                    "domain": domain_label
                })
                dataset["domain_labels"].append(domain_label)

        # Shuffle
        import random
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

    def print_examples(self, num_examples: int = 3):
        """Print example sequences from each domain."""
        print("=" * 70)
        print("3-DOMAIN SYNTHETIC DATASET EXAMPLES")
        print("=" * 70)

        for domain in ALL_DOMAINS:
            print(f"\n{domain.name.upper()} (color: {domain.color}):")
            print("-" * 70)

            for i in range(num_examples):
                token_ids, text, _ = self.generate_sequence(domain)
                print(f"  {i+1}. {text}")

        print("\n" + "=" * 70)


def create_train_val_split(
    dataset: Dict,
    val_ratio: float = 0.1
) -> Tuple[Dict, Dict]:
    """Split dataset into train and validation."""
    num_val = int(len(dataset["sequences"]) * val_ratio)

    train_dataset = {
        "sequences": dataset["sequences"][num_val:],
        "domain_labels": dataset["domain_labels"][num_val:],
        "metadata": dataset["metadata"].copy()
    }
    train_dataset["metadata"]["split"] = "train"

    val_dataset = {
        "sequences": dataset["sequences"][:num_val],
        "domain_labels": dataset["domain_labels"][:num_val],
        "metadata": dataset["metadata"].copy()
    }
    val_dataset["metadata"]["split"] = "val"

    return train_dataset, val_dataset


if __name__ == "__main__":
    # Create dataset generator
    dataset_gen = ThreeDomainDataset(seq_length=32, vocab_size=1000)

    # Print examples
    dataset_gen.print_examples(num_examples=3)

    # Generate small dataset for quick testing
    print("\nGenerating dataset...")
    dataset = dataset_gen.generate_dataset(
        num_sequences=1000,
        output_path="data/three_domain_dataset.json",
        balanced=True
    )

    # Split into train/val
    train_data, val_data = create_train_val_split(dataset, val_ratio=0.1)

    print(f"\nDataset created:")
    print(f"  Train: {len(train_data['sequences'])} sequences")
    print(f"  Val: {len(val_data['sequences'])} sequences")
    print(f"  Domains: {dataset['metadata']['domains']}")
    print(f"  Vocab size: {dataset['metadata']['vocab_size']}")
