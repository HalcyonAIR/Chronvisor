"""
Long Geeky Conversation Generator

Generates full conversational trajectories (500-1000 tokens) about geeky topics
instead of short fragments. Lets geology track conversational flow.

Topics:
- Compiler design (LLVM IR, SSA form, register allocation)
- Type systems (Hindley-Milner, dependent types, linear types)
- Distributed systems (Raft consensus, vector clocks, CRDTs)
- Functional programming (monads, algebraic effects, lazy evaluation)
"""

import numpy as np
from typing import List, Dict, Tuple


class LongGeekyConversationGenerator:
    """Generate full conversational arcs about geeky topics."""

    def __init__(self, vocab_size: int = 1000, min_length: int = 500, max_length: int = 1000):
        self.vocab_size = vocab_size
        self.min_length = min_length
        self.max_length = max_length

        # Conversational turns with geeky flavors
        self.turn_templates = {
            "Inquiry": [
                # Opening questions about technical topics
                "What's your take on {topic}? I've been diving into {subtopic} lately.",
                "I'm curious about {topic} - specifically how {subtopic} actually works under the hood.",
                "Can we talk about {topic}? There's something about {subtopic} that's been bugging me.",
            ],
            "Premise": [
                # Setting up technical context
                "So the core idea behind {topic} is that {principle}. The {subtopic} layer handles {function}.",
                "Here's how I understand {topic}: {principle}, and {subtopic} is basically {function}.",
                "The foundation of {topic} is {principle}. That's why {subtopic} needs to {function}.",
            ],
            "Complication": [
                # Introducing technical challenges
                "But here's where it gets tricky - {subtopic} has to deal with {problem}, and that breaks {assumption}.",
                "The issue is {subtopic} creates {problem}, which means {assumption} doesn't hold anymore.",
                "Right, except {subtopic} introduces {problem}, so you can't just {assumption}.",
            ],
            "Contradiction": [
                # Presenting alternative views
                "Actually, I think {alternative} is a better approach. {subtopic} doesn't need to {function} if you {method}.",
                "Wait - what if {alternative}? Then {subtopic} could avoid {problem} entirely by {method}.",
                "Hold on, {alternative} solves this differently. {subtopic} just {method} instead.",
            ],
            "Exception": [
                # Edge cases and special scenarios
                "True, but there's an edge case: when {condition}, {subtopic} has to fall back to {fallback}.",
                "Fair point, though if {condition} happens, {subtopic} can't {function} - it needs {fallback}.",
                "Agreed, except in cases where {condition}. Then {subtopic} reverts to {fallback}.",
            ],
            "Concession": [
                # Acknowledging tradeoffs
                "You're right about {alternative}, but the tradeoff is {cost}. {subtopic} optimizes for {priority}.",
                "I see your point on {alternative}. The cost is {cost}, but {subtopic} prioritizes {priority}.",
                "Yeah, {alternative} would work, at the expense of {cost}. {subtopic} chooses {priority} instead.",
            ],
            "Synthesis": [
                # Resolving and integrating
                "So the full picture is: {topic} uses {principle}, {subtopic} handles {function}, and when {condition} you {method}.",
                "Putting it together: {topic} is based on {principle}. {subtopic} manages {function}, with {method} for {condition}.",
                "The complete model is {topic} → {principle} → {subtopic} does {function}, falling back to {method} when {condition}.",
            ],
        }

        # Technical content pools
        self.topics = [
            # Compiler topics
            ("LLVM IR", "intermediate representation", "code generation", "type safety",
             "register pressure", "SSA form", "phi nodes", "dominance frontiers"),
            ("type inference", "Hindley-Milner", "unification", "polymorphism",
             "occurs check", "let-polymorphism", "type schemes", "principal types"),
            ("garbage collection", "mark-and-sweep", "generational GC", "write barriers",
             "heap fragmentation", "reference counting", "cycle detection", "finalization"),

            # Distributed systems topics
            ("Raft consensus", "leader election", "log replication", "linearizability",
             "network partitions", "quorum writes", "term numbers", "heartbeat timeouts"),
            ("vector clocks", "causal ordering", "concurrent updates", "happens-before",
             "version vectors", "conflict resolution", "eventual consistency", "merge semantics"),
            ("CRDTs", "commutative operations", "state-based replication", "convergence",
             "tombstones", "merge functions", "operation-based CRDTs", "causal stability"),

            # Type system topics
            ("linear types", "affine types", "resource tracking", "use-once semantics",
             "uniqueness", "ownership transfer", "borrowing", "lifetime annotations"),
            ("dependent types", "type-level computation", "proof terms", "normalization",
             "decidability", "universe levels", "indexed families", "equality types"),
            ("effect systems", "algebraic effects", "handler semantics", "delimited continuations",
             "monadic style", "effect rows", "capability tracking", "resumption"),
        ]

        self.principles = [
            "static single assignment",
            "referential transparency",
            "eventual consistency",
            "type soundness",
            "monotonicity",
            "commutativity",
            "idempotence",
            "linearizability",
        ]

        self.functions = [
            "guarantee correctness",
            "eliminate redundancy",
            "preserve invariants",
            "track dependencies",
            "resolve conflicts",
            "maintain ordering",
            "enforce uniqueness",
            "optimize throughput",
        ]

        self.problems = [
            "race conditions",
            "cyclic dependencies",
            "unbounded growth",
            "non-determinism",
            "divergence",
            "aliasing",
            "inconsistency",
            "deadlock",
        ]

        self.alternatives = [
            "lazy evaluation",
            "continuation-passing style",
            "monadic composition",
            "persistent data structures",
            "copy-on-write semantics",
            "weak references",
            "optimistic locking",
            "compensating transactions",
        ]

        self.methods = [
            "use a phase distinction",
            "apply normalization by evaluation",
            "introduce indirection",
            "cache intermediate results",
            "batch operations",
            "employ backtracking",
            "leverage structural recursion",
            "defer evaluation",
        ]

        self.conditions = [
            "polymorphic recursion is involved",
            "the heap is nearly full",
            "network latency spikes",
            "types are inhabited",
            "references escape scope",
            "mutations are concurrent",
            "invariants are violated",
            "resources are exhausted",
        ]

        self.costs = [
            "higher memory overhead",
            "increased latency",
            "worse cache locality",
            "runtime type checks",
            "verbose annotations",
            "slower compilation",
            "complex implementation",
            "reduced expressiveness",
        ]

        self.priorities = [
            "correctness over performance",
            "simplicity over generality",
            "predictability over optimization",
            "safety over expressiveness",
            "composability over efficiency",
            "determinism over parallelism",
            "explicitness over inference",
            "robustness over minimality",
        ]

    def generate_conversation(self) -> Tuple[List[int], List[int]]:
        """Generate a full conversational trajectory (500-1000 tokens)."""

        # Pick a random geeky topic cluster
        topic_cluster = self.topics[np.random.randint(len(self.topics))]
        topic = topic_cluster[0]
        subtopic = np.random.choice(topic_cluster[1:])

        # Fill in technical details
        principle = np.random.choice(self.principles)
        function = np.random.choice(self.functions)
        problem = np.random.choice(self.problems)
        alternative = np.random.choice(self.alternatives)
        method = np.random.choice(self.methods)
        condition = np.random.choice(self.conditions)
        fallback = np.random.choice(self.methods)
        cost = np.random.choice(self.costs)
        priority = np.random.choice(self.priorities)

        # Generate full conversational arc
        turn_types = ["Inquiry", "Premise", "Complication", "Contradiction",
                      "Exception", "Concession", "Synthesis"]

        conversation_tokens = []
        turn_boundaries = [0]

        for turn_type in turn_types:
            # Pick a template for this turn
            template = np.random.choice(self.turn_templates[turn_type])

            # Fill in the template
            utterance = template.format(
                topic=topic,
                subtopic=subtopic,
                principle=principle,
                function=function,
                problem=problem,
                alternative=alternative,
                method=method,
                condition=condition,
                fallback=fallback,
                cost=cost,
                priority=priority,
                assumption=f"assume {principle}",
            )

            # Tokenize (simplified: words → random vocab indices)
            # In real system, this would use actual tokenizer
            words = utterance.split()
            turn_tokens = [np.random.randint(self.vocab_size) for _ in words]

            # Add padding to reach target length
            # Each turn gets ~70-150 tokens to reach 500-1000 total
            target_turn_length = np.random.randint(70, 150)
            while len(turn_tokens) < target_turn_length:
                turn_tokens.append(np.random.randint(self.vocab_size))

            conversation_tokens.extend(turn_tokens)
            turn_boundaries.append(len(conversation_tokens))

        # Ensure we're in target length range
        while len(conversation_tokens) < self.min_length:
            conversation_tokens.append(np.random.randint(self.vocab_size))

        conversation_tokens = conversation_tokens[:self.max_length]

        # Create labels (next-token prediction)
        labels = conversation_tokens[1:] + [conversation_tokens[0]]  # Shifted

        return conversation_tokens, labels, turn_boundaries[:-1]  # Exclude final boundary

    def generate_dataset(self, num_conversations: int = 10) -> Dict:
        """Generate a dataset of full conversations."""

        sequences = []

        for i in range(num_conversations):
            tokens, labels, turn_boundaries = self.generate_conversation()

            sequences.append({
                "input_ids": np.array(tokens, dtype=np.int64),
                "labels": np.array(labels, dtype=np.int64),
                "turn_boundaries": turn_boundaries,
                "length": len(tokens),
            })

        return {
            "sequences": sequences,
            "vocab_size": self.vocab_size,
            "num_conversations": num_conversations,
            "avg_length": np.mean([s["length"] for s in sequences]),
        }


def main():
    """Test the generator."""
    print("="*70)
    print("LONG GEEKY CONVERSATION GENERATOR")
    print("="*70)

    gen = LongGeekyConversationGenerator(vocab_size=1000, min_length=500, max_length=1000)

    print("\nGenerating 5 sample conversations...\n")

    dataset = gen.generate_dataset(num_conversations=5)

    print(f"Dataset generated:")
    print(f"  Conversations: {dataset['num_conversations']}")
    print(f"  Vocab size: {dataset['vocab_size']}")
    print(f"  Avg length: {dataset['avg_length']:.1f} tokens")
    print(f"\nConversation lengths:")

    for i, seq in enumerate(dataset["sequences"]):
        print(f"  Conv {i+1}: {seq['length']} tokens, {len(seq['turn_boundaries'])} turns")
        print(f"    Turn boundaries: {seq['turn_boundaries']}")

    print(f"\n✅ Generator working! Ready for geological flow tracking.")


if __name__ == "__main__":
    main()
