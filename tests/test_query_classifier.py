"""Tests for query_classifier — no GPU, no embed server needed."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from serve.query_classifier import classify_query, should_skip_embeddings

# --- ARCHITECTURAL ground truth (from T-009 classification) ---
ARCH_QUERIES = [
    "If I change the gateway routing logic, what else is affected?",
    "What is the blast radius of changing the payment status enum?",
    "If I modify the order data type, what tests and modules need updating?",
    "Who owns the authentication module?",
    "What other modules co-change with euler_api_gateway?",
    "Predict what else needs to change if I update the lock acquisition code.",
    "What is the change risk of modifying the transaction boundary logic?",
    "Which modules historically change together with this file?",
]

# --- SEMANTIC ground truth ---
SEMANTIC_QUERIES = [
    "What modules handle UPI payments?",
    "How does the order creation flow work end to end?",
    "How do the microservices communicate with each other?",
    "Show me an example of a retry pattern in the codebase.",
    "Explain how the embedding server works.",
    "Where is the function that validates payment amounts?",
    "What is the design pattern used for connector plugins?",
    "How does the blast radius calculation work?",  # explains mechanism, not impact
]


def test_architectural_queries_classified_correctly():
    wrong = [q for q in ARCH_QUERIES if classify_query(q) != "ARCHITECTURAL"]
    assert not wrong, f"Misclassified as SEMANTIC: {wrong}"


def test_semantic_queries_classified_correctly():
    wrong = [q for q in SEMANTIC_QUERIES if classify_query(q) != "SEMANTIC"]
    assert not wrong, f"Misclassified as ARCHITECTURAL: {wrong}"


def test_skip_embeddings_off_by_default():
    os.environ.pop("HR_LIGHTWEIGHT_MODE", None)
    assert not should_skip_embeddings("What changes with X?")


def test_skip_embeddings_on_in_lightweight_mode():
    os.environ["HR_LIGHTWEIGHT_MODE"] = "1"
    assert should_skip_embeddings("If I change X, what else is affected?")
    os.environ.pop("HR_LIGHTWEIGHT_MODE")


def test_skip_embeddings_false_for_semantic_even_in_lightweight():
    os.environ["HR_LIGHTWEIGHT_MODE"] = "1"
    assert not should_skip_embeddings("How does the payment flow work?")
    os.environ.pop("HR_LIGHTWEIGHT_MODE")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
