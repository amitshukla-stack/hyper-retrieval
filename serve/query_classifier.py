"""
Query type classifier for HR's dual-mode retrieval router.

ARCHITECTURAL queries can be answered by BM25 + co-change graph alone (zero GPU).
SEMANTIC queries require embedding similarity.

T-009 finding: 42% of chat_50_questions.json are ARCHITECTURAL, 58% SEMANTIC.
This classifier routes the 42% away from the embedding path when HR_LIGHTWEIGHT_MODE=1.
"""
import os
import re

_ARCH_VERBS = re.compile(
    r'\b(blast.?radius|co.?change|chang\w+|affect\w*|break\w*|impact\w*|'
    r'own\w*|maintai\w*|critic\w*|depend\w*|coupl\w*|cascad\w*|propagat\w*|'
    r'rippl\w*|missing|predict\w*|histor\w*|evolv\w*|touch\w*|modif\w*|'
    r'who.?(review|wrote|owns?)|reviewer|author)\b',
    re.IGNORECASE,
)

_ARCH_PHRASES = re.compile(
    r'(what (else |other |modules? )?(change|break|need|affect|update))'
    r'|(if i (change|modify|update|remove|delete|rename))'
    r'|(what (happen|break)s? (if|when))'
    r'|(blast radius|change risk|change prediction|missing change'
    r'|what to review|who should review)',
    re.IGNORECASE,
)

_SEMANTIC_INDICATORS = re.compile(
    r'\b(how does|explain|what is|what does|show me|find|where is|'
    r'implement\w*|work\w*|example|pattern|flow|logic|design|architecture'
    r' of|communicat\w*|integrat\w*)\b',
    re.IGNORECASE,
)


_MECHANISM_OVERRIDE = re.compile(
    r'^(how does|how do|explain how|what is the (design|implementation|logic|algorithm|mechanism))',
    re.IGNORECASE,
)


def classify_query(query: str) -> str:
    """Return 'ARCHITECTURAL' or 'SEMANTIC' for a query string."""
    # "How does X work?" asks about mechanism → always SEMANTIC
    if _MECHANISM_OVERRIDE.match(query.strip()):
        return "SEMANTIC"
    if _ARCH_PHRASES.search(query):
        return "ARCHITECTURAL"
    arch_hits = len(_ARCH_VERBS.findall(query))
    sem_hits = len(_SEMANTIC_INDICATORS.findall(query))
    if arch_hits > sem_hits:
        return "ARCHITECTURAL"
    return "SEMANTIC"


def is_lightweight_mode() -> bool:
    """True when HR_LIGHTWEIGHT_MODE=1 is set (no GPU available)."""
    return os.getenv("HR_LIGHTWEIGHT_MODE", "0") == "1"


def should_skip_embeddings(query: str) -> bool:
    """
    True when the query is ARCHITECTURAL and we're in lightweight mode.
    In this case unified_search can skip vector lookup entirely and rely
    on BM25 + co-change expansion, which needs zero GPU.
    """
    return is_lightweight_mode() and classify_query(query) == "ARCHITECTURAL"
