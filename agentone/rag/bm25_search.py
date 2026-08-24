"""Okapi BM25 sparse keyword ranking engine for exact and lexical term matching."""

import math
import re
from typing import Dict, List, Optional, Tuple
from collections import Counter
from agentone.core.state import DocumentChunk


def _normalize_token(w: str) -> str:
    """Normalize word endings for robust matching."""
    s = w.lower().strip("$%.,!?:;'\"()[]")
    if len(s) > 4 and s.endswith("ies"):
        return s[:-3] + "y"
    if len(s) > 4 and s.endswith("es"):
        return s[:-2]
    if len(s) > 3 and s.endswith("s") and not s.endswith("ss"):
        return s[:-1]
    if len(s) > 5 and s.endswith("ing"):
        return s[:-3]
    if len(s) > 4 and s.endswith("ed"):
        return s[:-2]
    return s


def tokenize(text: str) -> List[str]:
    """Lowercase, tokenize, and normalize alphanumeric word tokens."""
    raw = re.findall(r"\b[a-zA-Z0-9$%]{2,}\b", text.lower())
    return [_normalize_token(w) for w in raw if len(_normalize_token(w)) >= 2]


class BM25Index:
    """Okapi BM25 search index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count: int = 0
        self.avg_doc_len: float = 0.0
        self.documents: Dict[str, DocumentChunk] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.term_freqs: Dict[str, Counter] = {}
        self.doc_freqs: Dict[str, int] = {}

    def add_documents(self, documents: List[DocumentChunk]):
        """Index a collection of document chunks."""
        for doc in documents:
            tokens = tokenize(doc.content)
            doc_len = len(tokens)
            self.documents[doc.doc_id] = doc
            self.doc_lengths[doc.doc_id] = max(1, doc_len)
            tf = Counter(tokens)
            self.term_freqs[doc.doc_id] = tf

            for term in tf.keys():
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.doc_count = len(self.documents)
        if self.doc_count > 0:
            self.avg_doc_len = sum(self.doc_lengths.values()) / float(self.doc_count)

    def _idf(self, term: str) -> float:
        """Calculate Robertson-Spärck Jones Inverse Document Frequency with smoothing."""
        df = self.doc_freqs.get(term, 0)
        return math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))

    def search(
        self,
        query: str,
        top_k: int = 4,
        category_filter: Optional[str] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Score and rank indexed documents using BM25 formula."""
        if not self.documents or self.doc_count == 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: List[Tuple[DocumentChunk, float]] = []

        for doc_id, doc in self.documents.items():
            if category_filter and doc.category.lower() != category_filter.lower():
                continue

            doc_len = self.doc_lengths[doc_id]
            tf = self.term_freqs[doc_id]
            score = 0.0

            for q_term in query_tokens:
                if q_term not in tf:
                    continue
                term_freq = tf[q_term]
                idf = self._idf(q_term)
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf * (numerator / denominator)

            if score > 0.0:
                doc_copy = doc.model_copy()
                doc_copy.sparse_score = round(score, 4)
                scores.append((doc_copy, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def count(self) -> int:
        """Return total indexed documents count."""
        return self.doc_count
