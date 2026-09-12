"""Hybrid retrieval over past (incoming, reply) pairs.

Semantic: Gemini/OpenAI embeddings + cosine similarity (numpy).
Lexical:   token-overlap score (captures order numbers, SKUs, product names —
           exact tokens that embeddings blur).

hybrid_score = w_sem * cos + w_lex * lexical   (w_sem = w_lex = 0.5 by default)

Embeddings are cached to disk (data/cache/embeddings.npz) so repeated runs
cost zero API calls. A pure-Python TF-IDF is used as lexical feature so the
retriever has no heavyweight dependencies.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..llm import embed_gemini, embed_openai

DATA = Path(__file__).resolve().parents[3] / "data"
CACHE = DATA / "cache"

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Light stopwords: keep domain tokens like "order", "refund", "size".
_STOP = {"the", "a", "an", "and", "or", "to", "of", "i", "my", "is", "it",
         "in", "on", "for", "was", "be", "have", "has", "do", "does", "you",
         "your", "we", "our", "this", "that", "with", "at", "as", "am", "me"}


def tokenize(text: str) -> list[str]:
    toks = _TOKEN_RE.findall(text.lower())
    return [t for t in toks if t not in _STOP and len(t) > 1]


@dataclass
class Retrieved:
    id: str
    intent: str
    score: float
    sem_score: float
    lex_score: float
    subject: str
    incoming: str
    reply: str
    facts: dict
    kb_anchors: list
    register: str


class HybridRetriever:
    def __init__(self, pairs: list[dict], w_sem: float = 0.5, embeddings: np.ndarray | None = None,
                 backend: str = "gemini"):
        self.pairs = pairs
        self.w_sem = w_sem
        self.backend = backend
        self.ids = [p["id"] for p in pairs]
        self.texts = [f"{p['subject']}\n{p['incoming']}" for p in pairs]
        # TF-IDF over corpus
        self.df = Counter()
        doc_toks = []
        for t in self.texts:
            toks = tokenize(t)
            doc_toks.append(toks)
            self.df.update(set(toks))
        self.doc_toks = doc_toks
        self.N = len(pairs)
        self.tfidf = [self._tfidf_vector(toks) for toks in doc_toks]
        self.embeddings = embeddings

    def _tfidf_vector(self, toks: list[str]) -> dict[str, float]:
        tf = Counter(toks)
        n = max(len(toks), 1)
        vec = {}
        for tok, c in tf.items():
            idf = math.log((self.N + 1) / (self.df.get(tok, 0) + 1)) + 1.0
            vec[tok] = (c / n) * idf
        return vec

    def _lex_score(self, query: str, idx: int) -> float:
        q_vec = self._tfidf_vector(tokenize(query))
        d_vec = self.tfidf[idx]
        num = sum(w * d_vec.get(tok, 0.0) for tok, w in q_vec.items())
        den = math.sqrt(sum(w * w for w in q_vec.values())) * \
              math.sqrt(sum(w * w for w in d_vec.values())) or 1.0
        return num / den

    async def ensure_embeddings(self, cache_name: str = "train_embeddings.npz"):
        if self.embeddings is not None:
            return
        CACHE.mkdir(parents=True, exist_ok=True)
        path = CACHE / cache_name
        if path.exists():
            data = np.load(path)
            if data["ids"].tolist() == self.ids:
                self.embeddings = data["vecs"]
                return
        if self.backend == "gemini":
            vecs = await embed_gemini(self.texts)
        else:
            vecs = await embed_openai(self.texts)
        arr = np.array(vecs, dtype=np.float32)
        arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
        np.savez(path, ids=np.array(self.ids), vecs=arr)
        self.embeddings = arr

    async def search(self, query: str, k: int = 6, embed_query: bool = True) -> list[Retrieved]:
        await self.ensure_embeddings()
        q_toks = tokenize(query)
        if embed_query and self.embeddings is not None:
            if self.backend == "gemini":
                q_vec = np.array((await embed_gemini([query]))[0], dtype=np.float32)
            else:
                q_vec = np.array((await embed_openai([query]))[0], dtype=np.float32)
            q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)
            sem = self.embeddings @ q_vec
        else:
            sem = np.zeros(self.N)
        lex = np.array([self._lex_score(query, i) for i in range(self.N)])
        hybrid = self.w_sem * sem + (1 - self.w_sem) * lex
        order = np.argsort(-hybrid)[:k]
        out = []
        for i in order:
            p = self.pairs[i]
            out.append(Retrieved(
                id=p["id"], intent=p["intent"], score=float(hybrid[i]),
                sem_score=float(sem[i]), lex_score=float(lex[i]),
                subject=p["subject"], incoming=p["incoming"], reply=p["reply"],
                facts=p["facts"], kb_anchors=p["kb_anchors"], register=p["register"],
            ))
        return out


def load_pairs(path: Path) -> list[dict]:
    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs
