"""
DataMind AI — Hybrid Retriever + Reranker
==========================================
Combines semantic search (ChromaDB) + BM25 keyword search
for best-of-both-worlds retrieval, then reranks with Cohere.

Pipeline:
  1. Semantic search  → top-k candidates
  2. BM25 search      → top-k candidates
  3. Reciprocal Rank Fusion → merged ranked list
  4. Cohere Reranker  → final top-n (optional)
"""

import os
import math
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    In-memory BM25 keyword retriever.
    Built from the same chunks as the vector store.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1     = k1
        self.b      = b
        self.corpus = []
        self.index  = {}
        self.avgdl  = 0
        self._built = False

    def build(self, chunks: list) -> None:
        """Build BM25 index from chunks."""
        self.corpus = chunks
        tokenized   = [self._tokenize(c.content) for c in chunks]
        self.avgdl  = sum(len(t) for t in tokenized) / max(len(tokenized), 1)

        # Build inverted index: term → {doc_idx: tf}
        self.index = defaultdict(dict)
        for doc_idx, tokens in enumerate(tokenized):
            tf = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            for token, freq in tf.items():
                self.index[token][doc_idx] = freq

        self._built = True
        logger.info(f"BM25 index built: {len(chunks)} docs, {len(self.index)} terms")

    def _tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r'\b[a-z]{2,}\b', text.lower())

    def search(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        if not self._built:
            logger.warning("BM25 index not built. Call build() first.")
            return []

        query_tokens = self._tokenize(query)
        scores       = defaultdict(float)
        N            = len(self.corpus)

        for token in query_tokens:
            if token not in self.index:
                continue
            df  = len(self.index[token])
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

            for doc_idx, tf in self.index[token].items():
                doc_len = len(self._tokenize(self.corpus[doc_idx].content))
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                )
                scores[doc_idx] += idf * tf_norm

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {
                "content":  self.corpus[idx].content,
                "metadata": self.corpus[idx].metadata,
                "score":    round(score, 4),
                "chunk_id": self.corpus[idx].chunk_id,
            }
            for idx, score in ranked
        ]


class HybridRetriever:
    """
    Combines semantic (dense) and BM25 (sparse) retrieval
    using Reciprocal Rank Fusion (RRF), then optionally reranks
    with Cohere for maximum precision.
    """

    def __init__(
        self,
        vectorstore,
        chunks: list = None,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
        top_k: int = 6,
        reranker_top_n: int = 3,
        use_reranker: bool = False,
        rrf_k: int = 60,
    ):
        self.vectorstore     = vectorstore
        self.semantic_weight = semantic_weight
        self.bm25_weight     = bm25_weight
        self.top_k           = top_k
        self.reranker_top_n  = reranker_top_n
        self.use_reranker    = use_reranker
        self.rrf_k           = rrf_k

        # Build BM25 index
        self.bm25 = BM25Retriever()
        if chunks:
            self.bm25.build(chunks)

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict],
        bm25_results: List[Dict],
    ) -> List[Dict]:
        """Merge two ranked lists using RRF scoring."""
        scores = defaultdict(float)
        docs   = {}

        for rank, doc in enumerate(semantic_results):
            cid = doc["chunk_id"]
            scores[cid] += self.semantic_weight * (1 / (self.rrf_k + rank + 1))
            docs[cid] = doc

        for rank, doc in enumerate(bm25_results):
            cid = doc["chunk_id"]
            scores[cid] += self.bm25_weight * (1 / (self.rrf_k + rank + 1))
            if cid not in docs:
                docs[cid] = doc

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            {**docs[cid], "rrf_score": round(score, 6)}
            for cid, score in ranked
        ]

    def _rerank(self, query: str, docs: List[Dict]) -> List[Dict]:
        """Use Cohere Reranker for final ordering."""
        try:
            import cohere
            co = cohere.Client(os.environ.get("COHERE_API_KEY", ""))
            response = co.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=[d["content"] for d in docs],
                top_n=self.reranker_top_n,
            )
            reranked = []
            for result in response.results:
                doc = docs[result.index].copy()
                doc["rerank_score"] = round(result.relevance_score, 4)
                reranked.append(doc)
            return reranked
        except Exception as e:
            logger.warning(f"Reranker failed ({e}), using RRF order")
            return docs[:self.reranker_top_n]

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Natural language query
            top_k: Override default top_k
            where: Metadata filter for semantic search

        Returns:
            Ranked list of relevant chunk dicts
        """
        k = top_k or self.top_k

        # Semantic retrieval
        semantic = self.vectorstore.similarity_search(query, top_k=k * 2, where=where)

        # BM25 retrieval
        bm25 = self.bm25.search(query, top_k=k * 2) if self.bm25._built else []

        # Fuse rankings
        if bm25:
            fused = self._reciprocal_rank_fusion(semantic, bm25)
        else:
            fused = semantic

        candidates = fused[:k * 2]

        # Optional reranking
        if self.use_reranker and candidates:
            return self._rerank(query, candidates)

        return candidates[:k]
