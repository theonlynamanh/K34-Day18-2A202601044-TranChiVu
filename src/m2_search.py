from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except Exception:
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = [segment_vietnamese(c["text"]).split() for c in chunks]
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus_tokens)
        except Exception:
            self.bm25 = None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or not self.documents:
            tokens = set(segment_vietnamese(query).lower().split())
            scored = []
            for doc in self.documents:
                doc_tokens = set(segment_vietnamese(doc["text"]).lower().split())
                overlap = len(tokens.intersection(doc_tokens))
                if overlap > 0:
                    scored.append((overlap, doc))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                SearchResult(text=d["text"], score=float(s), metadata=d.get("metadata", {}), method="bm25")
                for s, d in scored[:top_k]
            ]

        tokenized_query = segment_vietnamese(query).split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in top_indices:
            if scores[i] > 0:
                results.append(SearchResult(
                    text=self.documents[i]["text"],
                    score=float(scores[i]),
                    metadata=self.documents[i].get("metadata", {}),
                    method="bm25"
                ))
            if len(results) >= top_k:
                break
        return results


_GLOBAL_DENSE_ENCODERS = {}


class DenseSearch:
    def __init__(self):
        self.client = None
        try:
            from qdrant_client import QdrantClient
            try:
                self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
                self.client.get_collections()
            except Exception:
                self.client = QdrantClient(":memory:")
        except Exception:
            self.client = None

    def _get_encoder(self):
        global _GLOBAL_DENSE_ENCODERS
        if EMBEDDING_MODEL not in _GLOBAL_DENSE_ENCODERS:
            try:
                from sentence_transformers import SentenceTransformer
                try:
                    _GLOBAL_DENSE_ENCODERS[EMBEDDING_MODEL] = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
                except Exception:
                    _GLOBAL_DENSE_ENCODERS[EMBEDDING_MODEL] = SentenceTransformer(EMBEDDING_MODEL)
            except Exception:
                _GLOBAL_DENSE_ENCODERS[EMBEDDING_MODEL] = None
        return _GLOBAL_DENSE_ENCODERS[EMBEDDING_MODEL]

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        self._chunks = chunks
        if self.client is None:
            return

        from qdrant_client.models import Distance, VectorParams, PointStruct
        texts = [c["text"] for c in chunks]
        encoder = self._get_encoder()
        if encoder is not None:
            vectors = encoder.encode(texts, show_progress_bar=False)
            dim = len(vectors[0]) if len(vectors) > 0 else EMBEDDING_DIM
        else:
            import numpy as np
            vectors = [np.zeros(EMBEDDING_DIM) for _ in texts]
            dim = EMBEDDING_DIM

        self.client.recreate_collection(
            collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

        points = [
            PointStruct(
                id=i,
                vector=vectors[i].tolist() if hasattr(vectors[i], "tolist") else list(vectors[i]),
                payload={**c.get("metadata", {}), "text": c["text"]}
            )
            for i, c in enumerate(chunks)
        ]
        if points:
            self.client.upsert(collection, points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if self.client is None:
            return []
        encoder = self._get_encoder()
        if encoder is not None:
            query_vector = encoder.encode(query).tolist()
        else:
            import numpy as np
            query_vector = np.zeros(EMBEDDING_DIM).tolist()

        try:
            response = self.client.query_points(collection, query=query_vector, limit=top_k)
            return [
                SearchResult(
                    text=pt.payload.get("text", ""),
                    score=float(pt.score),
                    metadata=pt.payload,
                    method="dense"
                )
                for pt in response.points
            ]
        except Exception:
            return []


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {
                    "score": 0.0,
                    "result": result
                }
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)

    sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    merged = []
    for item in sorted_items[:top_k]:
        res = item["result"]
        merged.append(SearchResult(
            text=res.text,
            score=float(item["score"]),
            metadata=res.metadata,
            method="hybrid"
        ))
    return merged


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
