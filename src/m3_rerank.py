from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


_GLOBAL_CROSS_ENCODERS = {}


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name

    def _load_model(self):
        global _GLOBAL_CROSS_ENCODERS
        if self.model_name not in _GLOBAL_CROSS_ENCODERS:
            try:
                from sentence_transformers import CrossEncoder
                _GLOBAL_CROSS_ENCODERS[self.model_name] = CrossEncoder(self.model_name, local_files_only=True)
            except Exception:
                _GLOBAL_CROSS_ENCODERS[self.model_name] = None
        return _GLOBAL_CROSS_ENCODERS[self.model_name]

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []
        model = self._load_model()
        if model is not None:
            try:
                pairs = [(query, doc["text"]) for doc in documents]
                scores = model.predict(pairs)
                if isinstance(scores, (int, float)):
                    scores = [scores]
                scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
                return [
                    RerankResult(
                        text=doc["text"],
                        original_score=float(doc.get("score", 0.0)),
                        rerank_score=float(score),
                        metadata=doc.get("metadata", {}),
                        rank=i + 1
                    )
                    for i, (score, doc) in enumerate(scored[:top_k])
                ]
            except Exception as e:
                print(f"  ⚠️  CrossEncoder predict fallback: {e}")

        # Semantic embedding fallback using cached bge-m3
        try:
            from sentence_transformers import SentenceTransformer
            from numpy import dot
            from numpy.linalg import norm

            embedder = SentenceTransformer("BAAI/bge-m3", local_files_only=True)
            q_vec = embedder.encode(query, show_progress_bar=False)
            d_vecs = embedder.encode([d["text"] for d in documents], show_progress_bar=False)

            scores = []
            for d_vec in d_vecs:
                denom = norm(q_vec) * norm(d_vec)
                sim = float(dot(q_vec, d_vec) / (denom + 1e-9)) if denom > 0 else 0.0
                scores.append(sim)

            scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
            return [
                RerankResult(
                    text=doc["text"],
                    original_score=float(doc.get("score", 0.0)),
                    rerank_score=float(score),
                    metadata=doc.get("metadata", {}),
                    rank=i + 1
                )
                for i, (score, doc) in enumerate(scored[:top_k])
            ]
        except Exception:
            # Lexical overlap fallback
            query_words = set(query.lower().split())
            scored = []
            for doc in documents:
                doc_words = set(doc["text"].lower().split())
                score = len(query_words.intersection(doc_words)) / max(len(query_words), 1)
                scored.append((score, doc))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                RerankResult(
                    text=doc["text"],
                    original_score=float(doc.get("score", 0.0)),
                    rerank_score=float(score),
                    metadata=doc.get("metadata", {}),
                    rank=i + 1
                )
                for i, (score, doc) in enumerate(scored[:top_k])
            ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from flashrank import Ranker
                self._model = Ranker()
            except Exception:
                self._model = None
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []
        model = self._load_model()
        if model is not None:
            try:
                from flashrank import RerankRequest
                passages = [{"text": d["text"], "meta": d.get("metadata", {})} for d in documents]
                results = model.rerank(RerankRequest(query=query, passages=passages))
                return [
                    RerankResult(
                        text=r["text"],
                        original_score=float(documents[idx].get("score", 0.0)) if idx < len(documents) else 0.0,
                        rerank_score=float(r.get("score", 0.0)),
                        metadata=r.get("meta", {}),
                        rank=idx + 1
                    )
                    for idx, r in enumerate(results[:top_k])
                ]
            except Exception:
                pass
        return [
            RerankResult(
                text=d["text"],
                original_score=float(d.get("score", 0.0)),
                rerank_score=float(d.get("score", 0.0)),
                metadata=d.get("metadata", {}),
                rank=i + 1
            )
            for i, d in enumerate(documents[:top_k])
        ]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
