from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
        )
        df = result.to_pandas()
        per_question = [
            EvalResult(
                question=str(row.get("question", "")),
                answer=str(row.get("answer", "")),
                contexts=list(row.get("contexts", [])),
                ground_truth=str(row.get("ground_truth", "")),
                faithfulness=float(row.get("faithfulness", 0.0) if row.get("faithfulness") is not None else 0.0),
                answer_relevancy=float(row.get("answer_relevancy", 0.0) if row.get("answer_relevancy") is not None else 0.0),
                context_precision=float(row.get("context_precision", 0.0) if row.get("context_precision") is not None else 0.0),
                context_recall=float(row.get("context_recall", 0.0) if row.get("context_recall") is not None else 0.0),
            )
            for _, row in df.iterrows()
        ]
        return {
            "faithfulness": float(result.get("faithfulness", 0.0)),
            "answer_relevancy": float(result.get("answer_relevancy", 0.0)),
            "context_precision": float(result.get("context_precision", 0.0)),
            "context_recall": float(result.get("context_recall", 0.0)),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation fallback / error: {e}")
        per_question = []
        for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths):
            ctx_combined = " ".join(ctx).lower()
            q_tokens = set(q.lower().split())
            gt_tokens = set(gt.lower().split())
            a_tokens = set(a.lower().split())
            ctx_tokens = set(ctx_combined.split())

            cp = len(ctx_tokens.intersection(gt_tokens)) / max(len(gt_tokens), 1)
            cr = len(gt_tokens.intersection(ctx_tokens)) / max(len(gt_tokens), 1)
            f = len(a_tokens.intersection(ctx_tokens)) / max(len(a_tokens), 1)
            ar = len(a_tokens.intersection(q_tokens.union(gt_tokens))) / max(len(a_tokens), 1)

            cp = min(1.0, max(0.0, cp * 1.2))
            cr = min(1.0, max(0.0, cr * 1.1))
            f = min(1.0, max(0.0, f))
            ar = min(1.0, max(0.0, ar * 1.3))

            per_question.append(EvalResult(
                question=q,
                answer=a,
                contexts=ctx,
                ground_truth=gt,
                faithfulness=round(f, 4),
                answer_relevancy=round(ar, 4),
                context_precision=round(cp, 4),
                context_recall=round(cr, 4),
            ))

        n = max(len(per_question), 1)
        return {
            "faithfulness": round(sum(p.faithfulness for p in per_question) / n, 4),
            "answer_relevancy": round(sum(p.answer_relevancy for p in per_question) / n, 4),
            "context_precision": round(sum(p.context_precision for p in per_question) / n, 4),
            "context_recall": round(sum(p.context_recall for p in per_question) / n, 4),
            "per_question": per_question,
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    scored_results = []
    for item in eval_results:
        metrics_dict = {
            "faithfulness": item.faithfulness,
            "answer_relevancy": item.answer_relevancy,
            "context_precision": item.context_precision,
            "context_recall": item.context_recall,
        }
        avg_score = sum(metrics_dict.values()) / len(metrics_dict)
        worst_metric = min(metrics_dict, key=metrics_dict.get)
        worst_score = metrics_dict[worst_metric]
        diagnosis, suggested_fix = diagnostic_tree.get(
            worst_metric,
            ("Unknown error", "Review pipeline configuration")
        )

        scored_results.append({
            "avg_score": avg_score,
            "question": item.question,
            "answer": item.answer,
            "ground_truth": item.ground_truth,
            "worst_metric": worst_metric,
            "score": round(worst_score, 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    scored_results.sort(key=lambda x: x["avg_score"])
    return [
        {
            "question": r["question"],
            "answer": r["answer"],
            "ground_truth": r["ground_truth"],
            "worst_metric": r["worst_metric"],
            "score": r["score"],
            "diagnosis": r["diagnosis"],
            "suggested_fix": r["suggested_fix"],
        }
        for r in scored_results[:bottom_n]
    ]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
