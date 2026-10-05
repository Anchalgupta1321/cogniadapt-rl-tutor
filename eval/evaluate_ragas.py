import json
import numpy as np
from typing import Dict, Any, List
from app.services.vector_dedup import get_embedding, cosine_similarity

class RAGEvaluationFramework:
    """
    RAG Evaluation Framework computing standard benchmark metrics:
      1. Faithfulness: Degree to which generated response is grounded in source context.
      2. Answer Relevancy: Alignment between generated answer and student query.
      3. Context Precision: Fraction of retrieved chunks that are relevant.
      4. Context Recall: Degree to which all ground truth source facts are retrieved.
    """
    def __init__(self):
        pass

    def evaluate_sample(
        self,
        query: str,
        retrieved_contexts: List[str],
        generated_answer: str,
        ground_truth_answer: str = ""
    ) -> Dict[str, float]:
        query_vec = get_embedding(query)
        ans_vec = get_embedding(generated_answer)

        # 1. Answer Relevancy
        ans_relevancy = cosine_similarity(query_vec, ans_vec)

        # 2. Faithfulness
        faithfulness_scores = []
        for ctx in retrieved_contexts:
            ctx_vec = get_embedding(ctx)
            faithfulness_scores.append(cosine_similarity(ans_vec, ctx_vec))
        faithfulness = float(np.max(faithfulness_scores)) if faithfulness_scores else 0.0

        # 3. Context Precision
        precision_scores = []
        for ctx in retrieved_contexts:
            ctx_vec = get_embedding(ctx)
            precision_scores.append(1.0 if cosine_similarity(query_vec, ctx_vec) > 0.30 else 0.0)
        context_precision = float(np.mean(precision_scores)) if precision_scores else 0.0

        # 4. Context Recall
        if ground_truth_answer:
            gt_vec = get_embedding(ground_truth_answer)
            recall_scores = [cosine_similarity(gt_vec, get_embedding(c)) for c in retrieved_contexts]
            context_recall = float(np.max(recall_scores)) if recall_scores else 0.0
        else:
            context_recall = faithfulness

        return {
            "faithfulness": round(float(faithfulness), 4),
            "answer_relevancy": round(float(ans_relevancy), 4),
            "context_precision": round(float(context_precision), 4),
            "context_recall": round(float(context_recall), 4)
        }

    def run_benchmark_suite(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = []
        for item in dataset:
            metrics = self.evaluate_sample(
                query=item["query"],
                retrieved_contexts=item["retrieved_contexts"],
                generated_answer=item["generated_answer"],
                ground_truth_answer=item.get("ground_truth", "")
            )
            results.append(metrics)

        avg_faithfulness = float(np.mean([r["faithfulness"] for r in results]))
        avg_relevancy = float(np.mean([r["answer_relevancy"] for r in results]))
        avg_precision = float(np.mean([r["context_precision"] for r in results]))
        avg_recall = float(np.mean([r["context_recall"] for r in results]))

        return {
            "evaluation_framework": "RAGAS & DeepEval Standalone Benchmark",
            "samples_evaluated": len(dataset),
            "summary_metrics": {
                "faithfulness": round(avg_faithfulness, 4),
                "answer_relevancy": round(avg_relevancy, 4),
                "context_precision": round(avg_precision, 4),
                "context_recall": round(avg_recall, 4)
            },
            "sample_results": results[:5]
        }

if __name__ == "__main__":
    evaluator = RAGEvaluationFramework()
    dummy_dataset = [
        {
            "query": "What is the function of chlorophyll?",
            "retrieved_contexts": ["Chlorophyll is the green pigment in plants responsible for absorbing light energy for photosynthesis."],
            "generated_answer": "Based on [Page 4], Chlorophyll is the green pigment in plants responsible for absorbing light energy for photosynthesis.",
            "ground_truth": "Chlorophyll absorbs light energy during photosynthesis."
        },
        {
            "query": "What is Quantum Mechanics?",
            "retrieved_contexts": ["This chapter covers cell biology."],
            "generated_answer": "Refusal: The provided course materials do not cover this query.",
            "ground_truth": "Not covered."
        }
    ]
    report = evaluator.run_benchmark_suite(dummy_dataset)
    print("=== RAGAS EVALUATION SUITE BENCHMARK REPORT ===")
    print(json.dumps(report, indent=2))
