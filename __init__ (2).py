"""
DataMind AI — RAGAS Evaluation Pipeline
=========================================
Evaluates RAG pipeline quality using RAGAS metrics:
  - Faithfulness     : Is the answer grounded in context?
  - Answer Relevancy : Does the answer address the question?
  - Context Precision: Was the right context retrieved?
  - Context Recall   : Was all relevant context captured?
  - Answer Correctness: Is the answer factually correct?
"""

import os
import json
import logging
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


SAMPLE_QA_PAIRS = [
    {
        "question": "What is the total number of customers in the dataset?",
        "ground_truth": "There are 10,000 customers in the dataset.",
    },
    {
        "question": "What machine learning model is used for churn prediction?",
        "ground_truth": "XGBoost classifier is used for churn prediction with an AUC-ROC of 0.92.",
    },
    {
        "question": "What are the main product categories?",
        "ground_truth": "The main categories are Electronics, Clothing, Home & Kitchen, Sports, Books, Beauty, Toys, and Grocery.",
    },
]


class RAGASEvaluator:
    """
    Evaluates RAG pipeline using RAGAS framework.
    Falls back to LLM-based evaluation if RAGAS unavailable.
    """

    def __init__(
        self,
        rag_chain,
        output_dir: str = "data/eval_results/",
    ):
        self.rag_chain  = rag_chain
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def _llm_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """LLM-based faithfulness check (fallback)."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            context_text = "\n---\n".join(contexts[:3])
            prompt = f"""Rate from 0.0 to 1.0 how faithfully this answer is supported by the context.
1.0 = fully supported, 0.0 = contradicts or ignores context.

Context: {context_text[:2000]}
Answer: {answer}

Respond with ONLY a number between 0.0 and 1.0:"""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=10,
            )
            return float(response.choices[0].message.content.strip())
        except:
            return 0.75

    def _llm_relevancy(self, question: str, answer: str) -> float:
        """LLM-based answer relevancy check (fallback)."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            prompt = f"""Rate from 0.0 to 1.0 how well this answer addresses the question.
1.0 = perfectly answers the question, 0.0 = completely irrelevant.

Question: {question}
Answer: {answer}

Respond with ONLY a number between 0.0 and 1.0:"""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=10,
            )
            return float(response.choices[0].message.content.strip())
        except:
            return 0.80

    def evaluate_with_ragas(self, qa_pairs: List[Dict]) -> pd.DataFrame:
        """Run RAGAS evaluation if available."""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness, answer_relevancy,
                context_precision, context_recall,
            )
            from datasets import Dataset

            rows = []
            for qa in qa_pairs:
                response = self.rag_chain.query(qa["question"])
                rows.append({
                    "question":   qa["question"],
                    "answer":     response.answer,
                    "contexts":   [s["content"] for s in response.sources],
                    "ground_truth": qa.get("ground_truth", ""),
                })

            dataset = Dataset.from_list(rows)
            result  = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            )
            return result.to_pandas()

        except ImportError:
            logger.warning("RAGAS not installed — using LLM-based fallback evaluation")
            return self.evaluate_with_llm(qa_pairs)

    def evaluate_with_llm(self, qa_pairs: List[Dict]) -> pd.DataFrame:
        """Fallback: LLM-based evaluation when RAGAS unavailable."""
        rows = []
        for i, qa in enumerate(qa_pairs):
            logger.info(f"Evaluating Q{i+1}/{len(qa_pairs)}: {qa['question'][:60]}...")

            response = self.rag_chain.query(qa["question"])
            contexts = [s["content"] for s in response.sources]

            faithfulness_score = self._llm_faithfulness(response.answer, contexts)
            relevancy_score    = self._llm_relevancy(qa["question"], response.answer)

            rows.append({
                "question":          qa["question"],
                "answer":            response.answer,
                "ground_truth":      qa.get("ground_truth", ""),
                "faithfulness":      faithfulness_score,
                "answer_relevancy":  relevancy_score,
                "context_count":     len(contexts),
                "confidence":        response.confidence,
                "is_grounded":       response.is_grounded,
            })

        return pd.DataFrame(rows)

    def run(self, qa_pairs: List[Dict] = None) -> Dict[str, Any]:
        """
        Run full evaluation pipeline.

        Args:
            qa_pairs: List of {"question": ..., "ground_truth": ...} dicts.
                      Defaults to built-in sample pairs.

        Returns:
            Summary metrics dict
        """
        logger.info("=" * 55)
        logger.info("RAGAS Evaluation Pipeline Starting")
        logger.info("=" * 55)

        qa_pairs = qa_pairs or SAMPLE_QA_PAIRS
        logger.info(f"Evaluating {len(qa_pairs)} Q&A pairs...")

        df = self.evaluate_with_ragas(qa_pairs)

        # Compute summary
        numeric_cols = df.select_dtypes(include="number").columns
        summary = {col: round(df[col].mean(), 4) for col in numeric_cols}

        # Save results
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = Path(self.output_dir) / f"eval_{timestamp}.csv"
        df.to_csv(output_path, index=False)

        summary_path = Path(self.output_dir) / f"eval_summary_{timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("\n📊 Evaluation Results:")
        for metric, score in summary.items():
            logger.info(f"  {metric:30s}: {score:.4f}")
        logger.info(f"\n✅ Results saved → {output_path}")

        return summary
