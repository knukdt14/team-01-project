"""무작위 일괄 평가의 순수 로직 테스트."""

from __future__ import annotations

import math
import unittest

from batch_evaluation import (
    ALL_METRICS,
    BenchmarkQuestion,
    build_html_report,
    select_random_questions,
    summarize_results,
)
from prompt_templates import PromptVariant


def make_question(question_id: int) -> BenchmarkQuestion:
    return BenchmarkQuestion(
        question_id=str(question_id),
        vehicle="아반떼",
        car="avante",
        answerability="문서 내 답변",
        difficulty="쉬움",
        question_type="직접 사실",
        question=f"질문 {question_id}",
        reference_answer=f"모범답안 {question_id}",
        key_terms="핵심어",
        source_pdf="manual.pdf",
        source_page=str(question_id),
        source_section="테스트",
        evaluation_focus="직접 사실",
    )


def make_result(question_id: int, base: float) -> dict:
    variants = {}
    for index, variant in enumerate(PromptVariant):
        score = base + index * 0.01
        variants[variant.value] = {
            "answer": f"{variant.value} 답변",
            "bertscore_precision": score,
            "bertscore_recall": score,
            "bertscore_f1": score,
            "faithfulness": score,
            "answer_relevancy": score,
            "context_precision": 0.8,
            "context_recall": 0.9,
            "generation_time_seconds": 10.0 + index,
        }
    return {
        **make_question(question_id).to_dict(),
        "variants": variants,
    }


class BatchEvaluationTests(unittest.TestCase):
    def test_random_sample_is_reproducible_and_without_duplicates(self):
        questions = [make_question(index) for index in range(1, 101)]

        first = select_random_questions(
            questions,
            sample_size=25,
            seed=42,
        )
        second = select_random_questions(
            questions,
            sample_size=25,
            seed=42,
        )

        first_ids = [question.question_id for question in first]
        second_ids = [question.question_id for question in second]
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(len(first_ids), 25)
        self.assertEqual(len(set(first_ids)), 25)

    def test_sample_size_cannot_exceed_dataset(self):
        with self.assertRaises(ValueError):
            select_random_questions(
                [make_question(1)],
                sample_size=2,
                seed=42,
            )

    def test_summary_averages_each_variant(self):
        results = [make_result(1, 0.5), make_result(2, 0.7)]

        summary = summarize_results(results)

        self.assertAlmostEqual(
            summary["basic"]["bertscore_f1"]["mean"],
            0.6,
        )
        self.assertAlmostEqual(
            summary["few_shot"]["bertscore_f1"]["mean"],
            0.63,
        )
        self.assertEqual(
            summary["role"]["faithfulness"]["valid_count"],
            2,
        )
        self.assertAlmostEqual(
            summary["constraint"]["generation_time_seconds"]["mean"],
            12.0,
        )

    def test_summary_ignores_missing_and_nan_scores(self):
        results = [make_result(1, 0.5), make_result(2, 0.7)]
        results[0]["variants"]["basic"]["faithfulness"] = None
        results[1]["variants"]["basic"]["faithfulness"] = float("nan")

        summary = summarize_results(results)

        self.assertTrue(
            math.isnan(summary["basic"]["faithfulness"]["mean"])
        )
        self.assertEqual(
            summary["basic"]["faithfulness"]["valid_count"],
            0,
        )

    def test_html_report_contains_metrics_and_selected_questions(self):
        results = [make_result(1, 0.7)]
        summary = summarize_results(results)
        report = build_html_report(
            summary=summary,
            results=results,
            metadata={
                "sample_size": 1,
                "seed": 42,
                "answer_provider": "huggingface",
                "answer_model": "Qwen",
                "evaluation_model": "solar-pro3",
                "updated_at": "2026-01-01T00:00:00+09:00",
            },
        )

        self.assertIn("프롬프트 4종 성능 비교", report)
        self.assertIn("BERTScore F1", report)
        self.assertIn("응답 생성 시간", report)
        self.assertIn("질문 1", report)
        for metric in ALL_METRICS:
            self.assertIn(
                {
                    "bertscore_precision": "BERTScore Precision",
                    "bertscore_recall": "BERTScore Recall",
                    "bertscore_f1": "BERTScore F1",
                    "faithfulness": "Faithfulness",
                    "answer_relevancy": "Answer Relevancy",
                    "context_precision": "Context Precision",
                    "context_recall": "Context Recall",
                    "generation_time_seconds": "응답 생성 시간(초)",
                }[metric],
                report,
            )


if __name__ == "__main__":
    unittest.main()
