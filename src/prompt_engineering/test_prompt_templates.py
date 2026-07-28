"""prompt_templates.py의 표준 라이브러리 기반 단위 테스트."""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from evaluation_metrics import (
    EvaluationScores,
    build_ragas_payloads,
    format_score,
)
from prompt_templates import (
    PROMPT_LABELS,
    PromptVariant,
    build_prompt,
    format_context,
)
from retrieval_adapter import detect_car_from_question
from run_local_model import (
    DEFAULT_MODEL,
    DEFAULT_UPSTAGE_MODEL,
    answer_question,
    apply_output_guard,
    is_exit_command,
    load_upstage_generator,
    resolve_model_id,
)


class FakeDocument:
    """LangChain Document와 같은 최소 인터페이스를 가진 테스트 객체."""

    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


class PromptTemplateTests(unittest.TestCase):
    def setUp(self):
        self.chunk_dict = {
            "text": "일반 조건에서는 정해진 주기에 따라 엔진오일을 교환합니다.",
            "car": "avante",
            "page": 390,
            "chunk_id": "avante_p390_0",
        }
        self.langchain_document = FakeDocument(
            "가혹 조건에서는 교환 주기가 더 짧아질 수 있습니다.",
            {
                "car": "avante",
                "page": 391,
                "chunk_id": "avante_p391_0",
            },
        )

    def test_format_context_combines_multiple_document_types(self):
        context = format_context([self.chunk_dict, self.langchain_document])

        self.assertIn("[문서 1 | 차종: avante | 페이지: 390", context)
        self.assertIn("avante_p390_0", context)
        self.assertIn("[문서 2 | 차종: avante | 페이지: 391", context)
        self.assertIn("가혹 조건", context)

    def test_constraint_prompt_contains_question_car_and_sources(self):
        prompt = build_prompt(
            question="엔진오일 교환주기는?",
            documents=[self.chunk_dict, self.langchain_document],
            variant=PromptVariant.CONSTRAINT,
            car="avante",
        )

        self.assertIn('대상 차량은 "avante"', prompt)
        self.assertIn("엔진오일 교환주기는?", prompt)
        self.assertIn(
            "차량 취급설명서에서 해당 내용을 찾지 못했습니다.",
            prompt,
        )
        self.assertIn("avante_p391_0", prompt)
        self.assertIn("제공된 참고 문서에 있는 정보만 사용하세요", prompt)
        self.assertIn("[출처: (실제 차종) p.(실제 페이지)]", prompt)
        self.assertIn(
            '"차량 취급설명서에서 해당 내용을 찾지 못했습니다."만',
            prompt,
        )
        self.assertEqual(
            prompt.count(
                "차량 취급설명서에서 해당 내용을 찾지 못했습니다."
            ),
            1,
        )
        self.assertIn("관련 근거가 하나라도 있으면 반드시", prompt)
        self.assertIn('"회생제동 기능"과', prompt)
        self.assertIn("언급이 없는 것을", prompt)
        self.assertIn("청크 ID를 사용하지 마세요", prompt)
        self.assertIn('"p.6"으로 표시하세요', prompt)
        self.assertIn("관련 없는 문서의 출처를 붙이지 마세요", prompt)
        self.assertIn("질문과 직접 관련된 내용만", prompt)

    def test_string_variant_is_supported(self):
        prompt = build_prompt(
            question="질문",
            documents=[self.chunk_dict],
            variant="basic",
        )

        self.assertIn("다음 참고 문서를 바탕으로", prompt)

    def test_five_comparison_prompt_variants_are_available(self):
        self.assertEqual(
            [variant.value for variant in PromptVariant],
            ["basic", "role", "constraint", "few_shot", "verification"],
        )
        self.assertEqual(
            PROMPT_LABELS[PromptVariant.BASIC],
            "CONTEXT + QUESTION",
        )

    def test_few_shot_prompt_has_supported_and_unsupported_examples(self):
        prompt = build_prompt(
            question="엔진오일 교환주기는?",
            documents=[self.chunk_dict],
            variant=PromptVariant.FEW_SHOT,
            car="avante",
        )

        self.assertIn("[예시 1: 문서에 답이 있는 경우]", prompt)
        self.assertIn("[예시 2: 문서에 답이 없는 경우]", prompt)
        self.assertIn("[실제 참고 문서]", prompt)
        self.assertIn("예시의 차량·페이지·내용을 실제 질문의 근거로 사용하지 마세요", prompt)

    def test_verification_prompt_checks_claims_before_final_answer(self):
        prompt = build_prompt(
            question="엔진오일 교환주기는?",
            documents=[self.chunk_dict],
            variant=PromptVariant.VERIFICATION,
            car="avante",
        )

        self.assertIn("각 주장, 수치, 단위, 조건", prompt)
        self.assertIn("직접 확인되지 않는 주장", prompt)
        self.assertIn("검증 과정이나 초안은 보여주지 말고", prompt)
        self.assertIn("[검증을 마친 최종 답변]", prompt)

    def test_empty_documents_have_explicit_context(self):
        prompt = build_prompt("답이 있나요?", [], variant="constraint", car="ioniq6")

        self.assertIn("검색된 참고 문서 없음", prompt)

    def test_needs_review_metadata_is_exposed_to_model(self):
        review_document = FakeDocument(
            "표에서 추출된 내용",
            {
                "car": "nexo",
                "page": 100,
                "chunk_id": "nexo_p100_0",
                "needs_review": True,
            },
        )

        prompt = build_prompt(
            "표의 수치는?",
            [review_document],
            variant="constraint",
            car="nexo",
        )

        self.assertIn("검토 필요", prompt)
        self.assertIn("표 구조가 불완전할 수 있으므로", prompt)

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            build_prompt("   ", [self.chunk_dict])

    def test_interactive_exit_commands(self):
        for command in ("q", "QUIT", " exit ", "종료"):
            with self.subTest(command=command):
                self.assertTrue(is_exit_command(command))

        self.assertFalse(is_exit_command("타이어 공기압은?"))

    def test_car_is_detected_from_question(self):
        cases = {
            "투싼의 엔진 경고등이 들어왔어": "tucson",
            "아반떼의 와이퍼가 고장났어": "avante",
            "아반떼 하이브리드 연료 탱크 용량은?": "avante_hev",
            "아이오닉 6 충전 방법은?": "ioniq6",
            "넥쏘 수소 충전 방법은?": "nexo",
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(detect_car_from_question(question), expected)

        self.assertIsNone(detect_car_from_question("엔진 경고등이 들어왔어"))

    def test_provider_default_models_can_be_selected_without_code_changes(self):
        self.assertEqual(resolve_model_id("huggingface", None), DEFAULT_MODEL)
        self.assertEqual(
            resolve_model_id("upstage", None),
            DEFAULT_UPSTAGE_MODEL,
        )
        self.assertEqual(
            resolve_model_id("upstage", "solar-custom"),
            "solar-custom",
        )

    def test_constraint_no_information_output_is_normalized(self):
        raw_answer = "해당 정보 없음\n\n[출처: avante p.180]"

        self.assertEqual(
            apply_output_guard(raw_answer, PromptVariant.CONSTRAINT),
            "해당 정보 없음",
        )
        self.assertEqual(
            apply_output_guard(raw_answer, PromptVariant.BASIC),
            raw_answer,
        )
        self.assertEqual(
            apply_output_guard(
                "검토 필요\n[출처: tucson p.267]",
                PromptVariant.CONSTRAINT,
            ),
            "해당 정보 없음",
        )

    def test_upstage_generator_uses_selected_model_without_real_api_call(self):
        captured = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured["request"] = kwargs
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="모의 API 답변")
                        )
                    ]
                )

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured["client"] = kwargs
                self.chat = SimpleNamespace(completions=FakeCompletions())

        fake_openai_module = ModuleType("openai")
        fake_openai_module.OpenAI = FakeOpenAI

        with (
            patch.dict(os.environ, {"UPSTAGE_API_KEY": "test-key"}),
            patch.dict(sys.modules, {"openai": fake_openai_module}),
            redirect_stdout(io.StringIO()),
        ):
            generate = load_upstage_generator(
                model_id="solar-test",
                max_new_tokens=123,
            )
            answer = generate("테스트 프롬프트")

        self.assertEqual(answer, "모의 API 답변")
        self.assertEqual(
            captured["client"],
            {
                "api_key": "test-key",
                "base_url": "https://api.upstage.ai/v1",
            },
        )
        self.assertEqual(captured["request"]["model"], "solar-test")
        self.assertEqual(captured["request"]["max_tokens"], 123)
        self.assertEqual(
            captured["request"]["messages"],
            [{"role": "user", "content": "테스트 프롬프트"}],
        )

    def test_evaluation_score_format_handles_number_and_missing_value(self):
        self.assertEqual(format_score(0.87654), "0.877")
        self.assertEqual(format_score(float("nan")), "N/A")
        self.assertEqual(format_score(None), "N/A")

    def test_context_metrics_are_built_once_for_shared_retrieval(self):
        answer_payload, context_payload = build_ragas_payloads(
            question="질문",
            answers=["답변 1", "답변 2", "답변 3"],
            contexts=["문맥 1", "문맥 2"],
            reference_answer="모범답안",
        )

        self.assertEqual(len(answer_payload["answer"]), 3)
        self.assertEqual(len(context_payload["answer"]), 1)
        self.assertEqual(
            context_payload["contexts"],
            [["문맥 1", "문맥 2"]],
        )

    def test_answer_question_prints_evaluation_metrics(self):
        class FakeRetriever:
            def __init__(self):
                self.search_kwargs = {}

            def invoke(self, question):
                return [self_document]

        class FakeEvaluator:
            def evaluate(self, **kwargs):
                self.kwargs = kwargs
                return [
                    EvaluationScores(
                        bertscore_precision=0.91,
                        bertscore_recall=0.82,
                        bertscore_f1=0.86,
                        faithfulness=1.0,
                        answer_relevancy=0.9,
                        context_precision=0.8,
                        context_recall=0.7,
                    )
                ]

        self_document = self.langchain_document
        evaluator = FakeEvaluator()
        args = SimpleNamespace(top_k=1, show_prompt=False)

        with redirect_stdout(io.StringIO()) as output:
            succeeded = answer_question(
                question="엔진오일 교환주기는?",
                car="avante",
                args=args,
                variants=[PromptVariant.BASIC],
                retriever=FakeRetriever(),
                generate=lambda prompt: "테스트 답변",
                evaluator=evaluator,
                reference_answer="모범답안",
            )

        rendered = output.getvalue()
        self.assertTrue(succeeded)
        self.assertIn("[1번 청크]", rendered)
        self.assertIn("청크 ID: avante_p391_0", rendered)
        self.assertIn(self_document.page_content, rendered)
        self.assertIn("EVALUATION METRICS", rendered)
        self.assertIn("BERTScore P=0.910, R=0.820, F1=0.860", rendered)
        self.assertIn("Faithfulness: 1.000", rendered)
        self.assertIn("응답 생성 시간", rendered)


if __name__ == "__main__":
    unittest.main()
