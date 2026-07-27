"""prompt_templates.py의 표준 라이브러리 기반 단위 테스트."""

import unittest

from prompt_templates import PromptVariant, build_prompt, format_context
from retrieval_adapter import detect_car_from_question
from run_local_model import is_exit_command


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
        self.assertIn("해당 정보 없음", prompt)
        self.assertIn("avante_p391_0", prompt)

    def test_string_variant_is_supported(self):
        prompt = build_prompt(
            question="질문",
            documents=[self.chunk_dict],
            variant="basic",
        )

        self.assertIn("다음 참고 문서를 바탕으로", prompt)

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


if __name__ == "__main__":
    unittest.main()
