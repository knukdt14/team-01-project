"""수업자료의 BERTScore와 RAGAS 지표로 프롬프트 답변을 평가한다."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_EVALUATION_MODEL = "solar-pro3"
DEFAULT_EVALUATION_EMBEDDING = "intfloat/multilingual-e5-base"


@dataclass(frozen=True)
class EvaluationScores:
    """답변 하나에 대한 수업자료 기반 평가 결과."""

    bertscore_precision: float
    bertscore_recall: float
    bertscore_f1: float
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def _document_text(document: Any) -> str:
    """딕셔너리 또는 LangChain Document에서 본문을 추출한다."""

    if isinstance(document, Mapping):
        return str(
            document.get("text", document.get("page_content", "")) or ""
        ).strip()
    return str(getattr(document, "page_content", "") or "").strip()


def _as_float(value: Any) -> float:
    """RAGAS 결과의 숫자 또는 결측값을 float로 변환한다."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _configure_gitpython() -> None:
    """Windows conda 환경에서 RAGAS의 GitPython 초기화 오류를 방지한다."""

    if os.environ.get("GIT_PYTHON_GIT_EXECUTABLE"):
        return

    candidate = Path(sys.prefix) / "Library" / "bin" / "git.exe"
    if candidate.exists():
        os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = str(candidate)
    else:
        os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")


def build_ragas_payloads(
    *,
    question: str,
    answers: Sequence[str],
    contexts: Sequence[str],
    reference_answer: str,
) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    """답변별 지표와 질문 공통 검색 지표의 입력을 분리한다."""

    answer_count = len(answers)
    answer_payload = {
        "question": [question] * answer_count,
        "answer": list(answers),
        "contexts": [list(contexts) for _ in answers],
        "ground_truth": [reference_answer] * answer_count,
    }
    context_payload = {
        "question": [question],
        "answer": [answers[0]],
        "contexts": [list(contexts)],
        "ground_truth": [reference_answer],
    }
    return answer_payload, context_payload


class RAGEvaluator:
    """BERTScore 모델과 RAGAS 평가기를 한 번만 로드해 재사용한다."""

    def __init__(
        self,
        *,
        api_key: str,
        model_id: str = DEFAULT_EVALUATION_MODEL,
        embedding_model: str = DEFAULT_EVALUATION_EMBEDDING,
    ) -> None:
        clean_key = api_key.strip()
        if not clean_key:
            raise ValueError("RAGAS 평가에는 UPSTAGE_API_KEY가 필요합니다.")

        self.api_key = clean_key
        self.model_id = model_id
        self.embedding_model = embedding_model
        self._bert_scorer = None
        self._llm = None
        self._embeddings = None

    def _load(self) -> None:
        if self._bert_scorer is not None:
            return

        _configure_gitpython()

        try:
            from bert_score import BERTScorer
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "평가 패키지가 부족합니다. requirements.txt의 bert-score, "
                "ragas, langchain-openai, langchain-huggingface를 설치하세요."
            ) from exc

        print("\n평가 모델 준비:")
        print("- BERTScore: 한국어 지원 모델(CPU)")
        print(f"- RAGAS 평가 LLM: Upstage {self.model_id}")
        print(f"- RAGAS 임베딩: {self.embedding_model}(CPU)")
        print("※ RAGAS 평가 과정에서 Upstage API 사용량이 발생합니다.")

        self._bert_scorer = BERTScorer(lang="ko", device="cpu")
        self._llm = ChatOpenAI(
            model=self.model_id,
            api_key=self.api_key,
            base_url="https://api.upstage.ai/v1",
            temperature=0,
        )
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def evaluate_answer_metric(
        self,
        *,
        metric_name: str,
        question: str,
        answers: Sequence[str],
        documents: Iterable[Any],
    ) -> list[float]:
        """결측 복구용으로 답변 기반 RAGAS 지표 하나만 계산한다."""

        if metric_name not in {"faithfulness", "answer_relevancy"}:
            raise ValueError(
                "답변 지표는 faithfulness 또는 answer_relevancy여야 합니다."
            )
        if not answers:
            return []

        contexts = [
            text for document in documents if (text := _document_text(document))
        ]
        if not contexts:
            raise ValueError("평가에 사용할 검색 문맥이 없습니다.")

        self._load()
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, faithfulness
        except ImportError as exc:
            raise RuntimeError(
                "RAGAS 평가 패키지를 불러오지 못했습니다."
            ) from exc

        selected_metric = (
            faithfulness
            if metric_name == "faithfulness"
            else answer_relevancy
        )
        if metric_name == "answer_relevancy":
            answer_relevancy.strictness = 1

        payload = {
            "question": [question] * len(answers),
            "answer": list(answers),
            "contexts": [list(contexts) for _ in answers],
        }
        result = evaluate(
            dataset=Dataset.from_dict(payload),
            metrics=[selected_metric],
            llm=self._llm,
            embeddings=self._embeddings,
            show_progress=False,
            raise_exceptions=False,
        ).to_pandas()
        return [
            _as_float(result.iloc[index].get(metric_name))
            for index in range(len(answers))
        ]

    def evaluate_context_metrics(
        self,
        *,
        question: str,
        answer: str,
        documents: Iterable[Any],
        reference_answer: str,
    ) -> tuple[float, float]:
        """결측 복구용으로 검색 문맥 지표 두 개만 계산한다."""

        clean_reference = reference_answer.strip()
        if not clean_reference:
            raise ValueError("평가할 때는 모범답안이 필요합니다.")

        contexts = [
            text for document in documents if (text := _document_text(document))
        ]
        if not contexts:
            raise ValueError("평가에 사용할 검색 문맥이 없습니다.")

        self._load()
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import context_precision, context_recall
        except ImportError as exc:
            raise RuntimeError(
                "RAGAS 평가 패키지를 불러오지 못했습니다."
            ) from exc

        payload = {
            "question": [question],
            "answer": [answer],
            "contexts": [list(contexts)],
            "ground_truth": [clean_reference],
        }
        result = evaluate(
            dataset=Dataset.from_dict(payload),
            metrics=[context_precision, context_recall],
            llm=self._llm,
            embeddings=self._embeddings,
            show_progress=False,
            raise_exceptions=False,
        ).to_pandas()
        row = result.iloc[0]
        return (
            _as_float(row.get("context_precision")),
            _as_float(row.get("context_recall")),
        )

    def evaluate(
        self,
        *,
        question: str,
        answers: Sequence[str],
        documents: Iterable[Any],
        reference_answer: str,
    ) -> list[EvaluationScores]:
        """같은 질문·문맥에 대한 여러 프롬프트 답변을 한 번에 평가한다."""

        clean_reference = reference_answer.strip()
        if not clean_reference:
            raise ValueError("평가할 때는 모범답안이 필요합니다.")
        if not answers:
            return []

        contexts = [
            text for document in documents if (text := _document_text(document))
        ]
        if not contexts:
            raise ValueError("평가에 사용할 검색 문맥이 없습니다.")

        self._load()

        bert_precision, bert_recall, bert_f1 = self._bert_scorer.score(
            list(answers),
            [clean_reference] * len(answers),
        )

        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
        except ImportError as exc:
            raise RuntimeError(
                "RAGAS 평가 패키지를 불러오지 못했습니다."
            ) from exc

        # RAGAS 기본값은 Answer Relevancy 후보를 n=3으로 한 번에 요청한다.
        # Upstage Chat API는 n=1만 허용하므로 한 후보만 생성하도록 맞춘다.
        answer_relevancy.strictness = 1

        answer_payload, context_payload = build_ragas_payloads(
            question=question,
            answers=answers,
            contexts=contexts,
            reference_answer=clean_reference,
        )
        answer_result = evaluate(
            dataset=Dataset.from_dict(answer_payload),
            metrics=[
                faithfulness,
                answer_relevancy,
            ],
            llm=self._llm,
            embeddings=self._embeddings,
            show_progress=False,
            raise_exceptions=False,
        ).to_pandas()
        context_result = evaluate(
            dataset=Dataset.from_dict(context_payload),
            metrics=[
                context_precision,
                context_recall,
            ],
            llm=self._llm,
            embeddings=self._embeddings,
            show_progress=False,
            raise_exceptions=False,
        ).to_pandas()
        shared_context_scores = context_result.iloc[0]

        scores: list[EvaluationScores] = []
        for index in range(len(answers)):
            answer_scores = answer_result.iloc[index]
            scores.append(
                EvaluationScores(
                    bertscore_precision=float(bert_precision[index]),
                    bertscore_recall=float(bert_recall[index]),
                    bertscore_f1=float(bert_f1[index]),
                    faithfulness=_as_float(
                        answer_scores.get("faithfulness")
                    ),
                    answer_relevancy=_as_float(
                        answer_scores.get("answer_relevancy")
                    ),
                    context_precision=_as_float(
                        shared_context_scores.get("context_precision")
                    ),
                    context_recall=_as_float(
                        shared_context_scores.get("context_recall")
                    ),
                )
            )

        return scores


def format_score(value: Any) -> str:
    """0~1 지표를 소수 셋째 자리로 표시하고 결측값은 N/A로 표시한다."""

    numeric = _as_float(value)
    if numeric != numeric:
        return "N/A"
    return f"{numeric:.3f}"
