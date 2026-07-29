"""RAG 답변 생성을 위한 프롬프트 템플릿과 문맥 조립 함수.

이 모듈은 검색 및 LLM 구현과 분리되어 있다. 검색 담당 코드에서 반환한
청크 목록을 ``build_prompt``에 전달하면 LLM에 넣을 문자열을 생성한다.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Mapping


class PromptVariant(str, Enum):
    """비교 실험에 사용할 프롬프트 종류."""

    BASIC = "basic"
    ROLE = "role"
    CONSTRAINT = "constraint"
    FEW_SHOT = "few_shot"


PROMPT_LABELS = {
    PromptVariant.BASIC: "CONTEXT + QUESTION",
    PromptVariant.ROLE: "ROLE",
    PromptVariant.CONSTRAINT: "INSTRUCTION / CONSTRAINT",
    PromptVariant.FEW_SHOT: "FEW-SHOT",
}


BASIC_TEMPLATE = """다음 참고 문서를 바탕으로 질문에 답하세요.

[참고 문서]
{context}

[질문]
{question}

[답변]
"""


ROLE_TEMPLATE = """당신은 자동차 취급설명서를 설명하는 자동차 전문가입니다.
사용자가 이해하기 쉽도록 다음 참고 문서의 내용을 종합하여 답하세요.
문서에서 답을 찾을 수 없는 경우에는 추측하지 말고
"차량 취급설명서에서 해당 내용을 찾지 못했습니다."라고 답하세요.

[참고 문서]
{context}

[질문]
{question}

[답변]
"""


CONSTRAINT_TEMPLATE = """다음 참고 문서를 바탕으로 질문에 답하세요.

다음 규칙을 반드시 지키세요.
1. 모든 참고 문서를 먼저 확인하세요.

2. 질문에 직접 관련된 근거가 하나라도 있으면 반드시 그 근거를 사용해
   답하세요.

3. 문서마다 대상, 유형, 조건 또는 적용 기준이 다르면 임의로 하나로
   합치지 말고 각각 구분하여 답하세요.

4. 문서 일부가 불완전하거나 구조가 명확하지 않더라도 본문에서 직접
   확인할 수 있는 내용은 근거로 사용할 수 있습니다. 다만 문서에
   명시되지 않은 수치, 조건 또는 정보 간의 관계를 추측해 단정하지 마세요.

5. 모든 문서를 확인해도 관련 근거가 전혀 없을 때만
   "차량 취급설명서에서 해당 내용을 찾지 못했습니다."라고 답하세요.
[참고 문서]
{context}

[질문]
{question}

[답변]
"""


FEW_SHOT_TEMPLATE = """당신은 자동차 취급설명서 기반 질의응답 도우미입니다.
아래 두 예시는 판단 방법과 출력 형식만 보여줍니다.
예시의 차량·페이지·내용을 실제 질문의 근거로 사용하지 마세요.

[예시 1: 문서에 답이 있는 경우]
참고 문서: [차종: sample_car | 페이지: 10] 투싼의 험로 주행 모드에 대해서 설명하는 페이지
질문: 투싼의 주행 모드에는 어떤 모드가 있습니까?
답변: 투싼의 주행 모드에는 AUTO, SNOW, MUD, SAND 모드가 있습니다.
[출처: sample_car 취급 설명서 p.10]

[예시 2: 문서에 답이 없는 경우]
참고 문서: [차종: sample_car | 페이지: 20] 타이어 공기압 점검 방법이 설명되어 있는 페이지
질문: 아이오닉6의 연료통은 몇L 입니까?
답변: "차량 취급설명서에서는 해당 내용을 찾지 못했습니다."

[실제 참고 문서]
{context}

[실제 질문]
{question}

[답변]
"""


PROMPT_TEMPLATES = {
    PromptVariant.BASIC: BASIC_TEMPLATE,
    PromptVariant.ROLE: ROLE_TEMPLATE,
    PromptVariant.CONSTRAINT: CONSTRAINT_TEMPLATE,
    PromptVariant.FEW_SHOT: FEW_SHOT_TEMPLATE,
}


def _extract_document(document: Any) -> tuple[str, Mapping[str, Any]]:
    """청크 딕셔너리 또는 LangChain Document에서 본문과 메타데이터를 꺼낸다."""

    if isinstance(document, Mapping):
        text = document.get("text", document.get("page_content", ""))
        nested_metadata = document.get("metadata")
        metadata = nested_metadata if isinstance(nested_metadata, Mapping) else document
        return str(text or "").strip(), metadata

    if hasattr(document, "page_content"):
        text = str(getattr(document, "page_content", "") or "").strip()
        metadata = getattr(document, "metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        return text, metadata

    raise TypeError(
        "documents의 각 항목은 청크 딕셔너리 또는 page_content를 가진 객체여야 합니다."
    )


def format_context(documents: Iterable[Any]) -> str:
    """여러 검색 청크를 출처 정보가 포함된 하나의 문맥으로 만든다."""

    blocks: list[str] = []

    for index, document in enumerate(documents, start=1):
        text, metadata = _extract_document(document)
        if not text:
            continue

        car = metadata.get("car", "unknown")
        page = metadata.get("page", "unknown")
        chunk_id = metadata.get("chunk_id")
        needs_review = metadata.get("needs_review", False)

        label = f"[문서 {index} | 차종: {car} | 페이지: {page}"
        if chunk_id:
            label += f" | 청크: {chunk_id}"
        if needs_review:
            label += " | 검토 필요"
        label += "]"

        blocks.append(f"{label}\n{text}")

    if not blocks:
        return "검색된 참고 문서 없음"

    return "\n\n".join(blocks)


def build_prompt(
    question: str,
    documents: Iterable[Any],
    variant: PromptVariant | str = PromptVariant.CONSTRAINT,
    car: str | None = None,
) -> str:
    """질문과 검색 청크를 선택한 템플릿에 넣어 최종 프롬프트를 생성한다."""

    clean_question = question.strip()
    if not clean_question:
        raise ValueError("question은 비어 있을 수 없습니다.")

    try:
        selected_variant = PromptVariant(variant)
    except ValueError as exc:
        choices = ", ".join(item.value for item in PromptVariant)
        raise ValueError(f"variant는 다음 중 하나여야 합니다: {choices}") from exc

    context = format_context(documents)
    target_car = car.strip() if car and car.strip() else "지정되지 않음"

    return PROMPT_TEMPLATES[selected_variant].format(
        context=context,
        question=clean_question,
        car=target_car,
    )
