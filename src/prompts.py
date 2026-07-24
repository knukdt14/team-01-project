"""
prompts.py — [6] 프롬프트 구성  ─ 담당 D

같은 모델·같은 검색 결과에 프롬프트만 바꿔 성능 차이를 측정한다.
평가셋의 '답 없음' 유형 문항이 constrained 프롬프트의 효과를 직접 검증한다.

실행:  python src/prompts.py
"""

from langchain_core.prompts import ChatPromptTemplate

# ── 1. 기본형 ────────────────────────────────────────────────
BASIC = """다음 문맥을 참고하여 질문에 답하세요.

[문맥]
{context}

[질문]
{question}
"""

# ── 2. 역할 부여형 ────────────────────────────────────────────
ROLE = """당신은 현대·기아 자동차 취급설명서에 정통한 자동차 정비 전문가입니다.
운전자가 이해하기 쉽도록 정확하고 친절하게 답변하세요.

[문맥]
{context}

[질문]
{question}
"""

# ── 3. 제약형 (할루시네이션 억제) ────────────────────────────────
CONSTRAINED = """당신은 자동차 취급설명서 질의응답 어시스턴트입니다.
아래 규칙을 반드시 지키세요.

규칙:
1. 반드시 주어진 [문맥]에 있는 내용만으로 답하세요.
2. 문맥에 근거가 없으면 반드시 "해당 정보 없음"이라고만 답하세요.
3. 추측하거나 일반 상식으로 보충하지 마세요.
4. 질문의 차량과 문맥의 차량이 다르면 "해당 차량 정보 없음"이라고 답하세요.
5. 답변 끝에 근거가 된 문장을 [근거] 로 인용하세요.

[문맥]
{context}

[질문]
{question}
"""

TEMPLATES = {
    "basic": BASIC,
    "role": ROLE,
    "constrained": CONSTRAINED,
}


def get_prompt(prompt_type: str = "basic") -> ChatPromptTemplate:
    if prompt_type not in TEMPLATES:
        raise ValueError(f"알 수 없는 프롬프트 유형: {prompt_type} (가능: {list(TEMPLATES)})")
    return ChatPromptTemplate.from_template(TEMPLATES[prompt_type])


if __name__ == "__main__":
    import io
    import sys

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    for name in TEMPLATES:
        print(f"\n{'=' * 60}\n### {name}\n{'=' * 60}")
        print(TEMPLATES[name])
