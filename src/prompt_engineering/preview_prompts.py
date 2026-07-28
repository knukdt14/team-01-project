"""비교 실험용 프롬프트 4종을 터미널에서 미리 확인한다."""

from __future__ import annotations

import argparse

from prompt_templates import PROMPT_LABELS, PromptVariant, build_prompt


SAMPLE_DOCUMENTS = [
    {
        "text": (
            "예시 문서입니다. 일반 조건의 엔진오일 교환주기는 "
            "A km 또는 B개월입니다."
        ),
        "car": "avante",
        "page": 390,
        "chunk_id": "avante_p390_0",
    },
    {
        "text": (
            "예시 문서입니다. 가혹 조건의 엔진오일 교환주기는 "
            "C km 또는 D개월입니다."
        ),
        "car": "avante",
        "page": 391,
        "chunk_id": "avante_p391_0",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 프롬프트 4종 미리보기")
    parser.add_argument(
        "--variant",
        choices=["all", *(item.value for item in PromptVariant)],
        default="all",
        help="출력할 프롬프트 종류(기본값: all)",
    )
    parser.add_argument(
        "--question",
        default="아반떼 엔진오일 교환주기는?",
        help="미리보기에 사용할 질문",
    )
    parser.add_argument(
        "--car",
        default="avante",
        help="대상 차량 메타데이터",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = (
        list(PromptVariant)
        if args.variant == "all"
        else [PromptVariant(args.variant)]
    )

    print("※ 형식 확인용 예시이며, 실제 설명서 검색 결과나 LLM 답변이 아닙니다.")

    for variant in variants:
        prompt = build_prompt(
            question=args.question,
            documents=SAMPLE_DOCUMENTS,
            variant=variant,
            car=args.car,
        )
        title = f" {PROMPT_LABELS[variant]} "
        print(f"\n{title:=^76}")
        print(prompt)


if __name__ == "__main__":
    main()
