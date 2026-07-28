"""Few-shot 프롬프트를 사용하는 사용자용 RAG 질의응답 실행 파일."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path
from typing import Callable

from prompt_templates import PromptVariant, build_prompt
from retrieval_adapter import (
    DEFAULT_TOP_K,
    SUPPORTED_CARS,
    create_retriever,
    detect_car_from_question,
    retrieve_documents,
)
from run_local_model import (
    DEFAULT_ENV_FILE,
    DEFAULT_MODEL,
    DEFAULT_UPSTAGE_MODEL,
    MODEL_PROVIDERS,
    apply_output_guard,
    is_exit_command,
    load_answer_generator,
    load_env_file,
    normalize_huggingface_settings,
    resolve_model_id,
    usable_hf_token,
)


FINAL_VARIANT = PromptVariant.FEW_SHOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Few-shot 프롬프트 기반 자동차 취급설명서 질의응답"
    )
    parser.add_argument(
        "--question",
        default=None,
        help="단건 실행 질문(생략하면 대화형 모드)",
    )
    parser.add_argument(
        "--car",
        choices=SUPPORTED_CARS,
        default=None,
        help="질문에 차종이 없을 때 사용할 초기 차량",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"검색에 사용할 청크 수(기본값: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--provider",
        choices=MODEL_PROVIDERS,
        default="huggingface",
        help="답변 모델 제공자(기본값: huggingface)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "모델 ID(생략 시 huggingface는 "
            f"{DEFAULT_MODEL}, upstage는 {DEFAULT_UPSTAGE_MODEL})"
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="답변 최대 생성 토큰 수(기본값: 512)",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="불러올 .env 경로",
    )
    return parser.parse_args()


def initialize(args: argparse.Namespace):
    """검색기와 답변 모델을 기술 로그 없이 준비한다."""

    load_env_file(args.env_file)
    token = usable_hf_token(os.environ.get("HF_TOKEN"))
    model_id = resolve_model_id(args.provider, args.model)

    hidden_output = io.StringIO()
    with (
        contextlib.redirect_stdout(hidden_output),
        contextlib.redirect_stderr(hidden_output),
    ):
        if args.provider == "huggingface":
            normalize_huggingface_settings()
        retriever = create_retriever(car=None, k=args.top_k)
        generate = load_answer_generator(
            provider=args.provider,
            model_id=model_id,
            token=token,
            max_new_tokens=args.max_new_tokens,
        )

    return retriever, generate


def generate_few_shot_answer(
    *,
    question: str,
    car: str,
    top_k: int,
    retriever,
    generate: Callable[[str], str],
) -> str:
    """검색 문맥에 Few-shot 템플릿을 적용해 최종 답변만 반환한다."""

    documents = retrieve_documents(
        question=question,
        car=car,
        k=top_k,
        retriever=retriever,
    )
    if not documents:
        return "차량 취급설명서에서 해당 내용을 찾지 못했습니다."

    prompt = build_prompt(
        question=question,
        documents=documents,
        variant=FINAL_VARIANT,
        car=car,
    )
    return apply_output_guard(generate(prompt), FINAL_VARIANT)


def print_answer(answer: str) -> None:
    print("\n[모델 답변]")
    print(answer or "차량 취급설명서에서 해당 내용을 찾지 못했습니다.")


def answer_or_report_error(
    *,
    question: str,
    car: str,
    args: argparse.Namespace,
    retriever,
    generate: Callable[[str], str],
) -> bool:
    try:
        answer = generate_few_shot_answer(
            question=question,
            car=car,
            top_k=args.top_k,
            retriever=retriever,
            generate=generate,
        )
    except Exception:
        print_answer(
            "답변을 불러오지 못했습니다. 잠시 후 다시 질문해 주세요."
        )
        return False

    print_answer(answer)
    return True


def main() -> int:
    args = parse_args()
    if args.max_new_tokens < 1:
        print("--max-new-tokens는 1 이상이어야 합니다.", file=sys.stderr)
        return 2
    if args.top_k < 1:
        print("--top-k는 1 이상이어야 합니다.", file=sys.stderr)
        return 2

    print("답변 시스템을 준비하고 있습니다...")
    try:
        retriever, generate = initialize(args)
    except Exception as exc:
        print(f"답변 시스템을 준비하지 못했습니다: {exc}", file=sys.stderr)
        return 1

    if args.question is not None:
        target_car = detect_car_from_question(args.question) or args.car
        if target_car is None:
            print_answer(
                "질문에 아반떼, 아반떼 하이브리드, 아이오닉6, 넥쏘, "
                "투싼 중 하나를 포함해 주세요."
            )
            return 2

        succeeded = answer_or_report_error(
            question=args.question,
            car=target_car,
            args=args,
            retriever=retriever,
            generate=generate,
        )
        return 0 if succeeded else 1

    print("준비가 완료되었습니다. 차량과 궁금한 내용을 질문해 주세요.")
    print("빈 줄 또는 q, quit, exit, 종료를 입력하면 끝납니다.\n")

    active_car = args.car
    while True:
        try:
            question = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n질의응답을 종료합니다.")
            break

        if not question or is_exit_command(question):
            print("질의응답을 종료합니다.")
            break

        detected_car = detect_car_from_question(question)
        if detected_car is not None:
            active_car = detected_car
        elif active_car is None:
            print_answer(
                "첫 질문에는 아반떼, 아반떼 하이브리드, 아이오닉6, 넥쏘, "
                "투싼 중 하나를 포함해 주세요."
            )
            print()
            continue

        answer_or_report_error(
            question=question,
            car=active_car,
            args=args,
            retriever=retriever,
            generate=generate,
        )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
