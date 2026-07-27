"""Hugging Face 로컬 모델로 프롬프트별 답변을 생성한다.

기본 모델은 Qwen/Qwen2.5-3B-Instruct이다. OpenAI API나 Upstage API를
호출하지 않으므로 답변 생성 과정에서 외부 API 사용료가 발생하지 않는다.
최초 실행 시 Hugging Face에서 모델 파일을 내려받는다.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

from prompt_templates import PromptVariant, build_prompt
from retrieval_adapter import (
    CAR_LABELS,
    DEFAULT_TOP_K,
    SUPPORTED_CARS,
    create_retriever,
    detect_car_from_question,
    retrieve_documents,
)


DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_ENV_FILE = Path(__file__).with_name(".env")
EXIT_COMMANDS = {"q", "quit", "exit", "종료"}


def load_env_file(path: Path) -> None:
    """간단한 KEY=VALUE 형식의 .env를 현재 프로세스에만 불러온다."""

    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


def usable_hf_token(value: str | None) -> str | None:
    """자리표시자처럼 짧은 값은 인증 토큰으로 전달하지 않는다."""

    if not value:
        return None
    clean_value = value.strip()
    if not clean_value.startswith("hf_") or len(clean_value) < 20:
        return None
    return clean_value


def normalize_huggingface_settings() -> None:
    """선택적 가속 패키지가 없으면 일반 다운로드 방식으로 되돌린다."""

    transfer_enabled = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1"
    transfer_installed = importlib.util.find_spec("hf_transfer") is not None
    if transfer_enabled and not transfer_installed:
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
        print("안내: hf_transfer가 없어 Hugging Face 일반 다운로드를 사용합니다.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen2.5 로컬 모델로 RAG 프롬프트 답변 생성"
    )
    parser.add_argument(
        "--variant",
        choices=["all", *(item.value for item in PromptVariant)],
        default=PromptVariant.CONSTRAINT.value,
        help="실행할 프롬프트 종류(기본값: constraint)",
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
        help=f"답변에 사용할 검색 청크 수(기본값: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face 모델 ID(기본값: {DEFAULT_MODEL})",
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
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="모델 답변과 함께 완성된 프롬프트도 출력",
    )
    return parser.parse_args()


def is_exit_command(value: str) -> bool:
    """대화형 입력이 종료 명령인지 확인한다."""

    return value.strip().lower() in EXIT_COMMANDS


def load_model(model_id: str, token: str | None):
    """토크나이저와 인과언어모델을 GPU 우선으로 불러온다."""

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "torch 또는 transformers가 없습니다. TF_ENV 환경을 활성화하세요."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"모델 로딩: {model_id}")
    print(f"실행 장치: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    transformers_major = int(transformers.__version__.split(".", 1)[0])
    dtype_argument = (
        {"dtype": dtype}
        if transformers_major >= 5
        else {"torch_dtype": dtype}
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        **dtype_argument,
    )
    model.to(device)
    model.eval()

    return tokenizer, model, torch, device


def generate_answer(
    prompt: str,
    tokenizer,
    model,
    torch_module,
    device: str,
    max_new_tokens: int,
) -> str:
    """채팅 템플릿을 적용하고 프롬프트 뒤에 생성된 답변만 반환한다."""

    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer(rendered, return_tensors="pt").to(device)
    input_length = model_inputs["input_ids"].shape[1]

    with torch_module.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, input_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def answer_question(
    question: str,
    *,
    car: str,
    args: argparse.Namespace,
    variants: list[PromptVariant],
    retriever,
    tokenizer,
    model,
    torch_module,
    device: str,
) -> bool:
    """한 질문을 검색하고 선택된 프롬프트별 답변을 출력한다."""

    try:
        documents = retrieve_documents(
            question=question,
            car=car,
            k=args.top_k,
            retriever=retriever,
        )
    except Exception as exc:
        print(f"\n문서 검색 실패: {exc}", file=sys.stderr)
        return False

    if not documents:
        print("검색된 문서가 없습니다.", file=sys.stderr)
        return False

    print(f"\n검색된 청크: {len(documents)}개")
    for index, document in enumerate(documents, start=1):
        metadata = getattr(document, "metadata", {})
        review = " (검토 필요)" if metadata.get("needs_review") else ""
        print(
            f"  {index}. {metadata.get('car', 'unknown')} "
            f"p.{metadata.get('page', 'unknown')}{review}"
        )

    for variant in variants:
        prompt = build_prompt(
            question=question,
            documents=documents,
            variant=variant,
            car=car,
        )
        answer = generate_answer(
            prompt=prompt,
            tokenizer=tokenizer,
            model=model,
            torch_module=torch_module,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )

        title = f" {variant.value.upper()} "
        print(f"\n{title:=^76}")
        if args.show_prompt:
            print("[완성된 프롬프트]")
            print(prompt)
            print()
        print("[모델 답변]")
        print(answer or "(빈 답변)")

    return True


def main() -> int:
    args = parse_args()
    if args.max_new_tokens < 1:
        print("--max-new-tokens는 1 이상이어야 합니다.", file=sys.stderr)
        return 2
    if args.top_k < 1:
        print("--top-k는 1 이상이어야 합니다.", file=sys.stderr)
        return 2

    load_env_file(args.env_file)
    normalize_huggingface_settings()
    token = usable_hf_token(os.environ.get("HF_TOKEN"))

    variants = (
        list(PromptVariant)
        if args.variant == "all"
        else [PromptVariant(args.variant)]
    )

    print(f"FAISS retriever 로딩: top_k={args.top_k}")
    print("※ 최초 실행은 모델 다운로드 때문에 시간이 걸릴 수 있습니다.\n")

    try:
        retriever = create_retriever(
            car=None,
            k=args.top_k,
        )
    except Exception as exc:
        print(f"\nFAISS retriever 로딩 실패: {exc}", file=sys.stderr)
        return 1

    try:
        tokenizer, model, torch_module, device = load_model(args.model, token)
    except Exception as exc:
        print(f"\n모델 로딩 실패: {exc}", file=sys.stderr)
        return 1

    if args.question is not None:
        target_car = detect_car_from_question(args.question) or args.car
        if target_car is None:
            print(
                "질문에서 차종을 찾지 못했습니다. 질문에 아반떼, 아반떼 "
                "하이브리드, 아이오닉6, 넥쏘, 투싼 중 하나를 포함하거나 "
                "--car 옵션을 사용하세요.",
                file=sys.stderr,
            )
            return 2

        print(f"인식된 차량: {CAR_LABELS[target_car]} ({target_car})")
        succeeded = answer_question(
            question=args.question,
            car=target_car,
            args=args,
            variants=variants,
            retriever=retriever,
            tokenizer=tokenizer,
            model=model,
            torch_module=torch_module,
            device=device,
        )
        return 0 if succeeded else 1

    print("\n대화형 모드")
    if args.car is None:
        print("- 현재 차량: 미선택(첫 질문에 차종을 포함하세요)")
    else:
        print(f"- 초기 차량: {CAR_LABELS[args.car]} ({args.car})")
    print("- 질문에서 차종을 인식하면 해당 차량으로 자동 전환합니다.")
    print("- 이후 질문에 차종을 생략하면 마지막으로 인식한 차량을 유지합니다.")
    print("- 빈 줄 또는 q, quit, exit, 종료를 입력하면 끝납니다.")
    print("- 각 질문은 이전 대화와 분리하여 독립적으로 검색합니다.\n")

    active_car = args.car
    while True:
        try:
            question = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n대화형 모드를 종료합니다.")
            break

        if not question or is_exit_command(question):
            print("대화형 모드를 종료합니다.")
            break

        detected_car = detect_car_from_question(question)
        if detected_car is not None:
            if detected_car != active_car:
                print(
                    f"차량 인식: {CAR_LABELS[detected_car]} "
                    f"({detected_car})"
                )
            active_car = detected_car
        elif active_car is None:
            print(
                "차종을 인식하지 못했습니다. 질문에 아반떼, 아반떼 하이브리드, "
                "아이오닉6, 넥쏘, 투싼 중 하나를 포함해 주세요.\n"
            )
            continue
        else:
            print(f"현재 차량 유지: {CAR_LABELS[active_car]} ({active_car})")

        answer_question(
            question=question,
            car=active_car,
            args=args,
            variants=variants,
            retriever=retriever,
            tokenizer=tokenizer,
            model=model,
            torch_module=torch_module,
            device=device,
        )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
