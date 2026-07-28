"""RAG 질문 100개 중 25개를 뽑아 프롬프트 4종을 일괄 평가한다."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from batch_evaluation import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_SAMPLE_SIZE,
    BenchmarkQuestion,
    dataset_sha256,
    load_benchmark_questions,
    load_checkpoint,
    select_random_questions,
    summarize_results,
    utc_timestamp,
    write_checkpoint,
    write_detail_csv,
    write_html_report,
    write_summary_csv,
)
from evaluation_metrics import DEFAULT_EVALUATION_MODEL, RAGEvaluator, format_score
from prompt_templates import PROMPT_LABELS, PromptVariant, build_prompt
from retrieval_adapter import (
    DEFAULT_TOP_K,
    RETRIEVER_LABEL,
    create_retriever,
    retrieve_documents,
)
from run_local_model import (
    DEFAULT_ENV_FILE,
    DEFAULT_MODEL,
    MODEL_PROVIDERS,
    apply_output_guard,
    load_answer_generator,
    load_env_file,
    normalize_huggingface_settings,
    resolve_model_id,
    usable_hf_token,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "RAG_Question_100.xlsx"
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("evaluation_results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RAG_Question_100.xlsx에서 같은 질문 표본을 뽑아 "
            "프롬프트 4종의 평균 성능을 비교"
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"질문·모범답안 xlsx 경로(기본값: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"무작위 추출 질문 수(기본값: {DEFAULT_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"무작위 추출 재현용 seed(기본값: {DEFAULT_RANDOM_SEED})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"질문당 검색 청크 수(기본값: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--provider",
        choices=MODEL_PROVIDERS,
        default="huggingface",
        help="답변 생성 모델 제공자(기본값: huggingface)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"답변 모델 ID(기본 로컬 모델: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="답변당 최대 생성 토큰 수(기본값: 512)",
    )
    parser.add_argument(
        "--evaluation-model",
        default=DEFAULT_EVALUATION_MODEL,
        help=(
            "RAGAS 평가용 Upstage 모델"
            f"(기본값: {DEFAULT_EVALUATION_MODEL})"
        ),
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="UPSTAGE_API_KEY와 HF_TOKEN을 읽을 .env 경로",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"JSON·CSV·HTML 저장 폴더(기본값: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="모델·API를 호출하지 않고 선정 질문만 확인",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="같은 seed의 중단된 평가를 체크포인트에서 계속",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="같은 seed의 기존 결과를 새 평가로 덮어씀",
    )
    parser.add_argument(
        "--repair-missing",
        action="store_true",
        help="저장된 답변은 유지하고 결측 RAGAS 지표만 다시 평가",
    )
    parser.add_argument(
        "--repair-attempts",
        type=int,
        default=3,
        help="결측 RAGAS 지표의 최대 재시도 횟수(기본값: 3)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="API 사용 안내 확인 질문을 생략하고 바로 실행",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.sample_size < 1:
        raise ValueError("--sample-size는 1 이상이어야 합니다.")
    if args.top_k < 1:
        raise ValueError("--top-k는 1 이상이어야 합니다.")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens는 1 이상이어야 합니다.")
    if args.repair_attempts < 1:
        raise ValueError("--repair-attempts는 1 이상이어야 합니다.")
    if args.resume and args.overwrite:
        raise ValueError("--resume과 --overwrite는 동시에 사용할 수 없습니다.")
    if args.repair_missing and (args.resume or args.overwrite):
        raise ValueError(
            "--repair-missing은 --resume 또는 --overwrite와 함께 사용할 수 없습니다."
        )


def _checkpoint_path(output_dir: Path, seed: int) -> Path:
    return output_dir / f"batch_evaluation_seed{seed}.json"


def _serialize_document(document: Any) -> dict[str, Any]:
    if isinstance(document, Mapping):
        text = document.get("text", document.get("page_content", ""))
        nested_metadata = document.get("metadata")
        metadata = nested_metadata if isinstance(nested_metadata, Mapping) else document
    else:
        text = getattr(document, "page_content", "")
        raw_metadata = getattr(document, "metadata", {})
        metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}

    return {
        "text": str(text or ""),
        "metadata": {
            "car": metadata.get("car"),
            "page": metadata.get("page"),
            "chunk_id": metadata.get("chunk_id"),
            "needs_review": bool(metadata.get("needs_review", False)),
        },
    }


def _new_checkpoint(
    *,
    args: argparse.Namespace,
    sampled_questions: list[BenchmarkQuestion],
    answer_model: str,
    dataset_hash: str,
) -> dict[str, Any]:
    timestamp = utc_timestamp()
    return {
        "schema_version": 1,
        "status": "running",
        "created_at": timestamp,
        "updated_at": timestamp,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": dataset_hash,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "sampled_ids": [
            question.question_id for question in sampled_questions
        ],
        "top_k": args.top_k,
        "retriever_backend": RETRIEVER_LABEL,
        "answer_provider": args.provider,
        "answer_model": answer_model,
        "evaluation_model": args.evaluation_model,
        "max_new_tokens": args.max_new_tokens,
        "variants": [variant.value for variant in PromptVariant],
        "results": [],
    }


def _validate_resume(
    checkpoint: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    sampled_questions: list[BenchmarkQuestion],
    answer_model: str,
    dataset_hash: str,
) -> None:
    expected = {
        "dataset_sha256": dataset_hash,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "sampled_ids": [
            question.question_id for question in sampled_questions
        ],
        "top_k": args.top_k,
        "retriever_backend": RETRIEVER_LABEL,
        "answer_provider": args.provider,
        "answer_model": answer_model,
        "evaluation_model": args.evaluation_model,
        "max_new_tokens": args.max_new_tokens,
        "variants": [variant.value for variant in PromptVariant],
    }
    mismatches = [
        key
        for key, value in expected.items()
        if checkpoint.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "체크포인트와 현재 실행 조건이 다릅니다: "
            + ", ".join(mismatches)
            + ". 기존 조건으로 실행하거나 --overwrite를 사용하세요."
        )


def _print_sample(sampled_questions: list[BenchmarkQuestion]) -> None:
    print("\n무작위 선정 질문")
    for index, question in enumerate(sampled_questions, start=1):
        print(
            f"{index:>2}. ID {question.question_id} | "
            f"{question.vehicle} | {question.answerability} | "
            f"{question.question}"
        )


def _confirm_paid_evaluation(args: argparse.Namespace) -> bool:
    if args.yes:
        return True

    answer_count = args.sample_size * len(PromptVariant)
    print("\n실행 전 확인")
    print(
        f"- 질문 {args.sample_size}개 × 프롬프트 {len(PromptVariant)}종 "
        f"= 로컬 답변 {answer_count}개를 생성합니다."
    )
    print("- BERTScore는 로컬에서, RAGAS는 Upstage Solar API로 평가합니다.")
    print("- 실행 중단 시 같은 명령에 --resume을 붙이면 완료 지점부터 계속합니다.")
    try:
        confirmation = input("API 사용량이 발생합니다. 계속할까요? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return confirmation.strip().lower() in {"y", "yes", "예", "네"}


def _print_question_scores(
    *,
    question_index: int,
    sample_size: int,
    question: BenchmarkQuestion,
    variant_results: Mapping[str, Mapping[str, Any]],
) -> None:
    print(
        f"\n[{question_index}/{sample_size}] ID {question.question_id} "
        f"| {question.vehicle} | {question.question}"
    )
    for variant in PromptVariant:
        result = variant_results[variant.value]
        print(
            f"- {PROMPT_LABELS[variant]}: "
            f"BERT F1={format_score(result['bertscore_f1'])}, "
            f"Faith={format_score(result['faithfulness'])}, "
            f"Relevancy={format_score(result['answer_relevancy'])}, "
            f"시간={result['generation_time_seconds']:.2f}초"
        )


def _generate_and_evaluate_question(
    *,
    question: BenchmarkQuestion,
    args: argparse.Namespace,
    retriever: Any,
    generate: Any,
    evaluator: RAGEvaluator,
) -> dict[str, Any]:
    documents = retrieve_documents(
        question=question.question,
        car=question.car,
        k=args.top_k,
        retriever=retriever,
    )
    if not documents:
        raise RuntimeError("검색된 문서가 없습니다.")

    answers: list[str] = []
    generation_times: list[float] = []
    for variant in PromptVariant:
        prompt = build_prompt(
            question=question.question,
            documents=documents,
            variant=variant,
            car=question.car,
        )
        started_at = time.perf_counter()
        answer = apply_output_guard(generate(prompt), variant)
        elapsed_seconds = time.perf_counter() - started_at
        answers.append(answer)
        generation_times.append(elapsed_seconds)

    scores = evaluator.evaluate(
        question=question.question,
        answers=answers,
        documents=documents,
        reference_answer=question.reference_answer,
    )
    if len(scores) != len(PromptVariant):
        raise RuntimeError(
            "평가 결과 수가 프롬프트 수와 일치하지 않습니다."
        )

    variant_results: dict[str, dict[str, Any]] = {}
    for variant, answer, elapsed_seconds, score in zip(
        PromptVariant,
        answers,
        generation_times,
        scores,
    ):
        metrics = asdict(score)
        variant_results[variant.value] = {
            "answer": answer,
            "generation_time_seconds": elapsed_seconds,
            **{
                metric: (
                    value if math.isfinite(float(value)) else None
                )
                for metric, value in metrics.items()
            },
        }

    return {
        **question.to_dict(),
        "retrieved_documents": [
            _serialize_document(document) for document in documents
        ],
        "variants": variant_results,
    }


def _write_final_outputs(
    *,
    output_dir: Path,
    seed: int,
    checkpoint: Mapping[str, Any],
) -> tuple[Path, Path, Path]:
    results = checkpoint.get("results", [])
    summary = summarize_results(results)
    summary_path = output_dir / f"batch_summary_seed{seed}.csv"
    detail_path = output_dir / f"batch_details_seed{seed}.csv"
    report_path = output_dir / f"batch_report_seed{seed}.html"

    write_summary_csv(summary_path, summary)
    write_detail_csv(detail_path, results)
    write_html_report(
        report_path,
        summary=summary,
        results=results,
        metadata=checkpoint,
    )
    return summary_path, detail_path, report_path


def _is_missing(value: Any) -> bool:
    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError):
        return True


def _retry_ragas_call(
    *,
    label: str,
    attempts: int,
    operation: Any,
    is_complete: Any,
) -> Any:
    latest_result = None
    for attempt in range(1, attempts + 1):
        try:
            latest_result = operation()
        except Exception as exc:
            print(
                f"  {label} 재평가 {attempt}/{attempts} 실패: {exc}",
                file=sys.stderr,
            )
        else:
            if is_complete(latest_result):
                return latest_result
            print(
                f"  {label} 재평가 {attempt}/{attempts}: 결과가 N/A입니다.",
                file=sys.stderr,
            )

        if attempt < attempts:
            time.sleep(min(2**attempt, 8))
    return latest_result


def _confirm_repair(args: argparse.Namespace, missing_count: int) -> bool:
    if args.yes:
        return True
    print(
        f"\n결측 RAGAS 값 {missing_count}개만 Solar API로 다시 평가합니다."
    )
    print("저장된 Qwen 답변은 다시 생성하지 않습니다.")
    try:
        confirmation = input("API 사용량이 발생합니다. 계속할까요? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return confirmation.strip().lower() in {"y", "yes", "예", "네"}


def _repair_missing_metrics(
    *,
    args: argparse.Namespace,
    sampled_questions: list[BenchmarkQuestion],
    answer_model: str,
    dataset_hash: str,
) -> int:
    checkpoint_path = _checkpoint_path(args.output_dir, args.seed)
    if not checkpoint_path.is_file():
        print(
            f"복구할 체크포인트가 없습니다: {checkpoint_path}",
            file=sys.stderr,
        )
        return 2

    try:
        checkpoint = load_checkpoint(checkpoint_path)
        _validate_resume(
            checkpoint,
            args=args,
            sampled_questions=sampled_questions,
            answer_model=answer_model,
            dataset_hash=dataset_hash,
        )
    except Exception as exc:
        print(f"체크포인트 확인 실패: {exc}", file=sys.stderr)
        return 2

    results = checkpoint.get("results", [])
    if len(results) != args.sample_size:
        print(
            "25개 질문 평가가 끝난 체크포인트에서만 결측 복구를 실행할 수 "
            "있습니다. 먼저 --resume으로 본 평가를 완료하세요.",
            file=sys.stderr,
        )
        return 2

    ragas_metrics = (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    )
    missing_count = sum(
        _is_missing(variant_result.get(metric))
        for result in results
        for variant_result in result.get("variants", {}).values()
        for metric in ragas_metrics
    )
    if missing_count == 0:
        print("복구할 결측 RAGAS 지표가 없습니다.")
        _write_final_outputs(
            output_dir=args.output_dir,
            seed=args.seed,
            checkpoint=checkpoint,
        )
        return 0
    if not _confirm_repair(args, missing_count):
        print("결측 지표 복구를 취소했습니다.")
        return 0

    load_env_file(args.env_file)
    api_key = os.environ.get("UPSTAGE_API_KEY", "").strip()
    if not api_key:
        print(
            "평가 준비 실패: .env에 UPSTAGE_API_KEY가 필요합니다.",
            file=sys.stderr,
        )
        return 2

    evaluator = RAGEvaluator(
        api_key=api_key,
        model_id=args.evaluation_model,
    )
    checkpoint["status"] = "repairing"
    checkpoint["updated_at"] = utc_timestamp()
    write_checkpoint(checkpoint_path, checkpoint)

    try:
        for question_index, result in enumerate(results, start=1):
            variants = result.get("variants", {})
            documents = result.get("retrieved_documents", [])
            question = str(result.get("question", ""))
            reference_answer = str(result.get("reference_answer", ""))
            repaired_this_question = 0

            for metric_name in ("faithfulness", "answer_relevancy"):
                missing_variants = [
                    variant
                    for variant in PromptVariant
                    if _is_missing(
                        variants.get(variant.value, {}).get(metric_name)
                    )
                ]
                if not missing_variants:
                    continue

                answers = [
                    str(variants[variant.value].get("answer", ""))
                    for variant in missing_variants
                ]
                repaired_scores = _retry_ragas_call(
                    label=f"ID {result.get('question_id')} {metric_name}",
                    attempts=args.repair_attempts,
                    operation=lambda metric_name=metric_name, answers=answers: (
                        evaluator.evaluate_answer_metric(
                            metric_name=metric_name,
                            question=question,
                            answers=answers,
                            documents=documents,
                        )
                    ),
                    is_complete=lambda values: (
                        values is not None
                        and len(values) == len(answers)
                        and all(not _is_missing(value) for value in values)
                    ),
                )
                if repaired_scores is None:
                    continue
                for variant, value in zip(
                    missing_variants,
                    repaired_scores,
                ):
                    if not _is_missing(value):
                        variants[variant.value][metric_name] = float(value)
                        repaired_this_question += 1

            context_missing = any(
                _is_missing(
                    variants.get(variant.value, {}).get(metric_name)
                )
                for variant in PromptVariant
                for metric_name in ("context_precision", "context_recall")
            )
            if context_missing:
                basic_answer = str(
                    variants.get(PromptVariant.BASIC.value, {}).get(
                        "answer",
                        "",
                    )
                )
                context_scores = _retry_ragas_call(
                    label=f"ID {result.get('question_id')} context",
                    attempts=args.repair_attempts,
                    operation=lambda: evaluator.evaluate_context_metrics(
                        question=question,
                        answer=basic_answer,
                        documents=documents,
                        reference_answer=reference_answer,
                    ),
                    is_complete=lambda values: (
                        values is not None
                        and len(values) == 2
                        and all(not _is_missing(value) for value in values)
                    ),
                )
                if context_scores is not None:
                    context_precision_score, context_recall_score = context_scores
                    for variant in PromptVariant:
                        if not _is_missing(context_precision_score):
                            if _is_missing(
                                variants[variant.value].get(
                                    "context_precision"
                                )
                            ):
                                repaired_this_question += 1
                            variants[variant.value]["context_precision"] = float(
                                context_precision_score
                            )
                        if not _is_missing(context_recall_score):
                            if _is_missing(
                                variants[variant.value].get("context_recall")
                            ):
                                repaired_this_question += 1
                            variants[variant.value]["context_recall"] = float(
                                context_recall_score
                            )

            if repaired_this_question:
                checkpoint["updated_at"] = utc_timestamp()
                write_checkpoint(checkpoint_path, checkpoint)
                print(
                    f"[{question_index}/{args.sample_size}] "
                    f"ID {result.get('question_id')}: "
                    f"결측 {repaired_this_question}개 복구"
                )
    except Exception as exc:
        checkpoint["status"] = "repair_interrupted"
        checkpoint["updated_at"] = utc_timestamp()
        write_checkpoint(checkpoint_path, checkpoint)
        print(f"결측 지표 복구 중단: {exc}", file=sys.stderr)
        return 1

    remaining_missing = sum(
        _is_missing(variant_result.get(metric))
        for result in results
        for variant_result in result.get("variants", {}).values()
        for metric in ragas_metrics
    )
    checkpoint["status"] = "complete"
    checkpoint["updated_at"] = utc_timestamp()
    checkpoint["remaining_missing_ragas_values"] = remaining_missing
    write_checkpoint(checkpoint_path, checkpoint)
    summary_path, detail_path, report_path = _write_final_outputs(
        output_dir=args.output_dir,
        seed=args.seed,
        checkpoint=checkpoint,
    )

    print("\n" + " MISSING METRICS REPAIR COMPLETE ".center(76, "="))
    print(f"남은 결측 RAGAS 값: {remaining_missing}개")
    print(f"평균 지표 CSV: {summary_path}")
    print(f"질문별 상세 CSV: {detail_path}")
    print(f"시각화 HTML: {report_path}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        _validate_args(args)
        questions = load_benchmark_questions(args.dataset)
        sampled_questions = select_random_questions(
            questions,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"평가 데이터 준비 실패: {exc}", file=sys.stderr)
        return 2

    print(f"전체 질문: {len(questions)}개")
    print(
        f"선정 방식: 단순 무작위 {args.sample_size}개 "
        f"(seed={args.seed})"
    )
    _print_sample(sampled_questions)
    if args.dry_run:
        print("\nDRY RUN: 모델과 Upstage API를 호출하지 않았습니다.")
        return 0

    answer_model = resolve_model_id(args.provider, args.model)
    current_dataset_hash = dataset_sha256(args.dataset)
    if args.repair_missing:
        return _repair_missing_metrics(
            args=args,
            sampled_questions=sampled_questions,
            answer_model=answer_model,
            dataset_hash=current_dataset_hash,
        )

    if not _confirm_paid_evaluation(args):
        print("평가를 취소했습니다.")
        return 0

    load_env_file(args.env_file)
    if args.provider == "huggingface":
        normalize_huggingface_settings()
    if not os.environ.get("UPSTAGE_API_KEY", "").strip():
        print(
            "평가 준비 실패: .env에 UPSTAGE_API_KEY가 필요합니다.",
            file=sys.stderr,
        )
        return 2

    checkpoint_path = _checkpoint_path(args.output_dir, args.seed)

    if checkpoint_path.exists() and not (args.resume or args.overwrite):
        print(
            f"기존 체크포인트가 있습니다: {checkpoint_path}\n"
            "계속하려면 --resume, 처음부터 다시 하려면 --overwrite를 "
            "사용하세요.",
            file=sys.stderr,
        )
        return 2

    try:
        if args.resume:
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"재개할 체크포인트가 없습니다: {checkpoint_path}"
                )
            checkpoint = load_checkpoint(checkpoint_path)
            _validate_resume(
                checkpoint,
                args=args,
                sampled_questions=sampled_questions,
                answer_model=answer_model,
                dataset_hash=current_dataset_hash,
            )
            checkpoint["status"] = "running"
            checkpoint["updated_at"] = utc_timestamp()
        else:
            checkpoint = _new_checkpoint(
                args=args,
                sampled_questions=sampled_questions,
                answer_model=answer_model,
                dataset_hash=current_dataset_hash,
            )
        write_checkpoint(checkpoint_path, checkpoint)
    except Exception as exc:
        print(f"체크포인트 준비 실패: {exc}", file=sys.stderr)
        return 2

    completed_ids = {
        str(result.get("question_id"))
        for result in checkpoint.get("results", [])
    }
    remaining_count = sum(
        question.question_id not in completed_ids
        for question in sampled_questions
    )
    print(f"\n완료: {len(completed_ids)}개 / 남음: {remaining_count}개")
    if remaining_count == 0:
        checkpoint["status"] = "complete"
        checkpoint["updated_at"] = utc_timestamp()
        write_checkpoint(checkpoint_path, checkpoint)
        summary_path, detail_path, report_path = _write_final_outputs(
            output_dir=args.output_dir,
            seed=args.seed,
            checkpoint=checkpoint,
        )
        print(f"이미 평가가 완료되어 보고서를 다시 만들었습니다: {report_path}")
        print(f"요약 CSV: {summary_path}")
        print(f"상세 CSV: {detail_path}")
        return 0

    try:
        print(
            f"\n{RETRIEVER_LABEL} retriever 로딩: "
            f"top_k={args.top_k}"
        )
        retriever = create_retriever(car=None, k=args.top_k)
        generate = load_answer_generator(
            provider=args.provider,
            model_id=answer_model,
            token=usable_hf_token(os.environ.get("HF_TOKEN")),
            max_new_tokens=args.max_new_tokens,
        )
        evaluator = RAGEvaluator(
            api_key=os.environ.get("UPSTAGE_API_KEY", ""),
            model_id=args.evaluation_model,
        )
    except Exception as exc:
        print(f"모델 또는 평가기 준비 실패: {exc}", file=sys.stderr)
        return 1

    try:
        for question_index, question in enumerate(
            sampled_questions,
            start=1,
        ):
            if question.question_id in completed_ids:
                continue

            result = _generate_and_evaluate_question(
                question=question,
                args=args,
                retriever=retriever,
                generate=generate,
                evaluator=evaluator,
            )
            checkpoint["results"].append(result)
            completed_ids.add(question.question_id)
            checkpoint["updated_at"] = utc_timestamp()
            write_checkpoint(checkpoint_path, checkpoint)
            _print_question_scores(
                question_index=question_index,
                sample_size=args.sample_size,
                question=question,
                variant_results=result["variants"],
            )
            print(f"체크포인트 저장: {checkpoint_path}")
    except Exception as exc:
        checkpoint["status"] = "interrupted"
        checkpoint["updated_at"] = utc_timestamp()
        write_checkpoint(checkpoint_path, checkpoint)
        print(f"\n일괄 평가 중단: {exc}", file=sys.stderr)
        print(
            "같은 옵션에 --resume을 붙여 완료된 다음 질문부터 계속할 수 있습니다.",
            file=sys.stderr,
        )
        return 1

    checkpoint["status"] = "complete"
    checkpoint["updated_at"] = utc_timestamp()
    write_checkpoint(checkpoint_path, checkpoint)
    summary_path, detail_path, report_path = _write_final_outputs(
        output_dir=args.output_dir,
        seed=args.seed,
        checkpoint=checkpoint,
    )

    print("\n" + " BATCH EVALUATION COMPLETE ".center(76, "="))
    print(f"원본·체크포인트 JSON: {checkpoint_path}")
    print(f"평균 지표 CSV: {summary_path}")
    print(f"질문별 상세 CSV: {detail_path}")
    print(f"시각화 HTML: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
