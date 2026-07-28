"""main.py — RAG 질의응답 · [7] LLM 3종 비교 · [8] 평가 진입점. (담당 A)

사용법:
  # 단건 질의
  python main.py ask "아반떼 엔진오일 교환주기는?" --model upstage:solar-pro

  # 한 모델 평가 → eval/results_<model>.csv
  python main.py eval --model upstage:solar-pro --variant constraint

  # [7] 3종 모델 비교 (Qwen-7B / EXAONE-7.8B / Solar) → 각 results + 비교표
  python main.py compare

모델 스펙:  local:<hf_id> / upstage:<model> / gemini:<model>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import run  # noqa: E402
from rag_chain import RagChain, make_llm  # noqa: E402

# [7] 비교 대상 3종 (선정 기준: 오픈 vs 상용, 모델 크기 7B vs 3B)
COMPARE_MODELS = [
    "local:Qwen/Qwen2.5-7B-Instruct",  # 오픈·대형 7B (로컬 4-bit, 무료)
    "local:Qwen/Qwen2.5-3B-Instruct",  # 오픈·소형 3B (로컬 4-bit, 무료)
    "upstage:solar-pro",               # 상용·한국어 특화 (API)
]


def _print_table(rows: list[dict]) -> None:
    """[model, bertscore_f1, hit@k, refused, latency] 딕셔너리 목록을 표로 출력."""
    print("\n" + "=" * 78)
    print(f"{'모델':<38}{'BERTScore':>11}{'hit@k':>8}{'답없음':>8}{'응답(s)':>9}")
    print("-" * 78)
    for s in rows:
        print(f"{s['model']:<38}{s['bertscore_f1']:>11.4f}{s['hit@k']:>8.3f}"
              f"{s['refused']:>8.3f}{s['latency']:>9.2f}")
    print("=" * 78)


def _avg(rows, key, subset=None):
    subset = rows if subset is None else subset
    vals = []
    for r in subset:
        try:
            vals.append(float(r[key]))
        except (ValueError, TypeError, KeyError):
            pass
    return sum(vals) / len(vals) if vals else float("nan")


def cmd_ask(a: argparse.Namespace) -> int:
    chain = RagChain(make_llm(a.model, max_new_tokens=a.max_new_tokens),
                     variant=a.variant, k=a.top_k)
    r = chain.ask(a.question, car=a.car)
    print(f"\n[차종] {r.car}")
    print(f"[답변]\n{r.answer}")
    print(f"\n[출처] {', '.join(r.sources)}  ({r.latency:.2f}s)")
    return 0


def cmd_eval(a: argparse.Namespace) -> int:
    run(a.model, a.variant, a.top_k, a.max_new_tokens)
    return 0


def cmd_compare(a: argparse.Namespace) -> int:
    summary = []
    for spec in a.models:
        print("\n" + "#" * 70 + f"\n# {spec}\n" + "#" * 70)
        try:
            results = run(spec, a.variant, a.top_k, a.max_new_tokens)
        except Exception as exc:  # 한 모델 실패해도 나머지 진행
            print(f"[건너뜀] {spec} 실패: {exc}", file=sys.stderr)
            continue

        no_ans = [r for r in results if r["type"] == "답 없음"]
        summary.append({
            "model": spec,
            "bertscore_f1": _avg(results, "bertscore_f1"),
            "hit@k": _avg(results, "hit@k"),
            "latency": _avg(results, "latency"),
            "refused": _avg(results, "refused", no_ans) if no_ans else float("nan"),
        })

    _print_table(summary)  # [7] 최종 비교표
    return 0


def cmd_summary(a: argparse.Namespace) -> int:
    """재실행 없이, 이미 저장된 eval/results_*.csv 들로 비교표를 만든다."""
    import csv

    from evaluate import EVAL_DIR

    files = sorted(EVAL_DIR.glob("results_*.csv"))
    if not files:
        print("결과 CSV가 없습니다. 먼저 eval을 돌리세요.", file=sys.stderr)
        return 1

    summary = []
    for f in files:
        rows = list(csv.DictReader(open(f, encoding="utf-8-sig")))
        no_ans = [r for r in rows if r.get("type") == "답 없음"]
        summary.append({
            "model": f.stem.replace("results_", ""),
            "bertscore_f1": _avg(rows, "bertscore_f1"),
            "hit@k": _avg(rows, "hit@k"),
            "latency": _avg(rows, "latency"),
            "refused": _avg(rows, "refused", no_ans) if no_ans else float("nan"),
        })

    print(f"저장된 결과 {len(files)}개로 비교표 생성:")
    _print_table(summary)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="자동차 취급설명서 RAG — 질의·평가·모델비교")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--variant", default="constraint", choices=["basic", "role", "constraint"])
    common.add_argument("--top-k", type=int, default=3)
    common.add_argument("--max-new-tokens", type=int, default=512,
                        help="답변 최대 토큰(로컬 모델 속도↔길이 조절, 예: 256)")

    pa = sub.add_parser("ask", parents=[common], help="단건 질의")
    pa.add_argument("question")
    pa.add_argument("--model", default="upstage:solar-pro")
    pa.add_argument("--car", default=None, help="차종 미인식 시 지정")
    pa.set_defaults(func=cmd_ask)

    pe = sub.add_parser("eval", parents=[common], help="한 모델 평가")
    pe.add_argument("--model", default="upstage:solar-pro")
    pe.set_defaults(func=cmd_eval)

    pc = sub.add_parser("compare", parents=[common], help="[7] 3종 모델 비교(재실행)")
    pc.add_argument("--models", nargs="+", default=COMPARE_MODELS)
    pc.set_defaults(func=cmd_compare)

    ps = sub.add_parser("summary", help="저장된 results_*.csv로 비교표만 생성(재실행 X)")
    ps.set_defaults(func=cmd_summary)

    a = p.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
