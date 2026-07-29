"""ragas_eval.py — 저장된 results_*.csv로 RAGAS 지표를 사후 계산한다. (담당 A, [8])

팀의 RAGEvaluator(evaluation_metrics.py)를 재사용한다. 모델을 다시 돌리지 않고,
평가 때 CSV에 저장해 둔 검색 문맥(contexts 컬럼)으로 RAGAS를 계산한다.

지표:
  · faithfulness       답변이 검색 문맥에 충실한가 (답변 LLM별로 다름 → 모델 비교 유효)
  · answer_relevancy   답변이 질문에 관련 있나          (답변 LLM별로 다름 → 모델 비교 유효)
  · context_precision  검색 문맥 정밀도  (검색이 3모델 고정이라 질문당 1번만 계산)
  · context_recall     검색 문맥 재현율  (동일)

⚠️ RAGAS는 Upstage judge API를 문항마다 호출 → 시간·API 사용량 발생.
필요 패키지: ragas, langchain-openai, langchain-huggingface, datasets (requirements.txt)

실행:  python src/llm_eval/ragas_eval.py            # 저장된 모든 모델 결과에 RAGAS
       python src/llm_eval/ragas_eval.py --limit 5  # 앞 5문항만(빠른 확인용)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompt_engineering"
sys.path.insert(0, str(_PROMPT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import CTX_DELIM, EVAL_DIR, ROOT  # noqa: E402  (문맥 구분자·경로 재사용)

try:  # .env 로드(UPSTAGE_API_KEY)
    import rag_chain  # noqa: F401  (import 시 _load_env 실행)
except Exception:
    pass


def _model_name(path: Path) -> str:
    return path.stem.replace("results_", "")


def _to_documents(contexts_cell: str) -> list[dict]:
    """CSV의 contexts 칸(구분자로 이어진 문자열)을 RAGEvaluator용 문서 리스트로 복원."""
    return [{"text": c} for c in str(contexts_cell).split(CTX_DELIM) if c.strip()]


def load_joined() -> tuple[list[str], dict[str, dict]]:
    """results_*.csv들을 id 기준으로 조인. 반환: (모델목록, {id: {정보 + model별 answer}})."""
    files = sorted(EVAL_DIR.glob("results_*.csv"))
    if not files:
        raise SystemExit("results_*.csv가 없습니다. 먼저 eval을 돌리세요.")

    models = [_model_name(f) for f in files]
    joined: dict[str, dict] = {}
    for f in files:
        model = _model_name(f)
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            qid = r["id"]
            item = joined.setdefault(qid, {
                "id": qid, "type": r.get("type", ""),
                "question": r.get("question", ""), "reference": r.get("reference", ""),
                "contexts": r.get("contexts", ""), "answers": {},
            })
            item["answers"][model] = r.get("answer", "")
            if not item["contexts"]:  # 문맥은 3모델 고정이라 아무 거나 채워두면 됨
                item["contexts"] = r.get("contexts", "")
    return models, joined


def main() -> int:
    ap = argparse.ArgumentParser(description="저장된 결과로 RAGAS 사후 계산")
    ap.add_argument("--limit", type=int, default=None, help="앞 N문항만(빠른 확인)")
    ap.add_argument("--model", default="solar-pro3", help="RAGAS judge Upstage 모델")
    args = ap.parse_args()

    if not os.environ.get("UPSTAGE_API_KEY"):
        print("UPSTAGE_API_KEY가 없습니다(.env 확인). RAGAS는 Upstage judge가 필요합니다.",
              file=sys.stderr)
        return 1

    from evaluation_metrics import RAGEvaluator

    models, joined = load_joined()
    ids = sorted(joined)
    if args.limit:
        ids = ids[: args.limit]

    # contexts 컬럼이 있는지 확인(옛 결과면 없음)
    if not any(joined[q]["contexts"] for q in ids):
        print("결과 CSV에 contexts 컬럼이 없습니다. 최신 evaluate.py로 eval을 다시 돌리세요.",
              file=sys.stderr)
        return 1

    evaluator = RAGEvaluator(api_key=os.environ["UPSTAGE_API_KEY"], model_id=args.model)

    # 누적: 모델별 faithfulness/answer_relevancy, 질문 공통 context_precision/recall
    faith = {m: [] for m in models}
    arel = {m: [] for m in models}
    cprec, crec = [], []
    detail_rows = []

    for i, qid in enumerate(ids, start=1):
        item = joined[qid]
        docs = _to_documents(item["contexts"])
        reference = item["reference"].strip()
        answer_list = [item["answers"].get(m, "") for m in models]

        if not docs or not reference:
            print(f"  [{i}/{len(ids)}] {qid} 건너뜀(문맥/정답 없음)")
            continue

        try:
            scores = evaluator.evaluate(
                question=item["question"], answers=answer_list,
                documents=docs, reference_answer=reference,
            )
        except Exception as exc:
            print(f"  [{i}/{len(ids)}] {qid} RAGAS 실패: {exc}", file=sys.stderr)
            continue

        for m, s in zip(models, scores):
            faith[m].append(s.faithfulness)
            arel[m].append(s.answer_relevancy)
            detail_rows.append({"id": qid, "model": m, "faithfulness": s.faithfulness,
                                "answer_relevancy": s.answer_relevancy})
        cprec.append(scores[0].context_precision)  # 문맥 지표는 3모델 공통
        crec.append(scores[0].context_recall)
        print(f"  [{i}/{len(ids)}] {qid} 완료")

    def nanmean(xs):
        xs = [x for x in xs if x == x]  # NaN 제외
        return sum(xs) / len(xs) if xs else float("nan")

    # 저장
    out = EVAL_DIR / "ragas_details.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "model", "faithfulness", "answer_relevancy"])
        w.writeheader()
        w.writerows(detail_rows)

    print("\n" + "=" * 70)
    print("RAGAS 결과 (답변 LLM별: faithfulness / answer_relevancy)")
    print("-" * 70)
    print(f"{'모델':<40}{'faithful':>10}{'ans_rel':>10}")
    for m in models:
        print(f"{m:<40}{nanmean(faith[m]):>10.3f}{nanmean(arel[m]):>10.3f}")
    print("-" * 70)
    print(f"검색 문맥(3모델 공통): context_precision={nanmean(cprec):.3f}  "
          f"context_recall={nanmean(crec):.3f}")
    print("=" * 70)
    print(f"상세 저장 → {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
