"""
evaluate.py — [8] 평가  ─ 담당 A

평가를 '검색 단계'와 '생성 단계'로 분리해서 측정한다.
  · 검색 지표(Hit@k, MRR) → B·C의 실험 효과를 LLM과 분리해 증명
  · 생성 지표(BERTScore)  → A·D의 실험 효과를 측정

입력:  eval/questions.csv
       question,reference_answer,evidence_sentence,source_vehicle,source_page,question_type
출력:  eval/results_<tag>.csv

실행:  python src/evaluate.py
"""

from __future__ import annotations

import io
import re
import sys

import pandas as pd

from config import BASELINE, EVAL_DIR, Config
from rag_chain import RagChain, RagResult

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NO_ANSWER_MARKERS = ("해당 정보 없음", "해당 차량 정보 없음", "정보 없음", "알 수 없")


def _norm(text: str) -> str:
    """모든 공백(스페이스·줄바꿈·탭)을 제거해 비교용으로 정규화.

    회전된 정비 주기표는 셀 사이에 줄바꿈이 많이 들어가므로,
    공백 하나만 지워서는 근거 문장이 매칭되지 않는다.
    """
    return re.sub(r"\s+", "", str(text))


# ─────────────────────────── 검색 지표 ───────────────────────────
def hit_at_k(result: RagResult, row: pd.Series) -> int:
    """정답 근거가 검색된 청크 안에 있으면 1.

    근거 문장의 앞부분이 청크에 포함되는지로 판정한다.
    (근거 문장 전체는 청크 경계로 잘릴 수 있음)
    """
    needle = _norm(row["evidence_sentence"])[:20]
    if not needle:
        return 0
    return int(any(needle in _norm(d.page_content) for d in result.contexts))


def reciprocal_rank(result: RagResult, row: pd.Series) -> float:
    """정답 근거가 몇 번째로 검색됐는지의 역수."""
    needle = _norm(row["evidence_sentence"])[:20]
    if not needle:
        return 0.0
    for rank, d in enumerate(result.contexts, start=1):
        if needle in _norm(d.page_content):
            return 1.0 / rank
    return 0.0


def vehicle_correct(result: RagResult, row: pd.Series) -> int:
    """검색된 청크가 정답 차량 문서에서 나왔는가 (혼입 방지 측정)."""
    target = str(row["source_vehicle"]).strip()
    if not target or target == "nan":
        return 1
    return int(any(d.metadata.get("vehicle") == target for d in result.contexts))


# ─────────────────────────── 생성 지표 ───────────────────────────
def bert_scores(answers: list[str], references: list[str]) -> pd.DataFrame:
    from bert_score import score

    P, R, F1 = score(cands=answers, refs=references, lang="ko", verbose=False)
    return pd.DataFrame({
        "bertscore_p": P.tolist(),
        "bertscore_r": R.tolist(),
        "bertscore_f1": F1.tolist(),
    })


def refused(answer: str) -> int:
    """'해당 정보 없음' 류로 답했는가 (할루시네이션 억제 측정)."""
    return int(any(m in answer for m in NO_ANSWER_MARKERS))


# ──────────────────────────── 실행 ────────────────────────────
def run(cfg: Config = BASELINE, questions_csv: str = "questions.csv") -> pd.DataFrame:
    path = EVAL_DIR / questions_csv
    if not path.exists():
        raise FileNotFoundError(
            f"평가셋이 없습니다: {path}\n"
            "eval/questions.csv 를 먼저 작성하세요."
        )

    df = pd.read_csv(path).fillna("")
    print(f"평가셋 {len(df)}문항 | 설정: {cfg.tag()}\n")

    from build_vectorstore import load

    chain = RagChain(load(cfg), cfg)

    rows = []
    for i, row in df.iterrows():
        r = chain.invoke(str(row["question"]))
        rows.append({
            "question": r.question,
            "question_type": row.get("question_type", ""),
            "reference": row["reference_answer"],
            "answer": r.answer,
            "sources": " | ".join(r.sources),
            "latency": round(r.latency, 3),
            "hit@k": hit_at_k(r, row),
            "rr": round(reciprocal_rank(r, row), 3),
            "vehicle_ok": vehicle_correct(r, row),
            "refused": refused(r.answer),
        })
        print(f"  [{i + 1:>2}/{len(df)}] {r.question[:34]:<34} "
              f"hit={rows[-1]['hit@k']} {r.latency:.1f}s")

    out = pd.DataFrame(rows)
    out = pd.concat([out, bert_scores(out["answer"].tolist(),
                                      out["reference"].tolist())], axis=1)

    # ── 요약 ──
    print("\n" + "=" * 62)
    print(f"{'검색 — Hit@k':<24}{out['hit@k'].mean():>8.3f}")
    print(f"{'검색 — MRR':<24}{out['rr'].mean():>8.3f}")
    print(f"{'검색 — 차량 정확도':<22}{out['vehicle_ok'].mean():>8.3f}")
    print(f"{'생성 — BERTScore F1':<23}{out['bertscore_f1'].mean():>8.3f}")
    print(f"{'운영 — 평균 응답시간':<21}{out['latency'].mean():>8.2f}s")

    no_ans = out[out["question_type"] == "답 없음"]
    if len(no_ans):
        print(f"{'억제 — 답없음 정답률':<21}{no_ans['refused'].mean():>8.3f}")
    print("=" * 62)

    EVAL_DIR.mkdir(exist_ok=True)
    dest = EVAL_DIR / f"results_{cfg.tag()}.csv"
    out.to_csv(dest, index=False, encoding="utf-8-sig")
    print(f"저장 → {dest.name}")
    return out


if __name__ == "__main__":
    run(BASELINE)
