"""evaluate.py — 평가셋으로 RAG 체인을 돌려 지표를 계산하고 결과 CSV 저장. (담당 A, [8])

입력: eval/questions.csv, eval/references.csv  (id로 조인)
지표:
  - BERTScore F1  : 생성 답변 vs 정답 의미 유사도 (lang="ko")
  - hit@k         : 정답 근거가 검색된 청크에 포함됐는지 (검색 품질 분리 측정)
  - refused       : '답 없음' 문항에 "해당 정보 없음"으로 답했는지 (할루시네이션 억제)
  - latency       : 질의당 응답 시간(초)
출력: eval/results_<model>.csv  +  콘솔 요약
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_chain import RagChain, make_llm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]  # 레포 루트
EVAL_DIR = ROOT / "eval"

# '정보 없음/문서에 없음'을 뜻하는 다양한 한국어 표현 (Few-shot 답변은 표현이 제각각)
NO_ANSWER_MARKERS = (
    "해당 정보 없음", "해당 차량 정보 없음", "정보 없음", "해당 없음", "알 수 없",
    "찾지 못", "찾을 수 없", "찾아볼 수 없", "확인되지 않", "확인할 수 없",
    "명시되어 있지 않", "명시되지 않", "명시하지 않",
    "설명되어 있지 않", "설명하지 않", "나와 있지 않", "나와있지 않",
    "정보를 찾", "정보가 없", "제공되지 않", "포함되어 있지 않",
    "기재되어 있지 않", "언급되어 있지 않", "언급되지 않",
)

CTX_DELIM = "\n===CHUNK===\n"  # 검색 문맥을 CSV 한 칸에 저장할 때 청크 구분자(RAGAS 사후계산용)


def _norm(text: str) -> str:
    """모든 공백 제거(회전 표는 줄바꿈이 많아 공백 하나만으론 매칭 안 됨)."""
    return re.sub(r"\s+", "", str(text))


def load_eval() -> list[dict]:
    """questions.csv + references.csv를 id로 조인."""
    def read(name):
        path = EVAL_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"평가셋 없음: {path}")
        return {r["id"]: r for r in csv.DictReader(open(path, encoding="utf-8-sig"))}

    questions, refs = read("questions.csv"), read("references.csv")
    rows = []
    for qid, q in questions.items():
        rows.append({**q, **refs.get(qid, {})})
    return rows


def hit_at_k(evidence: str, contexts) -> object:
    """정답 근거 앞부분이 검색된 청크에 있으면 1, 없으면 0. 근거 없으면 ''(N/A)."""
    needle = _norm(evidence)[:20]
    if not needle:
        return ""  # 답 없음 등 근거가 없는 문항
    return int(any(needle in _norm(getattr(d, "page_content", "")) for d in contexts))


def refused(answer: str) -> int:
    return int(any(m in answer for m in NO_ANSWER_MARKERS))


def add_bertscore(results: list[dict]) -> None:
    """생성 답변 vs 정답으로 BERTScore F1 계산(가능할 때만)."""
    try:
        from bert_score import score
    except ImportError:
        print("[경고] bert-score 미설치 → BERTScore 생략 (pip install bert-score)")
        for r in results:
            r["bertscore_f1"] = ""
        return

    cands = [r["answer"] for r in results]
    refs = [r["reference"] for r in results]
    _, _, f1 = score(cands=cands, refs=refs, lang="ko", verbose=False)
    for r, v in zip(results, f1.tolist()):
        r["bertscore_f1"] = round(v, 4)


def summarize(results: list[dict], model_spec: str) -> None:
    def avg(key, rows=results):
        vals = [r[key] for r in rows if isinstance(r[key], (int, float))]
        return sum(vals) / len(vals) if vals else float("nan")

    no_ans = [r for r in results if r["type"] == "답 없음"]
    print("\n" + "=" * 60)
    print(f"모델: {model_spec}   문항: {len(results)}")
    print("-" * 60)
    print(f"  BERTScore F1 (평균) : {avg('bertscore_f1'):.4f}")
    print(f"  hit@k        (평균) : {avg('hit@k'):.3f}")
    print(f"  응답시간(초) (평균) : {avg('latency'):.2f}")
    if no_ans:
        print(f"  답없음 정답률(refused): {avg('refused', no_ans):.3f}  (n={len(no_ans)})")
    print("=" * 60)


def write_csv(path: Path, results: list[dict]) -> None:
    cols = ["id", "type", "car", "question", "reference", "answer",
            "sources", "bertscore_f1", "hit@k", "refused", "latency", "contexts"]
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({c: r.get(c, "") for c in cols})


def run(model_spec: str, variant: str = "few_shot", k: int = 7,
        max_new_tokens: int = 512) -> list[dict]:
    rows = load_eval()
    print(f"평가셋 {len(rows)}문항 | 모델 {model_spec} | 프롬프트 {variant} | top_k={k}\n")

    chain = RagChain(make_llm(model_spec, max_new_tokens=max_new_tokens), variant=variant, k=k)

    results = []
    for i, r in enumerate(rows, start=1):
        res = chain.ask(r["question"], car=(r.get("car") or None))
        row = {
            "id": r["id"], "type": r.get("question_type", ""), "car": res.car,
            "question": r["question"], "reference": r.get("reference_answer", ""),
            "answer": res.answer, "sources": " | ".join(res.sources),
            "hit@k": hit_at_k(r.get("evidence", ""), res.contexts),
            "refused": refused(res.answer), "latency": round(res.latency, 3),
            # RAGAS 사후계산용: 검색 문맥 전체를 청크 구분자로 이어 저장
            "contexts": CTX_DELIM.join(
                str(getattr(d, "page_content", "") or "") for d in res.contexts
            ),
        }
        results.append(row)
        print(f"  [{i:>2}/{len(rows)}] {r['id']} {r['question'][:28]:<28} "
              f"hit={row['hit@k']} {res.latency:.1f}s")

    add_bertscore(results)
    tag = model_spec.replace(":", "_").replace("/", "-")
    out = EVAL_DIR / f"results_{tag}.csv"
    write_csv(out, results)
    summarize(results, model_spec)
    print(f"저장 → {out.relative_to(ROOT)}")
    return results


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="RAG 평가")
    p.add_argument("--model", default="upstage:solar-pro",
                   help="local:<id> / gemini:<model> / upstage:<model>")
    p.add_argument("--variant", default="few_shot",
                   choices=["basic", "role", "constraint", "few_shot"])
    p.add_argument("--top-k", type=int, default=7)
    a = p.parse_args()
    run(a.model, a.variant, a.top_k)
