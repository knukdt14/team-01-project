"""make_review_sheet.py — 모델별 결과 CSV를 합쳐 '답변 비교 + 사람평가' 시트를 만든다. (담당 A, [8])

eval/results_*.csv (모델별) → eval/answers_review.csv (질문당 한 줄, 모델 답변 나란히 + 사람평가 빈칸)

용도:
  1) 각 모델이 같은 질문에 어떻게 답했는지 한눈에 비교
  2) 팀원이 모델별 '사람평가' 열에 1~5점을 채워 넣는 채점 시트

실행:  python src/llm_eval/make_review_sheet.py
채점 후 집계:  python src/llm_eval/make_review_sheet.py --aggregate
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "eval"
SHEET = EVAL_DIR / "answers_review.csv"


def _model_name(path: Path) -> str:
    return path.stem.replace("results_", "")


def build_sheet() -> None:
    files = sorted(EVAL_DIR.glob("results_*.csv"))
    if not files:
        raise SystemExit("results_*.csv가 없습니다. 먼저 eval을 돌리세요.")

    models = [_model_name(f) for f in files]
    # id 기준으로 문항 정보 + 모델별 답변 모으기
    rows: dict[str, dict] = {}
    for f in files:
        model = _model_name(f)
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            qid = r["id"]
            base = rows.setdefault(qid, {
                "id": qid, "type": r.get("type", ""), "car": r.get("car", ""),
                "question": r.get("question", ""), "reference": r.get("reference", ""),
            })
            base[f"{model} | 답변"] = r.get("answer", "")
            base[f"{model} | 사람점수(1-5)"] = ""  # 팀원이 채울 칸

    # 컬럼 순서: 문항정보 → 모델별(답변 + 사람점수)
    cols = ["id", "type", "car", "question", "reference"]
    for m in models:
        cols += [f"{m} | 답변", f"{m} | 사람점수(1-5)"]

    EVAL_DIR.mkdir(exist_ok=True)
    with open(SHEET, "w", encoding="utf-8-sig", newline="") as out:
        w = csv.DictWriter(out, fieldnames=cols)
        w.writeheader()
        for qid in sorted(rows):
            w.writerow({c: rows[qid].get(c, "") for c in cols})

    print(f"[생성] {SHEET.relative_to(ROOT)}  ({len(rows)}문항 × {len(models)}모델)")
    print(f"  모델: {', '.join(models)}")
    print("  → 각 '사람점수(1-5)' 열을 팀원이 채운 뒤, --aggregate 로 평균을 냅니다.")


def aggregate() -> None:
    if not SHEET.exists():
        raise SystemExit(f"{SHEET.name} 없음. 먼저 시트를 생성/채점하세요.")

    reader = list(csv.DictReader(open(SHEET, encoding="utf-8-sig")))
    score_cols = [c for c in reader[0] if c.endswith("사람점수(1-5)")]

    print("\n===== 사람평가 평균 (1~5) =====")
    for c in score_cols:
        model = c.split(" | ")[0]
        vals = []
        for r in reader:
            try:
                vals.append(float(r[c]))
            except (ValueError, TypeError):
                pass
        avg = sum(vals) / len(vals) if vals else float("nan")
        print(f"  {model:<44} {avg:.2f}  (채점 {len(vals)}/{len(reader)})")
    print("=" * 40)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="답변 비교 + 사람평가 시트")
    p.add_argument("--aggregate", action="store_true", help="채점된 시트로 평균 집계")
    a = p.parse_args()
    aggregate() if a.aggregate else build_sheet()
