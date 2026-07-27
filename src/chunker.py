"""
chunker.py
LLM 구조화 결과(pages_llmhybrid.json) → 청크 JSON

[출력] 각 청크에 car/page/needs_review/table_structured 메타 부착.
       car 는 검색 시 차종 필터로 반드시 사용(5종 혼재).
"""

import json
from pathlib import Path

# ----------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------

IO_MAP = {
    "output/pages_llmhybrid.json": "output/chunks_llmhybrid.json",
}

# 청킹 파라미터
CHUNK_SIZE = 256     # 청크 최대 글자 수
OVERLAP = 50         # 인접 청크 겹침 글자 수(문맥 끊김 방지)

# 재귀 분할 구분자 우선순위: 문단 → 줄 → 문장 → 어절 → 글자
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def split_text(text, chunk_size, overlap, seps=SEPARATORS):
    
    # 1) 구분자 우선순위대로 재귀 분할 → '기본 조각' 생성
    def _split(txt, sep_idx):
        if len(txt) <= chunk_size:                     # 충분히 짧으면 그대로
            return [txt]
        if sep_idx >= len(seps) or seps[sep_idx] == "":  # 더 못 자르면 글자 단위 절단
            return [txt[i:i + chunk_size] for i in range(0, len(txt), chunk_size)]

        parts = txt.split(seps[sep_idx])
        result = []
        for part in parts:
            if len(part) <= chunk_size:
                result.append(part)
            else:
                result.extend(_split(part, sep_idx + 1))  # 더 작은 구분자로
        return result

    raw_parts = [p for p in _split(text, 0) if p.strip()]

    # 2) 조각들을 chunk_size 한도 내에서 합치고 overlap 적용
    chunks = []
    current = ""
    for part in raw_parts:
        candidate = (current + " " + part).strip() if current else part
        if len(candidate) <= chunk_size:               # 한도 안이면 이어 붙임
            current = candidate
        else:
            if current:
                chunks.append(current)
            if overlap > 0 and chunks:                 # 직전 청크 끝을 겹쳐 문맥 유지
                tail = chunks[-1][-overlap:]
                current = (tail + " " + part).strip()
            else:
                current = part
    if current:
        chunks.append(current)

    return chunks


def chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """페이지 리스트 → 청크 리스트. 표 페이지는 통짜, 일반은 분할."""
    chunks = []

    for p in pages:
        text = p.get("text", "").strip()
        if not text:                                   # 빈 페이지 스킵
            continue

        is_table = p.get("needs_review", False)        # 표 페이지 여부

        if is_table:
            pieces = [text]                            # 표: 통째로 1청크
        else:
            pieces = split_text(text, chunk_size, overlap)  # 일반: 분할

        for idx, piece in enumerate(pieces):
            chunks.append({
                "car": p["car"],                                 # 차종(필터 필수)
                "page": p["page"],                               # 출처 페이지
                "chunk_id": f"{p['car']}_p{p['page']}_{idx}",    # 고유 ID
                "n_chars": len(piece),
                "text": piece,                                   # 임베딩 대상
                "needs_review": is_table,                        # 표 플래그
                "table_structured": p.get("table_structured", False),  # LLM 구조화 여부
            })

    return chunks


def process(in_path, out_path, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """페이지 JSON 1개 → 청킹 → 저장 + 통계 출력"""
    in_p, out_p = Path(in_path), Path(out_path)
    if not in_p.exists():
        print(f"  건너뜀(입력 없음): {in_p}")
        return

    with open(in_p, encoding="utf-8") as f:
        pages = json.load(f)

    chunks = chunk_pages(pages, chunk_size, overlap)

    out_p.parent.mkdir(exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)

    table_chunks = sum(1 for c in chunks if c["needs_review"])
    avg = sum(c["n_chars"] for c in chunks) / len(chunks) if chunks else 0
    print(f"  {in_p.name} → {out_p.name}")
    print(f"    페이지 {len(pages):,} → 청크 {len(chunks):,}개 "
          f"(평균 {avg:.0f}자, 표청크 {table_chunks}개)")


def main():
    print(f"청킹 설정: chunk_size={CHUNK_SIZE}, overlap={OVERLAP}\n")
    for in_path, out_path in IO_MAP.items():
        process(in_path, out_path)


if __name__ == "__main__":
    main()

