"""
chunker.py
LLM 구조화 결과(pages_llmhybrid.json) → 청크 JSON

[출력] 각 청크에 car/page/needs_review/table_structured 메타 부착.
       car 는 검색 시 차종 필터로 반드시 사용(5종 혼재).
"""

import json
from pathlib import Path

from bs4 import BeautifulSoup

# ----------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------

# 실행 위치(cwd)와 무관하게 항상 프로젝트 루트의 output/ 을 보도록 절대경로로 고정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

IO_MAP = {
    OUTPUT_DIR / "pages_llmhybrid.json": OUTPUT_DIR / "chunks_llmhybrid.json",
}

# 청킹 파라미터
CHUNK_SIZE = 512     # 청크 최대 글자 수
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


def _grid_from_rows(trs):
    """<tr> 목록을 rowspan/colspan 반영한 2차원 텍스트 그리드로 변환.
    (정비 명세표는 항목명/서브사양이 rowspan으로, 교체주기가 colspan으로 병합돼 있어
    단순 텍스트 추출로는 어느 항목의 값인지 알 수 없음 -> 그리드로 펼쳐서 위치를 복원)
    """
    grid = []
    active = {}  # col -> [남은 행 수, 텍스트]
    for tr in trs:
        row = {}
        for col, (_remaining, text) in active.items():
            row[col] = text
        active = {c: [r - 1, t] for c, (r, t) in active.items() if r - 1 > 0}

        col = 0
        for cell in tr.find_all(["td", "th"]):
            while col in row:
                col += 1
            text = " ".join(cell.get_text(" ", strip=True).split())
            colspan = int(cell.get("colspan", 1) or 1)
            rowspan = int(cell.get("rowspan", 1) or 1)
            for k in range(colspan):
                row[col + k] = text
                if rowspan > 1:
                    active[col + k] = [rowspan - 1, text]
            col += colspan

        ncols = max(row.keys(), default=-1) + 1
        grid.append([row.get(c, "") for c in range(ncols)])
    return grid


def _table_to_row_texts(table):
    """<table> 1개 -> 행(정비 항목) 단위 텍스트 리스트.
    헤더 라벨(예: 매60,000 km)과 값을 묶어 "항목 - 라벨: 값" 형태의 문장으로 만듦.
    """
    thead = table.find("thead")
    header_trs = thead.find_all("tr") if thead else []
    header_grid = _grid_from_rows(header_trs)
    header = header_grid[-1] if header_grid else []

    tbody = table.find("tbody")
    body_trs = tbody.find_all("tr") if tbody else [tr for tr in table.find_all("tr") if tr not in header_trs]
    body_grid = _grid_from_rows(body_trs)

    def label(col):
        return header[col] if col < len(header) and header[col] else f"col{col}"

    # 항목명 컬럼: 헤더에 "항목"이 들어간 컬럼(없으면 0번 컬럼을 항목명으로 간주)
    item_cols = [c for c, h in enumerate(header) if "항목" in h] or [0]

    row_texts = []
    for row in body_grid:
        seen, item_parts = set(), []
        for c in item_cols:
            if c < len(row) and row[c] and row[c] not in seen:
                item_parts.append(row[c])
                seen.add(row[c])
        item_name = " ".join(item_parts).strip()
        if not item_name:
            continue

        # 나머지 컬럼: colspan으로 같은 값이 이어지면 범위로 묶고, 빈 칸은 제외
        details = []
        c, ncols = 0, len(row)
        while c < ncols:
            if c in item_cols or not row[c]:
                c += 1
                continue
            start, value = c, row[c]
            while c + 1 < ncols and row[c + 1] == value and (c + 1) not in item_cols:
                c += 1
            details.append(f"{label(start)}: {value}" if c == start else f"{label(start)}~{label(c)}: {value}")
            c += 1

        row_texts.append(item_name + (" - " + "; ".join(details) if details else ""))

    return row_texts


def table_page_to_pieces(text):
    """표 페이지 원문(HTML) -> 행 단위 텍스트 조각 리스트.
    표 전체를 1청크로 두면 15개 안팎의 서로 다른 정비 항목이 한 벡터에 뭉개져서
    "엔진오일"/"점화플러그"처럼 특정 항목을 찾는 검색에서 유사도가 희석됨 ->
    행 단위로 쪼개 항목별로 독립된 청크를 만듦. 파싱 실패/표 없음이면 원문 그대로 반환.
    """
    soup = BeautifulSoup(text, "lxml")
    tables = soup.find_all("table")
    if not tables:
        return [text]

    # 표 위 제목/범례만 문맥으로 붙이고(짧고 유용), 각주 목록(footnote)은 청크 비대화 방지 위해 제외
    context = " / ".join(
        " ".join(tag.get_text(" ", strip=True).split())
        for tag in soup.find_all(["h1", "p"])
        if tag.get("data-category") != "list" and not tag.find_parent("table")
    )[:200]

    pieces = []
    for table in tables:
        for row_text in _table_to_row_texts(table):
            pieces.append(f"{context} - {row_text}" if context else row_text)

    return pieces or [text]


def chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """페이지 리스트 → 청크 리스트. 표 페이지는 행 단위, 일반은 분할."""
    chunks = []

    for p in pages:
        text = p.get("text", "").strip()
        if not text:                                   # 빈 페이지 스킵
            continue

        is_table = p.get("needs_review", False)        # 표 페이지 여부

        if is_table:
            pieces = table_page_to_pieces(text)         # 표: 행 단위로 분할
            # 그래도 너무 길면(비정형 표 등) 문자 기준으로 한번 더 자름
            pieces = [q for piece in pieces for q in (
                split_text(piece, chunk_size, overlap) if len(piece) > chunk_size else [piece]
            )]
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

