"""
chunker.py
파서 결과(페이지 JSON) → 청크 JSON  [임베딩 직전 단계]

[역할] loader/hybrid 가 만든 '페이지 단위' JSON을 받아, 임베딩에 쓸
       '청크 단위'로 잘라 저장한다. 이 파일이 전처리 파이프라인의
       마지막 단계이며, 결과물을 다음 담당자(임베딩)가 그대로 받는다.

[청킹 전략]
  - 일반 페이지            : 512자 단위로 분할(문맥 유지 위해 겹침 50자)
  - 눕힌 표 페이지(needs_review=True)
                          : 자르지 않고 페이지 통째로 1개 청크로 유지
    → 표는 이미 셀 구조가 깨진 상태라 억지로 자르면 더 손상됨.
      플래그를 그대로 넘겨 임베딩 단계에서 검수/별도처리하게 한다.

[파라미터] chunk_size / overlap 은 나중에 성능 실험 대상이므로
           함수 인자로 받게 해 두었다(코드 수정 없이 숫자만 바꿔 실험).

[출력] 각 청크에 car/page/needs_review 등 메타데이터를 붙인다.
       car 는 검색 시 차종 필터로 반드시 사용해야 한다(5종 혼재).
"""

import json
from pathlib import Path

# 한국어 문서에 맞춘 재귀 분할기 (문단→줄→문장→어절 순으로 자름)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------
# 입력(페이지 JSON) → 출력(청크 JSON) 매핑.
# 세 파서 결과를 각각 청킹해 두면, 임베딩 단계에서 셋을 비교할 수 있다.
IO_MAP = {
    "output/pages_pymupdf.json":    "output/chunks_pymupdf.json",
    "output/pages_pdfplumber.json": "output/chunks_pdfplumber.json",
    "output/pages_hybrid.json":     "output/chunks_hybrid.json",
}

# 청킹 파라미터 (나중에 실험으로 최적값 탐색)
CHUNK_SIZE = 512     # 청크 최대 글자 수
OVERLAP = 50         # 인접 청크가 겹치는 글자 수(문맥 끊김 방지)

# 한국어 우선 구분자: 문단 → 줄 → 문장부호 → 공백 → 글자
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def make_splitter(chunk_size, overlap):
    """분할기 생성 - 파라미터를 받아 실험 때 값만 바꾸면 되게 함"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=SEPARATORS,
        length_function=len,        # 글자 수 기준(토큰 아님)
    )


def chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """
    페이지 리스트 → 청크 리스트.
    일반 페이지는 분할, 눕힌 표 페이지는 통째로 1청크로 둔다.
    """
    splitter = make_splitter(chunk_size, overlap)
    chunks = []

    for p in pages:
        text = p.get("text", "").strip()
        if not text:                          # 빈 페이지는 건너뜀
            continue

        # 표 페이지 여부(hybrid에만 있는 키. 없으면 False로 간주)
        is_table = p.get("needs_review", False)

        if is_table:
            # 표 페이지: 자르지 않고 페이지 전체를 1개 청크로
            pieces = [text]
        else:
            # 일반 페이지: 512자 단위로 분할
            pieces = splitter.split_text(text)

        # 각 조각을 메타데이터와 함께 청크로 저장
        for idx, piece in enumerate(pieces):
            chunks.append({
                "car": p["car"],                       # 차종(검색 필터용, 필수)
                "page": p["page"],                     # 출처 페이지
                "chunk_id": f"{p['car']}_p{p['page']}_{idx}",  # 고유 ID
                "n_chars": len(piece),                 # 청크 글자 수
                "text": piece,                         # 청크 본문(임베딩 대상)
                "needs_review": is_table,              # 표 페이지 플래그 유지
                "ocr_applied": p.get("ocr_applied", False),  # OCR 교체 여부 유지
            })

    return chunks


def process(in_path, out_path, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """페이지 JSON 1개를 읽어 청킹 후 저장. 통계도 출력."""
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

    # 통계
    table_chunks = sum(1 for c in chunks if c["needs_review"])
    avg = sum(c["n_chars"] for c in chunks) / len(chunks) if chunks else 0
    print(f"  {in_p.name} → {out_p.name}")
    print(f"    페이지 {len(pages):,} → 청크 {len(chunks):,}개 "
          f"(평균 {avg:.0f}자, 표청크 {table_chunks}개)")


def main():
    print(f"청킹 설정: chunk_size={CHUNK_SIZE}, overlap={OVERLAP}\n")
    for in_path, out_path in IO_MAP.items():   # 세 파서 결과를 각각 청킹
        process(in_path, out_path)
    print("\n완료. chunks_*.json 을 임베딩 단계로 전달.")


if __name__ == "__main__":
    main()