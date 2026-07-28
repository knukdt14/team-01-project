"""
single_page_experiment.py
특정 페이지 1장만 → Upstage로 HTML 구조화 → chunker.py 방식으로 행 단위 청킹

[목적] 전체 2,584p를 다 LLM 처리하는 건 비용 큼 → 표 있는 특정 페이지만 골라
       "정자(정상) 표도 LLM 구조화 + 행 청킹하면 검색 성능이 오르나"를 검증

[방식] chunker.py 와 동일하게 HTML <table> 을 파싱 → rowspan/colspan 복원 → 행 단위
       (chunker.py 의 table_page_to_pieces 를 그대로 import 해서 씀)

[출력]
  output/compare_single.txt   : before(PyMuPDF) vs after(Upstage HTML) 비교
  output/chunks_single.json   : 행 단위로 쪼갠 청크

[준비]
  pip install pymupdf pillow requests python-dotenv beautifulsoup4 lxml
  .env 에 UPSTAGE_API_KEY=up_...
"""

import os
import io
import sys
import json
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:
    import fitz

import requests
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# chunker.py 의 표 청킹 함수 재사용 -----------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "chunking"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
try:
    from chunker import table_page_to_pieces, split_text, CHUNK_SIZE, OVERLAP
except ImportError:
    print("chunker.py import 실패 - 표를 통짜 1청크로 처리(fallback)")
    table_page_to_pieces = None
    CHUNK_SIZE, OVERLAP = 512, 50

# ----------------------------------------------------------------------
# 실험 대상 (여기만 바꾸면 다른 페이지도 됨)
# ----------------------------------------------------------------------
TARGET_PDF = PROJECT_ROOT / "data" / "CE1_2025_ko_KR.pdf"
TARGET_PAGE = 62
CAR = "ioniq6"

OUT_CHUNKS = PROJECT_ROOT / "output" / "chunks_single.json"
OUT_COMPARE = PROJECT_ROOT / "output" / "compare_single.txt"

API_URL = "https://api.upstage.ai/v1/document-digitization"
API_KEY = os.environ.get("UPSTAGE_API_KEY", "")
RENDER_DPI = 300


def extract_pymupdf(pdf_path, page_no):
    doc = fitz.open(pdf_path)
    text = doc[page_no - 1].get_text("text").strip()
    doc.close()
    return text


def render_page_bytes(pdf_path, page_no):
    doc = fitz.open(pdf_path)
    page = doc[page_no - 1]
    zoom = RENDER_DPI / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    doc.close()
    return buf.getvalue()


def parse_table_html(image_bytes):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    files = {"document": ("page.png", image_bytes, "image/png")}
    data = {
        "ocr": "force",
        "output_formats": "['html','text']",
        "model": "document-parse",
    }
    resp = requests.post(API_URL, headers=headers, files=files, data=data, timeout=60)
    resp.raise_for_status()
    result = resp.json()
    content = result.get("content", {})
    if isinstance(content, dict):
        return (content.get("html") or content.get("text") or "").strip()
    return ""


def make_chunks(structured_html, car, page):
    if table_page_to_pieces:
        pieces = table_page_to_pieces(structured_html)
        pieces = [q for piece in pieces for q in (
            split_text(piece, CHUNK_SIZE, OVERLAP) if len(piece) > CHUNK_SIZE else [piece]
        )]
    else:
        pieces = [structured_html]

    return [{
        "car": car,
        "page": page,
        "chunk_id": f"{car}_p{page}_{idx}",
        "n_chars": len(piece),
        "text": piece,
        "needs_review": True,
        "table_structured": True,
    } for idx, piece in enumerate(pieces)]


def main():
    if not API_KEY:
        print("UPSTAGE_API_KEY 없음. .env 확인")
        return
    if not TARGET_PDF.exists():
        print(f"PDF 없음: {TARGET_PDF}")
        return

    print(f"실험 대상: {TARGET_PDF.name} p.{TARGET_PAGE}\n")

    print("1) PyMuPDF 원본 추출 (before)...")
    before = extract_pymupdf(TARGET_PDF, TARGET_PAGE)

    print("2) Upstage HTML 구조화 (after)...")
    img = render_page_bytes(TARGET_PDF, TARGET_PAGE)
    after = parse_table_html(img)

    OUT_COMPARE.parent.mkdir(exist_ok=True)
    with open(OUT_COMPARE, "w", encoding="utf-8") as f:
        f.write("===== BEFORE (PyMuPDF 원본) =====\n")
        f.write(before + "\n\n")
        f.write("===== AFTER (Upstage HTML) =====\n")
        f.write(after + "\n")
    print(f"   비교 저장 -> {OUT_COMPARE}")

    print("3) 행 단위 청킹 (chunker 방식)...")
    chunks = make_chunks(after, CAR, TARGET_PAGE)
    with open(OUT_CHUNKS, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)

    print(f"\n완료: {len(chunks)}개 청크 -> {OUT_CHUNKS}")
    print("\n--- 생성된 청크 미리보기 ---")
    for c in chunks[:8]:
        print(f"  [{c['chunk_id']}] ({c['n_chars']}자) {c['text'][:70]}")


if __name__ == "__main__":
    main()