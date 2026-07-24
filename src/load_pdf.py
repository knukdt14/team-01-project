"""
load_pdf.py — [1] PDF 로드 + [2] 청킹  ─ 담당 B

취급설명서 5권을 읽어 LangChain Document 리스트로 변환한다.
각 청크에는 차량·파워트레인·페이지 메타데이터가 붙어, 검색 단계에서
'차량 간 정보 혼입(cross-contamination)'을 막는 필터 기준이 된다.

⚠️ 사전 검증 결과: 정기 점검 주기표가 90° 회전 인쇄되어 있어
   pdfplumber는 글자가 역순으로 깨진다. PyMuPDF만 정상 처리하므로 기본값으로 둔다.

실행:  python src/load_pdf.py
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import BASELINE, DATA_DIR, VEHICLES, Config


# ────────────────────────── 로더별 페이지 텍스트 추출 ──────────────────────────
def _extract_pymupdf(path: Path) -> list[str]:
    """기본 로더. 회전된 표를 올바른 읽기 순서로 복원한다."""
    import fitz

    with fitz.open(path) as doc:
        return [page.get_text() for page in doc]


def _extract_pypdf(path: Path) -> list[str]:
    import pypdf

    reader = pypdf.PdfReader(path)
    return [(page.extract_text() or "") for page in reader.pages]


def _extract_pdfplumber(path: Path) -> list[str]:
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        return [(page.extract_text() or "") for page in pdf.pages]


EXTRACTORS = {
    "pymupdf": _extract_pymupdf,
    "pypdf": _extract_pypdf,
    "pdfplumber": _extract_pdfplumber,
}


# ──────────────────────────────── 표 처리 ────────────────────────────────
def extract_tables_markdown(path: Path, page_no: int) -> list[str]:
    """지정 페이지의 표를 마크다운 문자열로 변환한다.

    표는 청크 경계에서 잘리면 의미가 무너지므로, 통째로 하나의 청크가 되게 한다.
    TODO(B): 회전 표는 pdfplumber 기본 전략으로 잡히지만 셀 텍스트가 역순이 된다.
             PyMuPDF의 find_tables() 결과와 비교해 더 나은 쪽을 선택할 것.
    """
    import pdfplumber

    md_tables: list[str] = []
    with pdfplumber.open(path) as pdf:
        page = pdf.pages[page_no]
        for table in page.extract_tables():
            rows = [[(c or "").replace("\n", " ").strip() for c in row] for row in table]
            rows = [r for r in rows if any(r)]
            if len(rows) < 2:
                continue
            header, *body = rows
            md = ["| " + " | ".join(header) + " |",
                  "|" + "---|" * len(header)]
            md += ["| " + " | ".join(r) + " |" for r in body]
            md_tables.append("\n".join(md))
    return md_tables


# ──────────────────────────────── 정제 ────────────────────────────────
def clean(text: str) -> str:
    """도면 라벨(OCN7030001L 등)·과도한 공백 제거."""
    text = re.sub(r"\b[A-Z]{2,4}\d{6,}[A-Z]?\b", " ", text)  # 이미지 코드
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ──────────────────────────────── 메인 ────────────────────────────────
def load_documents(cfg: Config = BASELINE, data_dir: Path = DATA_DIR) -> list[Document]:
    """5권을 페이지 단위 Document로 로드 (청킹 전)."""
    extractor = EXTRACTORS[cfg.loader]
    docs: list[Document] = []

    for filename, meta in VEHICLES.items():
        path = data_dir / filename
        if not path.exists():
            print(f"  [건너뜀] 파일 없음: {filename}")
            continue

        pages = extractor(path)
        print(f"  {meta['vehicle']:<10} {len(pages):>4}p 로드")

        for i, raw in enumerate(pages):
            text = clean(raw)
            if len(text) < 30:            # 이미지 전용 페이지 제외
                continue
            docs.append(Document(
                page_content=text,
                metadata={**meta, "source": filename, "page": i + 1},
            ))

    return docs


def split_documents(docs: list[Document], cfg: Config = BASELINE) -> list[Document]:
    """청킹. 메타데이터는 청크에 그대로 승계된다."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,       # TODO(B): 토큰 기준으로 바꿔 비교해볼 것
    )
    chunks = splitter.split_documents(docs)
    for j, c in enumerate(chunks):
        c.metadata["chunk_id"] = j
    return chunks


def build_corpus(cfg: Config = BASELINE) -> list[Document]:
    print(f"[1] PDF 로드 (loader={cfg.loader})")
    docs = load_documents(cfg)
    print(f"[2] 청킹 (chunk_size={cfg.chunk_size}, overlap={cfg.chunk_overlap})")
    chunks = split_documents(docs, cfg)
    print(f"    페이지 {len(docs)}개 → 청크 {len(chunks)}개")
    return chunks


if __name__ == "__main__":
    import io
    import sys

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    chunks = build_corpus()
    if not chunks:
        raise SystemExit("청크가 없습니다. data/ 에 PDF를 넣었는지 확인하세요.")

    print("\n── 차량별 청크 수 ──")
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c.metadata["vehicle"]] = counts.get(c.metadata["vehicle"], 0) + 1
    for v, n in counts.items():
        print(f"  {v:<12} {n:>6,}개")

    print("\n── 샘플 청크 ──")
    sample = chunks[len(chunks) // 2]
    print(f"  {sample.metadata}")
    print(f"  {sample.page_content[:200]}…")
