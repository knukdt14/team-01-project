"""
single_page_v2.py
특정 페이지 → Upstage HTML 구조화 → 표별로 행 단위 청킹 (개선판)

[이전 문제] chunker.py의 table_page_to_pieces가 페이지의 모든 제목을 헤더로
            긁어와서, 청크마다 "타이어효율 / 알아두기 / 속도등급..." 잡동사니가 붙음

[개선] 표(<table>)를 각각 독립적으로 처리:
       - 표마다 자기 헤더(thead)만 사용
       - 각 데이터 행(<tr>)을 "헤더:값" 형태의 한 문장 청크로
       - 표 밖 문단은 별도 청크로
       - 중첩 표도 처리
"""
import os, io, sys, json
from pathlib import Path
try:
    import pymupdf as fitz
except ImportError:
    import fitz
import requests
from PIL import Image
from bs4 import BeautifulSoup
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name=="src" else Path.cwd()
TARGET_PDF = ROOT/"data"/"CE1_2025_ko_KR.pdf"
TARGET_PAGE = 62
CAR = "ioniq6"
OUT = ROOT/"output"/"chunks_single.json"
API_KEY = os.environ.get("UPSTAGE_API_KEY","")


def get_html(pdf, page_no):
    doc = fitz.open(pdf)
    pix = doc[page_no-1].get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    doc.close()
    resp = requests.post(
        "https://api.upstage.ai/v1/document-digitization",
        headers={"Authorization": f"Bearer {API_KEY}"},
        files={"document": ("p.png", buf.getvalue(), "image/png")},
        data={"ocr":"force","output_formats":"['html','text']","model":"document-parse"},
        timeout=60)
    resp.raise_for_status()
    c = resp.json().get("content", {})
    return c.get("html","") if isinstance(c, dict) else ""


def table_to_row_chunks(table, title=""):
    """표 하나 → 행별 '헤더:값' 문장 리스트"""
    rows = table.find_all("tr")
    if not rows:
        return []

    # 헤더 추출: thead의 마지막 tr, 없으면 첫 tr
    thead = table.find("thead")
    if thead and thead.find_all("tr"):
        header_cells = thead.find_all("tr")[-1].find_all("td")
        body_rows = table.find("tbody").find_all("tr") if table.find("tbody") else []
    else:
        header_cells = rows[0].find_all("td")
        body_rows = rows[1:]

    headers = [c.get_text(strip=True) for c in header_cells]
    chunks = []
    prefix = f"[{title}] " if title else ""

    for tr in body_rows:
        # 중첩 표가 있으면 재귀 처리
        nested = tr.find("table")
        if nested:
            chunks.extend(table_to_row_chunks(nested, title))
            continue
        cells = [c.get_text(strip=True) for c in tr.find_all("td")]
        if not any(cells):
            continue
        # 헤더:값 짝지어 문장으로
        if len(headers) == len(cells):
            pairs = ", ".join(f"{h}: {v}" for h, v in zip(headers, cells) if v)
        else:
            pairs = ", ".join(c for c in cells if c)
        chunks.append(prefix + pairs)
    return chunks


def make_chunks(html, car, page):
    soup = BeautifulSoup(html, "html.parser")   # lxml 대신 내장 파서
    chunks = []
    current_title = ""

    for el in soup.find_all(["h1","h2","p","table"]):
        if el.name in ("h1","h2"):
            current_title = el.get_text(strip=True)
        elif el.name == "p":
            txt = el.get_text(" ", strip=True)
            if len(txt) > 15:   # 짧은 건 스킵
                chunks.append(f"[{current_title}] {txt}" if current_title else txt)
        elif el.name == "table":
            # 중첩 표 방지: 최상위 table만 (부모에 table 없는 것)
            if el.find_parent("table"):
                continue
            chunks.extend(table_to_row_chunks(el, current_title))

    result = []
    for idx, text in enumerate(chunks):
        if not text.strip():
            continue
        result.append({
            "car": car, "page": page,
            "chunk_id": f"{car}_p{page}_{idx}",
            "n_chars": len(text), "text": text,
            "needs_review": True, "table_structured": True,
        })
    return result


def main():
    if not API_KEY:
        print("API_KEY 없음"); return
    print(f"{TARGET_PDF.name} p.{TARGET_PAGE} 처리...")
    html = get_html(TARGET_PDF, TARGET_PAGE)
    chunks = make_chunks(html, CAR, TARGET_PAGE)
    OUT.parent.mkdir(exist_ok=True)
    json.dump(chunks, open(OUT,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n{len(chunks)}개 청크 생성\n")
    for c in chunks:
        print(f"  [{c['chunk_id']}] {c['text'][:90]}")


if __name__ == "__main__":
    main()