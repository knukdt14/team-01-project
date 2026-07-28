"""
replace_table_chunks.py
chunks_llmhybrid.json에서 '정자 표 페이지' 청크만 Upstage 행단위로 교체

[유지되는 것]
  · 줄글(일반 페이지) 청크 - 그대로
  · 눕힌 표 11장 청크 - 팀원이 자른 것 그대로 (건드리지 않음)

[교체되는 것]
  · 정자 표 페이지 청크 - PyMuPDF 뭉갠 것 → Upstage 행단위로 교체

[안전]
  · 페이지별 처리 후 바로 저장 → 끊겨도 재개 (이미 한 건 스킵)
  · 원본 백업(chunks_llmhybrid_backup.json) 먼저 생성

[실행]
  pip install pymupdf pdfplumber pillow requests python-dotenv beautifulsoup4
  python src/replace_table_chunks.py
"""
import os, io, json, time, shutil
from pathlib import Path
try:
    import pymupdf as fitz
except ImportError:
    import fitz
import pdfplumber
import requests
from PIL import Image
from bs4 import BeautifulSoup
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name=="src" else Path.cwd()
DATA = ROOT/"data"
CHUNKS = ROOT/"output"/"chunks_llmhybrid.json"
BACKUP = ROOT/"output"/"chunks_llmhybrid_backup.json"
PROGRESS = ROOT/"output"/"replace_progress.json"
FAILED = ROOT/"output"/"replace_failed.txt"

API_KEY = os.environ.get("UPSTAGE_API_KEY","")
API_URL = "https://api.upstage.ai/v1/document-digitization"

CAR_MAP = {"CN7HEV":"avante_hev","CN7":"avante","CE1":"ioniq6","NX4":"tucson","NH2":"nexo"}
def get_car(fn):
    for k,v in CAR_MAP.items():
        if k in fn: return v
    return fn

# 눕힌 표 11장 - 팀원이 이미 처리, 건드리지 않음
ROTATED = {
    ("avante", 390),("avante",391),("avante",392),("avante",393),("avante",394),("avante",395),
    ("avante_hev",228),("avante_hev",396),("avante_hev",397),("avante_hev",398),("avante_hev",399),
}


def find_table_pages(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as doc:
        for i, page in enumerate(doc.pages):
            if page.find_tables():
                pages.append(i+1)
    return pages


def get_html(pdf_path, page_no):
    doc = fitz.open(pdf_path)
    pix = doc[page_no-1].get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    doc.close()
    resp = requests.post(API_URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        files={"document": ("p.png", buf.getvalue(), "image/png")},
        data={"ocr":"force","output_formats":"['html','text']","model":"document-parse"},
        timeout=90)
    resp.raise_for_status()
    c = resp.json().get("content", {})
    return c.get("html","") if isinstance(c, dict) else ""


def table_to_row_chunks(table, title=""):
    rows = table.find_all("tr")
    if not rows: return []
    thead = table.find("thead")
    if thead and thead.find_all("tr"):
        header_cells = thead.find_all("tr")[-1].find_all("td")
        tbody = table.find("tbody")
        body_rows = tbody.find_all("tr") if tbody else []
    else:
        header_cells = rows[0].find_all("td")
        body_rows = rows[1:]
    headers = [c.get_text(strip=True) for c in header_cells]
    prefix = f"[{title}] " if title else ""
    out = []
    for tr in body_rows:
        nested = tr.find("table")
        if nested:
            out.extend(table_to_row_chunks(nested, title)); continue
        cells = [c.get_text(strip=True) for c in tr.find_all("td")]
        if not any(cells): continue
        if len(headers) == len(cells):
            pairs = ", ".join(f"{h}: {v}" for h,v in zip(headers,cells) if v)
        else:
            pairs = ", ".join(c for c in cells if c)
        out.append(prefix + pairs)
    return out


def html_to_chunks(html, car, page):
    soup = BeautifulSoup(html, "html.parser")
    texts, title = [], ""
    for el in soup.find_all(["h1","h2","p","table"]):
        if el.name in ("h1","h2"):
            title = el.get_text(strip=True)
        elif el.name == "p":
            t = el.get_text(" ", strip=True)
            if len(t) > 15:
                texts.append(f"[{title}] {t}" if title else t)
        elif el.name == "table":
            if el.find_parent("table"): continue
            texts.extend(table_to_row_chunks(el, title))
    return [{
        "car": car, "page": page, "chunk_id": f"{car}_p{page}_{i}",
        "n_chars": len(t), "text": t,
        "needs_review": True, "table_structured": True,
    } for i, t in enumerate(texts) if t.strip()]


def main():
    if not API_KEY:
        print("UPSTAGE_API_KEY 없음"); return
    if not CHUNKS.exists():
        print(f"{CHUNKS} 없음"); return

    # 1. 원본 백업 (한 번만)
    if not BACKUP.exists():
        shutil.copy(CHUNKS, BACKUP)
        print(f"원본 백업 → {BACKUP}")

    # 2. 정자 표 페이지 목록 (표 전체 - 눕힌 표 11장)
    target_pages = set()   # {(car, page)}
    for pdf in sorted(DATA.glob("*.pdf")):
        car = get_car(pdf.stem)
        for p in find_table_pages(pdf):
            if (car, p) not in ROTATED:      # 눕힌 표 제외
                target_pages.add((car, p))
    print(f"교체 대상 정자 표 페이지: {len(target_pages)}개")

    # 3. 재개용 진행 기록
    done = set()
    if PROGRESS.exists():
        done = set(tuple(x) for x in json.load(open(PROGRESS, encoding="utf-8")))

    # 4. 기존 청크 로드, 교체 대상 페이지 청크는 일단 제거
    chunks = json.load(open(CHUNKS, encoding="utf-8"))
    # 아직 처리 안 한 대상 페이지 청크만 남기고, 이미 처리한 건 유지
    def is_target_unprocessed(c):
        key = (c["car"], c["page"])
        return key in target_pages and key not in done
    kept = [c for c in chunks if not is_target_unprocessed(c)]

    # 5. 대상 페이지를 Upstage로 재청킹
    todo = [(car,p) for (car,p) in target_pages if (car,p) not in done]
    print(f"남은 페이지: {len(todo)}개\n")

    pdf_by_car = {get_car(pdf.stem): pdf for pdf in DATA.glob("*.pdf")}

    for n, (car, page) in enumerate(sorted(todo), 1):
        try:
            html = get_html(pdf_by_car[car], page)
            new_chunks = html_to_chunks(html, car, page)
            kept.extend(new_chunks)
            done.add((car, page))
            json.dump(kept, open(CHUNKS,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
            json.dump(list(done), open(PROGRESS,"w",encoding="utf-8"), ensure_ascii=False)
            print(f"[{n}/{len(todo)}] {car} p{page}: {len(new_chunks)}청크 교체")
            time.sleep(0.3)
        except Exception as e:
            with open(FAILED,"a",encoding="utf-8") as f:
                f.write(f"{car} p{page}: {e}\n")
            print(f"[{n}/{len(todo)}] {car} p{page}: 실패 ({e})")
            continue

    print(f"\n완료! 총 {len(kept)}청크 → {CHUNKS}")
    print(f"원본은 {BACKUP}에 백업됨")


if __name__ == "__main__":
    main()