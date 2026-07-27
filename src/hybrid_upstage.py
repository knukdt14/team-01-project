"""
table_upstage.py
눕힌 표 11장만 Upstage LLM으로 구조화 → 나머지는 그대로 → pages_llmhybrid.json
"""

import os
import io
import json
import time
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:
    import fitz

import requests
from PIL import Image

# .env 에서 키 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ----------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------
DATA_DIR = Path("data")                          
HYBRID_PATH = Path("output/pages_hybrid.json")   
OUT_PATH = Path("output/pages_llmhybrid.json")     

API_URL = "https://api.upstage.ai/v1/document-digitization"  # Document Parse 엔드포인트
API_KEY = os.environ.get("UPSTAGE_API_KEY", "")  

RENDER_DPI = 300          # 표 이미지 렌더 해상도(높을수록 정확·용량↑)
DIR_TOLERANCE = 0.3       # 글자방향 |sin| 이 크면 눕은 글자
TILTED_RATIO = 30         # 라인의 30% 이상 눕으면 눕힌 페이지(방향 보정용)

# 파일명(코드) → 차종. CN7HEV 가 CN7 보다 먼저 와야 함
CAR_MAP = {
    "CN7HEV": "avante_hev", "CN7": "avante", "CE1": "ioniq6",
    "NX4": "tucson", "NH2": "nexo",
}


def get_car_name(fn):
    """파일명에서 차종 코드 추출"""
    for k, v in CAR_MAP.items():
        if k in fn:
            return v
    return fn


def tilt_direction(page):
    """눕은 방향 추정용 sin 합 반환 (이미지 회전 방향 결정에 사용)"""
    data = page.get_text("dict")
    sin_sum = 0.0
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            dx, dy = line.get("dir", (1, 0))   # 라인 쓰기 방향 벡터
            if abs(dy) > DIR_TOLERANCE:
                sin_sum += dy
    return sin_sum


def render_page_bytes(page, sin_hint):
    """페이지를 똑바로 세운 PNG 바이트로 렌더 (눕은 방향에 맞춰 회전)"""
    zoom = RENDER_DPI / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))   # 페이지 렌더
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    angle = 270 if sin_hint >= 0 else 90                    # 방향 보정
    img = img.rotate(angle, expand=True)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def parse_table(image_bytes):
    """Upstage Document Parse 호출 → 표 HTML(없으면 텍스트) 반환"""
    headers = {"Authorization": f"Bearer {API_KEY}"}
    files = {"document": ("page.png", image_bytes, "image/png")}
    data = {                                    # 표 구조를 HTML 로 받기 위한 옵션
        "ocr": "force",
        "output_formats": "['html','text']",
        "model": "document-parse",
    }
    resp = requests.post(API_URL, headers=headers, files=files, data=data, timeout=60)
    resp.raise_for_status()                     # HTTP 에러면 예외 발생
    result = resp.json()

    # 응답에서 표 HTML 우선, 없으면 일반 텍스트
    content = result.get("content", {})
    html = content.get("html") if isinstance(content, dict) else None
    text = content.get("text") if isinstance(content, dict) else None
    return (html or text or "").strip()


def main():
    # 키 확인
    if not API_KEY:
        print("UPSTAGE_API_KEY 없음.")
        print("  1) pip install python-dotenv")
        return

    # 입력 확인
    if not HYBRID_PATH.exists():
        print(f"입력 없음: {HYBRID_PATH} — hybrid.py 먼저 실행")
        return

    with open(HYBRID_PATH, encoding="utf-8") as f:
        pages = json.load(f)                    # 원본 로드(리스트)

    # (차종, 페이지)로 레코드 빠르게 찾기
    index = {(p["car"], p["page"]): p for p in pages}
    done = 0

    # 원본 PDF 를 다시 열어, 눕힌 표 페이지만 골라 구조화
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        car = get_car_name(pdf_path.stem)
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            rec = index.get((car, i + 1))
            # needs_review=True (눕힌 표) 인 페이지만 처리, 나머지는 건너뜀
            if rec is None or not rec.get("needs_review"):
                continue

            sin_hint = tilt_direction(page)              # 회전 방향
            img_bytes = render_page_bytes(page, sin_hint)  # 이미지화

            try:
                table_html = parse_table(img_bytes)       # Upstage 구조화
            except Exception as e:
                print(f"  실패: {car} p.{i+1}  ({e})")
                continue

            if table_html:
                if "text_pymupdf" not in rec:
                    rec["text_pymupdf"] = rec["text"]     # 원본 텍스트 보존
                rec["text"] = table_html                  # 구조화 결과로 교체
                rec["n_chars"] = len(table_html)
                rec["table_structured"] = True            # 구조화 완료 표시
                done += 1
                print(f"  구조화: {car} p.{i+1}  ({len(table_html)}자)")

            time.sleep(0.3)                               # API 레이트리밋 여유
        doc.close()

    # 새 파일로 저장 (원본 pages_hybrid.json 은 그대로)
    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=1)

    print(f"\n완료: 표 {done}장 구조화 → {OUT_PATH}")

if __name__ == "__main__":
    main()