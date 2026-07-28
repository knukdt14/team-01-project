"""
loader_pdfplumber.py
현대·기아 취급설명서 5종 PDF 텍스트 추출 - pdfplumber 버전

[출력] output/pages_pdfplumber.json
"""

import json
import time
from pathlib import Path

import pdfplumber

# ----------------------------------------------------------------------
# 설정 (loader_pymupdf.py 와 동일하게 유지해야 공정한 비교가 됨)
# ----------------------------------------------------------------------
DATA_DIR = Path("data")
OUT_PATH = Path("output/pages_pdfplumber.json")

# 차종 매핑 - CN7HEV가 CN7보다 먼저 와야 함 
CAR_MAP = {
    "CN7HEV": "avante_hev",
    "CN7":    "avante",
    "CE1":    "ioniq6",
    "NX4":    "tucson",
    "NH2":    "nexo",
}

# 눕힌 페이지 샘플 확인용 - 여기서 글자 역순 현상이 눈으로 보인다
CHECK_PAGES = {
    "avante": [390, 391],
    "avante_hev": [396, 397],
}


def get_car_name(filename):
    """파일명에서 차종 코드 추출 (넣은 순서대로 검사)"""
    for keyword, car in CAR_MAP.items():
        if keyword in filename:
            return car
    return filename


def get_rotation(page):
    rot = getattr(page, "rotation", None)    # page에 rotation이라는 속성 있으면 가져오기
    if rot is None:
        rot = page.page_obj.get("/Rotate", 0)    # 구버전: PDF 객체에서 직접
    return int(rot or 0)


def extract_pdf(pdf_path):
    """PDF 1권을 페이지 단위 dict 리스트로 변환"""
    car = get_car_name(pdf_path.stem)
    pages = []

    with pdfplumber.open(pdf_path) as pdf:       # with 로 열어 자동 닫힘
        for i, page in enumerate(pdf.pages):     # 페이지 순회
            text = page.extract_text() or ""     # 추출 실패 시 None → 빈 문자열 방어
            text = text.strip()
            pages.append({
                "car": car,
                "page": i + 1,
                "rotation": get_rotation(page),
                "n_chars": len(text),
                "text": text,
            })

    return pages


def print_stats(pages, parser_name, elapsed):
    """추출 결과 통계 출력 (pymupdf 버전과 동일 포맷)"""
    total = len(pages)
    total_chars = sum(p["n_chars"] for p in pages)
    empty = sum(1 for p in pages if p["n_chars"] == 0)
    rotated = sum(1 for p in pages if p["rotation"] != 0)
    rotated_ok = sum(1 for p in pages if p["rotation"] != 0 and p["n_chars"] > 0)

    print(f"\n{'=' * 55}")
    print(f"[{parser_name}] 추출 결과")
    print(f"{'=' * 55}")
    print(f"총 페이지        : {total:,}")
    print(f"총 문자 수       : {total_chars:,}")
    print(f"페이지당 평균    : {total_chars / total:.1f}자")
    print(f"빈 페이지        : {empty} ({empty / total * 100:.1f}%)")
    print(f"회전 페이지      : {rotated}")
    if rotated:
        print(f"회전 페이지 성공 : {rotated_ok}/{rotated} "
              f"({rotated_ok / rotated * 100:.1f}%)")
    print(f"처리 시간        : {elapsed:.1f}초")

    print("\n[차종별]")
    for car in sorted(set(p["car"] for p in pages)):
        sub = [p for p in pages if p["car"] == car]
        chars = sum(p["n_chars"] for p in sub)
        print(f"  {car:12s} {len(sub):5,}p  {chars:9,}자  "
              f"평균 {chars / len(sub):.0f}자/p")



def main():
    start = time.time()
    all_pages = []

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"PDF 없음: {DATA_DIR.resolve()}")
        return

    for pdf_path in pdf_files:
        print(f"처리 중: {pdf_path.name}")
        all_pages.extend(extract_pdf(pdf_path))

    elapsed = time.time() - start

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_pages, f, ensure_ascii=False, indent=1)

    print_stats(all_pages, "pdfplumber", elapsed)
    print(f"\n저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()