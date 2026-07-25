"""
loader_pymupdf.py
현대·기아 취급설명서 5종 PDF 텍스트 추출 - PyMuPDF 버전

[역할] 파서 비교 실험의 A안. PDF 바이너리에서 페이지 단위로 텍스트를 뽑아
       JSON으로 저장한다. loader_pdfplumber.py 와 출력 형식이 동일해야
       나중에 두 결과를 나란히 비교할 수 있다.

[출력] output/pages_pymupdf.json
       [{"car":"avante","page":1,"rotation":0,"n_chars":512,"text":"..."}, ...]
"""

import json
import time
from pathlib import Path

# PyMuPDF import - 버전에 따라 모듈명이 달라서 두 경우 모두 처리
try:
    import pymupdf as fitz          # PyMuPDF 1.24 이상은 pymupdf 로 import
except ImportError:
    import fitz                     # 구버전은 fitz 로 import

# ----------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------
DATA_DIR = Path("data")                      # 원본 PDF 5종이 있는 폴더
OUT_PATH = Path("output/pages_pymupdf.json") # 추출 결과 저장 경로

# 파일명(프로젝트 코드)에 이 문자열이 들어있으면 해당 차종으로 인식.
# 주의: "CN7HEV_2026" 에는 "CN7" 도 포함되므로, CN7HEV 를 CN7 보다
#       먼저 둬야 HEV가 일반 아반떼로 잘못 잡히지 않는다 (dict는 넣은 순서대로 검사).
CAR_MAP = {
    "CN7HEV": "avante_hev",
    "CN7":    "avante",
    "CE1":    "ioniq6",
    "NX4":    "tucson",
    "NH2":    "nexo",
}

# 회전(눕힌) 페이지 육안 확인용 샘플 위치. 추출된 글자 순서가
# 정상인지 콘솔에서 바로 보기 위한 것. 실제 표 페이지 번호로 맞추면 됨.
CHECK_PAGES = {
    "avante": [390, 391],
    "avante_hev": [396, 397],
}


def get_car_name(filename):
    """파일명에서 차종 코드 추출. CAR_MAP을 넣은 순서대로 검사(CN7HEV 먼저)."""
    for keyword, car in CAR_MAP.items():
        if keyword in filename:
            return car
    return filename                          # 못 찾으면 파일명 그대로 사용


def extract_pdf(pdf_path):
    """PDF 1권을 열어 페이지 단위 dict 리스트로 변환"""
    car = get_car_name(pdf_path.stem)        # 이 PDF의 차종 코드
    doc = fitz.open(pdf_path)                # PDF 열기
    pages = []

    for i, page in enumerate(doc):           # 페이지를 하나씩 순회
        text = page.get_text("text")         # 텍스트 추출(회전 페이지도 정방향 보정됨)
        text = text.strip()                  # 앞뒤 공백 제거
        pages.append({
            "car": car,                      # 차종
            "page": i + 1,                   # 페이지 번호(1부터)
            "rotation": page.rotation,       # 페이지 회전 속성(/Rotate 값)
            "n_chars": len(text),            # 추출된 문자 수
            "text": text,                    # 추출된 텍스트 본문
        })

    doc.close()                              # PDF 닫기(메모리 해제)
    return pages


def print_stats(pages, parser_name, elapsed):
    """추출 결과 통계를 콘솔에 출력 (비교 자료용)"""
    total = len(pages)
    total_chars = sum(p["n_chars"] for p in pages)
    empty = sum(1 for p in pages if p["n_chars"] == 0)          # 빈 페이지 수
    rotated = sum(1 for p in pages if p["rotation"] != 0)       # 회전속성 페이지 수
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

    # 차종별 통계
    print("\n[차종별]")
    for car in sorted(set(p["car"] for p in pages)):
        sub = [p for p in pages if p["car"] == car]
        chars = sum(p["n_chars"] for p in sub)
        print(f"  {car:12s} {len(sub):5,}p  {chars:9,}자  "
              f"평균 {chars / len(sub):.0f}자/p")


def print_samples(pages):
    """회전(눕힌) 페이지 샘플 텍스트 출력 - 글자 순서 육안 확인용"""
    print("\n[회전 페이지 샘플 - 앞 80자]")
    for car, page_nums in CHECK_PAGES.items():
        for num in page_nums:
            hit = [p for p in pages if p["car"] == car and p["page"] == num]
            if hit:
                print(f"  {car} p.{num}: {hit[0]['text'][:80]}")


def main():
    start = time.time()                      # 처리 시간 측정 시작
    all_pages = []

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))   # data 폴더의 PDF 목록
    if not pdf_files:
        print(f"PDF 없음: {DATA_DIR.resolve()}")
        return

    for pdf_path in pdf_files:               # PDF를 한 권씩 처리
        print(f"처리 중: {pdf_path.name}")
        all_pages.extend(extract_pdf(pdf_path))

    elapsed = time.time() - start            # 총 소요 시간

    OUT_PATH.parent.mkdir(exist_ok=True)     # output 폴더 없으면 생성
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_pages, f, ensure_ascii=False, indent=1)  # 한글 그대로 저장

    print_stats(all_pages, "PyMuPDF", elapsed)
    print_samples(all_pages)
    print(f"\n저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()