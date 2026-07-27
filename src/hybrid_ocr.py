"""
hybrid.py
인수인계용 3번째 파일 생성 - PyMuPDF 텍스트 + 눕힌 표 페이지 OCR 복구

[처리 흐름]
  1) 전량 PyMuPDF 텍스트를 불러온다 (pages_pymupdf.json).
     - 텍스트 '순서'는 PyMuPDF가 pdfplumber보다 안정적이라 기본으로 채택.
  2) '글자만 90° 눕힌' 표 페이지를 글자방향 벡터(dir)로 탐지한다.
     - 페이지 회전속성이 아니라 글자만 눕은 경우라 dir 검사가 필요.
  3) 그 페이지만 이미지로 렌더 → 똑바로 회전 → EasyOCR로 다시 읽는다.
     - 눕힌 표는 선이 없어 두 파서 모두 셀 구조 인식 실패 → OCR로 우회.
  4) OCR 결과가 원본보다 나으면 교체하고, 각 페이지에 플래그를 붙인다.

[OCR 엔진] EasyOCR (딥러닝 기반, pip만으로 설치됨 - 별도 프로그램 불필요)
           설치:  pip install easyocr
           최초 실행 시 한글/영문 인식 모델을 자동 다운로드한다.

[출력] output/pages_hybrid.json
[주의] OCR도 셀 구조(행/열 관계)까지 복원하진 못한다. 숫자·단위 오인식도
       있을 수 있으므로 needs_review 플래그를 유지하고, 검색 평가 전
       육안 검수를 권장한다.
"""

import json
import io
from pathlib import Path

# PyMuPDF - PDF 열기/렌더용
try:
    import pymupdf as fitz
except ImportError:
    import fitz

# ---- OCR 의존성 (없으면 OCR 없이 플래그만 부착하고 넘어감) ------------
try:
    import easyocr                    # OCR 엔진
    from PIL import Image            # 페이지 이미지 회전용
    import numpy as np               # EasyOCR 입력은 numpy 배열
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ----------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------
DATA_DIR = Path("data")                     # 원본 PDF 폴더
IN_PATH = Path("output/pages_pymupdf.json") # 입력: PyMuPDF 추출 결과
OUT_PATH = Path("output/pages_hybrid.json") # 출력: 하이브리드 결과

DIR_TOLERANCE = 0.3      # 글자방향 벡터의 |sin|이 이보다 크면 '눕은 글자'로 판단
TILTED_RATIO = 30        # 한 페이지에서 눕은 라인이 30%를 넘으면 '눕힌 페이지'
OCR_DPI = 300            # OCR용 이미지 렌더 해상도(높을수록 정확·느림)
OCR_LANGS = ["ko", "en"] # 인식 언어: 한글 + 영문/숫자

# 차종 매핑 - CN7HEV가 CN7보다 먼저 (loader 파일들과 동일 규칙)
CAR_MAP = {
    "CN7HEV": "avante_hev",
    "CN7":    "avante",
    "CE1":    "ioniq6",
    "NX4":    "tucson",
    "NH2":    "nexo",
}

# EasyOCR Reader는 무거우므로 전역에서 1회만 생성해 재사용 (지연 초기화)
_reader = None


def get_reader():
    """EasyOCR Reader를 최초 호출 때 한 번만 생성 (모델 로딩 비용 절약)"""
    global _reader
    if _reader is None:
        # gpu=False → CPU 사용. GPU 있으면 True로 바꾸면 훨씬 빠름
        _reader = easyocr.Reader(OCR_LANGS, gpu=False)
    return _reader


def get_car_name(filename):
    """파일명에서 차종 코드 추출"""
    for keyword, car in CAR_MAP.items():
        if keyword in filename:
            return car
    return filename


def is_tilted(page):
    """
    페이지가 '눕힌 표'인지 판정.
    반환: (눕힘 여부, sin 합) - sin 합의 부호로 눕은 방향을 짐작한다.
    """
    data = page.get_text("dict")             # 글자 위치·방향이 담긴 구조
    total = tilt = 0
    sin_sum = 0.0
    for block in data.get("blocks", []):     # 블록 → 라인 순회
        for line in block.get("lines", []):
            dx, dy = line.get("dir", (1, 0))  # 라인의 쓰기 방향 벡터(cos, sin)
            total += 1
            if abs(dy) > DIR_TOLERANCE:       # sin 성분이 크면 눕은 라인
                tilt += 1
                sin_sum += dy
    ratio = (tilt / total * 100) if total else 0
    # 페이지 회전속성이 있거나, 눕은 라인 비율이 기준을 넘으면 눕힌 페이지
    tilted = page.rotation != 0 or ratio > TILTED_RATIO
    return tilted, sin_sum


def score_text(text):
    """텍스트 품질 점수 = 한글 + 숫자 글자 수 (OCR/원본 중 나은 쪽 판정용)"""
    cnt = 0
    for ch in text:
        if "가" <= ch <= "힣" or ch.isdigit():
            cnt += 1
    return cnt


def ocr_page(page, sin_hint):
    """
    페이지를 이미지로 렌더 → 90/270도 회전 → OCR.
    두 방향 중 점수(한글+숫자 수)가 높은 결과를 채택한다.
    (눕은 방향을 정확히 몰라도 두 방향 다 시도하므로 안전)
    """
    zoom = OCR_DPI / 72                       # 72dpi 기준 → 목표 dpi 배율
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))   # 페이지 렌더
    img = Image.open(io.BytesIO(pix.tobytes("png")))        # PIL 이미지로

    # sin 부호로 우선 시도할 방향을 정하되, 두 방향 모두 시도
    angles = [270, 90] if sin_hint >= 0 else [90, 270]
    reader = get_reader()
    best_text, best_score = "", -1
    for ang in angles:
        rotated = img.rotate(ang, expand=True)            # 글자가 똑바로 서게 회전
        # detail=0 → 좌표 없이 텍스트만, paragraph=True → 줄을 문단으로 묶음
        lines = reader.readtext(np.array(rotated), detail=0, paragraph=True)
        text = "\n".join(lines).strip()
        s = score_text(text)
        if s > best_score:                    # 더 많은 한글/숫자를 읽은 쪽 채택
            best_text, best_score = text, s
    return best_text


def main():
    # 입력 파일(=PyMuPDF 결과)이 있어야 진행 가능
    if not IN_PATH.exists():
        print(f"입력 없음: {IN_PATH} — 먼저 loader_pymupdf.py 실행")
        return

    with open(IN_PATH, encoding="utf-8") as f:
        pages = json.load(f)                  # 페이지 리스트 로드
    print(f"불러옴: {IN_PATH} ({len(pages):,} 페이지)")

    if not OCR_AVAILABLE:
        print("\n[경고] easyocr 미설치 — OCR 없이 플래그만 부착합니다.")
        print("       pip install easyocr 후 재실행하면 OCR 복구가 적용됩니다.")

    # (차종, 페이지)로 빠르게 찾기 위한 인덱스
    index = {(p["car"], p["page"]): p for p in pages}

    ocr_done = flagged = 0                     # 카운터

    # 원본 PDF를 다시 열어 눕힘 판정 + OCR 수행
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        car = get_car_name(pdf_path.stem)
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            rec = index.get((car, i + 1))     # 해당 페이지의 JSON 레코드
            if rec is None:
                continue

            tilted, sin_hint = is_tilted(page)         # 눕힘 판정
            rec["needs_review"] = tilted               # 검수 필요 여부 플래그
            rec["review_reason"] = "눕힌표_셀구조손실" if tilted else ""
            rec["ocr_applied"] = False                 # OCR 교체 여부(기본 False)

            if tilted:
                flagged += 1
                if OCR_AVAILABLE:
                    ocr_text = ocr_page(page, sin_hint)         # OCR 재추출
                    # OCR이 원본보다 한글/숫자를 더 많이 읽었을 때만 교체
                    if score_text(ocr_text) > score_text(rec["text"]):
                        rec["text_pymupdf"] = rec["text"]       # 원본 보존
                        rec["text"] = ocr_text                  # OCR 결과로 교체
                        rec["n_chars"] = len(ocr_text)
                        rec["ocr_applied"] = True
                        ocr_done += 1
                    print(f"  OCR: {car} p.{i+1}  "
                          f"({'교체됨' if rec['ocr_applied'] else '원본유지'})")
        doc.close()

    # 결과 저장
    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=1)

    # 요약 출력
    print(f"\n{'=' * 52}")
    print("하이브리드 파일 생성 완료")
    print(f"{'=' * 52}")
    print(f"전체 페이지        : {len(pages):,}")
    print(f"눕힌 표 페이지     : {flagged}  (needs_review=True)")
    print(f"OCR 교체 적용      : {ocr_done}")
    print(f"텍스트 출처        : 일반=PyMuPDF, 눕힌표=OCR(교체분)")
    print(f"저장: {OUT_PATH}")



if __name__ == "__main__":
    main()