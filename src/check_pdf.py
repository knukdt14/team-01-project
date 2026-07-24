"""
check_pdf.py — 취급설명서 PDF 5종 추출 가능성 진단

프로젝트 착수 전, 대상 PDF가 RAG 파이프라인에 투입 가능한지 검증한다.
  1) 텍스트 레이어 존재 여부 (스캔 이미지 PDF가 아닌지)
  2) 한글 추출 정상 여부 (폰트 ToUnicode 매핑 문제 여부)
  3) 로더별 추출 성능 비교 (PyMuPDF / pypdf / pdfplumber)
  4) 표(Table) 검출 가능 여부

실행:  python src/check_pdf.py
"""

import io
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
import pypdf

# Windows 콘솔(cp949)에서 한글 깨짐 방지
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PDF_DIR = Path(r"C:\Users\KDT16\Downloads")

# 파일명 → (차량, 파워트레인)
TARGETS = {
    "CN7_2026_ko_KR.pdf": ("아반떼", "가솔린"),
    "CN7HEV_2026_ko_KR.pdf": ("아반떼 HEV", "하이브리드"),
    "CE1_2025_ko_KR.pdf": ("아이오닉6", "전기(EV)"),
    "NX4_2025_ko_KR.pdf": ("투싼", "디젤"),
    "NH2_2026_ko_KR.pdf": ("넥쏘", "수소(FCEV)"),
}

HANGUL = re.compile(r"[가-힣]")
SAMPLE_COUNT = 12  # 문서 전체에 고르게 분포시킬 샘플 페이지 수


def sample_pages(total: int, n: int = SAMPLE_COUNT) -> list[int]:
    """표지·목차를 피해 본문 구간에서 고르게 페이지를 뽑는다."""
    start, end = int(total * 0.15), int(total * 0.95)
    step = max(1, (end - start) // n)
    return list(range(start, end, step))[:n]


def count_ko(text: str) -> int:
    return len(HANGUL.findall(text or ""))


def check_one(path: Path, vehicle: str, powertrain: str) -> dict:
    doc = fitz.open(path)
    total = doc.page_count
    pages = sample_pages(total)

    ko_mu = ko_pypdf = ko_plumber = 0
    tables = 0

    reader = pypdf.PdfReader(path)
    with pdfplumber.open(path) as plumber:
        for i in pages:
            ko_mu += count_ko(doc[i].get_text())
            ko_pypdf += count_ko(reader.pages[i].extract_text())

            p = plumber.pages[i]
            ko_plumber += count_ko(p.extract_text())
            tables += len(p.find_tables())

    doc.close()

    return {
        "vehicle": vehicle,
        "powertrain": powertrain,
        "pages": total,
        "size_mb": path.stat().st_size / 1024 / 1024,
        "sampled": len(pages),
        "ko_mu": ko_mu,
        "ko_pypdf": ko_pypdf,
        "ko_plumber": ko_plumber,
        "tables": tables,
    }


def main() -> None:
    print("=" * 78)
    print("취급설명서 PDF 추출 진단")
    print("=" * 78)

    results = []
    for filename, (vehicle, powertrain) in TARGETS.items():
        path = PDF_DIR / filename
        if not path.exists():
            print(f"[누락] {filename}")
            continue
        print(f"  검사 중… {vehicle}")
        results.append(check_one(path, vehicle, powertrain))

    # --- 요약 ---
    print("\n" + "=" * 78)
    print(f"{'차량':<12}{'파워트레인':<12}{'페이지':>6}{'MB':>7}"
          f"{'PyMuPDF':>10}{'pypdf':>9}{'plumber':>9}{'표':>6}")
    print("-" * 78)
    for r in results:
        print(f"{r['vehicle']:<12}{r['powertrain']:<12}{r['pages']:>6}"
              f"{r['size_mb']:>7.1f}{r['ko_mu']:>10,}{r['ko_pypdf']:>9,}"
              f"{r['ko_plumber']:>9,}{r['tables']:>6}")

    # --- 판정 ---
    print("\n" + "=" * 78)
    print("판정")
    print("-" * 78)
    for r in results:
        best = max(r["ko_mu"], r["ko_pypdf"], r["ko_plumber"])
        per_page = best / r["sampled"]
        if per_page >= 100:
            verdict = "✅ 양호 — RAG 투입 가능"
        elif per_page >= 20:
            verdict = "⚠️  빈약 — 이미지 위주 페이지 많음"
        else:
            verdict = "❌ 실패 — OCR 필요"
        print(f"{r['vehicle']:<12} 페이지당 한글 {per_page:>6.0f}자   {verdict}")

    total_ko = sum(r["ko_mu"] for r in results)
    total_tb = sum(r["tables"] for r in results)
    print("-" * 78)
    print(f"합계: 한글 {total_ko:,}자 (샘플 기준) / 표 {total_tb}개 검출")
    print("=" * 78)


if __name__ == "__main__":
    main()
