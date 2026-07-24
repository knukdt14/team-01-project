"""
config.py — 팀 공통 베이스라인 설정 (변수 통제의 핵심)

⚠️ 이 파일의 BASELINE 값은 팀 합의 없이 수정하지 않는다.
   각자 실험할 때는 variant()로 '자기 변수만' 바꿔서 쓴다.

   예)  B 담당 :  cfg = variant(chunk_size=1024)
        C 담당 :  cfg = variant(embedding_model="nlpai-lab/KURE-v1")
        D 담당 :  cfg = variant(prompt_type="constrained")
        A 담당 :  cfg = variant(llm="Qwen/Qwen2.5-7B-Instruct")
"""

from dataclasses import dataclass, replace
from pathlib import Path

# ─────────────────────────────── 경로 ───────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = PROJECT_ROOT / "eval"
INDEX_DIR = PROJECT_ROOT / "index"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"

# ──────────────────────── 대상 문서 & 메타데이터 ────────────────────────
# 파일명 → 차량 메타데이터. 검색 시 메타데이터 필터링의 기준이 된다.
VEHICLES: dict[str, dict[str, str]] = {
    "CN7_2026_ko_KR.pdf":    {"vehicle": "아반떼",      "powertrain": "가솔린",  "body": "세단"},
    "CN7HEV_2026_ko_KR.pdf": {"vehicle": "아반떼 HEV",  "powertrain": "하이브리드", "body": "세단"},
    "CE1_2025_ko_KR.pdf":    {"vehicle": "아이오닉6",   "powertrain": "전기",    "body": "세단"},
    "NX4_2025_ko_KR.pdf":    {"vehicle": "투싼",       "powertrain": "디젤",    "body": "SUV"},
    "NH2_2026_ko_KR.pdf":    {"vehicle": "넥쏘",       "powertrain": "수소",    "body": "SUV"},
}

# 질문에서 차량을 추출할 때 쓰는 별칭 (C 담당: 메타데이터 필터링)
VEHICLE_ALIASES: dict[str, str] = {
    "아반떼": "아반떼", "엘란트라": "아반떼",
    "아반떼 하이브리드": "아반떼 HEV", "아반떼HEV": "아반떼 HEV", "아반떼 hev": "아반떼 HEV",
    "아이오닉6": "아이오닉6", "아이오닉 6": "아이오닉6", "ioniq6": "아이오닉6",
    "투싼": "투싼", "tucson": "투싼",
    "넥쏘": "넥쏘", "nexo": "넥쏘",
}


# ──────────────────────────── 설정 ────────────────────────────
@dataclass(frozen=True)
class Config:
    """RAG 파이프라인 전체 설정. 담당자별 실험 변수가 구간별로 묶여 있다."""

    # ── [1][2] 문서 로드·청킹 ─ 담당 B ──────────────────────────
    loader: str = "pymupdf"          # pymupdf | pypdf | pdfplumber
    chunk_size: int = 512            # 토큰 기준
    chunk_overlap: int = 50          # 토큰 기준 (약 10%)
    table_mode: str = "markdown"     # markdown | raw | none

    # ── [3][4] 임베딩·벡터스토어 ─ 담당 C ────────────────────────
    embedding_model: str = "BAAI/bge-m3"
    vectorstore: str = "faiss"       # faiss | chroma
    similarity: str = "cosine"       # cosine | l2 | ip

    # ── [5] 검색 ─ 담당 C ──────────────────────────────────────
    top_k: int = 3
    search_type: str = "similarity"  # similarity | mmr
    use_metadata_filter: bool = False

    # ── [6] 프롬프트 ─ 담당 D ───────────────────────────────────
    prompt_type: str = "basic"       # basic | role | constrained

    # ── [7] LLM ─ 담당 A ──────────────────────────────────────
    # ⚠️ GPU(VRAM) 여건에 따라 로컬 7B 모델이 안 돌 수 있음.
    #    그럴 경우 API 모델을 베이스라인으로 사용한다.
    llm: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_new_tokens: int = 512

    def tag(self) -> str:
        """실험 결과 파일명에 쓸 짧은 식별자."""
        return (f"{self.loader}_cs{self.chunk_size}_ov{self.chunk_overlap}"
                f"_{self.embedding_model.split('/')[-1]}_{self.vectorstore}"
                f"_k{self.top_k}_{self.prompt_type}_{self.llm.split('/')[-1]}")


BASELINE = Config()


def variant(**overrides) -> Config:
    """베이스라인에서 지정한 값만 바꾼 설정을 만든다.

    >>> variant(chunk_size=1024, chunk_overlap=100)
    """
    unknown = set(overrides) - set(BASELINE.__dataclass_fields__)
    if unknown:
        raise ValueError(f"알 수 없는 설정 항목: {unknown}")
    return replace(BASELINE, **overrides)


if __name__ == "__main__":
    print("BASELINE:", BASELINE.tag())
    print("변형 예시:", variant(chunk_size=1024).tag())
