"""
retriever.py — [5] 검색  ─ 담당 C

top_k 유사 청크를 검색한다. 본 프로젝트의 차별화 지점인
'차량 간 정보 혼입(cross-contamination)' 방지 로직이 여기 있다.

  예) "아반떼 타이어 공기압은?" → 질문에서 '아반떼' 추출
      → 아반떼 문서로만 검색 범위 제한

실행:  python src/retriever.py
"""

from __future__ import annotations

import re

from langchain_core.documents import Document

from config import BASELINE, VEHICLE_ALIASES, Config


# ─────────────────────── 질문에서 차량 추출 ───────────────────────
def detect_vehicle(question: str) -> str | None:
    """질문에 언급된 차량명을 찾는다. 없으면 None(전체 검색).

    긴 별칭부터 매칭해야 '아반떼 하이브리드'가 '아반떼'로 잘못 잡히지 않는다.
    """
    q = question.replace(" ", "").lower()
    for alias in sorted(VEHICLE_ALIASES, key=len, reverse=True):
        if alias.replace(" ", "").lower() in q:
            return VEHICLE_ALIASES[alias]
    return None


# ──────────────────────────── 검색 ────────────────────────────
def search(store, question: str, cfg: Config = BASELINE) -> list[Document]:
    """설정에 따라 검색을 수행한다."""
    kwargs: dict = {"k": cfg.top_k}

    if cfg.use_metadata_filter:
        vehicle = detect_vehicle(question)
        if vehicle:
            kwargs["filter"] = {"vehicle": vehicle}

    if cfg.search_type == "mmr":
        kwargs["fetch_k"] = cfg.top_k * 4
        return store.max_marginal_relevance_search(question, **kwargs)

    return store.similarity_search(question, **kwargs)


def format_docs(docs: list[Document]) -> str:
    """검색 결과를 프롬프트에 넣을 문자열로 변환. 출처를 함께 표기한다."""
    parts = []
    for d in docs:
        m = d.metadata
        parts.append(f"[{m.get('vehicle')} / {m.get('powertrain')} / p.{m.get('page')}]\n"
                     f"{d.page_content}")
    return "\n\n---\n\n".join(parts)


def get_retriever(store, cfg: Config = BASELINE):
    """LCEL 체인에 꽂을 수 있는 Runnable 형태로 반환."""
    from langchain_core.runnables import RunnableLambda

    return RunnableLambda(lambda q: search(store, q, cfg))


if __name__ == "__main__":
    import io
    import sys

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    # 차량 추출 로직만 단독 테스트 (인덱스 불필요)
    tests = [
        "아반떼 타이어 공기압은?",
        "아반떼 하이브리드 연비는?",
        "아이오닉6 급속충전 시간은?",
        "넥쏘 수소 충전 방법은?",
        "엔진오일 교환주기는?",          # 차량 미지정 → None
    ]
    print("── 질문에서 차량 추출 테스트 ──")
    for t in tests:
        print(f"  {t:<28} → {detect_vehicle(t)}")
