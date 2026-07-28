"""팀원의 Chroma retriever를 프롬프트 엔지니어링 코드에 연결한다.

팀원 파일은 수정하지 않고 동적으로 불러온다. ``vectorstore_search.py``의
Chroma 경로가 해당 파일 기준의 상대경로이므로, 로딩과 검색 중에 작업
디렉터리를 벡터스토어 폴더로 변경한 뒤 즉시 원래 위치로 복원한다.
"""

from __future__ import annotations

import importlib.util
import os
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator


SUPPORTED_CARS = ("avante", "avante_hev", "ioniq6", "nexo", "tucson")
CAR_LABELS = {
    "avante": "아반떼",
    "avante_hev": "아반떼 하이브리드",
    "ioniq6": "아이오닉6",
    "nexo": "넥쏘",
    "tucson": "투싼",
}
CAR_ALIASES = {
    "아반떼 하이브리드": "avante_hev",
    "아반떼 hev": "avante_hev",
    "아반떼hev": "avante_hev",
    "avante hev": "avante_hev",
    "avante_hev": "avante_hev",
    "아반떼": "avante",
    "avante": "avante",
    "아이오닉 6": "ioniq6",
    "아이오닉6": "ioniq6",
    "ioniq 6": "ioniq6",
    "ioniq6": "ioniq6",
    "넥쏘": "nexo",
    "nexo": "nexo",
    "투싼": "tucson",
    "tucson": "tucson",
}
DEFAULT_TOP_K = 5
RETRIEVER_LABEL = "Chroma/bge-m3"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VECTORSTORE_DIR = PROJECT_ROOT / "src" / "embedding_&_vectorstore"
VECTORSTORE_MODULE_PATH = VECTORSTORE_DIR / "vectorstore_search.py"


@contextmanager
def _vectorstore_working_directory() -> Iterator[None]:
    """상대경로 Chroma 인덱스가 항상 같은 위치를 보도록 작업 경로를 고정한다."""

    previous_directory = Path.cwd()
    try:
        os.chdir(VECTORSTORE_DIR)
        yield
    finally:
        os.chdir(previous_directory)


def _load_vectorstore_module() -> ModuleType:
    if not VECTORSTORE_MODULE_PATH.is_file():
        raise RuntimeError(
            f"벡터 검색 파일을 찾을 수 없습니다: {VECTORSTORE_MODULE_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "team_vectorstore_search",
        VECTORSTORE_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("팀원의 벡터 검색 모듈을 불러올 수 없습니다.")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise RuntimeError(
            "벡터 검색 패키지가 부족합니다. TF_ENV에서 프로젝트의 "
            "requirements.txt를 설치했는지 확인하세요."
        ) from exc
    return module


def _normalize_car_text(value: str) -> str:
    return "".join(value.lower().replace("_", " ").replace("-", " ").split())


def detect_car_from_question(question: str) -> str | None:
    """질문에 포함된 한글·영문 차종명을 검색용 메타데이터 값으로 바꾼다."""

    normalized_question = _normalize_car_text(question)
    for alias in sorted(CAR_ALIASES, key=lambda item: len(_normalize_car_text(item)), reverse=True):
        if _normalize_car_text(alias) in normalized_question:
            return CAR_ALIASES[alias]
    return None


def create_retriever(car: str | None = None, k: int = DEFAULT_TOP_K):
    """차종 필터가 적용된 Chroma retriever를 한 번 로드한다."""

    if car is not None and car not in SUPPORTED_CARS:
        choices = ", ".join(SUPPORTED_CARS)
        raise ValueError(f"지원하지 않는 차종입니다. 가능한 값: {choices}")
    if k < 1:
        raise ValueError("검색할 청크 수(k)는 1 이상이어야 합니다.")

    module = _load_vectorstore_module()
    try:
        with _vectorstore_working_directory():
            retriever = module.load_retriever(car=car, k=k)
    except Exception as exc:
        raise RuntimeError(f"{RETRIEVER_LABEL} retriever 로딩 실패: {exc}") from exc

    return retriever


def retrieve_documents(
    question: str,
    car: str,
    k: int = DEFAULT_TOP_K,
    retriever=None,
) -> list[Any]:
    """질문과 차종으로 Chroma 상위 ``k``개 문서를 검색한다."""

    clean_question = question.strip()
    if not clean_question:
        raise ValueError("질문은 비어 있을 수 없습니다.")

    active_retriever = retriever or create_retriever(car=car, k=k)
    if car not in SUPPORTED_CARS:
        choices = ", ".join(SUPPORTED_CARS)
        raise ValueError(f"지원하지 않는 차종입니다. 가능한 값: {choices}")

    search_kwargs = getattr(active_retriever, "search_kwargs", None)
    if not isinstance(search_kwargs, dict):
        raise RuntimeError("Chroma retriever의 검색 조건을 설정할 수 없습니다.")
    search_kwargs["k"] = k
    search_kwargs["filter"] = {"car": car}

    try:
        with _vectorstore_working_directory():
            documents = list(active_retriever.invoke(clean_question))
    except Exception as exc:
        raise RuntimeError(f"문서 검색 실패: {exc}") from exc

    return documents
