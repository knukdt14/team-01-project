"""팀원의 Chroma retriever를 프롬프트 엔지니어링 코드에 연결한다.

팀원 파일은 수정하지 않고 ``vectorstore_search.py``를 동적으로 불러온다.
Windows의 한글·OneDrive 경로에서 Chroma HNSW 파일을 열지 못하는 문제를
피하기 위해 원본 DB의 정확한 복사본을 영문 임시 캐시에 두고 검색한다.
"""

from __future__ import annotations

import importlib.util
import hashlib
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

from term_synonyms import normalize_terms

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
DEFAULT_TOP_K = 7
RETRIEVER_LABEL = "Chroma/bge-m3"
FAISS_RETRIEVER_LABEL = "FAISS/bge-m3"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VECTORSTORE_DIR = PROJECT_ROOT / "src" / "embedding_&_vectorstore"
VECTORSTORE_MODULE_PATH = VECTORSTORE_DIR / "vectorstore_search.py"
TEAM_CHROMA_DIR = VECTORSTORE_DIR / "chroma_db_bgem3"
CHROMA_CACHE_ROOT = Path(tempfile.gettempdir()) / "team01_chroma_bgem3_cache"


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


def load_team_vectorstore_module() -> ModuleType:
    """팀원의 문서 로더와 임베딩 설정을 읽기 전용으로 불러온다."""

    return _load_vectorstore_module()


def _directory_sha256(directory: Path) -> str:
    """DB 전체 내용으로 캐시 버전을 구분하는 지문을 만든다."""

    files = sorted(
        (path for path in directory.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    if not files:
        raise RuntimeError(f"팀원의 Chroma DB 파일이 없습니다: {directory}")

    digest = hashlib.sha256()
    for path in files:
        relative_path = path.relative_to(directory).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(path.stat().st_size.to_bytes(8, "little"))
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _prepare_chroma_cache() -> Path:
    """팀원 DB를 내용 기반 버전의 영문 임시 폴더에 안전하게 복사한다."""

    if not (TEAM_CHROMA_DIR / "chroma.sqlite3").is_file():
        raise RuntimeError(f"팀원의 Chroma DB를 찾을 수 없습니다: {TEAM_CHROMA_DIR}")

    signature = _directory_sha256(TEAM_CHROMA_DIR)
    CHROMA_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_dir = CHROMA_CACHE_ROOT / signature[:16]
    ready_file = cache_dir / ".source_sha256"

    if ready_file.is_file():
        cached_signature = ready_file.read_text(encoding="ascii").strip()
        if cached_signature == signature:
            return cache_dir

    if cache_dir.exists():
        raise RuntimeError(
            "완성되지 않은 Chroma 임시 캐시가 있습니다. "
            f"Python을 모두 종료한 뒤 이 폴더를 삭제하세요: {cache_dir}"
        )

    staging_dir = CHROMA_CACHE_ROOT / (
        f".building_{signature[:16]}_{uuid.uuid4().hex}"
    )
    try:
        shutil.copytree(TEAM_CHROMA_DIR, staging_dir)
        (staging_dir / ".source_sha256").write_text(signature, encoding="ascii")
        staging_dir.rename(cache_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

    return cache_dir


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
    """차종 필터가 적용된 팀원 Chroma 복사본을 한 번 로드한다."""

    if car is not None and car not in SUPPORTED_CARS:
        choices = ", ".join(SUPPORTED_CARS)
        raise ValueError(f"지원하지 않는 차종입니다. 가능한 값: {choices}")
    if k < 1:
        raise ValueError("검색할 청크 수(k)는 1 이상이어야 합니다.")

    module = load_team_vectorstore_module()
    try:
        cache_dir = _prepare_chroma_cache()
        embeddings = module.make_embeddings()
        vectorstore = module.Chroma(
            collection_name=module.COLLECTION_NAME,
            persist_directory=str(cache_dir),
            embedding_function=embeddings,
        )
    except Exception as exc:
        raise RuntimeError(f"{RETRIEVER_LABEL} retriever 로딩 실패: {exc}") from exc

    search_kwargs: dict[str, Any] = {"k": k}
    if car is not None:
        search_kwargs["filter"] = {"car": car}
    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def create_faiss_retriever(car: str | None = None, k: int = DEFAULT_TOP_K):
    """최신 FAISS 인덱스를 영문 임시 경로에서 로드한다.

    Windows FAISS는 한글이 포함된 프로젝트 경로에서 인덱스 파일을 열지
    못하므로, 팀원의 인덱스 원본은 수정하지 않고 임시 폴더에 복사한다.
    """

    if car is not None and car not in SUPPORTED_CARS:
        choices = ", ".join(SUPPORTED_CARS)
        raise ValueError(f"지원하지 않는 차종입니다. 가능한 값: {choices}")
    if k < 1:
        raise ValueError("검색할 청크 수(k)는 1 이상이어야 합니다.")

    module = _load_vectorstore_module()
    source_dir = Path(module.FAISS_DIR)
    cache_dir = Path(tempfile.gettempdir()) / "team01_faiss_index_bgem3"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for filename in ("index.faiss", "index.pkl"):
        source = source_dir / filename
        if not source.is_file():
            raise RuntimeError(f"FAISS 인덱스 파일을 찾을 수 없습니다: {source}")
        shutil.copy2(source, cache_dir / filename)

    try:
        embeddings = module.make_embeddings()
        vectorstore = module.FAISS.load_local(
            str(cache_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"{FAISS_RETRIEVER_LABEL} retriever 로딩 실패: {exc}"
        ) from exc

    search_kwargs: dict[str, Any] = {
        "k": k,
        "fetch_k": max(200, k * 50),
    }
    if car is not None:
        search_kwargs["filter"] = {"car": car}
    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def retrieve_documents(
    question: str,
    car: str,
    k: int = DEFAULT_TOP_K,
    retriever=None,
) -> list[Any]:
    """질문과 차종으로 Chroma 상위 ``k``개 문서를 검색한다."""

    clean_question = normalize_terms(question.strip())
    if not clean_question:
        raise ValueError("질문은 비어 있을 수 없습니다.")

    active_retriever = retriever or create_retriever(car=car, k=k)
    if car not in SUPPORTED_CARS:
        choices = ", ".join(SUPPORTED_CARS)
        raise ValueError(f"지원하지 않는 차종입니다. 가능한 값: {choices}")

    search_kwargs = getattr(active_retriever, "search_kwargs", None)
    if not isinstance(search_kwargs, dict):
        raise RuntimeError("벡터 retriever의 검색 조건을 설정할 수 없습니다.")
    search_kwargs["k"] = k
    search_kwargs["filter"] = {"car": car}

    try:
        with _vectorstore_working_directory():
            documents = list(active_retriever.invoke(clean_question))
    except Exception as exc:
        raise RuntimeError(f"문서 검색 실패: {exc}") from exc

    return documents
