"""
build_vectorstore.py — [3] 임베딩 + [4] 벡터스토어  ─ 담당 C

청크를 임베딩하여 FAISS/Chroma 인덱스로 저장한다.
설정(tag)별로 인덱스를 따로 저장하므로, 임베딩 모델이나 청킹을 바꾸면
자동으로 별도 인덱스가 만들어져 실험 간 간섭이 없다.

실행:  python src/build_vectorstore.py
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import BASELINE, INDEX_DIR, Config
from load_pdf import build_corpus


# ──────────────────────────── 임베딩 ────────────────────────────
def get_embeddings(cfg: Config = BASELINE) -> Embeddings:
    """임베딩 모델 로드.

    TODO(C): 비교 대상
      - BAAI/bge-m3                        (기본, 다국어)
      - nlpai-lab/KURE-v1                  (한국어 특화)
      - jhgan/ko-sroberta-multitask        (한국어 SBERT)
      - intfloat/multilingual-e5-large     (다국어)
      - OpenAI text-embedding-3-small      (상용 비교군)
    """
    name = cfg.embedding_model

    if name.startswith("text-embedding"):
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=name)

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=name,
        encode_kwargs={"normalize_embeddings": cfg.similarity == "cosine"},
    )


# ──────────────────────────── 벡터스토어 ────────────────────────────
def index_path(cfg: Config) -> Path:
    return INDEX_DIR / cfg.tag()


def build(cfg: Config = BASELINE, force: bool = False):
    """청크 → 임베딩 → 인덱스 저장."""
    path = index_path(cfg)
    if path.exists() and not force:
        print(f"[건너뜀] 인덱스가 이미 있습니다: {path.name}")
        print("         다시 만들려면 force=True 로 실행하세요.")
        return load(cfg)

    chunks: list[Document] = build_corpus(cfg)
    embeddings = get_embeddings(cfg)

    print(f"[3][4] 임베딩·인덱싱 (model={cfg.embedding_model}, store={cfg.vectorstore})")
    print(f"       청크 {len(chunks):,}개 — 시간이 걸립니다…")

    if cfg.vectorstore == "faiss":
        from langchain_community.vectorstores import FAISS

        store = FAISS.from_documents(chunks, embeddings)
        path.mkdir(parents=True, exist_ok=True)
        store.save_local(str(path))

    elif cfg.vectorstore == "chroma":
        from langchain_chroma import Chroma

        store = Chroma.from_documents(chunks, embeddings, persist_directory=str(path))

    else:
        raise ValueError(f"알 수 없는 벡터스토어: {cfg.vectorstore}")

    print(f"       저장 완료 → {path}")
    return store


def load(cfg: Config = BASELINE):
    """저장된 인덱스 로드."""
    path = index_path(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"인덱스가 없습니다: {path}\n먼저 build()를 실행하세요."
        )
    embeddings = get_embeddings(cfg)

    if cfg.vectorstore == "faiss":
        from langchain_community.vectorstores import FAISS

        return FAISS.load_local(
            str(path), embeddings, allow_dangerous_deserialization=True
        )

    from langchain_chroma import Chroma

    return Chroma(persist_directory=str(path), embedding_function=embeddings)


if __name__ == "__main__":
    import io
    import sys

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    build(BASELINE)
