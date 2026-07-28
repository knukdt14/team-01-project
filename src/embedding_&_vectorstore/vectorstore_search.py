# 벡터스토어 비교 (FAISS vs Chroma) + 차종 메타데이터 필터
# 임베딩 모델: bge-m3 (실험1에서 정밀도 가장 높게 나와서 선택)
# 최종 선택: Chroma (검색 속도 + 차종 필터가 DB 단에서 걸려서 결과가 항상 k개 채워짐)
# 실행: python vectorstore_search.py

import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import torch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma

# 실행 위치(cwd)와 무관하게 항상 같은 곳을 보도록 스크립트 위치 기준 절대경로로 고정
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

CHUNKS_PATH = PROJECT_ROOT / "output" / "chunks_llmhybrid.json"
EMBED_MODEL = "BAAI/bge-m3"   # 실험1에서 고른 모델 (차원 1024)
TOP_K = 10
CARS = ["avante", "avante_hev", "ioniq6", "nexo", "tucson"]   # 차종 5종
K_SHOW = 10   # 필터 없을 때 차종이 얼마나 섞이는지 볼 상위 개수

# 저장 경로 (bge-m3는 차원이 1024라 기존 768짜리 인덱스와 호환 안 됨 -> 폴더 분리)
FAISS_DIR = SCRIPT_DIR / "faiss_index_bgem3"
CHROMA_DIR = SCRIPT_DIR / "chroma_db_bgem3"
COLLECTION_NAME = "car_manual"   # Chroma 컬렉션 이름 (load 할 때 똑같이 줘야 함)

# GPU 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32   # bge-m3는 모델이 커서 VRAM 8GB 기준 32로 시작 (OOM 안 나면 올려도 됨)

# e5 계열 모델만 query: / passage: 프리픽스가 필요함. bge-m3는 프리픽스 없이 그대로 넣어야 함
USE_PREFIX = "e5" in EMBED_MODEL.lower()

# 테스트할 때만 숫자 넣기 (예: 300). 전체 다 쓰면 None
SAMPLE_SIZE = None

# 필터 효과 보여줄 질문 (타이어 공기압은 5차종에 다 있어서 필터 없으면 섞임)
# 질문에는 일부러 차종을 안 넣음 -> 그래야 필터 없을 때 딴 차 값이 섞이는 게 보임
FILTER_DEMO_Q = "타이어 공기압 얼마로 맞춰야 하나요"
TARGET_CAR = "avante"   # 이 차종 값을 원한다고 가정 (아반떼 아닌 결과 = 틀린 값)

# 검색 속도 비교용 질문들
TEST_QUERIES = [
    "수소 충전 방법",
    "급속충전 시간",
    "엔진오일 교환주기",
    "타이어 공기압",
]


# 모델에 따라 프리픽스를 붙일지 말지만 처리해주는 래퍼
# USE_PREFIX 가 False면(=bge-m3) 원문 그대로 임베딩함
class ManualEmbeddings(HuggingFaceEmbeddings):
    def embed_documents(self, texts):
        if USE_PREFIX:
            texts = ["passage: " + t for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        if USE_PREFIX:
            text = "query: " + text
        return super().embed_query(text)


def make_embeddings():
    # 임베딩 모델을 GPU에 올려서 생성 (FAISS/Chroma 인덱스 자체는 CPU 연산이라 그대로 둠)
    return ManualEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": BATCH_SIZE},
    )


def load_documents():
    data = json.load(open(CHUNKS_PATH, encoding="utf-8"))
    if SAMPLE_SIZE:
        import random
        random.seed(42)
        data = random.sample(data, SAMPLE_SIZE)

    docs = []
    for c in data:
        meta = {
            "car": c["car"],            # 차종 필터에 쓸 값
            "page": c["page"],          # 출처 페이지 표시용
            "chunk_id": c["chunk_id"],
            "needs_review": c["needs_review"],   # 눕힌 표는 True (평가때 따로 볼 수 있게 남겨둠)
        }
        docs.append(Document(page_content=c["text"], metadata=meta))

    review_cnt = sum(1 for c in data if c["needs_review"])
    print("문서 개수: " + str(len(docs)) + " (needs_review 청크: " + str(review_cnt) + "개)")
    return docs


def search(db, query, is_faiss, k, car=None):
    # 필터 유무 + FAISS/Chroma 차이를 여기서 한 번에 처리
    # 필터가 없을 때는 두 쪽 다 똑같은 조건으로 검색해야 속도 비교가 공정함
    if car is None:
        return db.similarity_search(query, k=k)
    # FAISS 는 후보를 먼저 뽑고 나중에 거르는 방식이라 fetch_k 를 크게 줘야 함
    if is_faiss:
        return db.similarity_search(query, k=k, fetch_k=200, filter={"car": car})
    # Chroma 는 DB 단에서 바로 걸러줌 (후보 부족으로 결과가 비는 일이 없음)
    return db.similarity_search(query, k=k, filter={"car": car})


def build_faiss(docs, embeddings):
    t0 = time.time()
    db = FAISS.from_documents(docs, embeddings)
    build_time = time.time() - t0
    db.save_local(str(FAISS_DIR))
    return db, build_time


def build_chroma(docs, embeddings):
    # 같은 폴더에 또 넣으면 문서가 중복으로 쌓이므로 새로 만들기 전에 폴더를 비움
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    t0 = time.time()
    db = Chroma.from_documents(docs, embeddings,
                               collection_name=COLLECTION_NAME,
                               persist_directory=str(CHROMA_DIR))
    build_time = time.time() - t0   # persist_directory 주면 자동 저장됨
    return db, build_time


def load_retriever(car=None, k=TOP_K):
    # 저장된 Chroma 인덱스를 불러와서 retriever 로 돌려줌 (임베딩 다시 안 함)
    # 팀원은 이 함수만 부르면 됨. car 를 주면 그 차종만 검색되게 필터가 걸림
    embeddings = make_embeddings()
    db = Chroma(collection_name=COLLECTION_NAME,
                persist_directory=str(CHROMA_DIR),
                embedding_function=embeddings)
    search_kwargs = {"k": k}
    if car is not None:
        search_kwargs["filter"] = {"car": car}
    return db.as_retriever(search_kwargs=search_kwargs)


def avg_search_time(db, is_faiss):
    # 질문 1건 검색하는 데 걸리는 평균 시간 (FAISS vs Chroma 검색 속도 비교용)
    times = []
    for q in TEST_QUERIES:
        t0 = time.time()
        search(db, q, is_faiss, TOP_K)
        times.append(time.time() - t0)
    return sum(times) / len(times)


def show_filter_effect(db, is_faiss):
    name = "FAISS" if is_faiss else "Chroma"
    print("\n===== 차종 필터 효과 (" + name + ") =====")
    print("질문: " + FILTER_DEMO_Q + "  (아반떼 값을 원한다고 가정)")

    # 1) 필터 없음 -> 상위 결과가 어느 차종에서 나왔는지 (아반떼 아닌 건 틀린 값)
    res = search(db, FILTER_DEMO_Q, is_faiss, 5)
    cars = [r.metadata["car"] for r in res]
    pages = [r.metadata["page"] for r in res]
    print("\n[필터 없음] 상위 5개 결과")
    print("  차종 -> " + str(cars) + "  page=" + str(pages))
    print("  -> 아반떼 말고 딴 차 값이 섞여 나옴 (아반떼 아닌 건 틀린 답)")

    # 2) 필터 있음 -> 차종별로 걸면 그 차종만 나옴
    #    결과 개수도 같이 찍어서 k개가 제대로 채워지는지 확인 (FAISS는 여기서 모자랄 수 있음)
    print("\n[필터 있음] 차종별로 필터 걸고 상위 " + str(TOP_K) + "개")
    for car in CARS:
        res = search(db, FILTER_DEMO_Q, is_faiss, TOP_K, car=car)
        out_cars = [r.metadata["car"] for r in res]
        pages = [r.metadata["page"] for r in res]
        print("  car=" + car.ljust(12) + " 개수=" + str(len(res)) +
              " -> " + str(out_cars) + "  page=" + str(pages))
    print("  -> 필터 건 차종만 정확히 나옴")


if __name__ == "__main__":
    # GPU 확인 (cpu 로 뜨면 torch 가 CPU 빌드라서 재설치 필요)
    print("device = " + DEVICE)
    if DEVICE == "cuda":
        print("GPU = " + torch.cuda.get_device_name(0))
    else:
        print("경고: GPU를 못 찾아서 CPU로 돌아감 (torch CUDA 빌드 확인 필요)")

    print("\n문서 로딩...")
    docs = load_documents()

    print("임베딩 모델 로딩... (" + EMBED_MODEL + ", batch_size=" + str(BATCH_SIZE) + ")")
    embeddings = make_embeddings()

    print("\nFAISS 만드는 중... (비교용)")
    faiss_db, faiss_build = build_faiss(docs, embeddings)
    print("FAISS 생성시간 = " + str(round(faiss_build, 1)) + "초  저장위치=" + str(FAISS_DIR))

    print("\nChroma 만드는 중... (최종 선택)")
    chroma_db, chroma_build = build_chroma(docs, embeddings)
    print("Chroma 생성시간 = " + str(round(chroma_build, 1)) + "초  저장위치=" + str(CHROMA_DIR))

    # 검색 속도 비교 (질문 1건 검색에 걸리는 평균 시간)
    faiss_search = avg_search_time(faiss_db, is_faiss=True)
    chroma_search = avg_search_time(chroma_db, is_faiss=False)

    # 차종 필터 효과 보여주기 (최종 선택인 Chroma 기준으로 시연)
    show_filter_effect(chroma_db, is_faiss=False)

    # 최종 비교 (그래프 대신 터미널 표)
    print("\n===== FAISS vs Chroma 비교 =====")
    print("항목".ljust(16) + "FAISS".ljust(12) + "Chroma")
    print("생성시간(초)".ljust(15) + str(round(faiss_build, 1)).ljust(12) + str(round(chroma_build, 1)))
    print("1회 검색(ms)".ljust(15) + str(round(faiss_search * 1000, 1)).ljust(12) + str(round(chroma_search * 1000, 1)))
    print("\n-> 최종 선택: Chroma (load_retriever 가 Chroma 를 불러오도록 설정됨)")