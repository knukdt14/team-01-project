# 벡터스토어 비교 (FAISS vs Chroma) + 차종 메타데이터 필터
# 임베딩 모델: multilingual-e5 (실험1에서 선택)
# 실행: python vectorstore_search.py

import json
import time
from collections import Counter

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma

CHUNKS_PATH = "../../output/chunks_hybrid.json"
EMBED_MODEL = "intfloat/multilingual-e5-base"   # 실험1에서 고른 모델
TOP_K = 3
CARS = ["avante", "avante_hev", "ioniq6", "nexo", "tucson"]   # 차종 5종
K_SHOW = 10   # 필터 없을 때 차종이 얼마나 섞이는지 볼 상위 개수

# 테스트할 때만 숫자 넣기 (예: 300). 전체 다 쓰면 None
SAMPLE_SIZE = 300

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


# e5 모델은 문서 앞에 passage:, 질문 앞에 query: 를 붙여야 제 성능이 나옴 (e5 공식 사용법)
# HuggingFaceEmbeddings 를 상속받아서 프리픽스만 자동으로 붙여주게 만듦
class E5Embeddings(HuggingFaceEmbeddings):
    def embed_documents(self, texts):
        texts = ["passage: " + t for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        return super().embed_query("query: " + text)


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
    # FAISS 는 필터 쓸 때 후보를 넉넉히 뽑아야 해서 fetch_k 를 크게 줌
    if car is None:
        if is_faiss:
            return db.similarity_search(query, k=k, fetch_k=200)
        return db.similarity_search(query, k=k)
    if is_faiss:
        return db.similarity_search(query, k=k, fetch_k=200, filter={"car": car})
    return db.similarity_search(query, k=k, filter={"car": car})


def build_faiss(docs, embeddings):
    t0 = time.time()
    db = FAISS.from_documents(docs, embeddings)
    build_time = time.time() - t0
    db.save_local("faiss_index")   # 팀원이 FAISS.load_local 로 불러쓸 수 있게 저장
    return db, build_time


def build_chroma(docs, embeddings):
    t0 = time.time()
    db = Chroma.from_documents(docs, embeddings, persist_directory="chroma_db")
    build_time = time.time() - t0   # persist_directory 주면 자동 저장됨
    return db, build_time


def avg_search_time(db, is_faiss):
    # 질문 1건 검색하는 데 걸리는 평균 시간 (FAISS vs Chroma 검색 속도 비교용)
    times = []
    for q in TEST_QUERIES:
        t0 = time.time()
        search(db, q, is_faiss, TOP_K)
        times.append(time.time() - t0)
    return sum(times) / len(times)


def show_filter_effect(db, is_faiss):
    print("\n===== 차종 필터 효과 =====")
    print("질문: " + FILTER_DEMO_Q + "  (아반떼 값을 원한다고 가정)")

    # 1) 필터 없음 -> 상위 결과가 어느 차종에서 나왔는지 (아반떼 아닌 건 틀린 값)
    res = search(db, FILTER_DEMO_Q, is_faiss, 5)
    cars = [r.metadata["car"] for r in res]
    pages = [r.metadata["page"] for r in res]
    print("\n[필터 없음] 상위 5개 결과")
    print("  차종 -> " + str(cars) + "  page=" + str(pages))
    print("  -> 아반떼 말고 딴 차 값이 섞여 나옴 (아반떼 아닌 건 틀린 답)")

    # 2) 필터 있음 -> 차종별로 걸면 그 차종만 나옴
    print("\n[필터 있음] 차종별로 필터 걸고 상위 " + str(TOP_K) + "개")
    for car in CARS:
        res = search(db, FILTER_DEMO_Q, is_faiss, TOP_K, car=car)
        out_cars = [r.metadata["car"] for r in res]
        pages = [r.metadata["page"] for r in res]
        print("  car=" + car.ljust(12) + " -> " + str(out_cars) + "  page=" + str(pages))
    print("  -> 필터 건 차종만 정확히 나옴")


if __name__ == "__main__":
    print("문서 로딩...")
    docs = load_documents()

    print("임베딩 모델 로딩... (" + EMBED_MODEL + ")")
    embeddings = E5Embeddings(model_name=EMBED_MODEL,
                              encode_kwargs={"normalize_embeddings": True})

    print("\nFAISS 만드는 중...")
    faiss_db, faiss_build = build_faiss(docs, embeddings)
    print("FAISS 생성시간 = " + str(round(faiss_build, 1)) + "초")

    print("\nChroma 만드는 중...")
    chroma_db, chroma_build = build_chroma(docs, embeddings)
    print("Chroma 생성시간 = " + str(round(chroma_build, 1)) + "초")

    # 검색 속도 비교 (질문 1건 검색에 걸리는 평균 시간)
    faiss_search = avg_search_time(faiss_db, is_faiss=True)
    chroma_search = avg_search_time(chroma_db, is_faiss=False)

    # 차종 필터 효과 보여주기 (FAISS 기준으로 시연)
    show_filter_effect(faiss_db, is_faiss=True)

    # 최종 비교 (그래프 대신 터미널 표)
    print("\n===== FAISS vs Chroma 비교 =====")
    print("항목".ljust(16) + "FAISS".ljust(12) + "Chroma")
    print("생성시간(초)".ljust(15) + str(round(faiss_build, 1)).ljust(12) + str(round(chroma_build, 1)))
    print("1회 검색(ms)".ljust(15) + str(round(faiss_search * 1000, 1)).ljust(12) + str(round(chroma_search * 1000, 1)))