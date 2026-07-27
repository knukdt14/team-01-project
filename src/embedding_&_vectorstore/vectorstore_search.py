# 벡터스토어 비교 (FAISS vs Chroma) + 차종 메타데이터 필터
# 임베딩 모델은 실험1에서 고른 걸로 하나만 씀 (아래 EMBED_MODEL 에서 바꾸기)
# 실행: python vectorstore_search.py

import json
import time
import matplotlib.pyplot as plt

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma

CHUNKS_PATH = "chunks_hybrid.json"
EMBED_MODEL = "BAAI/bge-m3"   # 실험1에서 고른 모델로 바꾸기 (e5 쓸거면 프리픽스 필요, 실험1 코드 참고)
TOP_K = 3

# 테스트할 때만 숫자 넣기 (예: 300). 전체 다 쓰면 None
SAMPLE_SIZE = None

# 필터가 왜 필요한지 보여줄 질문 (타이어 공기압은 5차종에 다 들어있어서 필터 없으면 섞임)
FILTER_DEMO_Q = "타이어 공기압 얼마로 맞춰야 하나요"
FILTER_CAR = "avante"

# 검색 속도 비교용 질문들
TEST_QUERIES = [
    "수소 충전 방법",
    "급속충전 시간",
    "엔진오일 교환주기",
    "타이어 공기압",
]


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
    times = []
    for q in TEST_QUERIES:
        t0 = time.time()
        if is_faiss:
            db.similarity_search(q, k=TOP_K, fetch_k=200)
        else:
            db.similarity_search(q, k=TOP_K)
        times.append(time.time() - t0)
    return sum(times) / len(times)


def show_filter_effect(db, is_faiss):
    # 필터 없이 검색
    print("\n[필터 없음] 질문: " + FILTER_DEMO_Q)
    if is_faiss:
        res = db.similarity_search(FILTER_DEMO_Q, k=TOP_K, fetch_k=200)
    else:
        res = db.similarity_search(FILTER_DEMO_Q, k=TOP_K)
    for r in res:
        print("  차종=" + r.metadata["car"] + " page=" + str(r.metadata["page"]))

    # car 필터 걸고 검색 -> 해당 차종만 나옴
    print("[필터 car=" + FILTER_CAR + "] 질문: " + FILTER_DEMO_Q)
    if is_faiss:
        res = db.similarity_search(FILTER_DEMO_Q, k=TOP_K, fetch_k=200, filter={"car": FILTER_CAR})
    else:
        res = db.similarity_search(FILTER_DEMO_Q, k=TOP_K, filter={"car": FILTER_CAR})
    for r in res:
        print("  차종=" + r.metadata["car"] + " page=" + str(r.metadata["page"]))


def draw_chart(faiss_build, chroma_build, faiss_search, chroma_search):
    import matplotlib.font_manager as fm
    # 한글 폰트 설정 (윈도우 맑은고딕 / 맥 애플고딕 / 리눅스 나눔고딕 중 있는거 사용)
    have = [f.name for f in fm.fontManager.ttflist]
    for f in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
        if f in have:
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    b1 = ax[0].bar(["FAISS", "Chroma"], [faiss_build, chroma_build], color=["tab:blue", "tab:orange"])
    ax[0].set_title("생성 시간(초)")
    ax[0].bar_label(b1, fmt="%.1f")
    b2 = ax[1].bar(["FAISS", "Chroma"], [faiss_search * 1000, chroma_search * 1000], color=["tab:blue", "tab:orange"])
    ax[1].set_title("검색 평균시간(ms)")
    ax[1].bar_label(b2, fmt="%.1f")
    plt.tight_layout()
    plt.savefig("vectorstore_compare.png", dpi=120)
    print("\n그래프 저장: vectorstore_compare.png")


if __name__ == "__main__":
    print("문서 로딩...")
    docs = load_documents()

    print("임베딩 모델 로딩... (" + EMBED_MODEL + ")")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL,
                                       encode_kwargs={"normalize_embeddings": True})

    print("\nFAISS 만드는 중...")
    faiss_db, faiss_build = build_faiss(docs, embeddings)
    print("FAISS 생성시간 = " + str(round(faiss_build, 1)) + "초")

    print("\nChroma 만드는 중...")
    chroma_db, chroma_build = build_chroma(docs, embeddings)
    print("Chroma 생성시간 = " + str(round(chroma_build, 1)) + "초")

    # 검색 속도 비교
    faiss_search = avg_search_time(faiss_db, is_faiss=True)
    chroma_search = avg_search_time(chroma_db, is_faiss=False)
    print("\n검색 평균시간  FAISS=" + str(round(faiss_search * 1000, 1)) + "ms  Chroma=" + str(round(chroma_search * 1000, 1)) + "ms")

    # 차종 필터 효과 보여주기 (FAISS 기준으로 시연)
    show_filter_effect(faiss_db, is_faiss=True)

    # 그래프
    draw_chart(faiss_build, chroma_build, faiss_search, chroma_search)
