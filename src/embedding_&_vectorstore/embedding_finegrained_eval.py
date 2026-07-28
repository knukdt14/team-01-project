# 세부 항목(와이퍼/엔진오일/타이어공기압 등) 단위로 bge-m3 dense 임베딩이
# "같은 차종 안에서" top10 안에 정답 청크를 제대로 찾아내는지 검증
#
# 차종 필터는 이미 완벽하다고 가정 -> 그 변수는 고정하고, 그 안에서
# dense 임베딩 자체의 성능만 따로 떼어서 봄
#
# 정답(ground truth) 정의: 사람이 라벨링하지 않고, 그 차종 청크들 중
# 키워드가 실제로 텍스트에 들어있는 청크를 정답으로 간주 (객관적/재현 가능)
#
# 질문은 일부러 키워드와 다른 표현으로 씀 -> "표현이 달라도 의미로 잘 찾는지"를 봄
#
# 모델: FlagEmbedding의 BGEM3FlagModel (SentenceTransformer 아님, dense만 사용)
#   pip install FlagEmbedding 필요
#
# 실행: python embedding_finegrained_eval.py

import json
import numpy as np
import torch
import transformers.modeling_utils as _hf_modeling_utils
from FlagEmbedding import BGEM3FlagModel

# FlagEmbedding 1.4.0이 AutoModel.from_pretrained(..., dtype=...)로 호출하는데,
# 설치된 transformers(4.x)는 이 이름을 모르고 그대로 모델 생성자에 넘겨버려서
# "unexpected keyword argument 'dtype'" 에러가 남 -> torch_dtype으로 바꿔서 흡수
_orig_from_pretrained = _hf_modeling_utils.PreTrainedModel.from_pretrained.__func__


def _from_pretrained_dtype_compat(cls, *args, **kwargs):
    if "dtype" in kwargs:
        kwargs.setdefault("torch_dtype", kwargs.pop("dtype"))
    return _orig_from_pretrained(cls, *args, **kwargs)


_hf_modeling_utils.PreTrainedModel.from_pretrained = classmethod(_from_pretrained_dtype_compat)

CHUNKS_PATH = "../../output/chunks_llmhybrid.json"
TOP_K = 10
CANDIDATE_K = 50  # 1차 dense로 이만큼 후보를 뽑은 뒤 cross-encoder로 재정렬
MAX_LEN = 512   # 512 토큰이면 충분

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# (차종, 항목명, 정답 판정 키워드, 실제로 던질 질문)
# 항목명=키워드로 같게 둬도 되지만, 정답 기준(키워드)과 질문 표현을 분리해서 보여주려고 나눠둠
TEST_ITEMS = [
    # 공통 소모품 (내연기관/전동화 관계없이 5차종 다 있음)
    ("avante",     "와이퍼",       "와이퍼",     "빗물이 안 닦일 때 이 부품 어떻게 교체하나요"),
    ("avante_hev", "와이퍼",       "와이퍼",     "빗물이 안 닦일 때 이 부품 어떻게 교체하나요"),
    ("ioniq6",     "와이퍼",       "와이퍼",     "빗물이 안 닦일 때 이 부품 어떻게 교체하나요"),
    ("nexo",       "와이퍼",       "와이퍼",     "빗물이 안 닦일 때 이 부품 어떻게 교체하나요"),
    ("tucson",     "와이퍼",       "와이퍼",     "빗물이 안 닦일 때 이 부품 어떻게 교체하나요"),

    ("avante",     "타이어 공기압", "타이어 공기압", "바퀴 바람을 얼마나 넣어야 하나요"),
    ("avante_hev", "타이어 공기압", "타이어 공기압", "바퀴 바람을 얼마나 넣어야 하나요"),
    ("ioniq6",     "타이어 공기압", "타이어 공기압", "바퀴 바람을 얼마나 넣어야 하나요"),
    ("nexo",       "타이어 공기압", "타이어 공기압", "바퀴 바람을 얼마나 넣어야 하나요"),
    ("tucson",     "타이어 공기압", "타이어 공기압", "바퀴 바람을 얼마나 넣어야 하나요"),

    ("avante",     "브레이크 패드", "브레이크 패드", "제동장치가 마모되면 어떻게 하나요"),
    ("avante_hev", "브레이크 패드", "브레이크 패드", "제동장치가 마모되면 어떻게 하나요"),
    ("ioniq6",     "브레이크 패드", "브레이크 패드", "제동장치가 마모되면 어떻게 하나요"),
    ("nexo",       "브레이크 패드", "브레이크 패드", "제동장치가 마모되면 어떻게 하나요"),
    ("tucson",     "브레이크 패드", "브레이크 패드", "제동장치가 마모되면 어떻게 하나요"),

    # 내연기관 전용 (avante / avante_hev / tucson)
    ("avante",     "엔진오일",     "엔진오일",   "엔진 기름을 몇 킬로마다 갈아야 하나요"),
    ("avante_hev", "엔진오일",     "엔진오일",   "엔진 기름을 몇 킬로마다 갈아야 하나요"),
    ("tucson",     "엔진오일",     "엔진오일",   "엔진 기름을 몇 킬로마다 갈아야 하나요"),
    ("avante",     "점화플러그",   "점화플러그", "시동이 잘 안 걸릴 때 점검할 부품"),
    ("tucson",     "점화플러그",   "점화플러그", "시동이 잘 안 걸릴 때 점검할 부품"),

    # 전동화 전용
    ("ioniq6",     "완속충전",     "완속충전",   "집에서 천천히 충전하는 방법"),
    ("ioniq6",     "급속충전",     "급속충전",   "고속도로 휴게소에서 빠르게 충전하는 방법"),
    ("avante_hev", "회생제동",     "회생제동",   "브레이크 밟을 때 배터리가 충전되는 기능"),
    ("nexo",       "수소 충전",    "수소 충전",  "수소를 넣는 방법"),
    ("nexo",       "연료전지",     "연료전지",   "수소로 전기를 만드는 장치 점검 방법"),
]


def load_chunks():
    data = json.load(open(CHUNKS_PATH, encoding="utf-8"))
    texts = [c["text"] for c in data]
    cars = np.array([c["car"] for c in data])
    pages = [c["page"] for c in data]
    return texts, cars, pages


def build_model():
    print("BGEM3FlagModel 로딩... (device=" + DEVICE + ")")
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=(DEVICE == "cuda"))


def build_reranker():
    from FlagEmbedding import FlagReranker
    print("bge-reranker-v2-m3 로딩... (device=" + DEVICE + ")")
    return FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=(DEVICE == "cuda"))


def encode_dense(model, texts, batch_size=12):
    # dense_vecs 는 이미 정규화되어 나옴 -> 내적이 곧 코사인 유사도
    out = model.encode(
        texts,
        batch_size=batch_size,
        max_length=MAX_LEN,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    return np.asarray(out["dense_vecs"])


def evaluate_item(car, item_label, keyword, query, texts, cars, pages, doc_dense, model,
                   reranker=None, candidate_k=CANDIDATE_K):
    idx_in_car = np.where(cars == car)[0]
    car_texts = [texts[i] for i in idx_in_car]

    # 정답 청크 = 그 차종 청크 중 키워드가 실제로 들어있는 청크 (local index 기준)
    gt_local = [i for i, t in enumerate(car_texts) if keyword in t]
    if not gt_local:
        print(f"  [경고] {car}/{item_label}: 키워드 '{keyword}' 포함 청크가 아예 없음 -> 스킵")
        return None

    q_dense = encode_dense(model, [query])[0]
    car_dense = doc_dense[idx_in_car]           # 차종 필터 시뮬레이션 (car_dense만 비교)
    sims = car_dense @ q_dense
    order = np.argsort(-sims)                    # local index, dense 유사도 내림차순

    rank_of_local = {local_idx: rank for rank, local_idx in enumerate(order, start=1)}
    best_rank_dense = min(rank_of_local[i] for i in gt_local)
    hit10_dense = best_rank_dense <= TOP_K
    top10_pages_dense = [pages[idx_in_car[i]] for i in order[:TOP_K]]

    result = {
        "car": car,
        "item": item_label,
        "query": query,
        "gt_chunks": len(gt_local),
        "best_rank_dense": best_rank_dense,
        "hit10_dense": hit10_dense,
        "top10_pages_dense": top10_pages_dense,
    }

    if reranker is not None:
        # 1차: dense로 top-candidate_k 후보만 추리고, 그 안에서만 cross-encoder로 재정렬
        # (전체 청크에 cross-encoder를 돌리면 너무 느려서 후보 압축이 필수)
        cand_local = order[:candidate_k]
        pairs = [[query, car_texts[i]] for i in cand_local]
        rerank_scores = np.asarray(reranker.compute_score(pairs, normalize=True))
        rerank_order = np.argsort(-rerank_scores)                # 후보 내부 순위(0-base)
        reranked_local = [cand_local[i] for i in rerank_order]   # 재정렬된 local index 목록

        rerank_rank_of_local = {local_idx: rank for rank, local_idx in enumerate(reranked_local, start=1)}
        gt_in_candidates = [i for i in gt_local if i in rerank_rank_of_local]
        # 1차 후보(top candidate_k)에 정답이 아예 없으면 rerank로도 못 살림 -> None
        best_rank_rerank = min(rerank_rank_of_local[i] for i in gt_in_candidates) if gt_in_candidates else None
        hit10_rerank = best_rank_rerank is not None and best_rank_rerank <= TOP_K
        top10_pages_rerank = [pages[idx_in_car[i]] for i in reranked_local[:TOP_K]]

        result.update({
            "best_rank_rerank": best_rank_rerank,
            "hit10_rerank": hit10_rerank,
            "top10_pages_rerank": top10_pages_rerank,
        })

    return result


def main():
    print("청크 로딩...")
    texts, cars, pages = load_chunks()
    print(f"청크 개수: {len(texts)}")

    model = build_model()
    reranker = build_reranker()

    print(f"전체 청크 dense 임베딩 중... ({len(texts)}개, max_length={MAX_LEN})")
    doc_dense = encode_dense(model, texts, batch_size=12)

    results = []
    print("\n===== 세부 항목별 top10 검증 (dense 단독 vs dense→rerank) =====")
    for car, item_label, keyword, query in TEST_ITEMS:
        r = evaluate_item(car, item_label, keyword, query, texts, cars, pages, doc_dense, model,
                           reranker=reranker)
        if r is None:
            continue
        results.append(r)
        mark_d = "O" if r["hit10_dense"] else "X"
        mark_r = "O" if r["hit10_rerank"] else "X"
        print(f"[dense {mark_d} / rerank {mark_r}] {r['car']:12s} {r['item']:10s} 정답청크수={r['gt_chunks']:3d}  "
              f"dense순위={r['best_rank_dense']:4d}  rerank순위={r['best_rank_rerank']}  질문='{r['query']}'")
        if not r["hit10_rerank"]:
            print(f"      -> rerank top10 페이지={r['top10_pages_rerank']}")

    if not results:
        print("검증할 항목이 없음 (키워드가 데이터에 하나도 없음)")
        return

    hit_dense = sum(1 for r in results if r["hit10_dense"])
    hit_rerank = sum(1 for r in results if r["hit10_rerank"])
    n = len(results)

    print("\n===== 요약 =====")
    print(f"dense 단독      : {n}개 중 적중 {hit_dense}개  recall@10 = {round(hit_dense / n, 3)}")
    print(f"dense→rerank    : {n}개 중 적중 {hit_rerank}개  recall@10 = {round(hit_rerank / n, 3)}")

    misses = [r for r in results if not r["hit10_rerank"]]
    if misses:
        print("\n미적중 항목 (rerank까지 거쳐도 top10 밖):")
        for r in misses:
            print(f"  {r['car']:12s} {r['item']:10s} dense순위={r['best_rank_dense']}  rerank순위={r['best_rank_rerank']}")


if __name__ == "__main__":
    main()
