# 임베딩 모델 3개 비교 (ko-sbert / bge-m3 / multilingual-e5)
# 목적: 어떤 임베딩 모델이 차종별 문서를 잘 구분해서 검색하는지 보고 하나 고르기
# 실행: python embedding_compare.py

import os
import json
import time
import gc
import glob
import random
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "../../output/chunks_llmhybrid.json"   # 한이가 넘겨준 파일
RESULT_DIR = "result_embedding"      # 모델별 결과 저장 폴더
TOP_K = 10                           # 정밀도 볼 때 위에서 몇 개까지 볼지

# 테스트할 때만 숫자 넣기 (예: 300). 전체 다 쓰면 None
SAMPLE_SIZE = None

# 검색 테스트용 질문 + 기대 차종 (기대 차종은 실제 데이터에서 키워드 분포 확인한 값)
# 차종: avante / avante_hev / ioniq6(전기) / nexo(수소) / tucson
TEST_QUERIES = [
    # 수소차(nexo)에만 있는 내용
    ("수소 충전은 어떻게 하나요", ["nexo"]),
    ("연료전지 시스템 점검 방법", ["nexo"]),
    ("수소 탱크 용량이 얼마인가요", ["nexo"]),
    ("수소가 누출되면 어떻게 하나요", ["nexo"]),
    # 전기차(ioniq6)에만 있는 내용
    ("전기차 급속충전 시간이 얼마나 걸려요", ["ioniq6"]),
    ("완속충전은 몇 시간 걸리나요", ["ioniq6"]),
    # 전동화 차량(전기·하이브리드·수소) 공통
    ("회생제동은 어떻게 작동하나요", ["ioniq6", "avante_hev", "nexo"]),
    ("고전압 배터리 관리 방법", ["ioniq6", "avante_hev", "nexo"]),
    ("V2L 기능 사용법", ["ioniq6", "nexo"]),
    ("충전소에서 충전하는 방법", ["ioniq6", "nexo"]),
    # 하이브리드(avante_hev)
    ("하이브리드 시스템은 어떻게 작동하나요", ["avante_hev"]),
    # 엔진 있는 가솔린 차(avante / avante_hev / tucson)
    ("엔진오일 교환주기 알려줘", ["avante", "avante_hev", "tucson"]),
    ("점화플러그 교체 시기", ["avante", "avante_hev", "tucson"]),
    ("주유구 여는 방법", ["avante", "avante_hev", "tucson"]),
    ("연료 필터 점검 방법", ["avante", "avante_hev", "tucson"]),
]


def load_chunks():
    data = json.load(open(CHUNKS_PATH, encoding="utf-8"))
    if SAMPLE_SIZE:
        random.seed(42)
        data = random.sample(data, SAMPLE_SIZE)
    texts = [c["text"] for c in data]
    cars = [c["car"] for c in data]
    return texts, cars


def add_prefix(texts, is_e5, kind):
    # e5 모델만 문장 앞에 query: / passage: 를 붙여야 함 (e5 공식 사용법)
    if not is_e5:
        return texts
    tag = "query: " if kind == "query" else "passage: "
    return [tag + t for t in texts]


def run_one_model(name, model_path, texts, cars, is_e5=False):
    print("\n===== " + name + " 시작 =====")

    # 1. 모델 로딩 시간
    t0 = time.time()
    model = SentenceTransformer(model_path)
    load_time = time.time() - t0
    print("로딩 완료  로딩시간=" + str(round(load_time, 1)) + "초")

    # 2. 전체 청크 임베딩 시간
    doc_input = add_prefix(texts, is_e5, "passage")
    t0 = time.time()
    doc_emb = model.encode(doc_input, batch_size=64,
                           normalize_embeddings=True, show_progress_bar=True)
    embed_time = time.time() - t0
    dim = doc_emb.shape[1]   # 임베딩 벡터 길이가 곧 차원
    print("임베딩 완료  차원=" + str(dim) + "  청크수=" + str(len(texts)) + "  임베딩시간=" + str(round(embed_time, 1)) + "초")

    # 3. 검색 테스트
    #  top1 정확도  : 1등이 기대 차종에서 나왔나 (다 잘하면 1.0 근처라 변별력 약함)
    #  상위 K 정밀도 : 상위 K개 중 기대 차종 비율 (딴 차종 섞이는 정도 -> 변별력 있음)
    #  질문 처리시간 : 질문 1개 임베딩하는 평균 시간 (서비스 응답속도랑 연결)
    car_arr = np.array(cars)
    top1_hit = 0
    precisions = []
    query_times = []
    for q, expected in TEST_QUERIES:
        t0 = time.time()
        q_emb = model.encode(add_prefix([q], is_e5, "query"), normalize_embeddings=True)
        query_times.append(time.time() - t0)

        # 정규화했으니 내적이 곧 코사인 유사도
        sims = (q_emb @ doc_emb.T)[0]
        top_idx = np.argsort(-sims)[:TOP_K]
        top_cars = car_arr[top_idx].tolist()

        if top_cars[0] in expected:
            top1_hit += 1
        correct = sum(1 for c in top_cars if c in expected)
        precisions.append(correct / TOP_K)
        print("  질문: " + q)
        print("    1등=" + top_cars[0] + "  기대=" + str(expected) + "  상위" + str(TOP_K) + " 정밀도=" + str(round(correct / TOP_K, 2)))

    top1_acc = top1_hit / len(TEST_QUERIES)
    mean_precision = sum(precisions) / len(precisions)
    mean_query_ms = (sum(query_times) / len(query_times)) * 1000
    print("top1 정확도=" + str(round(top1_acc, 2)) +
          "  기대차종 정밀도(상위" + str(TOP_K) + ")=" + str(round(mean_precision, 2)) +
          "  평균 질문처리=" + str(round(mean_query_ms, 1)) + "ms")

    # 4. 결과 저장
    result = {
        "name": name,
        "dim": dim,
        "load_time": round(load_time, 2),
        "embed_time": round(embed_time, 2),
        "query_ms": round(mean_query_ms, 2),
        "top1_acc": round(top1_acc, 3),
        "precision": round(mean_precision, 3),
        "num_chunks": len(texts),
    }
    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = os.path.join(RESULT_DIR, name + ".json")
    json.dump(result, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("결과 저장: " + out_path)

    # 5. 다음 모델 로딩 전에 메모리 정리
    del model, doc_emb
    gc.collect()
    return result


def print_summary():
    # 저장된 모델별 결과를 모아서 마지막에 표로 한 번에 보여주기
    files = sorted(glob.glob(os.path.join(RESULT_DIR, "*.json")))
    results = [json.load(open(f, encoding="utf-8")) for f in files]
    if not results:
        print("저장된 결과가 없음")
        return

    print("\n===== 최종 비교 =====")
    # 제목 줄
    print("모델".ljust(16) + "차원".ljust(8) + "임베딩(초)".ljust(12) +
          "질문(ms)".ljust(12) + "top1".ljust(8) + "정밀도")
    for r in results:
        print(r["name"].ljust(16) +
              str(r["dim"]).ljust(8) +
              str(r["embed_time"]).ljust(12) +
              str(r["query_ms"]).ljust(12) +
              str(r["top1_acc"]).ljust(8) +
              str(r["precision"]))


if __name__ == "__main__":
    texts, cars = load_chunks()
    print("불러온 청크 수: " + str(len(texts)))

    # 메모리가 부족하면 아래 3줄 중 2개를 주석 처리하고 하나씩 실행해도 됨
    # 결과가 파일로 저장되니까 마지막 요약(print_summary)은 돌린 모델만 모아서 보여줌
    # run_one_model("ko-sbert", "jhgan/ko-sbert-nli", texts, cars)
    # run_one_model("bge-m3", "BAAI/bge-m3", texts, cars)
    run_one_model("multilingual-e5", "intfloat/multilingual-e5-base", texts, cars, is_e5=True)

    print_summary()