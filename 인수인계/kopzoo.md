# 임베딩·벡터스토어 → LLM 인수인계 (용주 → 다음 담당자)

## 1. 한 줄 요약
`chunks_llmhybrid.json`을 multilingual-e5로 임베딩해서 **FAISS 벡터스토어**를 만들었음.
차종(car) 메타데이터 필터까지 붙어 있어서, 저장된 인덱스(`faiss_index/`)를 불러다
`load_retriever()`로 쓰면 됨. **임베딩 다시 안 돌려도 됨.**

- 임베딩 모델: ko-sbert / bge-m3 / multilingual-e5 셋을 비교(`embedding_compare.py`)한 뒤 **e5 채택**
- 벡터스토어: FAISS vs Chroma 비교한 뒤 **FAISS 채택**

---

## 2. 넘기는 것

| 파일 / 폴더 | 내용 |
|---|---|
| `vectorstore_search.py` | 빌드 + 검색 + 필터 스크립트. `E5Embeddings` 클래스랑 `load_retriever()` 함수가 들어있음 |
| `faiss_index/` | 저장된 FAISS 인덱스. 이걸 불러쓰면 재빌드 안 해도 됨 |
| `chunks_llmhybrid.json` | 임베딩에 쓴 원본 청크 데이터 (인덱스 다시 빌드할 때만 필요) |
| `embedding_compare.py` | 임베딩 모델 3개 비교 코드 (참고용, 안 돌려도 됨) |

- 임베딩 모델: `intfloat/multilingual-e5-base` — 단, 반드시 **`E5Embeddings` 클래스로 감싸서** 씀 (5번 주의 참고)
- 벡터스토어: FAISS

---

## 3. 설치

```
pip install langchain langchain-community langchain-huggingface faiss-cpu sentence-transformers
```

- 처음 실행할 때 e5 모델(약 1.1GB)이 자동으로 받아짐. 인터넷 필요.
- (Chroma는 안 쓰지만 `vectorstore_search.py` 맨 위에서 import 함. 그냥 두려면 `pip install langchain-chroma chromadb`도 같이 깔고, 아예 안 쓸 거면 그 import 줄이랑 `build_chroma` 부분만 지우면 됨)

---

## 4. 벡터스토어 얻는 법 (둘 중 하나)

### A. `faiss_index/` 가 git에 같이 올라온 경우 (제일 편함)
`git pull` 하면 인덱스가 딸려옴. **아무것도 빌드 안 해도 됨.** 바로 5번으로.

### B. 인덱스가 없는 경우 (직접 빌드)
```
python vectorstore_search.py
```
- `chunks_llmhybrid.json`을 이 스크립트랑 같은 폴더에 두고 실행
- 청크를 e5로 임베딩해서 `faiss_index/` 폴더가 생김 (CPU면 30분 안팎, GPU면 훨씬 빠름)
- 다 돌면 차종 필터 시연이랑 FAISS/Chroma 비교표가 터미널에 찍힘
- 한 번 빌드하면 그 뒤론 A처럼 불러쓰기만 하면 됨

> 참고: 인덱스 폴더가 좀 큼(수십 MB). git 용량 부담되면 `.gitignore`에 넣고 각자 B로 빌드하거나, 인덱스를 구글드라이브로 공유해도 됨.

---

## 5. 쓰는 법 (핵심)

`load_retriever()`만 부르면 됨. 다시 빌드 안 함.

```python
from vectorstore_search import load_retriever

# car 를 주면 그 차종만 검색됨 (필터). 안 주면 전체 검색.
retriever = load_retriever(car="avante")

docs = retriever.invoke("타이어 공기압 얼마로 맞춰야 하나요")
for d in docs:
    print(d.metadata["car"], d.metadata["page"])
    print(d.page_content)
```

이 `retriever`를 LangChain 체인(RetrievalQA 등)에 그대로 넣으면 됨.

### 꼭 지킬 것
1. **`E5Embeddings`를 써야 함.** `load_retriever()`가 알아서 이걸 쓰긴 하는데,
   혹시 직접 인덱스를 로드할 거면 그냥 `HuggingFaceEmbeddings`를 쓰지 말 것.
   e5는 질문 앞에 `query:`를 붙여야 검색이 제대로 됨 → 그 처리를 `E5Embeddings`가 함.
2. **차종 필터 꼭 걸기.** 질문에 차종이 안 들어있으면 딴 차 값이 섞여 나옴(= 틀린 답).
   사용자가 어느 차인지 먼저 정한 다음 그 `car`로 필터를 걸어야 함.
   car 값은 `avante / avante_hev / ioniq6 / nexo / tucson` 5개.

---

## 6. 다음에 할 것 (LLM 담당 = 파이프라인 6~8단계)

retriever까지는 준비됨. 이제 이 `retriever`를 입력으로 받아서:

- **[6] 프롬프트 구성**: retriever로 뽑은 청크(문맥) + 사용자 질문 → 프롬프트 템플릿
- **[7] LLM 답변**: HuggingFace LLM 또는 API 모델 연결해서 답 생성 (retriever를 RetrievalQA 같은 체인에 연결)
- **[8] 평가**: BERTScore / 검색지표 / 응답시간 / 할루시네이션 / 사람평가

정리하면, 이 인수인계의 `retriever`가 6~8단계의 시작점임.

---

## 7. 참고 / 주의

- **차종 필터가 이 프로젝트 핵심.** 세부목표 3(할루시네이션 억제)이 여기 걸려 있음.
  필터 안 걸면 아반떼 물었는데 투싼 값이 나오는 식이 됨.
- **needs_review=true 청크(눕힌 표, 11개)** 는 셀 구조가 깨진 상태로 들어가 있음.
  메타데이터에 `needs_review` 플래그로 남겨놨으니, 표 관련 답변 정확도를 평가할 때 따로 감안할 것.
- **청크가 바뀌면** (`chunk_size` 튜닝 등으로 `chunks_llmhybrid.json`이 갱신되면)
  `python vectorstore_search.py`만 다시 돌려서 인덱스를 새로 빌드하면 됨.
  코드는 청크 파일만 입력으로 받게 돼 있어서 그 외에 손댈 건 없음.
- FAISS 인덱스를 불러올 때 `allow_dangerous_deserialization=True`가 필요함(우리가 만든 파일이라 안전).
  `load_retriever()`가 이미 처리해둠.
- 임베딩/벡터스토어 비교 결과가 궁금하면 `embedding_compare.py`를 돌려보거나 `result_embedding/`의 json을 참고.

---

문의는 용주한테.
