# 프롬프트 엔지니어링

기존 전처리·임베딩·검색 코드를 수정하지 않고 RAG 답변용 프롬프트를
독립적으로 관리하기 위한 폴더입니다.

## 구성

- `prompt_templates.py`: 기본형·역할 부여형·제약형 프롬프트와 문맥 조립 함수
- `preview_prompts.py`: API·모델 호출 없이 완성된 프롬프트 미리보기
- `retrieval_adapter.py`: 팀원의 FAISS `load_retriever()`를 안전하게 연결
- `run_local_model.py`: 실제 검색 청크와 Qwen2.5 로컬 모델로 RAG 답변 생성
- `test_prompt_templates.py`: 딕셔너리 청크와 LangChain `Document` 호환 테스트

## 사용 방법

검색 결과가 기존 JSON 청크와 같은 딕셔너리 목록인 경우:

```python
from prompt_templates import build_prompt

retrieved_chunks = [
    {
        "text": "검색된 설명서 내용",
        "car": "avante",
        "page": 390,
        "chunk_id": "avante_p390_0",
    },
    {
        "text": "두 번째로 검색된 설명서 내용",
        "car": "avante",
        "page": 391,
        "chunk_id": "avante_p391_0",
    },
]

prompt = build_prompt(
    question="아반떼 엔진오일 교환주기는?",
    documents=retrieved_chunks,
    variant="constraint",
    car="avante",
)
print(prompt)
```

`vectorstore_search.py`가 반환하는 LangChain `Document` 목록도 그대로
`documents`에 전달할 수 있습니다. 실제 실행 시에는 차량 필터를 적용해
FAISS에서 상위 3개 청크를 검색하고, 이 청크들을 하나의 프롬프트에 함께
넣어 답변을 생성합니다.

## 프롬프트 종류

- `basic`: 참고 문서와 질문만 제공
- `role`: 자동차 정비 전문가 역할 추가
- `constraint`: 차종 혼입·추측 방지, 정보 부족 응답, 근거 표시 규칙 추가

프롬프트 성능을 비교할 때는 검색 결과, LLM, 생성 설정, 질문을 동일하게
유지하고 `variant`만 변경해야 합니다.

## 테스트

프로젝트 루트에서 다음 명령을 실행합니다.

```bash
python -B src/prompt_engineering/test_prompt_templates.py
```

## 로컬 모델 답변 생성

RTX 4070 GPU와 필수 패키지가 준비된 `TF_ENV` 환경을 사용합니다.

```cmd
conda activate TF_ENV
chcp 65001
python -X utf8 -B src\prompt_engineering\run_local_model.py
```

기본 실행은 질문에서 차종을 자동으로 인식하는 제약형 대화형 모드입니다.
모델과 FAISS retriever를 처음 한 번만 로드한 뒤 `질문>`에 여러 질문을
연속으로 입력할 수 있습니다.

```text
질문> 투싼의 엔진 경고등이 들어왔어
차량 인식: 투싼 (tucson)

질문> 아반떼의 와이퍼가 고장났어
차량 인식: 아반떼 (avante)

질문> 엔진오일 교환주기는?
현재 차량 유지: 아반떼 (avante)

질문> 종료
```

빈 줄이나 `q`, `quit`, `exit`, `종료`를 입력하면 프로그램이 끝납니다.
각 질문은 이전 대화 내용과 분리하여 독립적으로 검색하고 답변합니다.
질문에 차종이 없으면 마지막으로 인식한 차량을 유지하며, 첫 질문에도
차종이 없으면 지원 차량명을 포함해 달라고 안내합니다.

질문에 차종이 없을 때 사용할 초기 차량을 지정할 수도 있습니다.

```cmd
python -X utf8 -B src\prompt_engineering\run_local_model.py --car tucson
```

대화형 모드 대신 질문 하나만 실행하려면:

```cmd
python -X utf8 -B src\prompt_engineering\run_local_model.py --car tucson --question "타이어 공기압은?"
```

차량 값은 `avante`, `avante_hev`, `ioniq6`, `nexo`, `tucson` 중 하나여야
합니다. 검색 청크 수를 바꾸려면 `--top-k 5`처럼 지정할 수 있습니다.
답변 최대 생성 길이는 기본 512토큰이며, 필요한 경우
`--max-new-tokens` 옵션으로 변경할 수 있습니다.

최초 실행 시 `Qwen/Qwen2.5-3B-Instruct` 모델을 Hugging Face 캐시에
다운로드합니다. 검색에 사용하는 `intfloat/multilingual-e5-base`도 캐시에
없으면 최초 한 번 다운로드합니다. 이후에는 캐시된 모델을 사용합니다.

제약형 프롬프트 하나만 실행하려면:

```cmd
python -X utf8 -B src\prompt_engineering\run_local_model.py --variant constraint
```

완성된 프롬프트까지 함께 출력하려면:

```cmd
python -X utf8 -B src\prompt_engineering\run_local_model.py --show-prompt
```

`.env`의 키는 코드나 Git에 넣지 않습니다. 기본 모델은 공개 모델이라
`HF_TOKEN`이 없어도 다운로드할 수 있지만, Hugging Face 인증과 다운로드
제한 완화를 위해 본인 토큰을 사용할 수 있습니다. 현재 로컬 실행에서
사용하는 키는 `HF_TOKEN`이며, `OPENAI_API_KEY`, `PINECONE_API_KEY`,
`UPSTAGE_API_KEY`는 이 실행 파일에서는 사용하지 않습니다.
