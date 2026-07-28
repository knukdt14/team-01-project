# 프롬프트 엔지니어링

기존 전처리·임베딩·검색 코드를 수정하지 않고 RAG 답변용 프롬프트를
독립적으로 관리하기 위한 폴더입니다.

## 구성

- `prompt_templates.py`: RAG 비교용 프롬프트 4종과 문맥 조립 함수
- `preview_prompts.py`: API·모델 호출 없이 완성된 프롬프트 미리보기
- `retrieval_adapter.py`: 팀원의 Chroma DB를 영문 임시 캐시로 연결
- `run_local_model.py`: 실제 검색 청크로 Qwen 로컬 또는 Upstage API 답변 생성
- `evaluation_metrics.py`: 수업자료 방식의 BERTScore·RAGAS 평가
- `batch_evaluation.py`: Excel 질문 로딩, 무작위 추출, 평균 집계, HTML 시각화
- `run_batch_evaluation.py`: 질문 25개 × 프롬프트 4종 일괄 평가 실행
- `test_prompt_templates.py`: 딕셔너리 청크와 LangChain `Document` 호환 테스트
- `test_batch_evaluation.py`: 표본 추출·평균 집계·HTML 보고서 테스트

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
Chroma에서 상위 7개 청크를 검색하고, 이 청크들을 하나의 프롬프트에 함께
넣어 답변을 생성합니다.

## 프롬프트 종류

| 실행값 | 대응 기법 | 비교 목적 |
|---|---|---|
| `basic` | Context + Question | 검색 문맥만 제공하는 RAG 기준선 |
| `role` | Role Prompt | 자동차 정비 전문가 역할의 영향 확인 |
| `constraint` | Instruction Prompt | 추측 금지와 출력 규칙의 영향 확인 |
| `few_shot` | Few-shot Prompt | 근거 있음·없음 예시 2개의 영향 확인 |

프롬프트 성능을 비교할 때는 검색 결과, LLM, 생성 설정, 질문을 동일하게
유지하고 `variant`만 변경해야 합니다.

제약형에는 정확성 평가 안내, 문서 밖 추측 금지, 질문과 직접 관련된 근거만
사용, 실제 차종·페이지 출처 표시, 답변 전 수치·조건 재검증 규칙이 포함됩니다.
모델이 정보 부족 응답 뒤에 불필요한 문장을 덧붙이더라도 최종 출력은
`해당 정보 없음` 한 문구로 정규화합니다.

### Qwen 1차 정성 비교

동일한 Qwen 모델, top-3, 생성 설정에서 `투싼의 엔진 경고등이 켜졌어`와
`투싼으로 달에 가는 방법은?`를 비교했습니다.

| 기법 | 문서에 답이 있는 질문 | 문서에 답이 없는 질문 | 관찰 |
|---|---|---|---|
| Context + Question | 관련 답변 생성 | 문서 밖 답변 생성 | 기준선이지만 할루시네이션 위험이 큼 |
| Role | 이해하기 쉬운 답변 생성 | 문서 밖 답변 생성 | 표현은 자연스럽지만 근거 제약이 약함 |
| Instruction / Constraint | 근거와 출처 제시 | `해당 정보 없음` | 짧고 안정적이며 출력 가드가 적용됨 |
| Few-shot | `해당 정보 없음` | `해당 정보 없음` | 할루시네이션은 줄지만 과도하게 보수적임 |

표본 질문 2개의 정성 비교이므로 일반적인 성능 순위로 단정할 수는 없습니다.
정량 평가 시에는 여러 질문의 정확성, 출처 일치율, 정보 부족 판단률을 함께
측정해야 합니다.

## 수업자료 기반 평가 지표

수업자료 `lab.06_rag_evaluation.ipynb`와 강의 PDF의 RAG 평가 부분에 맞춰
다음 지표를 출력할 수 있습니다.

- BERTScore Precision / Recall / F1
- RAGAS Faithfulness
- RAGAS Answer Relevancy
- RAGAS Context Precision
- RAGAS Context Recall
- 프롬프트별 응답 생성 시간

BERTScore와 Context Precision/Recall을 계산하려면 사람이 작성한 모범답안이
필요합니다. RAGAS는 별도 평가 LLM을 사용하므로 평가 결과 생성 과정에서
Upstage API 사용량이 발생합니다. 기본 실행에서는 유료 평가를 수행하지 않고
응답 생성 시간만 표시합니다.

프롬프트 4종이 같은 검색 결과를 공유하므로 Faithfulness와 Answer Relevancy는
답변별로 계산하고, Context Precision과 Context Recall은 질문당 한 번만 계산해
모든 프롬프트 결과에 공통으로 표시합니다.
Upstage가 한 요청당 `n=1`만 허용하므로 Answer Relevancy의 질문 후보 수도
1개로 설정해 Solar 호환성과 API 사용량을 함께 맞춥니다.

단건 질문의 프롬프트 4종을 평가하려면:

```cmd
python -X utf8 -B src\prompt_engineering\run_local_model.py --variant all --question "투싼의 엔진 경고등이 켜졌어" --evaluate --reference-answer "엔진 경고등이 3초 후에도 계속 켜져 있으면 엔진 제어 장치나 연료 공급 장치 이상일 수 있으므로 하이테크센터나 블루핸즈에서 점검받아야 합니다."
```

대화형 평가에서는 질문 다음에 `모범답안>` 입력란이 추가됩니다.

```cmd
python -X utf8 -B src\prompt_engineering\run_local_model.py --variant all --evaluate
```

최초 평가 시 BERTScore와 임베딩 모델을 내려받을 수 있습니다. 평가 결과가
`N/A`이면 해당 RAGAS 평가 호출이 실패했거나 판단 가능한 값이 없다는 뜻입니다.

## 질문 25개 일괄 평가와 시각화

`data/RAG_Question_100.xlsx`에서 질문 25개를 단순 무작위 추출한 뒤, 같은
질문과 같은 top-3 검색 결과로 프롬프트 4종의 답변을 생성하고 평가합니다.
기본 random seed는 `42`이므로 같은 데이터셋에서는 항상 같은 질문이
선정됩니다. 각 질문의 차량 열을 Chroma 차종 필터로 사용하므로 대화형 모드의
이전 차량 상태는 일괄 평가에 영향을 주지 않습니다.

먼저 모델과 API를 호출하지 않고 선정 질문만 확인할 수 있습니다.

```cmd
python -X utf8 -B src\prompt_engineering\run_batch_evaluation.py --dry-run
```

전체 평가를 실행하려면 다음 명령을 사용합니다.

```cmd
python -X utf8 -B src\prompt_engineering\run_batch_evaluation.py
```

질문 25개 × 프롬프트 4종으로 Qwen 답변 100개를 생성합니다. BERTScore는
로컬에서 계산하고 Faithfulness, Answer Relevancy, Context Precision,
Context Recall은 `.env`의 `UPSTAGE_API_KEY`를 이용해 Solar로 평가하므로
API 사용량과 긴 실행 시간이 발생합니다. 실행 전에 계속할지 확인하며,
확인 질문을 생략하려면 `--yes`를 추가할 수 있습니다.

평가는 질문 하나가 끝날 때마다 체크포인트에 저장됩니다. 중간에 종료된 경우
같은 옵션과 다음 옵션으로 이어서 실행합니다.

```cmd
python -X utf8 -B src\prompt_engineering\run_batch_evaluation.py --resume
```

Solar 연결 오류로 일부 RAGAS 값이 `N/A`가 된 경우에는 저장된 Qwen 답변을
다시 생성하지 않고 결측 지표만 재평가할 수 있습니다.

```cmd
python -X utf8 -B src\prompt_engineering\run_batch_evaluation.py --repair-missing
```

기본 결과는 `src/prompt_engineering/evaluation_results/`에 저장됩니다.

- `batch_evaluation_seed42.json`: 선정 질문, 검색 문맥, 답변, 개별 지표
- `batch_details_seed42.csv`: 질문 25개 × 프롬프트 4종의 상세 결과
- `batch_summary_seed42.csv`: 프롬프트별 지표 평균
- `batch_report_seed42.html`: 평균 지표 막대그래프와 비교표

HTML 보고서에서 BERTScore P/R/F1, Faithfulness, Answer Relevancy,
Context Precision, Context Recall은 0~1 범위로 시각화하고, 응답 생성 시간은
별도 그래프로 비교합니다. 오류나 결측값은 평균에서 제외하며 표에 실제 평균
계산에 사용된 질문 수 `n`을 함께 표시합니다.

## 테스트

프로젝트 루트에서 다음 명령을 실행합니다.

```bash
python -B src/prompt_engineering/test_prompt_templates.py
python -B src/prompt_engineering/test_batch_evaluation.py
```

## 로컬 모델 답변 생성

RTX 4070 GPU와 필수 패키지가 준비된 `TF_ENV` 환경을 사용합니다.

팀원의 Chroma DB는 한글·OneDrive 경로에서 HNSW 파일을 열지 못할 수 있어,
최초 실행 시 동일한 DB를 Windows 영문 임시 캐시에 복사한 뒤 검색합니다.
원본 DB는 수정하지 않으며, 원본 내용이 바뀌면 새 캐시가 자동 생성됩니다.

```cmd
conda activate TF_ENV
chcp 65001
python -X utf8 -B src\prompt_engineering\run_local_model.py
```

기본 실행은 질문에서 차종을 자동으로 인식하고, 같은 검색 결과로 프롬프트
4종(Context + Question / Role / Instruction·Constraint / Few-shot)의 답변을
모두 출력하는 대화형 모드입니다.
모델과 Chroma/bge-m3 retriever를 처음 한 번만 로드한 뒤 `질문>`에 여러 질문을
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

## 답변 모델 교체

기본값은 로컬 `Qwen/Qwen2.5-3B-Instruct`입니다. Hugging Face 모델은
코드 수정 없이 `--model`로 교체할 수 있습니다.

```cmd
python -X utf8 -B src\prompt_engineering\run_local_model.py --provider huggingface --model Qwen/Qwen2.5-3B-Instruct
```

`.env`에 `UPSTAGE_API_KEY`가 있으면 Upstage API 모델로도 실행할 수 있습니다.

```cmd
python -X utf8 -B src\prompt_engineering\run_local_model.py --provider upstage
```

Upstage 기본 모델은 `solar-pro3`이며 다른 모델 ID는 `--model`로 지정합니다.
API 제공자를 사용하면 질문마다 사용량이 발생합니다.

최초 실행 시 `Qwen/Qwen2.5-3B-Instruct` 모델을 Hugging Face 캐시에
다운로드합니다. 검색에 사용하는 `BAAI/bge-m3`도 캐시에
없으면 최초 한 번 다운로드합니다. 이후에는 캐시된 모델을 사용합니다.

제약형 프롬프트 하나만 실행하려면:

```cmd
python -X utf8 -B src\prompt_engineering\run_local_model.py --variant constraint
```

같은 검색 결과로 4종을 연속 비교하는 기본 실행:

```cmd
python -X utf8 -B src\prompt_engineering\run_local_model.py
```

`--variant all`을 명시해도 동일하게 동작합니다.

완성된 프롬프트까지 함께 출력하려면:

```cmd
python -X utf8 -B src\prompt_engineering\run_local_model.py --show-prompt
```

`.env`의 키는 코드나 Git에 넣지 않습니다. 기본 모델은 공개 모델이라
`HF_TOKEN`이 없어도 다운로드할 수 있지만, Hugging Face 인증과 다운로드
제한 완화를 위해 본인 토큰을 사용할 수 있습니다. 현재 로컬 실행에서
사용하는 키는 `HF_TOKEN`이며, `OPENAI_API_KEY`, `PINECONE_API_KEY`,
`UPSTAGE_API_KEY`는 이 실행 파일에서는 사용하지 않습니다.
