# 프롬프트 기법 비교 평가 결과

## 평가 목적

자동차 취급설명서 RAG 파이프라인에서 검색 결과가 같더라도 프롬프트 구성에
따라 답변의 정확성, 문서 근거 충실도, 질문 관련성 및 생성 시간이 달라지는지
확인하기 위해 프롬프트 4종을 비교했습니다.

비교한 프롬프트는 다음과 같습니다.

| 프롬프트 | 구성 |
|---|---|
| Context + Question | 검색 문맥과 질문만 제공하는 기준선 |
| Role | 자동차 취급설명서 전문가 역할 부여 |
| Instruction / Constraint | 문서 근거 사용과 추측 금지 등의 제약 부여 |
| Few-shot | 문서에 답이 있는 경우와 없는 경우의 예시 제공 |

프롬프트 템플릿은 `../prompt_templates.py`에 정의되어 있습니다.

## 프롬프트 엔지니어링 진행 방식

1. `data/RAG_Question_100.xlsx`의 질문 100개 중 seed 42로 25개를
   무작위 추출했습니다.
2. 모든 프롬프트에 동일한 질문과 동일한 Chroma/bge-m3 Top-7 검색 결과를
   사용했습니다.
3. 각 검색 청크를 하나의 문맥으로 결합한 뒤 프롬프트 4종만 바꿔
   `Qwen/Qwen2.5-3B-Instruct`의 답변을 생성했습니다.
4. Excel의 모범답안과 생성 답변을 BERTScore로 비교했습니다.
5. Upstage `solar-pro3`를 RAGAS 평가 모델로 사용해 Faithfulness,
   Answer Relevancy, Context Precision, Context Recall을 계산했습니다.
6. 프롬프트별 평균 지표와 평균 응답 생성 시간을 비교해 최종 프롬프트를
   선정했습니다.

평가 조건은 다음과 같습니다.

- 질문: 25개 무작위 추출, `seed=42`
- 검색: `Chroma/bge-m3`, 차종 필터 적용, `Top-7`
- 답변 모델: `Qwen/Qwen2.5-3B-Instruct`
- 최대 생성 길이: 512토큰
- RAGAS 평가 모델: Upstage `solar-pro3`

## 평가 지표

| 지표 | 의미 |
|---|---|
| BERTScore Precision | 생성 답변의 내용이 모범답안과 얼마나 정밀하게 겹치는지 평가 |
| BERTScore Recall | 모범답안의 핵심 내용을 생성 답변이 얼마나 포함하는지 평가 |
| BERTScore F1 | BERTScore Precision과 Recall의 균형 |
| Faithfulness | 답변의 주장이 검색 문맥으로 뒷받침되는 정도 |
| Answer Relevancy | 답변이 질문과 직접 관련된 정도 |
| Context Precision | 검색된 문맥 중 질문에 유용한 문맥의 비율 |
| Context Recall | 모범답안에 필요한 내용을 검색 문맥이 포함하는 정도 |
| 응답 생성 시간 | 프롬프트별 Qwen 답변 생성 시간 |

Context Precision과 Context Recall은 프롬프트 적용 전의 동일한 검색 결과를
평가합니다. 따라서 네 프롬프트에서 같은 값이 나오는 것이 정상입니다.

## 최신 비교 결과

| 프롬프트 | BERTScore P | BERTScore R | BERTScore F1 | Faithfulness | Answer Relevancy | Context Precision | Context Recall | 생성 시간 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Context + Question | 0.645 | 0.770 | 0.701 | **0.802** | 0.830 | 0.810 | 0.800 | 104.19초 |
| Role | **0.709** | 0.748 | 0.726 | 0.476 | 0.596 | 0.810 | 0.800 | **47.98초** |
| Instruction / Constraint | 0.684 | **0.782** | 0.729 | 0.670 | 0.753 | 0.810 | 0.800 | 75.94초 |
| Few-shot | 0.689 | 0.779 | **0.730** | 0.688 | **0.837** | 0.810 | 0.800 | 69.46초 |

이번 평가에서는 Few-shot이 BERTScore F1과 Answer Relevancy가 가장 높았고,
Constraint보다 Faithfulness가 높으면서 생성 시간도 짧았습니다. 따라서 답변
품질, 관련성, 근거 충실도 및 속도의 균형을 기준으로 Few-shot을 최종
프롬프트로 선정했습니다.

Context + Question은 Faithfulness가 가장 높았지만 가장 느렸고 BERTScore F1이
낮았습니다. Role은 가장 빨랐지만 Faithfulness와 Answer Relevancy가 크게
낮았습니다.

## 평가 관련 Python 파일

평가를 실행하거나 이 폴더의 결과물을 생성하는 Python 파일은 한 단계 위인
`src/prompt_engineering/`에 있습니다.

| 파일 | 역할 |
|---|---|
| `run_batch_evaluation.py` | 질문 추출, 검색, 프롬프트 4종 답변 생성, 평가, 체크포인트 저장을 총괄하는 실행 파일 |
| `batch_evaluation.py` | Excel 질문 로딩, 무작위 추출, 평균 계산, JSON·CSV·HTML 보고서 생성을 담당 |
| `evaluation_metrics.py` | BERTScore와 RAGAS 4개 지표를 계산하고 누락 지표 재평가 기능을 제공 |
| `prompt_templates.py` | 비교 대상 프롬프트 4종과 검색 문맥 조립 형식을 정의 |
| `retrieval_adapter.py` | 질문의 동의어를 정규화하고 팀 Chroma/bge-m3 검색기와 차종 필터를 연결 |
| `test_batch_evaluation.py` | 무작위 추출 재현성, 평균 계산, 결측값 처리 및 HTML 보고서 생성 테스트 |
| `run_local_model.py` | 청크와 프롬프트별 답변을 모두 보여주는 개발·분석용 실행 파일 |
| `final_prompt.py` | 최종 Few-shot만 사용하고 내부 청크와 평가 로그를 숨긴 사용자용 실행 파일 |

## 이 폴더의 결과 파일

| 파일 | 포함 내용 |
|---|---|
| `batch_evaluation_seed42.json` | 실행 조건, 선정 질문, 검색 청크, 프롬프트별 원문 답변, 모든 지표 및 체크포인트 상태 |
| `batch_summary_seed42.csv` | 프롬프트 4종의 지표별 평균값 |
| `batch_details_seed42.csv` | 질문 25개에 대한 프롬프트별 답변과 상세 지표 |
| `batch_report_seed42.html` | 평균 그래프, 비교표, 선정 질문을 포함한 웹 보고서 |
| `batch_report_seed42.png` | HTML 보고서를 한 장으로 확인할 수 있는 정적 이미지 |
| `README.md` | 평가 방법, 코드 역할, 결과 해석 및 재실행 방법 |

JSON은 중단 지점부터 다시 시작하는 체크포인트 역할도 합니다. 실행 조건이나
데이터셋 해시가 기존 체크포인트와 다르면 잘못된 결과가 섞이지 않도록 재개를
차단합니다.

## 실행 방법

프로젝트 최상위 폴더에서 실행합니다.

선정 질문만 확인:

```powershell
python -X utf8 -B src\prompt_engineering\run_batch_evaluation.py --dry-run
```

기존 결과를 새 평가로 교체:

```powershell
python -X utf8 -B src\prompt_engineering\run_batch_evaluation.py --overwrite --yes
```

중단된 체크포인트부터 재개:

```powershell
python -X utf8 -B src\prompt_engineering\run_batch_evaluation.py --resume --yes
```

저장된 답변을 유지하고 누락된 RAGAS 값만 복구:

```powershell
python -X utf8 -B src\prompt_engineering\run_batch_evaluation.py --repair-missing --yes
```

RAGAS 평가에는 `.env`의 `UPSTAGE_API_KEY`가 필요하며 Solar API 사용량이
발생합니다. `--overwrite`는 기존 체크포인트와 결과를 처음부터 교체하므로
기존 결과를 유지해야 할 때는 사용하지 않습니다.

## 결과 해석 시 주의사항

- 이번 결론은 질문 25개와 seed 42를 사용한 비교 결과입니다.
- RAGAS는 LLM 평가이므로 재실행 시 일부 점수가 달라질 수 있습니다.
- 프롬프트 비교에서는 검색 문맥을 동일하게 고정했습니다. 따라서 이 결과만으로
  청킹, 임베딩 또는 검색기의 우열을 판단할 수 없습니다.
- 다음 모델 비교에서는 `final_prompt.py`의 Few-shot과 동일한 검색 조건을
  고정한 뒤 답변 모델만 교체해야 공정하게 비교할 수 있습니다.
