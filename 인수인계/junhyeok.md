# 프롬프트 엔지니어링 인수인계

## 구현 내용

- Context + Question, Role, Instruction / Constraint, Few-shot 프롬프트
  4종을 구현함.
- 동일한 검색 결과를 사용해 프롬프트만 변경한 답변을 비교할 수 있도록 구성함.
- `run_local_model.py`는 검색 청크와 프롬프트별 답변을 모두 표시하는
  개발·분석용으로 유지함.
- `final_prompt.py`는 최종 선정한 Few-shot만 사용하며 청크, 기법명,
  응답 생성 시간 등의 내부 정보를 표시하지 않는 사용자용으로 추가함.

## 프롬프트 4종 비교 조건

- 데이터: `data/RAG_Question_100.xlsx`
- 표본: 전체 100문항 중 seed 42로 무작위 추출한 25문항
- 검색: `Chroma/bge-m3`, 차종 필터, Top-7
- 답변 모델: `Qwen/Qwen2.5-3B-Instruct`
- 평가: BERTScore, RAGAS, 평균 응답 생성 시간
- RAGAS 평가 모델: Upstage `solar-pro3`

모든 프롬프트에 동일한 25문항과 동일한 검색 청크를 제공해 프롬프트 구성의
차이만 비교함.

## 프롬프트 4종 비교 결과

| 프롬프트 | BERTScore F1 | Faithfulness | Answer Relevancy | Context Precision | Context Recall | 생성 시간 |
|---|---:|---:|---:|---:|---:|---:|
| Context + Question | 0.701 | **0.802** | 0.830 | 0.810 | 0.800 | 104.19초 |
| Role | 0.726 | 0.476 | 0.596 | 0.810 | 0.800 | **47.98초** |
| Instruction / Constraint | 0.729 | 0.670 | 0.753 | 0.810 | 0.800 | 75.94초 |
| Few-shot | **0.730** | 0.688 | **0.837** | 0.810 | 0.800 | 69.46초 |

Context Precision과 Context Recall은 프롬프트가 적용되기 전의 동일한 검색
결과를 평가하므로 네 기법에서 같은 값이 나옴.

전체 결과는 `src/prompt_engineering/evaluation_results/`의 JSON, CSV,
HTML 및 PNG에서 확인할 수 있음.

## 최종 선정: Few-shot

Few-shot을 최종 프롬프트로 선정함.

- BERTScore F1이 0.730으로 가장 높음.
- Answer Relevancy가 0.837로 가장 높음.
- Constraint보다 Faithfulness가 높고 응답 생성 시간도 짧음.
- 모범 예시를 통해 문서에 답이 있는 경우와 없는 경우의 답변 방식을 함께
  안내할 수 있음.

Context + Question은 Faithfulness가 가장 높았지만 가장 느리고 BERTScore F1이
낮았음. Role은 가장 빠르지만 Faithfulness와 Answer Relevancy가 낮아 최종
사용자용 프롬프트로 적합하지 않다고 판단함.

## 최종 RAG 동작 흐름

```text
PDF 원문
  → PyMuPDF·pdfplumber 하이브리드 추출
  → 회전된 표 11페이지를 Upstage Document Parse로 구조화
  → 1차 청킹
      - 일반 문서: 최대 512글자, 50글자 중첩
      - 회전 표: 구조화된 표를 항목별로 분할
  → 일반 방향 표 페이지 후처리
      - replace_table_chunks.py에서 Upstage Document Parse 결과를
        행 단위 청크로 교체
  → 문서 임베딩
      - BAAI/bge-m3
      - Chroma에 차종·페이지·청크 ID 메타데이터와 함께 저장
  → 사용자 질문 처리
      - 질문에서 차종 자동 인식
      - 동의어 사전으로 구어체·약어를 취급설명서 표준 용어로 정규화
      - 질문을 bge-m3로 임베딩
  → Chroma 유사도 검색
      - 해당 차종으로 필터링
      - 관련 청크 Top-7 검색
  → 검색 청크를 하나의 문맥으로 결합
  → Few-shot 프롬프트 적용
  → Qwen2.5-3B-Instruct 답변 생성
  → 사용자에게 모델 답변만 표시
```

현재 `output/chunks_llmhybrid.json`에서 일반 문서 청크는 최대 512글자이며
50글자 중첩 설정을 사용함. 표 청크는 Upstage의 행 단위 구조화 결과를
사용하므로 일부 청크가 512글자를 넘을 수 있음.

표 구조화에는 Upstage Document Parse API가 사용됨. 질문을 LLM으로 확장하는
과정은 없으며, 질문의 동의어 처리는 `term_synonyms.py`의 사전과 정규식으로
수행됨.

청킹 파일이 두 위치에 있어 설정을 혼동하지 않도록 주의해야 함.
현재 결과와 검색 인덱스의 입력인 `output/chunks_llmhybrid.json`은
`src/chunker.py`의 일반 문서 512글자·50글자 중첩 설정과 이후 표 청크 교체
결과를 기준으로 함. `src/chunking/chunker.py`에는 256글자·100글자 중첩
설정도 남아 있으므로 데이터를 다시 만들 때 사용할 파일을 먼저 확인해야 함.

## 사용자용 최종 파일

`src/prompt_engineering/final_prompt.py`

- Few-shot 프롬프트 고정
- 차종 자동 인식 및 마지막 차종 유지
- Chroma/bge-m3 Top-7 검색
- 청크 내용과 프롬프트 비교 로그를 숨김
- 답변 결과에는 `[모델 답변]`과 답변 본문만 표시
  (시스템 준비·질문 입력·종료 안내는 표시)
- 기본 답변 모델은 로컬 `Qwen/Qwen2.5-3B-Instruct`

`run_local_model.py`는 검색 결과와 네 기법을 비교·진단할 때 사용하고,
일반 사용자는 `final_prompt.py`를 사용하면 됨.

## 다음 작업: 답변 모델 비교

다음 단계에서는 `final_prompt.py`를 기준 파이프라인으로 사용해 답변 모델을
비교하면 됨. 청킹, 동의어 처리, Chroma/bge-m3 Top-7 검색, Few-shot
프롬프트와 질문 세트는 고정하고 답변 모델만 교체해야 공정하게 비교할 수 있음.

기존 `src/llm_eval/`의 모델 비교 코드는 기본값이 Constraint 프롬프트와
Top-3이므로 현재 최종 조건과 다름. 해당 코드를 그대로 실행하지 말고
`final_prompt.py`의 Few-shot, Chroma/bge-m3 Top-7 검색 조건에 맞춘 뒤
모델 비교를 진행해야 함.

비교 후보 예시는 로컬 Qwen과 Upstage Solar이며 다음 항목을 확인하면 됨.

- BERTScore Precision, Recall, F1
- Faithfulness
- Answer Relevancy
- 응답 생성 시간
- API 사용량 및 비용
- 한국어 문장 품질과 사용자 체감 응답 품질
