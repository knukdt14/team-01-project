# 인수인계: LLM 비교 & 평가 파이프라인 ([7][8], 담당 A)

검색·프롬프트까지 완성된 파이프라인 뒤에 **LLM 답변 생성 + 평가**를 붙였습니다.
LLM만 교체하며 3종을 비교하고, BERTScore·검색지표·할루시네이션을 자동 측정합니다.

---

## 1. 산출물

```
src/llm_eval/
  rag_chain.py    검색→프롬프트→LLM 연결 (LLM 백엔드 교체 가능)
  evaluate.py     BERTScore·hit@k·refused·latency 계산 → results_<model>.csv
  main.py         ask / eval / compare / summary 진입점
eval/
  questions.csv   25문항 (id·question·question_type·car)
  references.csv  정답·근거·페이지 (id로 조인)
  results_*.csv   모델별 평가 결과 (실행 시 생성)
```

기존 팀 모듈을 **그대로 재사용**(수정 없음): `retrieval_adapter`(검색+차종필터),
`prompt_templates`(프롬프트 3종), `run_local_model`(로컬 모델 생성).

---

## 2. 환경 세팅 (팀원이 그대로 따라 하기)

### A. GPU용 PyTorch (⭐ 필수 — 없으면 CPU fp32로 돌아 매우 느림)
```bash
pip uninstall -y torch torchvision torchaudio
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available())"   # True 여야 함
```
> RTX 4070 Laptop(8GB) 기준. `cu124`가 안 맞으면 `cu121`로 시도.

### B. 추가 패키지
```bash
pip install bitsandbytes openai bert-score
pip install -U transformers
```
| 패키지 | 용도 |
|---|---|
| bitsandbytes | 로컬 모델 4-bit 로딩(8GB GPU) |
| openai | Solar/Upstage API 호출 클라이언트(※ GPT 아님, OpenAI 호환 규격용) |
| bert-score | 평가 지표 |
| transformers(최신) | EXAONE 로딩에 필요 |

### C. `.env` (레포 루트)
```
UPSTAGE_API_KEY=up_...     # Solar용 (console.upstage.ai)
HF_TOKEN=hf_...            # 모델 다운로드용 (huggingface.co/settings/tokens, Read 권한)
```
> `.env`는 `.gitignore`에 있어 커밋되지 않음. 각자 로컬에 생성.

---

## 3. 실행 방법

```bash
# 개별 모델 평가 → eval/results_<model>.csv 생성
python src/llm_eval/main.py eval --model upstage:solar-pro
python src/llm_eval/main.py eval --model local:Qwen/Qwen2.5-7B-Instruct

# 저장된 결과들로 비교표만 (재실행 X, 1초)   ← 이걸 주로 사용
python src/llm_eval/main.py summary

# 3종 전체 재실행 후 비교 (느림 — 설정 바꿨을 때만)
python src/llm_eval/main.py compare

# 단건 질의 테스트
python src/llm_eval/main.py ask "아반떼 엔진오일 교환주기는?" --model upstage:solar-pro
```
공통 옵션: `--variant {basic,role,constraint}` · `--top-k N` · `--max-new-tokens N`

모델 스펙: `local:<hf_id>` / `upstage:<model>` / `gemini:<model>`

---

## 4. ⚠️ EXAONE 전용 주의 (EXAONE 돌릴 사람만)

EXAONE-3.5는 커스텀 코드가 최신 transformers와 충돌해서 **캐시 파일 2곳 수동 패치**가 필요합니다.

**파일**: 모델 최초 다운로드 후 생성됨
```
C:\Users\<사용자>\.cache\huggingface\modules\transformers_modules\
  LGAI_hyphen_EXAONE\EXAONE_hyphen_3_dot_5_hyphen_7_dot_8B_hyphen_Instruct\<해시>\modeling_exaone.py
```

**수정** — `create_causal_mask(...)` 호출부 (파일 상단쪽, 라인 420 근처):
1. `input_embeds=inputs_embeds,` → `inputs_embeds=inputs_embeds,`
2. 바로 아래 `cache_position=cache_position,` **줄 삭제**
   - ⚠️ `create_causal_mask` 안의 것만! `decoder_layer(...)` 호출부의 `cache_position`은 그대로 둘 것

**실행** (패치 재다운로드 방지):
```cmd
set HF_HUB_OFFLINE=1
python src/llm_eval/main.py eval --model local:LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct --max-new-tokens 256
```
> EXAONE는 커스텀 코드가 비최적화라 문항당 ~79초로 느림(Qwen은 ~3초). `--max-new-tokens 256`으로 절반 단축.
> **EXAONE 안 돌리면** 2단계까지만 하면 Qwen·Solar는 바로 됩니다.

---

## 5. 평가 지표 설명

| 지표 | 의미 |
|---|---|
| BERTScore F1 | 생성 답변 vs 정답 의미 유사도 (lang="ko") |
| hit@k | 정답 근거가 검색된 청크 top-k 안에 있었는가 (검색 품질 분리 측정) |
| refused | '답 없음' 문항에 "해당 정보 없음"으로 답했는가 (할루시네이션 억제) |
| latency | 질의당 응답 시간(초) |

---

## 6. 결과 요약 (chunk 256/50, top_k 3, constraint 프롬프트 기준)

| 모델 | BERTScore | hit@k | 답없음 | 응답(s) |
|---|---|---|---|---|
| **Qwen2.5-7B** | **0.660** | 0.304 | **1.000** | 3.1 |
| EXAONE-3.5-7.8B | 0.631 | 0.304 | 0.667 | 78.9 |
| Solar-pro | 0.624 | 0.304 | 0.667 | 1.3 |

- **Qwen2.5-7B 종합 우수** — 정확도·할루시네이션 억제·속도 모두 최상. 로컬·무료.
- **hit@k가 3모델 모두 0.304** → **검색이 공통 병목이자 천장.** 모델을 바꿔도 검색이 근거를 못 가져오면 못 맞힘. 특히 표 참조 문항(정비 주기표)이 top-3에 잘 안 들어옴.

---

## 7. 다음 담당자에게 (후속 과제)

- **검색 개선이 최우선** — `top_k` 3 → 5~8로 올리면 표 문항이 검색에 들어와 **전 모델 점수 동반 상승** 예상. (담당 C 검색과 연계)
- 개선 후 `python src/llm_eval/main.py compare`로 재평가 → `summary`로 전후 비교.
- 프롬프트 변형(`--variant`)별 성능 비교도 가능.
