# 전처리 → 임베딩 인수인계(한이-> 용주)

## 1. 한 줄 요약
PDF 5종을 파서(pymupdf랑 플럼버)로 텍스트 추출하고 청크로 잘라 `chunks_*.json` 3개 만들었음(각각 파서랑 하이브리드)

결과적으로 pymupdf가 성능이 모든 부분에서 우수했는데, 돌아간 표는 두 파서 다 성능 별로, 그래서 ocr 도 따로 모델선정해서 돌려봤는데 그냥 pymupdf가 나았음. 하이브리드 파일이 pymupdf결과랑 별반 다르진 않은데, 돌아간 표를 괜히 청킹했다가 성능 더 안 좋아질 거 같아서 돌아간 페이지는 1개 청크로 놔둠-> 이게 하이브리드.json 파일


**`chunks_hybrid.json`을 임베딩에 사용하면 됨.** 

---

## 2. 과정

### 파이프라인
```
PDF (data/) → 로더(텍스트 추출) → pages_*.json → chunker.py → chunks_*.json → [임베딩: 용주님]
```

### 파서를 3종 비교한 이유
취급설명서에 **글자를 90° 눕혀 인쇄한 표 페이지**가 11장 있습니다
(정비 주기표 등 / CN7 390~395, CN7HEV 228·396~399).

- **PyMuPDF**: 본문·눕힌 표 모두 텍스트 순서 안정적 → **채택**
- **pdfplumber**: 눕힌 표에서 글자 순서가 역순으로 깨짐 (청크 수도 더 많이 쪼개짐)
- 두 파서 모두 표의 **셀 구조(행/열 관계)** 는 자동 인식 불가 (표에 선이 없음)

### hybrid 파일이 하는 일
`pages_hybrid.json` = PyMuPDF 텍스트 + 눕힌 표 페이지에 플래그 부착.
(눕힌 표는 EasyOCR 재추출도 시도했으나 원본이 더 나아 텍스트는 PyMuPDF와 동일,
 차이는 `needs_review` 플래그입니다.)

---

## 3. 임베딩에 쓸 파일: `output/chunks_hybrid.json`

### 청크 구조
```json
{
  "car": "avante",              // 차종 (avante/avante_hev/ioniq6/tucson/nexo)
  "page": 390,                  // 출처 페이지
  "chunk_id": "avante_p390_0",  // 고유 ID
  "n_chars": 413,
  "text": "...",                // ← 이 필드를 임베딩
  "needs_review": true,         // 눕힌 표 페이지면 true
  "ocr_applied": false
}
```

### 용주님이 지킬거
1. **`text`** 를 임베딩한다.
2. **`car` 를 메타데이터 필터로 넣는다.** (5종 혼재 → 없으면 "공기압?"에 딴 차종 값이 섞임)
3. `page` 도 메타데이터로 넣으면 출처 표시(어느 페이지에서 답 나왔는지)가 가능하다.
4. **`needs_review=true` 청크(11개)** 는 눕힌 표라 셀 구조가 깨진 상태다.
   검색 평가 시 따로 집계하거나, 값 검수가 필요하다.

---

## 4. 재현 방법 (파일 다시 만들 때)
```bash
python src/loader_pymupdf.py       # → pages_pymupdf.json
python src/loader_pdfplumber.py    # → pages_pdfplumber.json
python src/hybrid.py               # → pages_hybrid.json  (loader_pymupdf 먼저 필요)
python src/chunker.py              # → chunks_*.json 3개
```
- 필요 패키지: `pymupdf`, `pdfplumber` (hybrid의 OCR까지 쓰려면 `easyocr pillow numpy`)
- `chunker.py` 는 외부 라이브러리 불필요 (순수 파이썬)

---

## 5. 참고: 청킹 파라미터는 아직 임시값(나중에 모델 다 만들면 바꾸면서 성능테스트 할 것)
현재 `chunk_size=512, overlap=50` 은 임시 기본값
파이프라인이 끝까지 연결된 뒤, **B가 실험②(Document Retrieval)에서**
이 값을 바꿔가며 최적값을 찾을 예정입니다.
→ C는 chunk_size에 의존하는 하드코딩 없이 `chunks_*.json` 을 입력으로만 받아주세요.

## 6. 알려진 한계 (향후 개선)
눕힌 표 11장은 셀 구조 복원이 안 돼 페이지 통째로 1청크 처리했습니다.
완전한 표 QA를 위해선 **멀티모달 LLM으로 표 이미지 구조화**가 필요하며,
이는 후속 작업할 거임 일단 저 배터리 13펀데 강의실에서 충전기를 안 들고 왔네요...............


원래 목표는 다 했고 제 욕심이 멀티모달까지라서 일단 넘겨준 파일대로 쭉 가면 될듯용