"""rag_chain.py — 검색 → 프롬프트 → LLM 답변을 하나로 잇는 RAG 체인. (담당 A)

팀원 코드를 그대로 재사용한다(수정하지 않음):
  - retrieval_adapter : 검색기 로드 + 차종 필터 검색            (junhyeok/Kopzoo)
  - prompt_templates  : 프롬프트 3종(basic/role/constraint)     (junhyeok)
  - run_local_model   : 로컬 HuggingFace 모델 로드/생성          (junhyeok)

[7] LLM 비교를 위해 백엔드만 교체 가능하게 추상화했다.
  local:<hf_id>    로컬 HuggingFace  (예: local:Qwen/Qwen2.5-7B-Instruct)
  gemini:<model>   Google Gemini API (예: gemini:gemini-3.5-flash)
  upstage:<model>  Upstage Solar API (예: upstage:solar-pro)
검색·프롬프트는 고정하고 LLM만 바꾸므로, 순수하게 '모델 성능'만 비교된다.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# 팀의 prompt_engineering 모듈을 import 경로에 추가 (파일 위치: src/llm_eval/)
_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompt_engineering"
sys.path.insert(0, str(_PROMPT_DIR))

from prompt_templates import PromptVariant, build_prompt  # noqa: E402
from retrieval_adapter import (  # noqa: E402
    SUPPORTED_CARS,
    create_retriever,
    detect_car_from_question,
    retrieve_documents,
)


def _load_env() -> None:
    """API 키를 위한 .env를 환경변수로 로드한다(레포 루트·prompt_engineering·현재 폴더)."""
    root = Path(__file__).resolve().parents[2]
    candidates = [root / ".env", root / "src" / "prompt_engineering" / ".env", Path.cwd() / ".env"]
    try:
        from dotenv import load_dotenv

        for p in candidates:
            if p.exists():
                load_dotenv(p, override=False)
        return
    except ImportError:
        pass
    for p in candidates:  # python-dotenv 없을 때 간단 파서
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip("\"'"))


_load_env()


# ────────────────────────────── LLM 백엔드 ──────────────────────────────
class LocalLLM:
    """로컬 HuggingFace 백엔드.

    8GB VRAM(RTX 4070 Laptop)에서 7B를 돌리기 위해 기본적으로 4-bit(NF4)로 로드한다.
      - CUDA + bitsandbytes 있으면 → 4-bit (Qwen2.5-7B ≈ 5~6GB, 8GB에 적합)
      - load_in_4bit=False → 강제 fp16 (작은 모델·큰 VRAM용)
    생성은 팀의 run_local_model.generate_answer를 그대로 재사용한다(팀 코드 무수정).
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        load_in_4bit: bool | None = None,
        max_new_tokens: int = 512,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from run_local_model import generate_answer, usable_hf_token

        token = usable_hf_token(os.environ.get("HF_TOKEN"))
        has_cuda = torch.cuda.is_available()
        if load_in_4bit is None:
            load_in_4bit = has_cuda  # GPU가 있으면 4-bit 기본(7B도 올라가게)

        # EXAONE 등 custom_code 모델은 trust_remote_code=True 필요(Qwen엔 무해)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=token, trust_remote_code=True
        )

        if load_in_4bit:
            if not has_cuda:
                raise RuntimeError(
                    "4-bit 로딩은 CUDA(GPU)가 필요합니다. "
                    "GPU가 없으면 LocalLLM(load_in_4bit=False)로 fp16을 쓰세요."
                )
            try:
                import bitsandbytes  # noqa: F401  (설치 여부 확인)
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError(
                    "bitsandbytes가 없습니다. 7B 4-bit 로딩을 위해 설치하세요:\n"
                    "    pip install bitsandbytes\n"
                    "(또는 작은 모델: local:Qwen/Qwen2.5-3B-Instruct)"
                ) from exc

            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,  # Ada(4070)는 bf16 지원
            )
            print(f"모델 로딩(4-bit NF4): {model_id}")
            # device_map={"":0} = 전체를 GPU0에 강제 배치(auto가 8GB를 보수적으로 보고
            # CPU로 떠넘겨 bnb 에러 내는 것 방지). EXAONE-7.8B 4-bit(~5~6GB)면 8GB에 들어감.
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, token=token, quantization_config=quant, device_map={"": 0},
                trust_remote_code=True,
            )
            self.device = "cuda"  # GPU에 배치됨(별도 .to() 금지)
        else:
            dtype = torch.float16 if has_cuda else torch.float32
            print(f"모델 로딩({'fp16' if has_cuda else 'fp32'}): {model_id}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, token=token, torch_dtype=dtype, trust_remote_code=True
            )
            self.device = "cuda" if has_cuda else "cpu"
            self.model.to(self.device)

        self.model.eval()
        self._torch = torch
        self._generate_answer = generate_answer
        self.max_new_tokens = max_new_tokens
        self.name = f"local:{model_id}" + ("(4bit)" if load_in_4bit else "")

    def generate(self, prompt: str) -> str:
        return self._generate_answer(
            prompt, self.tokenizer, self.model, self._torch, self.device, self.max_new_tokens
        )


class _CompatLLM:
    """Gemini·Upstage 공용 API 백엔드.

    두 서비스 모두 표준 Chat Completions 규격을 제공하므로, openai 파이썬
    패키지를 '범용 HTTP 클라이언트'로만 사용한다. (GPT 모델을 쓰는 것이 아님)
    temperature=0으로 재현성 확보.
    """

    def __init__(self, model: str, api_key: str | None, base_url: str, key_env: str = ""):
        if not api_key:
            raise RuntimeError(
                f"API 키가 없습니다: 환경변수 {key_env or '<KEY>'} 를 설정하세요.\n"
                f"  → 레포 루트에 .env 파일을 만들고  {key_env or 'XXX_API_KEY'}=... 를 넣으세요."
            )
        from openai import OpenAI  # 표준 Chat Completions 엔드포인트용 클라이언트 SDK

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, prompt: str, retries: int = 8) -> str:
        import re

        for attempt in range(retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                msg = str(exc)
                is_429 = ("429" in msg or "RESOURCE_EXHAUSTED" in msg
                          or getattr(exc, "status_code", None) == 429)
                if not (is_429 and attempt < retries):
                    raise
                # 무료 티어 분당 제한(RPM) → 서버가 알려준 대기 시간만큼 기다렸다 이어감
                m = re.search(r"retry in ([\d.]+)s", msg)
                wait = (float(m.group(1)) + 2) if m else 30.0
                wait = min(wait, 65.0)
                print(f"    [rate limit] {wait:.0f}s 대기 후 이어감 "
                      f"({attempt + 1}/{retries})...")
                time.sleep(wait)


class UpstageLLM(_CompatLLM):
    def __init__(self, model: str = "solar-pro"):
        super().__init__(model, api_key=os.environ.get("UPSTAGE_API_KEY"),
                         base_url="https://api.upstage.ai/v1", key_env="UPSTAGE_API_KEY")
        self.name = f"upstage:{model}"


class GeminiLLM(_CompatLLM):
    """Google Gemini API. 무료 티어(AI Studio) 사용 가능.

    키 발급: https://aistudio.google.com/  →  환경변수 GEMINI_API_KEY
    """

    def __init__(self, model: str = "gemini-3.5-flash"):
        super().__init__(
            model,
            api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            key_env="GEMINI_API_KEY",
        )
        self.name = f"gemini:{model}"


def make_llm(spec: str, max_new_tokens: int = 512):
    """'local:<id>' / 'gemini:<model>' / 'upstage:<model>' 문자열로 백엔드 생성."""
    kind, _, name = spec.partition(":")
    if kind == "local":
        return LocalLLM(name or "Qwen/Qwen2.5-7B-Instruct", max_new_tokens=max_new_tokens)
    if kind == "upstage":
        return UpstageLLM(name or "solar-pro")
    if kind == "gemini":
        return GeminiLLM(name or "gemini-3.5-flash")
    raise ValueError(
        f"알 수 없는 LLM 스펙: {spec!r} (local:/gemini:/upstage: 중 하나)"
    )


# ────────────────────────────── RAG 체인 ──────────────────────────────
@dataclass
class RagResult:
    question: str
    car: str
    answer: str
    contexts: list  # list[Document]
    latency: float

    @property
    def sources(self) -> list[str]:
        out = []
        for d in self.contexts:
            m = getattr(d, "metadata", {})
            out.append(f"{m.get('car', '?')} p.{m.get('page', '?')}")
        return out


class RagChain:
    """검색기를 한 번 로드해두고, LLM만 바꿔가며 질의에 답한다."""

    def __init__(self, llm, variant: PromptVariant | str = PromptVariant.CONSTRAINT, k: int = 3):
        self.llm = llm
        self.variant = PromptVariant(variant)
        self.k = k
        self.retriever = create_retriever(car=None, k=k)  # 필터는 질의 시점에 건다

    def ask(self, question: str, car: str | None = None) -> RagResult:
        car = car or detect_car_from_question(question)
        if car not in SUPPORTED_CARS:
            raise ValueError(
                f"차종을 인식/지정하지 못했습니다: {question!r} "
                f"(가능: {', '.join(SUPPORTED_CARS)})"
            )
        t0 = time.perf_counter()
        docs = retrieve_documents(question=question, car=car, k=self.k, retriever=self.retriever)
        prompt = build_prompt(question=question, documents=docs, variant=self.variant, car=car)
        answer = self.llm.generate(prompt)
        return RagResult(question, car, answer.strip(), docs, time.perf_counter() - t0)


if __name__ == "__main__":
    # 간단 단건 테스트: python rag_chain.py "아반떼 엔진오일 교환주기는?" upstage:solar-pro
    q = sys.argv[1] if len(sys.argv) > 1 else "아반떼 엔진오일 교환주기는?"
    spec = sys.argv[2] if len(sys.argv) > 2 else "upstage:solar-pro"
    chain = RagChain(make_llm(spec))
    r = chain.ask(q)
    print(f"[차종] {r.car}\n[답변] {r.answer}\n[출처] {r.sources}  ({r.latency:.2f}s)")
