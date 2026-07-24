"""
rag_chain.py — [7] LLM 답변 생성  ─ 담당 A

검색 → 프롬프트 → LLM → 파싱을 LCEL 체인으로 연결한다.
답변뿐 아니라 '검색된 문맥'도 함께 반환해야 평가([8])에서
검색 지표(Hit@k)와 생성 지표(BERTScore)를 분리해 계산할 수 있다.

실행:  python src/rag_chain.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from config import BASELINE, Config
from prompts import get_prompt
from retriever import format_docs, search

load_dotenv()


# ──────────────────────────── LLM ────────────────────────────
def get_llm(cfg: Config = BASELINE) -> BaseChatModel:
    """LLM 로드.

    TODO(A): 비교 대상
      API   — gpt-4o-mini, claude-*, solar-pro(Upstage)
      로컬 HF — Qwen/Qwen2.5-7B-Instruct, upstage/SOLAR-10.7B, yanolja/EEVE-Korean-10.8B

    ⚠️ 로컬 7B급은 VRAM이 부족하면 로드에 실패한다.
       GPU 여건을 먼저 확인하고, 안 되면 API 모델을 베이스라인으로 쓴다.
    """
    name = cfg.llm

    if name.startswith(("gpt-", "o1", "o3")):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=name, temperature=cfg.temperature)

    if name.startswith("claude"):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=name, temperature=cfg.temperature)

    if name.startswith("solar"):
        from langchain_upstage import ChatUpstage

        return ChatUpstage(model=name, temperature=cfg.temperature)

    # 로컬 HuggingFace 모델
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

    pipe = HuggingFacePipeline.from_model_id(
        model_id=name,
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": cfg.temperature > 0,
            "temperature": cfg.temperature or None,
        },
    )
    return ChatHuggingFace(llm=pipe)


# ──────────────────────────── 체인 ────────────────────────────
@dataclass
class RagResult:
    question: str
    answer: str
    contexts: list[Document]
    latency: float

    @property
    def context_text(self) -> str:
        return format_docs(self.contexts)

    @property
    def sources(self) -> list[str]:
        return [f"{d.metadata['vehicle']} p.{d.metadata['page']}" for d in self.contexts]


class RagChain:
    """검색 결과를 보존하면서 답변을 생성하는 RAG 체인."""

    def __init__(self, store, cfg: Config = BASELINE):
        self.store = store
        self.cfg = cfg
        self.llm = get_llm(cfg)
        self.chain = get_prompt(cfg.prompt_type) | self.llm | StrOutputParser()

    def invoke(self, question: str) -> RagResult:
        t0 = time.perf_counter()
        docs = search(self.store, question, self.cfg)
        answer = self.chain.invoke({
            "context": format_docs(docs),
            "question": question,
        })
        return RagResult(
            question=question,
            answer=answer.strip(),
            contexts=docs,
            latency=time.perf_counter() - t0,
        )


if __name__ == "__main__":
    import io
    import sys

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    from build_vectorstore import load

    chain = RagChain(load(BASELINE), BASELINE)
    for q in ["아반떼 엔진오일 교환주기는?", "아이오닉6 급속충전 시간은?"]:
        r = chain.invoke(q)
        print(f"\n[질문] {r.question}")
        print(f"[답변] {r.answer}")
        print(f"[출처] {r.sources}  ({r.latency:.2f}s)")
