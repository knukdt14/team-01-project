"""
main.py — 전체 파이프라인 실행 진입점

사용법:
  python src/main.py build              # 벡터스토어 구축
  python src/main.py ask "질문"          # 단건 질의
  python src/main.py chat               # 대화형 질의
  python src/main.py eval               # 평가셋 전체 평가

실험 시 자기 변수만 바꿔 실행:
  python src/main.py eval --chunk-size 1024
  python src/main.py eval --prompt constrained --filter
"""

from __future__ import annotations

import argparse
import io
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from config import BASELINE, variant  # noqa: E402


def build_cfg(args) -> object:
    """CLI 인자를 설정 오버라이드로 변환. 지정 안 한 값은 베이스라인 유지."""
    overrides = {}
    if args.loader:        overrides["loader"] = args.loader
    if args.chunk_size:    overrides["chunk_size"] = args.chunk_size
    if args.overlap:       overrides["chunk_overlap"] = args.overlap
    if args.embedding:     overrides["embedding_model"] = args.embedding
    if args.store:         overrides["vectorstore"] = args.store
    if args.top_k:         overrides["top_k"] = args.top_k
    if args.search:        overrides["search_type"] = args.search
    if args.prompt:        overrides["prompt_type"] = args.prompt
    if args.llm:           overrides["llm"] = args.llm
    if args.filter:        overrides["use_metadata_filter"] = True
    return variant(**overrides) if overrides else BASELINE


def main() -> None:
    p = argparse.ArgumentParser(description="자동차 취급설명서 RAG QA")
    p.add_argument("command", choices=["build", "ask", "chat", "eval"])
    p.add_argument("question", nargs="?", help="ask 명령의 질문")

    g = p.add_argument_group("실험 변수 (지정하지 않으면 베이스라인)")
    g.add_argument("--loader", choices=["pymupdf", "pypdf", "pdfplumber"])
    g.add_argument("--chunk-size", type=int)
    g.add_argument("--overlap", type=int)
    g.add_argument("--embedding")
    g.add_argument("--store", choices=["faiss", "chroma"])
    g.add_argument("--top-k", type=int)
    g.add_argument("--search", choices=["similarity", "mmr"])
    g.add_argument("--prompt", choices=["basic", "role", "constrained"])
    g.add_argument("--llm")
    g.add_argument("--filter", action="store_true", help="차량 메타데이터 필터링 사용")

    args = p.parse_args()
    cfg = build_cfg(args)
    print(f"설정: {cfg.tag()}{'  +필터' if cfg.use_metadata_filter else ''}\n")

    if args.command == "build":
        from build_vectorstore import build

        build(cfg, force=True)
        return

    if args.command == "eval":
        from evaluate import run

        run(cfg)
        return

    # ask / chat 은 인덱스와 LLM이 필요
    from build_vectorstore import load
    from rag_chain import RagChain

    chain = RagChain(load(cfg), cfg)

    if args.command == "ask":
        if not args.question:
            p.error("ask 명령에는 질문이 필요합니다.")
        r = chain.invoke(args.question)
        print(f"[답변]\n{r.answer}\n")
        print(f"[출처] {', '.join(r.sources)}  ({r.latency:.2f}s)")
        return

    # chat
    print("대화형 모드 — 종료하려면 빈 줄 입력\n")
    while True:
        try:
            q = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        r = chain.invoke(q)
        print(f"\n{r.answer}\n  └ 출처: {', '.join(r.sources)} ({r.latency:.2f}s)\n")


if __name__ == "__main__":
    main()
