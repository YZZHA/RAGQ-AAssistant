import json
from pathlib import Path

import pytest

from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase

from src.evaluate.judge_model import QwenJudgeModel
from src.retrieval.bm25 import BM25Index
from src.retrieval.embedding import vector_store, search_by_text
from src.retrieval.fusion import rrf_fusion
from src.retrieval.rewrite import rewrite
from src.reranker.cross_encoder import Reranker, dynamic_top_k
from src.generator.llm import LLM
from src.generator.prompts import build_rag_messages
from src.core.config import settings


judge = QwenJudgeModel()
llm = LLM()

bm25 = BM25Index()
reranker = Reranker()


def load_dataset(path: str = "data/qa_dataset/eval.jsonl") -> list[dict]:
    items = []
    p = Path(path)
    if not p.exists():
        return items
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_bm25_index():
    from src.ingestion.loader import load_directory
    from src.ingestion.chunker import chunk_document
    from src.ingestion.embedder import Embedder

    docs = load_directory("data/raw")
    all_chunks = []
    for text, meta in docs:
        chunks = chunk_document(text, meta["doc_id"])
        for c in chunks:
            c["doc_title"] = meta.get("filename", meta["doc_id"])
            c["tenant_id"] = "default"
        all_chunks.extend(chunks)
    bm25.build(all_chunks)

    try:
        embedder = Embedder()
        vector_store.build(all_chunks, embedder)
    except Exception as e:
        import logging
        logging.getLogger("rag_qa").warning("FAISS 索引构建跳过: %s", e)

    return len(all_chunks)


def run_rag_pipeline(question: str) -> tuple[str, list[str], str]:
    rewritten = rewrite(question)
    bm25_res = bm25.search(rewritten, top_k=10)
    emb_res = search_by_text(rewritten, top_k=10)
    fused = rrf_fusion(bm25_res, emb_res, top_k=5)
    reranked = reranker.rerank(rewritten, fused)
    top_chunks = dynamic_top_k(reranked, min_k=2, max_k=4)

    if not top_chunks:
        msgs = build_rag_messages(question, [])
        actual_output = llm.generate(msgs)
        return actual_output, [], rewritten

    retrieval_context = [c["content"] for c in top_chunks]
    msgs = build_rag_messages(rewritten, top_chunks)
    actual_output = llm.generate(msgs)
    return actual_output, retrieval_context, rewritten


@pytest.fixture(scope="session")
def eval_dataset():
    data = load_dataset()
    if not data:
        pytest.skip("评估数据集不存在: data/qa_dataset/eval.jsonl")
    return data


@pytest.fixture(scope="session", autouse=True)
def init_engine():
    count = build_bm25_index()
    return count


def get_test_params():
    data = load_dataset()
    if not data:
        return [("placeholder", "placeholder", ["placeholder"], "placeholder")]
    params = []
    for item in data[:10]:
        params.append((
            item.get("question", ""),
            item.get("expected_answer", ""),
            item.get("relevant_docs", []),
        ))
    return params


class TestRAGEval:
    @pytest.mark.parametrize(
        ("question", "expected_answer", "relevant_docs"),
        get_test_params(),
    )
    def test_rag_pipeline(self, question, expected_answer, relevant_docs):
        if not question or question == "placeholder":
            pytest.skip("无测试数据")

        actual_output, retrieval_context, rewritten = run_rag_pipeline(question)

        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=expected_answer,
            retrieval_context=retrieval_context,
        )

        assert_test(test_case, [
            AnswerRelevancyMetric(threshold=0.5, model=judge),
            FaithfulnessMetric(threshold=0.5, model=judge),
            ContextualPrecisionMetric(threshold=0.3, model=judge),
            ContextualRecallMetric(threshold=0.3, model=judge),
        ])
