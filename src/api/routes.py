# input:  FastAPI, schemas, generator/llm, retrieval, memory
# output: HTTP API endpoints (POST /api/chat, etc.)
# pos:    系统入口，接收用户请求 → 编排下游模块 → 返回 SSE 流

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from src.api.schemas import ChatRequest
from src.api.dependencies import get_or_create_session
from src.core.logging import logger
from src.generator.llm import LLM
from src.generator.prompts import build_rag_messages
from src.memory.short_term import add_message, get_history
from src.retrieval.bm25 import BM25Index
from src.retrieval.rewrite import rewrite as rewrite_query
from src.retrieval.fusion import rrf_fusion
from src.retrieval.embedding import search_by_text
from src.decompose.llm_splitter import decompose
from src.reranker.cross_encoder import Reranker, dynamic_top_k

from src.ingestion.loader import load_directory
from src.ingestion.chunker import chunk_document
from src.retrieval.local_store import LocalVectorStore
from src.knowledge.gap_detector import GapDetector
from src.knowledge.draft import generate_draft
from src.knowledge.validator import validate
from src.knowledge.auto_ingest import auto_ingest
from src.ingestion.embedder import Embedder
from src.retrieval.embedding import vector_store

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
app = FastAPI(title="FDI RAG 问答助手", version="0.1.0")

_bm25 = BM25Index()
_reranker = Reranker()
_llm = LLM()

_gap_detector = GapDetector()
_source_docs_cache = {}


def _init_bm25():
    docs = load_directory("data/raw")
    all_chunks = []
    for text, meta in docs:
        chunks = chunk_document(text, meta["doc_id"])
        for c in chunks:
            c["doc_title"] = meta.get("filename", meta["doc_id"])
            c["source_type"] = meta.get("source_type", "MD")
            c["tenant_id"] = "default"
        all_chunks.extend(chunks)
        _source_docs_cache[meta["doc_id"]] = meta.get("filename", meta["doc_id"])
    _bm25.build(all_chunks)
    logger.info("BM25 索引已构建: %d chunks", len(all_chunks))
    return all_chunks


def _init_vector_store(all_chunks):
    try:
        logger.info("向量索引构建中（首次加载模型约 30s）...")
        embedder = Embedder()
        count = vector_store.build(all_chunks, embedder)
        logger.info("FAISS 向量索引已构建: %d chunks", count)
    except Exception as e:
        logger.warning("FAISS 索引构建跳过: %s", e)


@app.on_event("startup")
async def startup():
    chunks = _init_bm25()
    from src.knowledge.llm_chunker import rebuild_all_cache

    import threading

    def _bg_init():
        _init_vector_store(chunks)
        rebuild_all_cache()

    threading.Thread(target=_bg_init, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse(request, "chat.html")


@app.get("/gaps", response_class=HTMLResponse)
async def gaps_page(request: Request):
    return templates.TemplateResponse(request, "gaps.html")


@app.get("/api/gaps")
async def list_gaps():
    records = _gap_detector.log.read_all()
    ingested_ids = set()
    ingested_path = Path("data/knowledge_gaps/ingested.jsonl")
    if ingested_path.exists():
        with ingested_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ingested_ids.add(json.loads(line).get("doc_id", ""))
    for r in records:
        r["status"] = "ingested" if any(i in str(r.get("question", "")) for i in ingested_ids) else "pending"
    return {"gaps": records, "total": len(records)}


@app.get("/api/gaps/count")
async def gaps_count():
    return {"count": _gap_detector.log.count()}


@app.delete("/api/gaps")
async def clear_gaps():
    _gap_detector.log.clear()
    ingested_path = Path("data/knowledge_gaps/ingested.jsonl")
    if ingested_path.exists():
        ingested_path.unlink()
    return {"status": "ok"}


@app.get("/chat", response_class=HTMLResponse)
async def chat_page_alt(request: Request):
    return templates.TemplateResponse(request, "chat.html")


@app.get("/health")
async def health():
    return {"status": "ok", "bm25_chunks": len(_bm25._docs)}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        _validate_input(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    session_id = get_or_create_session(req.session_id, req.tenant_id)
    history = get_history(session_id)

    return StreamingResponse(
        _chat_stream(session_id, req.question, history),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _chat_stream(session_id: str, question: str, history: list):
    add_message(session_id, "user", question)
    rewritten = rewrite_query(question, history)
    logger.info("Query: '%s' → Rewrite: '%s'", question, rewritten)

    sub_queries = decompose(rewritten)

    all_chunks = []
    for sq in sub_queries:
        bm25_res = _bm25.search(sq, top_k=10)
        emb_res = search_by_text(sq, top_k=10)
        fused = rrf_fusion(bm25_res, emb_res, top_k=15)
        all_chunks.extend(fused)

    all_chunks.sort(key=lambda c: c.get("rrf_score", 0), reverse=True)
    seen = set()
    deduped = []
    for c in all_chunks:
        if c["chunk_id"] not in seen:
            seen.add(c["chunk_id"])
            deduped.append(c)

    reranked = _reranker.rerank(rewritten, deduped)
    final_chunks = dynamic_top_k(reranked, min_k=2, max_k=5)

    sources = []
    for c in final_chunks:
        title = c.get("doc_title", c.get("doc_id", "未知"))
        sources.append({"doc_id": c.get("doc_id", ""), "title": title})

    if not final_chunks:
        yield _sse_error("RETRIEVAL_EMPTY", "未找到相关文档，基于通用知识回答", "llm_direct")
        messages = build_rag_messages(question, [], history)
    else:
        messages = build_rag_messages(rewritten, final_chunks, history)

    try:
        response_text = ""
        for token in _llm.generate_stream(messages):
            response_text += token
            yield _sse_token(token)
    except Exception as e:
        logger.error("LLM 生成失败: %s", e)
        yield _sse_token("服务暂时不可用，请稍后重试。")
        response_text = "服务暂时不可用，请稍后重试。"

    add_message(session_id, "assistant", response_text)

    import threading

    def _process_gap():
        gap = _gap_detector.check(question, response_text, final_chunks, session_id=session_id)
        if gap is None:
            return
        draft = generate_draft(question, response_text)
        if draft is None:
            return
        v = validate(draft.get("question", ""), draft.get("answer", ""))
        if not v.get("passed", False):
            return
        embedder = Embedder()
        auto_ingest(draft, _bm25, vector_store, embedder)

    threading.Thread(target=_process_gap, daemon=True).start()

    has_gap = not final_chunks or max((c.get("rerank_score", 0) for c in final_chunks), default=0) < 0.3
    yield _sse_done(sources, len(response_text), has_gap=has_gap)


from src.api.schemas import ChatRequest, FeedbackRequest, CreateSessionRequest


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest):
    from src.memory.short_term import get_history
    history = get_history(req.session_id)
    if not history:
        return {"status": "error", "message": "会话不存在"}

    last_answer = ""
    for msg in reversed(history):
        if msg["role"] == "assistant":
            last_answer = msg["content"]
            break

    import threading

    def _process_user_feedback():
        gap = _gap_detector.check(question=req.question, answer=last_answer, chunks=[], user_reported=True, session_id=req.session_id)
        if gap is None:
            return
        from src.knowledge.draft import generate_draft
        from src.knowledge.validator import validate
        from src.knowledge.auto_ingest import auto_ingest
        from src.ingestion.embedder import Embedder
        draft = generate_draft(req.question, last_answer)
        if draft is None:
            return
        v = validate(draft.get("question", ""), draft.get("answer", ""))
        if not v.get("passed", False):
            return
        embedder = Embedder()
        auto_ingest(draft, _bm25, vector_store, embedder)

    threading.Thread(target=_process_user_feedback, daemon=True).start()
    return {"status": "ok"}


@app.get("/api/sessions")
async def list_sessions():
    from src.memory.short_term import SESSION_TTL
    return {"sessions": [], "message": "会话列表功能（需 Redis 持久化）"}


@app.post("/api/sessions")
async def create_session(req: CreateSessionRequest):
    from src.memory.short_term import create_session as _create
    from src.api.dependencies import generate_session_id
    sid = req.session_id or generate_session_id()
    _create(sid, req.tenant_id)
    return {"session_id": sid, "status": "created"}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    from src.memory.short_term import get_history, get_tenant_id
    history = get_history(session_id)
    if not history:
        return {"error": "会话不存在"}, 404
    return {"session_id": session_id, "history": history, "tenant_id": get_tenant_id(session_id)}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    from src.memory.short_term import delete_session as _delete
    _delete(session_id)
    return {"status": "deleted"}


@app.get("/api/documents")
async def list_documents():
    from pathlib import Path
    docs = []
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        for f in sorted(raw_dir.iterdir()):
            if f.suffix.lower() in {".md", ".txt", ".pdf", ".docx"}:
                size = f.stat().st_size
                docs.append({"filename": f.name, "size": size, "source_type": f.suffix[1:].upper()})
    return {"documents": docs, "total": len(docs)}


@app.post("/api/documents/upload")
async def upload_document():
    return {"status": "error", "message": "上传功能需实现文件接收逻辑（本 demo 暂未开放）"}


@app.post("/api/evaluate/run")
async def run_evaluation():
    from src.evaluate.runner import start_evaluation

    task_id = start_evaluation(
        dataset_path="data/qa_dataset/eval.jsonl",
        k=10,
        bm25=_bm25,
        embedder=None,
        vector_store=vector_store if vector_store.size > 0 else None,
    )
    return {"task_id": task_id, "status": "pending"}


@app.get("/api/evaluate/status/{task_id}")
async def evaluate_status(task_id: str):
    from src.evaluate.runner import get_task

    task = get_task(task_id)
    if task is None:
        return {"error": "任务不存在"}, 404
    return task


def _validate_input(req: ChatRequest):
    import re
    if len(req.question) > 1000:
        raise ValueError("问题长度不能超过1000字")
    dangerous = re.search(r"(?i)(SELECT|DROP|DELETE|INSERT|UPDATE|ALTER|EXEC|<script|onerror)", req.question)
    if dangerous:
        raise ValueError("输入包含非法字符")


def _sse_token(token: str) -> str:
    return f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"


def _sse_done(sources: list, tokens_used: int, has_gap: bool = False) -> str:
    return f"event: done\ndata: {json.dumps({'sources': sources, 'tokens_used': tokens_used, 'has_gap': has_gap}, ensure_ascii=False)}\n\n"


def _sse_error(code: str, message: str, fallback: str) -> str:
    return f"event: error\ndata: {json.dumps({'code': code, 'message': message, 'fallback': fallback}, ensure_ascii=False)}\n\n"
