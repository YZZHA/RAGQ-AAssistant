# input:  eval dataset + BM25/FAISS/fusion instances
# output: evaluation metrics report (Recall@K, MRR, NDCG)
# pos:    评估层 → 异步评估运行器，编排检索→计算指标→聚合报告

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set
from threading import Thread

from src.evaluate.metrics import evaluate_retrieval
from src.core.logging import logger


_tasks: Dict[str, Dict] = {}


def _load_dataset(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        return []
    items = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def start_evaluation(
    dataset_path: str = "data/qa_dataset/eval.jsonl",
    k: int = 10,
    bm25=None,
    embedder=None,
    vector_store=None,
) -> str:
    task_id = f"eval_{uuid.uuid4().hex[:8]}"
    _tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0,
        "total": 0,
        "metrics": {},
        "error": None,
        "created_at": time.time(),
    }

    def _run():
        try:
            _tasks[task_id]["status"] = "running"
            items = _load_dataset(dataset_path)
            _tasks[task_id]["total"] = len(items)

            if not items:
                _tasks[task_id]["status"] = "completed"
                _tasks[task_id]["metrics"] = {"error": "数据集为空"}
                return

            all_metrics = []
            for i, item in enumerate(items):
                question = item.get("question", "")
                relevant_docs = set(item.get("relevant_docs", []))

                bm25_res = bm25.search(question, top_k=k) if bm25 else []
                emb_res = []
                if embedder and vector_store and vector_store.size > 0:
                    vec = embedder.encode(question)
                    emb_res = vector_store.search(vec, top_k=k)

                from src.retrieval.fusion import rrf_fusion
                fused = rrf_fusion(bm25_res, emb_res, top_k=k)
                retrieved_ids = [r.get("doc_id", "") for r in fused]

                metrics = evaluate_retrieval(
                    [{"doc_id": d} for d in retrieved_ids],
                    relevant_docs,
                    ks=[1, 3, 5, k],
                )
                all_metrics.append(metrics)
                _tasks[task_id]["progress"] = i + 1

            _tasks[task_id]["metrics"] = _aggregate(all_metrics)
            _tasks[task_id]["status"] = "completed"
            logger.info("评估完成: %d 条, 耗时 %.1fs", len(items), time.time() - _tasks[task_id]["created_at"])

        except Exception as e:
            _tasks[task_id]["status"] = "failed"
            _tasks[task_id]["error"] = str(e)
            logger.error("评估失败: %s", e)

    Thread(target=_run, daemon=True).start()
    return task_id


def get_task(task_id: str) -> Optional[Dict]:
    return _tasks.get(task_id)


def _aggregate(all_metrics: List[Dict]) -> Dict:
    if not all_metrics:
        return {}
    keys = all_metrics[0].keys()
    result = {}
    for key in keys:
        values = [m[key] for m in all_metrics]
        result[key] = round(sum(values) / len(values), 4)
    result["total_questions"] = len(all_metrics)
    return result
