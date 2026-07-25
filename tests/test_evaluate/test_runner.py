import time
from src.evaluate.runner import start_evaluation, get_task


class TestRunnerCore:
    def test_start_returns_task_id(self):
        tid = start_evaluation(dataset_path="data/qa_dataset/eval.jsonl")
        assert tid.startswith("eval_")

    def test_get_task_returns_status(self):
        tid = start_evaluation(dataset_path="data/qa_dataset/eval.jsonl")
        task = get_task(tid)
        assert task is not None
        assert task["status"] in ("pending", "running", "completed")

    def test_get_nonexistent_task(self):
        assert get_task("nonexistent") is None
