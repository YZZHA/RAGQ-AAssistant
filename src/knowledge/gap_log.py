# input:  gap records (question, answer, trigger info)
# output: JSONL file at data/knowledge_gaps/gaps.jsonl
# pos:    知识生产 → 缺口持久化，记录所有未命中 Q&A

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

GAPS_PATH = Path("data/knowledge_gaps/gaps.jsonl")


class GapLog:
    def __init__(self, path: str = str(GAPS_PATH)):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Dict) -> bool:
        record["created_at"] = record.get("created_at", time.time())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True

    def read_all(self) -> List[Dict]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def count(self) -> int:
        return len(self.read_all())

    def clear(self):
        if self.path.exists():
            self.path.unlink()
