# input:  eval dataset path, CLI args
# output: DeepEval test run output printed
# pos:    CLI 脚本 → 运行 RAG 评估

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


def main():
    parser = argparse.ArgumentParser(description="运行 RAG 评估")
    parser.add_argument("--dataset", default="data/qa_dataset/eval.jsonl", help="评估数据集路径")
    parser.add_argument("--test-file", default="tests/test_evaluate/test_rag_eval.py", help="测试文件路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--k", type=int, default=10, help="仅跑前 K 条")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"数据集不存在: {args.dataset}")
        sys.exit(1)

    count = sum(1 for _ in dataset_path.open("r", encoding="utf-8") if _.strip())
    print(f"数据集: {args.dataset} ({count} 条)")
    print(f"测试文件: {args.test_file}")
    print()

    pytest_args = [args.test_file, "-v" if args.verbose else "-q", "--tb=short"]
    exit_code = pytest.main(pytest_args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
