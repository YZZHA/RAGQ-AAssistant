# input:  source directory path, CLI args
# output: ingestion report printed to stdout
# pos:    CLI 脚本 → 批量文档摄入入口

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.pipeline import run_ingestion
from src.ingestion.embedder import Embedder
from src.retrieval.bm25 import BM25Index
from src.retrieval.embedding import vector_store


def main():
    parser = argparse.ArgumentParser(description="批量文档摄入")
    parser.add_argument("--source", default="data/raw", help="文档目录路径")
    parser.add_argument("--collection", default="fdi_knowledge_base", help="Milvus collection 名")
    parser.add_argument("--drop-existing", action="store_true", help="重建集合（全量模式）")
    parser.add_argument("--skip-vector", action="store_true", help="跳过向量索引构建")
    args = parser.parse_args()

    print(f"摄入源: {args.source}")
    print(f"Collection: {args.collection}")

    result = run_ingestion(
        source_dir=args.source,
        collection_name=args.collection,
        create_collection=not args.skip_vector,
        drop_existing=args.drop_existing,
    )

    print(f"\n完成: {result['documents_processed']}/{result['documents_found']} 文档")
    print(f"Chunks: {result['chunks_total']}")
    print(f"向量写入: {result['vectors_inserted']}")
    if result["errors"]:
        print(f"错误: {len(result['errors'])} 个")
        for e in result["errors"]:
            print(f"  ✗ {e}")


if __name__ == "__main__":
    main()
