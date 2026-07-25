# input:  chunks with dense vectors, Milvus client
# output: Milvus collection (created/inserted)
# pos:    摄入管道 → 索引写入，全量 drop&recreate / 增量 upsert

from typing import List, Dict, Optional

from src.core.config import settings


COLLECTION_NAME = "fdi_knowledge_base"


def _build_schema():
    from pymilvus import CollectionSchema, DataType, FieldSchema

    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="doc_title", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="parent_chunk_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="updated_at", dtype=DataType.INT64),
        FieldSchema(name="version", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="heading_chain", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=384),
    ]
    return CollectionSchema(fields, description="FDI knowledge base collection")


def _connect():
    from pymilvus import connections

    from src.core.milvus_helper import get_milvus_uri
    uri = get_milvus_uri()
    connections.connect(alias="default", uri=uri)


def create_collection(collection_name: str = COLLECTION_NAME, drop_if_exists: bool = True):
    from pymilvus import Collection, utility

    _connect()
    if drop_if_exists and utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    schema = _build_schema()
    collection = Collection(name=collection_name, schema=schema)
    collection.create_index(
        field_name="dense_vector",
        index_params={"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}},
    )
    try:
        collection.create_index(
            field_name="tenant_id",
            index_name="idx_tenant",
            index_params={"index_type": "INVERTED"},
        )
    except Exception:
        pass
    collection.load()
    return collection_name


def insert_chunks(chunks: List[Dict], collection_name: str = COLLECTION_NAME):
    from pymilvus import Collection

    _connect()
    collection = Collection(name=collection_name)

    entities = []
    for c in chunks:
        entities.append({
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "doc_title": c.get("doc_title", ""),
            "parent_chunk_id": c.get("parent_chunk_id", ""),
            "content": c["content"],
            "tenant_id": c.get("tenant_id", "default"),
            "source_type": c.get("source_type", "MD"),
            "updated_at": c.get("updated_at", 0),
            "version": c.get("version", "1.0"),
            "heading_chain": ",".join(c.get("heading_chain", [])),
            "dense_vector": c["dense_vector"],
        })

    result = collection.insert(entities)
    collection.flush()
    return {"insert_count": len(entities), "collection": collection_name}


def get_collection(collection_name: str = COLLECTION_NAME) -> Optional[object]:
    from pymilvus import Collection, utility

    _connect()
    if not utility.has_collection(collection_name):
        return None
    return Collection(name=collection_name)
