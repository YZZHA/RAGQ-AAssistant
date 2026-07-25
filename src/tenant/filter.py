# input:  tenant_id + collection schema
# output: Milvus query filter expression
# pos:    多租户 → 构建 metadata 过滤条件，隔离租户数据
