# api/ — FastAPI 接口层
对外暴露 HTTP API，接收用户请求、调用下游模块、返回 SSE 流式响应。
依赖: core/config, retrieval, generator, memory

## 文件清单
| 文件 | 地位 | 功能 |
|------|------|------|
| routes.py | 核心 | API 路由定义，POST /api/chat 等 |
| schemas.py | 核心 | Pydantic 请求/响应模型 |
| dependencies.py | 辅助 | 依赖注入（获取组件实例） |
