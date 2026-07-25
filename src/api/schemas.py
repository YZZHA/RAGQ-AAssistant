# input:  pydantic BaseModel
# output: request/response 类型定义
# pos:    API 层 → 数据契约，被 routes 和客户端引用

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(default="", description="会话ID，空则自动创建")
    question: str = Field(..., min_length=1, max_length=1000, description="用户问题")
    tenant_id: str = Field(default="default", description="租户标识")


class ChatError(BaseModel):
    code: str
    message: str
    fallback: str = ""


class TokenEvent(BaseModel):
    token: str


class DoneEvent(BaseModel):
    sources: list = []
    tokens_used: int = 0


class SourceItem(BaseModel):
    doc_id: str
    title: str = ""


class FeedbackRequest(BaseModel):
    session_id: str
    question: str


class CreateSessionRequest(BaseModel):
    session_id: str = Field(default="", description="会话ID，空则自动生成")
    tenant_id: str = Field(default="default", description="租户标识")
