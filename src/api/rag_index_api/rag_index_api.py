from fastapi import APIRouter

from agent.retrieval.doc_loader.web_doc_loader_impl import WebDocLoaderImpl
from agent.service.rag_index_service.rag_index_service_impl import RagIndexServiceImpl

rag_index_router = APIRouter()

@rag_index_router.get("/index_web_doc")
async def index_web_doc(url: str) -> bool:
    """RAG 建立索引服务"""
    rag_index_service = RagIndexServiceImpl()
    rag_index_service.set_doc_loader(WebDocLoaderImpl())
    ids = rag_index_service.rag_index(url)
    print(ids)
    return True