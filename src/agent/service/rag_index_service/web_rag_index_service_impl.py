from langchain_core.documents import Document

from agent.component.doc_loader.web_doc_loader_impl import WebDocLoaderImpl
from agent.service.rag_index_service.rag_index_service_impl import RagIndexServiceImpl


class WebRagIndexServiceImpl(RagIndexServiceImpl):

    def load_doc(self, path: str) -> list[Document]:
        """从网络地址加载"""
        web_doc_loader = WebDocLoaderImpl()
        docs = web_doc_loader.load_doc(path)
        print(f'从网页中加载导：\n{docs}')
        return docs
